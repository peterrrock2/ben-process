//! Input detection and the [`BenSource`] abstraction.
//!
//! `ben-process` accepts three on-disk inputs, distinguished by sniffing the leading magic bytes
//! (the file extension is never consulted):
//!
//! - a plain BEN file (`Standard` / `MkvChain` / `TwoDelta`, all auto-detected by the reader from
//!   its 17-byte banner),
//! - an XBEN file (an xz-framed BEN32 columnar stream), and
//! - a `.bendl` bundle: a seekable, CRC32C-checked container wrapping a BEN/XBEN assignment stream
//!   plus front-loaded assets (`graph.json`, metadata, ...).
//!
//! [`resolve`] turns a path into a [`BenSource`] (what to read) plus any embedded `graph.json`
//! bytes. A [`BenSource`] holds no live reader: every read pass re-opens the file fresh and, for a
//! bundle, bounds it to the embedded stream's declared `(offset, len)` range via `ExactLen`. The
//! `Box<dyn Read + Send>` each pass owns is `'static`, so there is no borrow of a short-lived
//! bundle reader to thread through the pipeline's multiple open passes.

use ben::format::banners::has_known_banner_prefix;
use ben::io::bundle::format::{ASSET_TYPE_GRAPH, BENDL_MAGIC};
use ben::io::bundle::{BendlReader, ExactLen};
use ben::io::reader::{BenStreamFrameReader, BenStreamReader, BenWireFormat};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// xz stream magic (`FD 37 7A 58 5A 00`). `xz2` does not export it, so it is defined here: an XBEN
/// file is an xz-framed BEN stream, so its first bytes are xz framing and the BEN banner lives
/// inside the compressed stream.
const XZ_MAGIC: [u8; 6] = [0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00];

/// What the leading magic bytes say the transport is. The BEN frame variant
/// (`Standard`/`MkvChain`/`TwoDelta`) is still auto-detected later by the reader from its banner;
/// for XBEN the embedded variant is only known after decompression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    Bundle,
    PlainBen,
    Xben,
}

/// Classify an input by its first up-to-17 bytes. The extension is ignored.
fn sniff(path: &Path) -> io::Result<InputKind> {
    let mut file = File::open(path)?;
    let mut buf = [0u8; 17];
    let n = read_up_to(&mut file, &mut buf)?;
    let buf = &buf[..n];

    if buf.starts_with(&BENDL_MAGIC) {
        Ok(InputKind::Bundle)
    } else if has_known_banner_prefix(buf) {
        Ok(InputKind::PlainBen)
    } else if buf.starts_with(&XZ_MAGIC) {
        Ok(InputKind::Xben)
    } else {
        Err(invalid_data(format!(
            "{:?} is not a recognized input: leading bytes {:02X?} match no known \
             BEN / XBEN / .bendl magic",
            path, buf
        )))
    }
}

/// Read until `buf` is full or EOF, returning the number of bytes read. A single `read` may return
/// fewer bytes than requested even mid-file, so loop; a short file (a valid xz header is only 6
/// bytes) just yields a shorter prefix for the magic checks.
fn read_up_to(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// Where a run reads its assignment stream from. Holds no live reader: each pass re-opens the file.
pub enum BenSource {
    /// A whole BEN or XBEN file on disk.
    File { path: PathBuf, wire: BenWireFormat },
    /// The embedded assignment-stream byte range of a `.bendl` bundle.
    Bundle {
        path: PathBuf,
        offset: u64,
        len: u64,
        wire: BenWireFormat,
        /// Finalized-bundle sample count, a progress/preallocation hint only. It never decides
        /// which frames are processed.
        header_samples: Option<usize>,
    },
}

impl BenSource {
    /// The on-disk path that was opened (the BEN/XBEN file, or the `.bendl` bundle). Output names
    /// and the `Reading ...` status line derive from this.
    pub fn path(&self) -> &Path {
        match self {
            BenSource::File { path, .. } | BenSource::Bundle { path, .. } => path,
        }
    }

    /// Open a fresh record reader, owning its `File` outright. `Box<dyn Read + Send>` is `'static`,
    /// so no borrow of a short-lived bundle reader is involved; for a bundle the file is bounded to
    /// the declared stream region.
    pub fn open_reader(&self) -> io::Result<BenStreamReader<Box<dyn Read + Send>>> {
        let (boxed, wire): (Box<dyn Read + Send>, BenWireFormat) = match self {
            BenSource::File { path, wire } => (Box::new(File::open(path)?), *wire),
            BenSource::Bundle {
                path,
                offset,
                len,
                wire,
                ..
            } => {
                let mut f = File::open(path)?;
                f.seek(SeekFrom::Start(*offset))?;
                // `resolve` already checked offset + len <= file length; `ExactLen` keeps this pass
                // bounded to exactly the declared stream region.
                (Box::new(ExactLen::bounded(f, *len)), *wire)
            }
        };
        let reader = match wire {
            BenWireFormat::Ben => BenStreamReader::from_ben(boxed),
            BenWireFormat::XBen => BenStreamReader::from_xben(boxed),
        }
        .map_err(io::Error::from)?;
        // We drive our own progress bar, so silence the reader's own spinner.
        Ok(reader.silent(true))
    }

    /// Open a fresh self-contained-frame reader (the serial pop; the parallel expand happens
    /// downstream).
    pub fn open_frames(&self) -> io::Result<BenStreamFrameReader<Box<dyn Read + Send>>> {
        Ok(self.open_reader()?.into_frames())
    }

    /// Total sample count (sum of repetition counts). For a finalized bundle this is the header
    /// hint; otherwise it walks the stream. Used only to size the progress bar.
    pub fn count_samples(&self) -> io::Result<usize> {
        if let BenSource::Bundle {
            header_samples: Some(n),
            ..
        } = self
        {
            return Ok(*n);
        }
        self.open_reader()?.count_samples()
    }

    /// Number of accepted frames (independent of repetition counts). Always walks the stream, so it
    /// is exact even for a finalized bundle whose header count is only a hint.
    pub fn count_frames(&self) -> io::Result<usize> {
        let mut n = 0usize;
        for frame in self.open_frames()? {
            frame?;
            n += 1;
        }
        Ok(n)
    }
}

/// A resolved input: the [`BenSource`] to read, plus any verified `graph.json` bytes the bundle
/// carried.
pub struct ResolvedInput {
    pub source: BenSource,
    pub embedded_graph: Option<Vec<u8>>,
}

/// Sniff `ben_file` and resolve it to a [`BenSource`] (and, for a bundle, its embedded graph). The
/// extension is never consulted.
pub fn resolve(ben_file: &str) -> crate::error::Result<ResolvedInput> {
    let path = PathBuf::from(ben_file);
    let resolved = match sniff(&path)? {
        InputKind::PlainBen => ResolvedInput {
            source: BenSource::File {
                path,
                wire: BenWireFormat::Ben,
            },
            embedded_graph: None,
        },
        InputKind::Xben => ResolvedInput {
            source: BenSource::File {
                path,
                wire: BenWireFormat::XBen,
            },
            embedded_graph: None,
        },
        InputKind::Bundle => resolve_bundle(path)?,
    };
    Ok(resolved)
}

fn resolve_bundle(path: PathBuf) -> crate::error::Result<ResolvedInput> {
    let file = File::open(&path)?;
    let file_len = file.metadata()?.len();
    // `BendlReader::open` fails with `BendlFormatError`, which does convert to `io::Error`.
    let mut reader = BendlReader::open(file).map_err(io::Error::from)?;

    // The embedded stream is one contiguous range. Validate it against the actual file length: a
    // BEN frame reader treats EOF at a frame boundary as a clean end, so a finalized bundle whose
    // declared range runs past EOF must fail here, not silently decode only the prefix.
    let (offset, len) = reader.assignment_stream_range()?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| invalid_data(format!("bundle stream range [{offset}, +{len}) overflows")))?;
    if end > file_len {
        return Err(invalid_data(format!(
            "bundle assignment stream range [{offset}, {end}) exceeds file length {file_len}"
        ))
        .into());
    }

    let wire: BenWireFormat = reader
        .assignment_format()
        .ok_or_else(|| invalid_data("bundle has no recognized assignment-stream format".into()))?
        .into();

    let header_samples = match reader.sample_count() {
        Some(n) => Some(usize::try_from(n).map_err(|_| {
            invalid_data(format!(
                "bundle sample_count {n} is negative or out of range"
            ))
        })?),
        None => None,
    };

    // The graph asset is read through the CRC32C-verified `asset_bytes` path, so a corrupt asset is
    // a hard error. Clone the directory entry to drop the immutable borrow before the `&mut` read.
    let graph_entry = reader.find_asset_by_type(ASSET_TYPE_GRAPH).cloned();
    let embedded_graph = match graph_entry {
        // `BendlReadError` has no `From<_> for io::Error`, so bridge it by message.
        Some(entry) => Some(
            reader
                .asset_bytes(&entry)
                .map_err(|e| invalid_data(e.to_string()))?,
        ),
        None => None,
    };

    Ok(ResolvedInput {
        source: BenSource::Bundle {
            path,
            offset,
            len,
            wire,
            header_samples,
        },
        embedded_graph,
    })
}

fn invalid_data(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::{sniff, InputKind};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn sniff_bytes(bytes: &[u8]) -> std::io::Result<InputKind> {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        sniff(f.path())
    }

    #[test]
    fn sniff_detects_bendl_magic() {
        assert_eq!(
            sniff_bytes(b"BENDL\0\0\x01 trailing").unwrap(),
            InputKind::Bundle
        );
    }

    #[test]
    fn sniff_detects_each_ben_banner() {
        assert_eq!(
            sniff_bytes(b"STANDARD BEN FILE").unwrap(),
            InputKind::PlainBen
        );
        assert_eq!(
            sniff_bytes(b"MKVCHAIN BEN FILE").unwrap(),
            InputKind::PlainBen
        );
        assert_eq!(
            sniff_bytes(b"TWODELTA BEN FILE").unwrap(),
            InputKind::PlainBen
        );
    }

    #[test]
    fn sniff_detects_xz_magic() {
        // The 6-byte xz header is enough even when the rest of the (compressed) stream is absent.
        assert_eq!(
            sniff_bytes(&[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00]).unwrap(),
            InputKind::Xben
        );
    }

    #[test]
    fn sniff_rejects_unknown_leading_bytes() {
        let err = sniff_bytes(b"not a real ensemble file").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn sniff_rejects_empty_file() {
        let err = sniff_bytes(b"").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
