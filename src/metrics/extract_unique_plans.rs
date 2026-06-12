//! Extract distinct *partitions* (label-invariant) and re-encode them as a Standard BEN file. Dedup
//! key is xxh3-128 of the canonical relabeling (districts numbered in order of first appearance);
//! the assignment that gets written is the **first-seen original**, so labels in the output match
//! how the plan first appeared in the input.

use crate::metrics::canonical::canonical_hash;
use crate::pipeline::{count_frames, run_sequential_accepted_frames, AssignmentLengthCheck};
use ben::encode::BenEncoder;
use ben::BenVariant;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

/// `Write` handle sharing one `BufWriter<File>` between the [`BenEncoder`] (which must own its
/// writer) and this module (which must flush explicitly at the end — `BufWriter`'s `Drop` flushes
/// but swallows errors, and a swallowed disk-full would mean a silently truncated output BEN).
struct SharedWriter(Rc<RefCell<BufWriter<File>>>);

impl Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.borrow_mut().flush()
    }
}

type Sink = (Rc<RefCell<BufWriter<File>>>, BenEncoder<SharedWriter>);

/// Open the output file and wrap it in an encoder. `BenEncoder::new` writes the BEN header
/// immediately, so this must not run before the first assignment decodes successfully (or until a
/// zero-frame run has completed) — a failed run must leave no output file.
fn make_sink(out_file_name: &str) -> io::Result<Sink> {
    let file = File::create(out_file_name)?;
    let shared = Rc::new(RefCell::new(BufWriter::new(file)));
    let encoder = BenEncoder::new(SharedWriter(Rc::clone(&shared)), BenVariant::Standard);
    Ok((shared, encoder))
}

pub fn extract_unique_plans(
    in_file_name: &str,
    out_file_name: &str,
    show_progress: bool,
) -> crate::error::Result<()> {
    let basename = Path::new(in_file_name)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    log::info!("Reading {:?}...", basename);

    let total_frames = count_frames(in_file_name)?;
    log::info!("Found {} accepted plans in {:?}", total_frames, basename);

    let mut sink: Option<Sink> = None;
    let mut seen: HashSet<u128> = HashSet::new();
    let mut written: u64 = 0;

    let total = run_sequential_accepted_frames(
        in_file_name,
        total_frames,
        None,
        // No graph is loaded in this mode, so the driver establishes the expected assignment
        // length from the first frame — mixed-length frames mean a corrupt ensemble and would
        // otherwise just hash as distinct plans. The dedup is label-invariant by design, so the
        // fixed district-set check stays off.
        AssignmentLengthCheck::UniformWithinFile,
        None,
        show_progress,
        |frame| {
            let hash = canonical_hash(&frame.assignment);
            if seen.insert(hash) {
                if sink.is_none() {
                    sink = Some(make_sink(out_file_name)?);
                }
                let (_, encoder) = sink
                    .as_mut()
                    .expect("sink should be initialized before the first unique plan is written");
                encoder.write_assignment(frame.assignment)?;
                written += 1;
            }
            Ok(())
        },
    )?;

    // A completed zero-frame run still emits a valid (header-only) Standard BEN.
    let (shared, mut encoder) = match sink {
        Some(sink) => sink,
        None => make_sink(out_file_name)?,
    };
    encoder.finish()?;
    drop(encoder);
    shared.borrow_mut().flush()?;

    log::info!(
        "Unique plans: {} (out of {} accepted frames)",
        written,
        total
    );
    log::info!("Wrote {}", out_file_name);
    Ok(())
}
