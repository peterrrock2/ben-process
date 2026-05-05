#!/usr/bin/env bash
# ben-concat — concatenate same-variant BEN files to stdout.
#
# BEN files have a 17-byte ASCII header ("STANDARD BEN FILE" or "MKVCHAIN
# BEN FILE") followed by self-describing frames. Files that share a variant
# can be combined by emitting the first file in full and stripping the
# header from each subsequent one.
#
# Usage:
#   ben-concat first.jsonl.ben second.jsonl.ben [...] > combined.jsonl.ben
#
# Caveats:
#   * All inputs must share the same variant (Standard or MkvChain).
#   * All inputs must be from the same graph (same node count). The format
#     does not store node count, so this script can't check it for you.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 file1.jsonl.ben [file2.jsonl.ben ...] > combined.jsonl.ben" >&2
    exit 1
fi

read_header() {
    head -c 17 -- "$1"
}

expected=$(read_header "$1")
case "$expected" in
    "STANDARD BEN FILE"|"MKVCHAIN BEN FILE") ;;
    *)
        echo "error: $1 is not a BEN file (header: '$expected')" >&2
        exit 1
        ;;
esac

for f in "$@"; do
    [[ -r "$f" ]] || { echo "error: cannot read $f" >&2; exit 1; }
    h=$(read_header "$f")
    if [[ "$h" != "$expected" ]]; then
        echo "error: $f has variant '$h', expected '$expected'" >&2
        exit 1
    fi
done

# First file: copy in full. Subsequent files: skip the 17-byte header.
cat -- "$1"
shift
for f in "$@"; do
    tail -c +18 -- "$f"
done
