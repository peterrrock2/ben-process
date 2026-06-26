# Ben Process

`ben-process` is a CLI for processing BEN files from the
[`binary-ensemble`](https://crates.io/crates/binary-ensemble) crate.

It currently supports:
- per-key district tallies
- cut-edge counts
- district-level Polsby-Popper scores
- changed-assignment counts
- region split and piece metrics
- unique-plan counting
- unique-plan extraction

## Build

```bash
cargo build --release
```

Or run directly with:

```bash
cargo run -- <args...>
```

## CLI

```text
ben-process <COMMAND> [OPTIONS] --ben-file <BEN_FILE>
```

Pick a command (one per metric), then pass its options. Run `ben-process <COMMAND> --help` to see the
exact flags a given command accepts.

Common options (every command):
- `-b`, `--ben-file <BEN_FILE>` (`.ben`, `.xben`, or `.bendl`; detected by content, not extension)
- `--output-dir <OUTPUT_DIR>`
- `-v`, `--verbose` (enable info-level status logging; off by default. `RUST_LOG` overrides the level)
- `-q`, `--quiet` (suppress the progress bar; errors and warnings still print)

Per-command options:
- `-g`, `--graph-file <GRAPH_FILE>` (graph-driven commands; optional when the input is a `.bendl` that embeds a graph)
- `-k`, `--keys <KEYS>...` (`tally-keys`, `region-splits`, `region-pieces`)
- `--max-samples <N>` (per-sample commands; stop after the first `N` expanded samples)
- `--high-compression` (Parquet-writing commands; Brotli instead of Snappy)

Commands:
- `tally-keys`
- `cut-edges`
- `polsby-popper`
- `changed-assignments`
- `region-splits`
- `region-pieces`
- `unique-plans`
- `extract-unique-plans`

## Input formats

`--ben-file` accepts three inputs, detected by their contents (the file extension is ignored, so any
name works):

- a **plain BEN** file, including all three encodings (`Standard`, `MkvChain`, and the newer
  `TwoDelta`);
- an **XBEN** file (xz-compressed BEN); and
- a **`.bendl` bundle**, which packages the assignment stream together with its dual graph
  (`graph.json`) and metadata in one seekable, checksummed file.

When the input is a `.bendl` that carries a `graph.json`, the graph-driven modes (`tally-keys`,
`cut-edges`, `polsby-popper`, `region-splits`, `region-pieces`) use that embedded graph, so
`--graph-file` becomes optional. An explicit `--graph-file` always takes precedence over the embedded
graph. Output file names derive from the input's base name (`plans.bendl` →
`plans_cut_edges.parquet`), never from the stream inside the bundle.

## Modes

### `tally-keys`

Tallies one or more numeric node attributes by district for each accepted plan.

Required:
- `--keys <KEYS>...`
- graph input: `--graph-file <GRAPH_FILE>` unless the `.bendl` input embeds `graph.json`

Output:
- creates a directory named `<ben_stem>_tallies`
- writes one parquet file per key:
  `<key>_tally_<ben_stem>.parquet`
- with `--max-samples <N>`:
  `<key>_tally_up_to_<N>_<ben_stem>.parquet`
- each file contains:
  `step`, `n_reps`, `accepted_count`, `district_*`

If `--output-dir` is set, the output layout is:

```text
<output_dir>/<ben_stem>_tallies/<key>_tally_<ben_stem>.parquet
```

Example:

```bash
ben-process tally-keys \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys pop area vap \
  --output-dir out
```

This produces files like:

```text
out/plans_tallies/pop_tally_plans.parquet
out/plans_tallies/area_tally_plans.parquet
out/plans_tallies/vap_tally_plans.parquet
```

### `cut-edges`

Counts cut edges for each accepted plan. By default this is an unweighted cut-edge count.

Required:
- graph input: `--graph-file <GRAPH_FILE>` unless the `.bendl` input embeds `graph.json`

Optional:
- `--edge-weight-key <EDGE_WEIGHT_KEY>`

Notes:
- with `--edge-weight-key`, edges missing the key (or carrying JSON `null`) fall back to a
  default weight of 1.0; a key that matches **zero** edges, a present value that doesn't parse
  to a finite number, or two directions of an edge that disagree on the value are all hard
  errors rather than silent fallbacks

Output:
- `<ben_stem>_cut_edges.parquet`
- columns:
  `step`, `n_reps`, `accepted_count`, `cut_edges`

Examples:

```bash
ben-process cut-edges \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben
```

```bash
ben-process cut-edges \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --edge-weight-key weight \
  --output-dir out
```

### `polsby-popper`

Computes district-level Polsby-Popper scores for each accepted plan.

Required:
- graph input: `--graph-file <GRAPH_FILE>` unless the `.bendl` input embeds `graph.json`
- one of:
  default GerryChain geometry keys
  `--perim-key <PERIM_KEY>`
  `--boundary-perim-key <BOUNDARY_PERIM_KEY>`

Output:
- `<ben_stem>_polsby_popper.parquet`
- columns:
  `step`, `n_reps`, `accepted_count`, `district_*`

Notes:
- default column names are:
  `area`, `boundary_perim`, and `shared_perim`
- with no Polsby-Popper key flags, `ben-process` assumes a standard
  GerryChain dual-graph export and derives node total perimeter from
  `boundary_perim + sum(shared perimeter on incident edges)`
- if `--perim-key` is provided, node total perimeter comes directly from
  that column instead
- the first assignment fixes the output district columns; if later plans
  introduce unseen district ids, the run fails fast
- a shared-perimeter key that matches zero edges is a hard error (it would
  silently produce boundary-only perimeters); a boundary-perimeter key that
  matches zero nodes warns but proceeds (legitimate for boundary-free grid
  graphs)
- a district whose computed perimeter is zero or negative fails the run —
  that means the geometry keys are wrong, and scoring it 0.0 would bury the
  data problem in plausible-looking output

Examples:

Use the default GerryChain geometry keys:

```bash
ben-process polsby-popper \
  --graph-file data/dual_graph.json \
  --ben-file runs/plans.jsonl.ben \
  --output-dir out
```

```bash
ben-process polsby-popper \
  --graph-file data/dual_graph.json \
  --ben-file runs/plans.jsonl.ben \
  --perim-key perim \
  --output-dir out
```

```bash
ben-process polsby-popper \
  --graph-file data/dual_graph.json \
  --ben-file runs/plans.jsonl.ben \
  --boundary-perim-key boundary_perim \
  --output-dir out
```

### `changed-assignments`

Counts how many times each node changes assignment across accepted plans.

Optional:
- `--normalize`
- `--max-accepted <MAX_ACCEPTED>`
- `--randomize-reassignments`
- `--seed <SEED>`

Output:
- `<ben_stem>_accept_<N>_changed_assignments.parquet`
- columns: `node`, `changed_assignments` (one row per node)

Notes:
- `N` is the number of accepted frames considered
- for MkvChain BEN files, this mode counts accepted frames, not repeated samples
- `--randomize-reassignments` is only appropriate for merge-split MCMC runs
- `--seed` makes `--randomize-reassignments` reproducible; without it the run is OS-seeded

Examples:

```bash
ben-process changed-assignments \
  --ben-file runs/plans.jsonl.ben
```

```bash
ben-process changed-assignments \
  --ben-file runs/plans.jsonl.ben \
  --normalize \
  --max-accepted 50000 \
  --output-dir out
```

### `region-splits`

Counts, for each requested region key, how many regions are split across more than one district in each plan.

Required:
- `--keys <KEYS>...`
- graph input: `--graph-file <GRAPH_FILE>` unless the `.bendl` input embeds `graph.json`

Output:
- `<ben_stem>_region_splits.parquet`
- columns:
  `step`, `n_reps`, `accepted_count`, `region_key`, `region_splits`

Example:

```bash
ben-process region-splits \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys county municipality \
  --output-dir out
```

### `region-pieces`

Counts, for each requested region key, the total number of district pieces across all regions in each plan.

Required:
- `--keys <KEYS>...`
- graph input: `--graph-file <GRAPH_FILE>` unless the `.bendl` input embeds `graph.json`

Output:
- `<ben_stem>_region_pieces.parquet`
- columns:
  `step`, `n_reps`, `accepted_count`, `region_key`, `region_pieces`

Example:

```bash
ben-process region-pieces \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys county \
  --output-dir out
```

### `unique-plans`

Counts label-invariant unique partitions in a BEN file.

Notes:
- deduplication is hash-based: each plan is canonically relabeled (districts numbered in order
  of first appearance) and hashed with xxh3-128, so two distinct partitions colliding on a
  digest would be silently counted as one. The xxh3-128 birthday bound makes this negligible
  for real ensembles — at 10^9 distinct plans the collision probability is ~1.5e-21 — but
  xxh3 is not cryptographic, so adversarially constructed inputs could collide deliberately
- all plans in the file must have the same assignment length; a mixed-length file is rejected
  as corrupt (this mode loads no graph, so the usual graph-length check does not apply)

Output:
- `<ben_stem>_unique_plans.parquet`
- a single row with columns:
  `unique_plans`, `total_accepted_frames`

Example:

```bash
ben-process unique-plans \
  --ben-file runs/plans.jsonl.ben \
  --output-dir out
```

### `extract-unique-plans`

Extracts the first occurrence of each label-invariant unique partition and writes them back out as a Standard BEN file.

Notes:
- the same hash-based deduplication and mixed-length rejection as `unique-plans` apply

Output:
- `<ben_stem>_unique.jsonl.ben`

Example:

```bash
ben-process extract-unique-plans \
  --ben-file runs/plans.jsonl.ben \
  --output-dir out
```

## Compression

Parquet-writing modes use Snappy by default.

Use `--high-compression` to switch to Brotli:

```bash
ben-process cut-edges \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --high-compression
```

Brotli is slower and is mainly useful when storage size matters more than CPU time.

## Examples

Tally multiple attributes:

```bash
ben-process tally-keys \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys pop area vap \
  --output-dir out
```

Weighted cut edges:

```bash
ben-process cut-edges \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --edge-weight-key weight
```

District-level Polsby-Popper with direct node perimeter:

```bash
ben-process polsby-popper \
  --graph-file data/dual_graph.json \
  --ben-file runs/plans.jsonl.ben \
  --perim-key perim \
  --output-dir out
```

Changed assignments on the first 100,000 accepted plans:

```bash
ben-process changed-assignments \
  --ben-file runs/plans.jsonl.ben \
  --max-accepted 100000 \
  --output-dir out
```

County split counts:

```bash
ben-process region-splits \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys county
```

Extract unique plans:

```bash
ben-process extract-unique-plans \
  --ben-file runs/plans.jsonl.ben \
  --output-dir out
```
