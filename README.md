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
ben-process [OPTIONS] --ben-file <BEN_FILE>
```

Common options:
- `--mode <MODE>`
- `--ben-file <BEN_FILE>`
- `--graph-file <GRAPH_FILE>`
- `--output-dir <OUTPUT_DIR>`
- `--no-progress`
- `--high-compression`

Available modes:
- `tally-keys`
- `cut-edges`
- `polsby-popper`
- `changed-assignments`
- `region-splits`
- `region-pieces`
- `unique-plans`
- `extract-unique-plans`

## Modes

### `tally-keys`

Tallies one or more numeric node attributes by district for each accepted plan.

Required:
- `--graph-file`
- `--keys <KEYS>...`

Output:
- creates a directory named `<ben_stem>_tallies`
- writes one parquet file per key:
  `<key>_tally_<ben_stem>.parquet`
- each file contains:
  `step`, `n_reps`, `accepted_count`, `district_*`

If `--output-dir` is set, the output layout is:

```text
<output_dir>/<ben_stem>_tallies/<key>_tally_<ben_stem>.parquet
```

Example:

```bash
ben-process \
  --mode tally-keys \
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
- `--graph-file`

Optional:
- `--edge-weight-key <EDGE_WEIGHT_KEY>`

Output:
- `<ben_stem>_cut_edges.parquet`
- columns:
  `step`, `n_reps`, `accepted_count`, `cut_edges`

Examples:

```bash
ben-process \
  --mode cut-edges \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben
```

```bash
ben-process \
  --mode cut-edges \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --edge-weight-key weight \
  --output-dir out
```

### `polsby-popper`

Computes district-level Polsby-Popper scores for each accepted plan.

Required:
- `--graph-file`
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

Examples:

Use the default GerryChain geometry keys:

```bash
ben-process \
  --mode polsby-popper \
  --graph-file data/dual_graph.json \
  --ben-file runs/plans.jsonl.ben \
  --output-dir out
```

```bash
ben-process \
  --mode polsby-popper \
  --graph-file data/dual_graph.json \
  --ben-file runs/plans.jsonl.ben \
  --perim-key perim \
  --output-dir out
```

```bash
ben-process \
  --mode polsby-popper \
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

Output:
- `<ben_stem>_accept_<N>_changed_assignments.txt`

Notes:
- `N` is the number of accepted frames considered
- for MkvChain BEN files, this mode counts accepted frames, not repeated samples
- `--randomize-reassignments` is only appropriate for merge-split MCMC runs

Examples:

```bash
ben-process \
  --mode changed-assignments \
  --ben-file runs/plans.jsonl.ben
```

```bash
ben-process \
  --mode changed-assignments \
  --ben-file runs/plans.jsonl.ben \
  --normalize \
  --max-accepted 50000 \
  --output-dir out
```

### `region-splits`

Counts, for each requested region key, how many regions are split across more than one district in each plan.

Required:
- `--graph-file`
- `--keys <KEYS>...`

Output:
- `<ben_stem>_region_splits.parquet`
- columns:
  `step`, `n_reps`, `accepted_count`, `region_key`, `region_splits`

Example:

```bash
ben-process \
  --mode region-splits \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys county municipality \
  --output-dir out
```

### `region-pieces`

Counts, for each requested region key, the total number of district pieces across all regions in each plan.

Required:
- `--graph-file`
- `--keys <KEYS>...`

Output:
- `<ben_stem>_region_pieces.parquet`
- columns:
  `step`, `n_reps`, `accepted_count`, `region_key`, `region_pieces`

Example:

```bash
ben-process \
  --mode region-pieces \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys county \
  --output-dir out
```

### `unique-plans`

Counts label-invariant unique partitions in a BEN file.

Output:
- `<ben_stem>_unique_plans.txt`

The output text file contains:
- `unique_plans: <count>`
- `total_accepted_frames: <count>`

Example:

```bash
ben-process \
  --mode unique-plans \
  --ben-file runs/plans.jsonl.ben \
  --output-dir out
```

### `extract-unique-plans`

Extracts the first occurrence of each label-invariant unique partition and writes them back out as a Standard BEN file.

Output:
- `<ben_stem>_unique.jsonl.ben`

Example:

```bash
ben-process \
  --mode extract-unique-plans \
  --ben-file runs/plans.jsonl.ben \
  --output-dir out
```

## Compression

Parquet-writing modes use Snappy by default.

Use `--high-compression` to switch to Brotli:

```bash
ben-process \
  --mode cut-edges \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --high-compression
```

Brotli is slower and is mainly useful when storage size matters more than CPU time.

## Examples

Tally multiple attributes:

```bash
ben-process \
  --mode tally-keys \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys pop area vap \
  --output-dir out
```

Weighted cut edges:

```bash
ben-process \
  --mode cut-edges \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --edge-weight-key weight
```

District-level Polsby-Popper with direct node perimeter:

```bash
ben-process \
  --mode polsby-popper \
  --graph-file data/dual_graph.json \
  --ben-file runs/plans.jsonl.ben \
  --perim-key perim \
  --output-dir out
```

Changed assignments on the first 100,000 accepted plans:

```bash
ben-process \
  --mode changed-assignments \
  --ben-file runs/plans.jsonl.ben \
  --max-accepted 100000 \
  --output-dir out
```

County split counts:

```bash
ben-process \
  --mode region-splits \
  --graph-file data/graph.json \
  --ben-file runs/plans.jsonl.ben \
  --keys county
```

Extract unique plans:

```bash
ben-process \
  --mode extract-unique-plans \
  --ben-file runs/plans.jsonl.ben \
  --output-dir out
```
