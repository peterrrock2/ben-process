#!/usr/bin/env python3
"""Run the Python scoring acceptance checks against a real RustReCom chain."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box

from ben_process import PlanScorer, PolsbyPopper, Reock, Tally


ROOT = Path(__file__).resolve().parents[1]
RUSTRECOM_REPO = "https://github.com/peterrrock2/frcw-dev.git"
RUSTRECOM_REF = "0.1.4"
STEPS = 1_000


def run(*args: str | Path, cwd: Path = ROOT) -> None:
    command = [str(arg) for arg in args]
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def output(*args: str | Path, cwd: Path) -> str:
    return subprocess.check_output(
        [str(arg) for arg in args],
        cwd=cwd,
        text=True,
    ).strip()


def prepare_grid(source: Path, graph_path: Path, geometry_path: Path) -> None:
    graph = json.loads(source.read_text())
    max_x = max(node["x"] for node in graph["nodes"])
    max_y = max(node["y"] for node in graph["nodes"])

    for node in graph["nodes"]:
        node["area"] = 1.0
        node["perim"] = 4.0
        node["boundary_perim"] = float(
            (node["x"] in {0, max_x}) + (node["y"] in {0, max_y})
        )
    for neighbors in graph["adjacency"]:
        for edge in neighbors:
            edge["shared_perim"] = 1.0
    graph_path.write_text(json.dumps(graph, separators=(",", ":")))

    geometry = gpd.GeoDataFrame(
        {"node_id": [node["id"] for node in graph["nodes"]]},
        geometry=[
            box(node["x"], node["y"], node["x"] + 1, node["y"] + 1)
            for node in graph["nodes"]
        ],
        crs="EPSG:3857",
    )
    geometry.to_parquet(geometry_path)


def only_parquet(directory: Path) -> Path:
    paths = list(directory.rglob("*.parquet"))
    if len(paths) != 1:
        raise AssertionError(f"expected one Parquet file under {directory}, found {paths}")
    return paths[0]


def assert_tables_close(label: str, actual_path: Path, expected_path: Path) -> None:
    actual = pq.read_table(actual_path).combine_chunks()
    expected = pq.read_table(expected_path).combine_chunks()
    if actual.column_names != expected.column_names:
        raise AssertionError(
            f"{label} columns differ: {actual.column_names} != {expected.column_names}"
        )
    if actual.num_rows != expected.num_rows:
        raise AssertionError(
            f"{label} row counts differ: {actual.num_rows} != {expected.num_rows}"
        )

    for name in actual.column_names:
        actual_values = actual[name].to_numpy(zero_copy_only=False)
        expected_values = expected[name].to_numpy(zero_copy_only=False)
        if pa.types.is_floating(actual[name].type):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"{label} column {name}",
            )
        else:
            np.testing.assert_array_equal(
                actual_values,
                expected_values,
                err_msg=f"{label} column {name}",
            )


def score_with_python(chain: Path, geometry_path: Path, output_dir: Path) -> None:
    geometry = gpd.read_parquet(geometry_path).iloc[::-1]
    scorer = PlanScorer.from_bendl(chain)
    scorer.add_gdf(geometry, node_id="node_id")
    scorer.add_metric(Tally(["population"]))
    scorer.add_metric(Reock())
    scorer.add_metric(PolsbyPopper(source="geometry"))
    scorer.score_ben_file(output_dir)


def score_graph_polsby_with_python(chain: Path, output_dir: Path) -> None:
    scorer = PlanScorer.from_bendl(chain)
    scorer.add_metric(PolsbyPopper(source="graph", perim_key="perim"))
    scorer.score_ben_file(output_dir)


def score_with_cli(binary: Path, chain: Path, geometry: Path, output_root: Path) -> None:
    commands = {
        "tally": ["tally-keys", "--keys", "population"],
        "reock": ["reock", "--geometry-file", str(geometry)],
        "polsby_geometry": [
            "polsby-popper",
            "--geometry-file",
            str(geometry),
        ],
        "polsby_graph": ["polsby-popper", "--perim-key", "perim"],
    }
    for name, command in commands.items():
        destination = output_root / name
        destination.mkdir(parents=True)
        run(
            binary,
            "--quiet",
            *command,
            "--ben-file",
            chain,
            "--output-dir",
            destination,
        )


def assert_run_metadata(run_directory: Path) -> int:
    manifest = json.loads((run_directory / "manifest.json").read_text())
    if [metric["metric_key"] for metric in manifest["metrics"]] != [
        "tally",
        "reock",
        "polsby_popper",
    ]:
        raise AssertionError("combined run manifest does not preserve metric order")

    tables = [
        pq.read_table(run_directory / "tally/0000.parquet"),
        pq.read_table(run_directory / "reock/scores.parquet"),
        pq.read_table(run_directory / "polsby_popper/scores.parquet"),
    ]
    prefix = ["step", "n_reps", "accepted_count"]
    reference = tables[0].select(prefix)
    if any(not table.select(prefix).equals(reference) for table in tables[1:]):
        raise AssertionError("combined metric tables do not share stream metadata")

    steps = reference["step"].to_numpy()
    n_reps = reference["n_reps"].to_numpy()
    accepted = reference["accepted_count"].to_numpy()
    if int(n_reps.sum()) != STEPS:
        raise AssertionError(f"expected {STEPS} expanded samples, got {n_reps.sum()}")
    if steps[0] != 1 or steps[-1] + n_reps[-1] - 1 != STEPS:
        raise AssertionError("step and repetition metadata do not span the full chain")
    np.testing.assert_array_equal(accepted, np.arange(1, len(accepted) + 1))
    return len(accepted)


def main(work: Path) -> None:
    checkout = work / "rustrecom"
    run(
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        RUSTRECOM_REF,
        RUSTRECOM_REPO,
        checkout,
    )
    manifest = tomllib.loads((checkout / "Cargo.toml").read_text())
    if manifest["package"]["version"] != RUSTRECOM_REF:
        raise AssertionError("RustReCom branch and package version differ")
    revision = output("git", "rev-parse", "HEAD", cwd=checkout)

    graph = work / "grid.json"
    geometry = work / "grid.parquet"
    prepare_grid(checkout / "test_fixtures/graphs/6x6.json", graph, geometry)

    run("cargo", "build", "--release", cwd=ROOT)
    run("cargo", "build", "--release", "--bin", "frcw", cwd=checkout)
    chain = work / "chain.bendl"
    run(
        checkout / "target/release/frcw",
        "--graph-json",
        graph,
        "--n-steps",
        str(STEPS),
        "--tol",
        "0.25",
        "--pop-col",
        "population",
        "--assignment-col",
        "district",
        "--rng-seed",
        "17",
        "--n-threads",
        "1",
        "--batch-size",
        "1",
        "--variant",
        "district-pairs-rmst",
        "--writer",
        "bendl",
        "--output-file",
        chain,
    )

    python_geometry = work / "python_geometry"
    python_graph = work / "python_graph"
    score_with_python(chain, geometry, python_geometry)
    score_graph_polsby_with_python(chain, python_graph)

    cli = work / "cli"
    score_with_cli(ROOT / "target/release/ben-process", chain, geometry, cli)
    comparisons = [
        ("Tally", python_geometry / "tally/0000.parquet", only_parquet(cli / "tally")),
        ("Reock", python_geometry / "reock/scores.parquet", only_parquet(cli / "reock")),
        (
            "geometry Polsby-Popper",
            python_geometry / "polsby_popper/scores.parquet",
            only_parquet(cli / "polsby_geometry"),
        ),
        (
            "graph Polsby-Popper",
            python_graph / "polsby_popper/scores.parquet",
            only_parquet(cli / "polsby_graph"),
        ),
    ]
    for label, python_path, cli_path in comparisons:
        assert_tables_close(label, python_path, cli_path)
    assert_tables_close(
        "Polsby-Popper input modes",
        python_geometry / "polsby_popper/scores.parquet",
        python_graph / "polsby_popper/scores.parquet",
    )

    accepted_frames = assert_run_metadata(python_geometry)
    print(
        f"PASS: RustReCom {RUSTRECOM_REF} {revision[:12]}, {STEPS} samples, "
        f"{accepted_frames} accepted frames, all scores match",
        flush=True,
    )


if __name__ == "__main__":
    work_directory = Path(tempfile.mkdtemp(prefix="ben-process-rustrecom-acceptance-"))
    try:
        main(work_directory)
    except BaseException:
        print(f"FAILED: artifacts retained at {work_directory}", file=sys.stderr)
        raise
    else:
        shutil.rmtree(work_directory)
