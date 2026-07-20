import io
import json
import math
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ben_process import PlanScorer, PolsbyPopper, Reock, Tally
from ben_process import _scorer as scorer_module


class Nodes:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        return self.data[key]


class Edges:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, data=False):
        if data:
            return iter(self.rows)
        return iter((u, v) for u, v, _ in self.rows)


class Graph:
    def __init__(self, nodes, edges=()):
        self.nodes = Nodes(nodes)
        self.edges = Edges(list(edges))

    def is_directed(self):
        return False

    def is_multigraph(self):
        return False


class Series(list):
    def __init__(self, values, crs=None):
        super().__init__(values)
        self.crs = crs


class Gdf:
    def __init__(self, geometry, **columns):
        self.geometry = Series(geometry)
        self.columns = columns

    def __getitem__(self, key):
        return self.columns[key]


class Partition:
    def __init__(self, assignment):
        self.assignment = assignment
        self.graph = object()


def square_wkb(x, y=0.0):
    points = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)]
    header = b"\x01" + struct.pack("<III", 3, 1, len(points))
    return header + b"".join(struct.pack("<dd", *point) for point in points)


def bundle_graph_bytes():
    return json.dumps(
        {
            "directed": False,
            "multigraph": False,
            "graph": [],
            "nodes": [
                {"id": "a", "POP": 1},
                {"id": "b", "POP": 2},
            ],
            "adjacency": [
                [{"id": "b", "shared_perim": 1.0}],
                [{"id": "a", "shared_perim": 1.0}],
            ],
        }
    ).encode()


class ScoringTests(unittest.TestCase):
    def test_from_bendl_selects_and_aligns_named_geometry_asset(self):
        requested_names = []

        def load_assets(path, name):
            self.assertEqual(path, "plans.bendl")
            requested_names.append(name)
            return bundle_graph_bytes(), ["units.parquet", "notes.txt"], b"geoparquet"

        gdf = Gdf(
            [square_wkb(2.0), square_wkb(0.0)],
            GEOID=["b", "a"],
        )
        with (
            patch.object(scorer_module._rust_backend, "load_bendl_assets", load_assets),
            patch.object(scorer_module, "_read_geoparquet_bytes", return_value=gdf),
        ):
            scorer = PlanScorer.from_bendl(
                "plans.bendl",
                geometry_asset_name="units.parquet",
                node_id="GEOID",
                allow_unknown_crs=True,
            )

        self.assertEqual(requested_names, ["units.parquet"])
        self.assertEqual(scorer._node_order, ("a", "b"))
        self.assertEqual(scorer._geometry.rows, (square_wkb(0.0), square_wkb(2.0)))
        self.assertEqual(scorer._default_ben_source, "plans.bendl")
        result = scorer.add_metric(Tally(["POP"])).compute({"a": 1, "b": 2})
        np.testing.assert_allclose(result.values, [[1.0, 2.0]])
        calls = []
        scorer._rust_backend_scorer = SimpleNamespace(
            score_ben_file=lambda *args: calls.append(args)
        )
        scorer.score_ben_file("scores")
        self.assertEqual(calls[0][:2], ("scores", "plans.bendl"))

    def test_from_bendl_requires_node_id_for_independent_graph_and_geometry(self):
        with patch.object(
            scorer_module._rust_backend,
            "load_bendl_assets",
            return_value=(bundle_graph_bytes(), ["units.parquet"], b"geoparquet"),
        ):
            with self.assertRaisesRegex(ValueError, "node_id is required"):
                PlanScorer.from_bendl(
                    "plans.bendl",
                    geometry_asset_name="units.parquet",
                )

    def test_from_bendl_geometry_establishes_order_without_graph(self):
        gdf = Gdf([square_wkb(0.0), square_wkb(2.0)])
        with (
            patch.object(
                scorer_module._rust_backend,
                "load_bendl_assets",
                return_value=(None, ["units.parquet"], b"geoparquet"),
            ),
            patch.object(scorer_module, "_read_geoparquet_bytes", return_value=gdf),
        ):
            scorer = PlanScorer.from_bendl(
                "plans.bendl",
                geometry_asset_name="units.parquet",
                allow_unknown_crs=True,
            )

        self.assertEqual(scorer._node_order, (0, 1))

    def test_from_bendl_does_not_guess_geometry_asset(self):
        with patch.object(
            scorer_module._rust_backend,
            "load_bendl_assets",
            return_value=(None, ["units.parquet", "notes.txt"], None),
        ):
            with self.assertRaisesRegex(ValueError, "units.parquet"):
                PlanScorer.from_bendl("plans.bendl")

    def test_bendl_geometry_metric_names_available_assets(self):
        with patch.object(
            scorer_module._rust_backend,
            "load_bendl_assets",
            return_value=(bundle_graph_bytes(), ["units.parquet", "blocks.parquet"], None),
        ):
            scorer = PlanScorer.from_bendl("plans.bendl").add_metric(Reock())

        with self.assertRaisesRegex(RuntimeError, "blocks.parquet"):
            scorer.compute([1, 2])

    def test_bendl_geoparquet_loader_uses_bytes_io(self):
        expected = object()

        def read_parquet(source):
            self.assertIsInstance(source, io.BytesIO)
            self.assertEqual(source.read(), b"geoparquet")
            return expected

        with patch.dict(
            sys.modules,
            {"geopandas": SimpleNamespace(read_parquet=read_parquet)},
        ):
            self.assertIs(
                scorer_module._read_geoparquet_bytes(b"geoparquet"),
                expected,
            )

    def test_run_manifest_preserves_metric_order_and_normalized_options(self):
        scorer = PlanScorer()
        scorer.add_metric(Tally(["POP", "VAP"]))
        scorer.add_metric(Reock())
        scorer.add_metric(
            PolsbyPopper(
                source="graph",
                area_key="AREA",
                perim_key="PERIM",
                shared_perim_key="SHARED",
            )
        )

        self.assertEqual(
            scorer._manifest_metrics(),
            [
                {
                    "metric_key": "tally",
                    "output_slug": "tally",
                    "options": {"keys": ["POP", "VAP"]},
                    "tables": [
                        {"subkey": "POP", "path": "tally/0000.parquet"},
                        {"subkey": "VAP", "path": "tally/0001.parquet"},
                    ],
                },
                {
                    "metric_key": "reock",
                    "output_slug": "reock",
                    "options": {},
                    "tables": [
                        {"subkey": None, "path": "reock/scores.parquet"}
                    ],
                },
                {
                    "metric_key": "polsby_popper",
                    "output_slug": "polsby_popper",
                    "options": {
                        "source": "graph",
                        "area_key": "AREA",
                        "perim_key": "PERIM",
                        "boundary_perim_key": None,
                        "shared_perim_key": "SHARED",
                    },
                    "tables": [
                        {
                            "subkey": None,
                            "path": "polsby_popper/scores.parquet",
                        }
                    ],
                },
            ],
        )

    def test_tally_normalizes_assignments_to_graph_order(self):
        graph = Graph(
            {
                "b": {"POP": 2, "VAP": 20},
                "a": {"POP": 1, "VAP": 10},
                "c": {"POP": 3, "VAP": 30},
            }
        )
        scorer = PlanScorer(graph).add_metric(Tally(["POP", "VAP"]))

        mapping = scorer.compute({"a": 1, "b": 1, "c": 3})
        partition = scorer.compute(Partition({"c": 3, "a": 1, "b": 1}))
        vectors = scorer.compute_many(np.array([[1, 1, 3], [1, 3, 3]]))
        iterable = scorer.compute_many(iter([[1, 1, 3], [1, 3, 3]]))

        np.testing.assert_array_equal(mapping.values, [[3, 3, 30, 30]])
        np.testing.assert_array_equal(partition.values, mapping.values)
        np.testing.assert_array_equal(vectors.values, [[3, 3, 30, 30], [2, 4, 20, 40]])
        np.testing.assert_array_equal(iterable.values, vectors.values)
        self.assertEqual(
            mapping.columns,
            [
                ("tally", "POP", 1),
                ("tally", "POP", 3),
                ("tally", "VAP", 1),
                ("tally", "VAP", 3),
            ],
        )

    def test_geometry_is_reordered_and_shared_by_reock_and_polsby_popper(self):
        graph = Graph({"a": {}, "b": {}}, [("a", "b", {})])
        gdf = Gdf([square_wkb(1), square_wkb(0)], GEOID=["b", "a"])
        scorer = PlanScorer(graph).add_gdf(gdf, node_id="GEOID", allow_unknown_crs=True)
        scorer.add_metric(Reock()).add_metric(PolsbyPopper(source="geometry"))

        result = scorer.compute([0, 0])

        np.testing.assert_allclose(
            result.values, [[8 / (5 * math.pi), 2 * math.pi / 9]]
        )
        self.assertEqual(
            result.columns,
            [("reock", None, 0), ("polsby_popper", None, 0)],
        )

        positional = PlanScorer(
            graph,
            geometry=Gdf([square_wkb(0), square_wkb(1)]),
            allow_unknown_crs=True,
        )
        positional.add_metric(Reock())
        self.assertAlmostEqual(
            positional.compute([0, 0]).values[0, 0], 8 / (5 * math.pi)
        )

    def test_graph_polsby_popper_supports_direct_and_derived_perimeters(self):
        graph = Graph(
            {
                0: {"area": 1, "perim": 4, "boundary_perim": 3},
                1: {"area": 1, "perim": 4, "boundary_perim": 3},
            },
            [(0, 1, {"shared_perim": 1})],
        )
        direct = PlanScorer(graph).add_metric(
            PolsbyPopper(source="graph", perim_key="perim")
        )
        derived = PlanScorer(graph).add_metric(PolsbyPopper(source="graph"))

        np.testing.assert_allclose(direct.compute([0, 0]).values, [[2 * math.pi / 9]])
        np.testing.assert_allclose(derived.compute([0, 0]).values, [[2 * math.pi / 9]])

    def test_compute_once_matches_prepared_scorer(self):
        graph = Graph({0: {"POP": 2}, 1: {"POP": 3}})

        prepared = PlanScorer(graph).add_metric(Tally(["POP"])).compute([0, 1])
        once = Tally.compute_once([0, 1], graph=graph, keys=["POP"])

        np.testing.assert_array_equal(once.values, prepared.values)
        self.assertEqual(once.columns, prepared.columns)

        gdf = Gdf([square_wkb(0), square_wkb(1)])
        reock = Reock.compute_once([0, 0], geometry=gdf, allow_unknown_crs=True)
        self.assertAlmostEqual(reock.values[0, 0], 8 / (5 * math.pi))

    def test_resource_snapshots_ignore_caller_mutation(self):
        graph = Graph({0: {"POP": 2}, 1: {"POP": 3}})
        scorer = PlanScorer(graph).add_metric(Tally(["POP"]))
        graph.nodes.data[0]["POP"] = 200

        np.testing.assert_array_equal(scorer.compute([0, 1]).values, [[2, 3]])

        geometry = Gdf([square_wkb(0), square_wkb(1)])
        reock = PlanScorer(geometry=geometry, allow_unknown_crs=True).add_metric(
            Reock()
        )
        geometry.geometry[0] = square_wkb(100)
        self.assertAlmostEqual(reock.compute([0, 0]).values[0, 0], 8 / (5 * math.pi))

    def test_graph_freeze_rejects_unprepared_inputs(self):
        graph = Graph({0: {"POP": 2, "VAP": 20}, 1: {"POP": 3, "VAP": 30}})
        scorer = PlanScorer(graph).add_metric(Tally(["POP"]))
        scorer.compute([0, 1])

        scorer.add_metric(Tally(["POP"]))
        self.assertEqual(len(scorer.compute([0, 1]).columns), 2)
        scorer.add_gdf(Gdf([square_wkb(0), square_wkb(1)]), allow_unknown_crs=True)
        scorer.add_metric(Reock())
        self.assertEqual(len(scorer.compute([0, 1]).columns), 4)
        with self.assertRaisesRegex(RuntimeError, "VAP"):
            scorer.add_metric(Tally(["VAP"]))
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            scorer.add_graph(graph)

    def test_validation_happens_before_rust_backend_scoring(self):
        graph = Graph({0: {"POP": 2}, 1: {"POP": 3}})
        scorer = PlanScorer(graph)
        with self.assertRaisesRegex(RuntimeError, "at least one metric"):
            scorer.compute([False, 1])

        scorer.add_metric(Tally(["POP"]))
        with self.assertRaisesRegex(ValueError, "district label"):
            scorer.compute([False, 1])
        with self.assertRaisesRegex(ValueError, "assignment length"):
            scorer.compute([1])
        with self.assertRaisesRegex(ValueError, "same district labels"):
            scorer.compute_many([[0, 1], [0, 0]])
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            scorer.compute_many(np.array([0, 1]))
        with self.assertRaisesRegex(ValueError, r"missing=\[1\]"):
            scorer.compute({0: 0, 2: 1})

    def test_empty_compute_many_and_result_adapters(self):
        graph = Graph({0: {"POP": 2}, 1: {"POP": 3}})
        scorer = PlanScorer(graph).add_metric(Tally(["POP"]))

        empty = scorer.compute_many(np.empty((0, 2), dtype=np.uint16))
        populated = scorer.compute([0, 1])

        self.assertEqual(empty.values.shape, (0, 0))
        self.assertEqual(empty.columns, [])
        self.assertEqual(populated.to_pandas().shape, (1, 2))
        self.assertEqual(
            populated.to_arrow().column_names,
            ['["tally","POP",0]', '["tally","POP",1]'],
        )


if __name__ == "__main__":
    unittest.main()
