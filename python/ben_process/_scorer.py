from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from . import _rust_backend
from ._result import ScoreResult
from .metrics import PolsbyPopper, Reock, Tally


@dataclass(frozen=True)
class _GraphSnapshot:
    node_data: tuple[dict, ...]
    edges: tuple[tuple[int, int], ...]
    edge_data: tuple[dict, ...]


@dataclass(frozen=True)
class _GeometrySnapshot:
    rows: tuple[bytes, ...]
    source_crs: str | None
    target_crs: str | None
    allow_geographic_crs: bool
    allow_unknown_crs: bool


class PlanScorer:
    def __init__(
        self,
        graph=None,
        *,
        geometry=None,
        node_id=None,
        geometry_column=None,
        source_crs=None,
        target_crs=None,
        allow_geographic_crs=False,
        allow_unknown_crs=False,
    ):
        self._node_order = None
        self._node_index = None
        self._graph = None
        self._geometry = None
        self._graph_frozen = False
        self._geometry_frozen = False
        self._prepared_node_keys = set()
        self._prepared_edge_keys = set()
        self._prepared_edges = False
        self._metric_order = []
        self._tally_keys = []
        self._metrics = {}
        self._rust_backend_scorer = None

        if graph is not None:
            self.add_graph(graph)
        if geometry is not None:
            self.add_gdf(
                geometry,
                node_id=node_id,
                geometry_column=geometry_column,
                source_crs=source_crs,
                target_crs=target_crs,
                allow_geographic_crs=allow_geographic_crs,
                allow_unknown_crs=allow_unknown_crs,
            )

    def add_graph(self, graph):
        if self._graph_frozen:
            raise RuntimeError("graph resource is frozen after scoring")
        if _graph_flag(graph, "is_directed"):
            raise ValueError("graph must be undirected")
        if _graph_flag(graph, "is_multigraph"):
            raise ValueError("graph must be a simple graph, not a multigraph")

        graph_order = tuple(graph.nodes)
        self._establish_or_validate_order(graph_order, "graph nodes")
        node_data = tuple(dict(graph.nodes[node]) for node in self._node_order)
        edges = []
        edge_data = []
        seen = set()
        for node_u, node_v, data in graph.edges(data=True):
            if node_u not in self._node_index or node_v not in self._node_index:
                raise ValueError(
                    f"graph edge ({node_u!r}, {node_v!r}) has an unknown endpoint"
                )
            u = self._node_index[node_u]
            v = self._node_index[node_v]
            if u == v:
                raise ValueError(f"graph contains self-loop at node {node_u!r}")
            edge = (min(u, v), max(u, v))
            if edge in seen:
                raise ValueError(
                    f"graph contains duplicate edge ({node_u!r}, {node_v!r})"
                )
            seen.add(edge)
            edges.append(edge)
            edge_data.append(dict(data))

        ordered = sorted(zip(edges, edge_data), key=lambda item: item[0])
        self._graph = _GraphSnapshot(
            node_data=node_data,
            edges=tuple(edge for edge, _ in ordered),
            edge_data=tuple(data for _, data in ordered),
        )
        self._prepared_node_keys.clear()
        self._prepared_edge_keys.clear()
        self._prepared_edges = False
        self._rust_backend_scorer = None
        return self

    def add_gdf(
        self,
        gdf,
        *,
        node_id=None,
        geometry_column=None,
        source_crs=None,
        target_crs=None,
        allow_geographic_crs=False,
        allow_unknown_crs=False,
    ):
        if self._geometry_frozen:
            raise RuntimeError("geometry resource is frozen after scoring")

        geometry = gdf.geometry if geometry_column is None else gdf[geometry_column]
        geometries = list(geometry)
        if node_id is None:
            if self._node_order is None:
                self._establish_or_validate_order(
                    tuple(range(len(geometries))), "geometry row positions"
                )
            elif len(geometries) != len(self._node_order):
                raise ValueError(
                    f"geometry row count is {len(geometries)} but graph node count is "
                    f"{len(self._node_order)}"
                )
            ordered_geometries = geometries
        else:
            keys = tuple(gdf[node_id])
            if any(_is_null(key) for key in keys):
                raise ValueError(
                    f"geometry node_id column {node_id!r} contains null values"
                )
            if len(set(keys)) != len(keys):
                raise ValueError(f"geometry node_id column {node_id!r} must be unique")
            self._establish_or_validate_order(keys, "geometry node ids")
            positions = {key: index for index, key in enumerate(keys)}
            ordered_geometries = [
                geometries[positions[key]] for key in self._node_order
            ]

        rows = tuple(
            _geometry_wkb(value, self._node_order[index])
            for index, value in enumerate(ordered_geometries)
        )
        effective_source_crs = source_crs
        if effective_source_crs is None:
            effective_source_crs = getattr(geometry, "crs", None)
        if effective_source_crs is not None:
            effective_source_crs = str(effective_source_crs)
        if target_crs is not None:
            target_crs = str(target_crs)

        self._geometry = _GeometrySnapshot(
            rows=rows,
            source_crs=effective_source_crs,
            target_crs=target_crs,
            allow_geographic_crs=allow_geographic_crs,
            allow_unknown_crs=allow_unknown_crs,
        )
        self._rust_backend_scorer = None
        return self

    def add_metric(self, metric):
        if not isinstance(metric, (Tally, Reock, PolsbyPopper)):
            raise TypeError("metric must be Tally, Reock, or PolsbyPopper")

        node_keys, edge_keys, needs_edges = _required_graph_inputs(metric)
        if self._graph_frozen:
            missing = sorted(node_keys - self._prepared_node_keys)
            missing += sorted(edge_keys - self._prepared_edge_keys)
            if needs_edges and not self._prepared_edges:
                missing.append("graph edges")
            if missing:
                raise RuntimeError(
                    "graph is frozen and is missing prepared inputs: "
                    + ", ".join(missing)
                )

        key = metric._metric_key
        if isinstance(metric, Tally):
            if key not in self._metric_order:
                self._metric_order.append(key)
            for tally_key in metric.keys:
                if tally_key not in self._tally_keys:
                    self._tally_keys.append(tally_key)
        else:
            if key in self._metrics:
                raise ValueError(f"duplicate prepared metric identity {key!r}")
            self._metric_order.append(key)
            self._metrics[key] = metric

        self._rust_backend_scorer = None
        return self

    def compute(self, assignment):
        backend = self._prepare_rust_backend()
        normalized = self._normalize_assignment(assignment)
        return self._score(backend, [normalized])

    def compute_many(self, assignments):
        backend = self._prepare_rust_backend()
        if isinstance(assignments, np.ndarray):
            if assignments.ndim != 2:
                raise ValueError("compute_many NumPy input must be two-dimensional")
            expected = len(self._node_order)
            if assignments.shape[1] != expected:
                raise ValueError(
                    f"assignment length is {assignments.shape[1]} but "
                    f"{self._resource_length_label()} is {expected}"
                )
            normalized = [self._normalize_assignment(row) for row in assignments]
        else:
            normalized = [self._normalize_assignment(row) for row in assignments]
        return self._score(backend, normalized)

    def _prepare_rust_backend(self):
        if not self._metric_order:
            raise RuntimeError("at least one metric must be registered before scoring")
        if self._rust_backend_scorer is not None:
            return self._rust_backend_scorer

        backend = _rust_backend.RustBackendScorer()
        prepared_node_keys = set(self._prepared_node_keys)
        prepared_edge_keys = set(self._prepared_edge_keys)
        prepared_edges = self._prepared_edges
        used_graph = False
        used_geometry = False

        for key in self._metric_order:
            if key == "tally":
                graph = self._require_graph("Tally")
                backend.add_tally(
                    [
                        _numeric_node_column(graph, tally_key)
                        for tally_key in self._tally_keys
                    ]
                )
                prepared_node_keys.update(self._tally_keys)
                used_graph = True
            elif key == "reock":
                geometry = self._require_geometry("Reock")
                backend.add_reock(geometry.rows, **_geometry_options(geometry))
                used_geometry = True
            else:
                metric = self._metrics[key]
                graph = self._require_graph("PolsbyPopper")
                if metric.source == "geometry":
                    geometry = self._require_geometry("PolsbyPopper(source='geometry')")
                    backend.add_polsby_popper_geometry(
                        geometry.rows,
                        graph.edges,
                        len(self._node_order),
                        **_geometry_options(geometry),
                    )
                    used_geometry = True
                    prepared_edges = True
                else:
                    area_key = metric.area_key or "area"
                    shared_key = metric.shared_perim_key or "shared_perim"
                    areas = _numeric_node_column(graph, area_key)
                    shared = _numeric_edge_column(graph, shared_key)
                    if metric.perim_key is not None:
                        totals = _numeric_node_column(graph, metric.perim_key)
                        boundaries = None
                        prepared_node_keys.update((area_key, metric.perim_key))
                    else:
                        boundary_key = metric.boundary_perim_key or "boundary_perim"
                        totals = None
                        boundaries = _numeric_node_column(graph, boundary_key)
                        prepared_node_keys.update((area_key, boundary_key))
                    backend.add_polsby_popper_graph(
                        areas,
                        totals,
                        boundaries,
                        graph.edges,
                        shared,
                    )
                    prepared_edge_keys.add(shared_key)
                    prepared_edges = True
                used_graph = True

        self._prepared_node_keys = prepared_node_keys
        self._prepared_edge_keys = prepared_edge_keys
        self._prepared_edges = prepared_edges
        self._graph_frozen |= used_graph
        self._geometry_frozen |= used_geometry
        self._rust_backend_scorer = backend
        return backend

    def _score(self, backend, assignments):
        rows, district_ids = backend.score_many(assignments)
        if not rows:
            return ScoreResult(np.empty((0, 0), dtype=np.float64), [])

        columns = []
        for key in self._metric_order:
            subkeys = self._tally_keys if key == "tally" else [None]
            for subkey in subkeys:
                columns.extend((key, subkey, district) for district in district_ids)
        return ScoreResult(np.asarray(rows, dtype=np.float64), columns)

    def _normalize_assignment(self, assignment):
        if isinstance(assignment, Mapping):
            mapping = assignment
        elif hasattr(assignment, "assignment") and isinstance(
            assignment.assignment, Mapping
        ):
            mapping = assignment.assignment
        else:
            array = np.asarray(assignment, dtype=object)
            if array.ndim != 1:
                raise ValueError("one assignment must be one-dimensional")
            if len(array) != len(self._node_order):
                raise ValueError(
                    f"assignment length is {len(array)} but "
                    f"{self._resource_length_label()} is {len(self._node_order)}"
                )
            return [_district_label(value) for value in array]

        missing = [key for key in self._node_order if key not in mapping]
        extra = [key for key in mapping if key not in self._node_index]
        if missing or extra:
            raise ValueError(
                f"assignment keys do not match node order; missing={missing}, extra={extra}"
            )
        return [_district_label(mapping[key]) for key in self._node_order]

    def _establish_or_validate_order(self, keys, label):
        if len(set(keys)) != len(keys):
            raise ValueError(f"{label} must be unique")
        if self._node_order is None:
            self._node_order = tuple(keys)
            self._node_index = {key: index for index, key in enumerate(keys)}
            return
        if len(keys) != len(self._node_order) or set(keys) != set(self._node_order):
            raise ValueError(f"{label} must exactly match the canonical node keys")

    def _require_graph(self, metric):
        if self._graph is None:
            raise RuntimeError(f"{metric} requires a graph resource")
        return self._graph

    def _require_geometry(self, metric):
        if self._geometry is None:
            raise RuntimeError(f"{metric} requires a geometry resource")
        return self._geometry

    def _resource_length_label(self):
        return "graph node count" if self._graph is not None else "geometry row count"


def _graph_flag(graph, name):
    value = getattr(graph, name, False)
    return bool(value() if callable(value) else value)


def _required_graph_inputs(metric):
    if isinstance(metric, Tally):
        return set(metric.keys), set(), False
    if isinstance(metric, Reock):
        return set(), set(), False
    if metric.source == "geometry":
        return set(), set(), True
    area_key = metric.area_key or "area"
    node_keys = {
        area_key,
        metric.perim_key or metric.boundary_perim_key or "boundary_perim",
    }
    return node_keys, {metric.shared_perim_key or "shared_perim"}, True


def _numeric_node_column(graph, key):
    return [
        _finite_number(data.get(key), f"graph node key {key!r}")
        for data in graph.node_data
    ]


def _numeric_edge_column(graph, key):
    return [
        _finite_number(data.get(key), f"graph edge key {key!r}")
        for data in graph.edge_data
    ]


def _finite_number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be present and numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _district_label(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"district label {value!r} must be an integer from 0 through 127"
        )
    value = int(value)
    if not 0 <= value <= 127:
        raise ValueError(
            f"district label {value!r} must be an integer from 0 through 127"
        )
    return value


def _geometry_wkb(geometry, key):
    if geometry is None:
        raise ValueError(f"geometry contains null value for node {key!r}")
    if isinstance(geometry, (bytes, bytearray, memoryview)):
        return bytes(geometry)
    try:
        return bytes(geometry.wkb)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"geometry for node {key!r} cannot be converted to WKB"
        ) from error


def _geometry_options(geometry):
    return {
        "source_crs": geometry.source_crs,
        "target_crs": geometry.target_crs,
        "allow_geographic_crs": geometry.allow_geographic_crs,
        "allow_unknown_crs": geometry.allow_unknown_crs,
    }


def _is_null(value):
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False
