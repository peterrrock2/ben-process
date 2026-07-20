from __future__ import annotations


class Tally:
    _metric_key = "tally"
    _output_slug = "tally"
    _requires = {"graph"}

    def __init__(self, keys):
        if isinstance(keys, str):
            raise ValueError(
                "Tally.keys must be an iterable of strings, not one string"
            )
        keys = tuple(keys)
        if not keys or any(not isinstance(key, str) for key in keys):
            raise ValueError("Tally.keys must contain at least one string")
        if len(set(keys)) != len(keys):
            raise ValueError("Tally.keys may not repeat a key")
        self.keys = keys

    @classmethod
    def compute_once(cls, assignment, *, graph, keys):
        from ._scorer import PlanScorer

        return PlanScorer(graph).add_metric(cls(keys)).compute(assignment)


class Reock:
    _metric_key = "reock"
    _output_slug = "reock"
    _requires = {"geometry"}

    @classmethod
    def compute_once(
        cls,
        assignment,
        *,
        geometry,
        node_id=None,
        geometry_column=None,
        source_crs=None,
        target_crs=None,
        allow_geographic_crs=False,
        allow_unknown_crs=False,
    ):
        from ._scorer import PlanScorer

        scorer = PlanScorer().add_gdf(
            geometry,
            node_id=node_id,
            geometry_column=geometry_column,
            source_crs=source_crs,
            target_crs=target_crs,
            allow_geographic_crs=allow_geographic_crs,
            allow_unknown_crs=allow_unknown_crs,
        )
        return scorer.add_metric(cls()).compute(assignment)


class PolsbyPopper:
    _metric_key = "polsby_popper"
    _output_slug = "polsby_popper"

    def __init__(
        self,
        *,
        source,
        area_key=None,
        perim_key=None,
        boundary_perim_key=None,
        shared_perim_key=None,
    ):
        if source not in {"graph", "geometry"}:
            raise ValueError("PolsbyPopper.source must be 'graph' or 'geometry'")
        graph_options = (area_key, perim_key, boundary_perim_key, shared_perim_key)
        if source == "geometry" and any(option is not None for option in graph_options):
            raise ValueError("graph key options must be None when source='geometry'")
        self.source = source
        self.area_key = area_key
        self.perim_key = perim_key
        self.boundary_perim_key = boundary_perim_key
        self.shared_perim_key = shared_perim_key
        self._requires = {"graph", "geometry"} if source == "geometry" else {"graph"}

    @classmethod
    def compute_once(
        cls,
        assignment,
        *,
        graph,
        geometry=None,
        node_id=None,
        geometry_column=None,
        source,
        source_crs=None,
        target_crs=None,
        allow_geographic_crs=False,
        allow_unknown_crs=False,
        area_key=None,
        perim_key=None,
        boundary_perim_key=None,
        shared_perim_key=None,
    ):
        from ._scorer import PlanScorer

        scorer = PlanScorer(graph)
        if geometry is not None:
            scorer.add_gdf(
                geometry,
                node_id=node_id,
                geometry_column=geometry_column,
                source_crs=source_crs,
                target_crs=target_crs,
                allow_geographic_crs=allow_geographic_crs,
                allow_unknown_crs=allow_unknown_crs,
            )
        metric = cls(
            source=source,
            area_key=area_key,
            perim_key=perim_key,
            boundary_perim_key=boundary_perim_key,
            shared_perim_key=shared_perim_key,
        )
        return scorer.add_metric(metric).compute(assignment)
