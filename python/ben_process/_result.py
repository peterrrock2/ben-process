from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

ScoreColumn = tuple[str, str | None, int]


@dataclass(frozen=True)
class ScoreResult:
    values: np.ndarray
    columns: list[ScoreColumn]
    steps: np.ndarray | None = None
    n_reps: np.ndarray | None = None
    accepted_count: np.ndarray | None = None

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("ScoreResult.to_pandas() requires pandas") from error

        columns = pd.MultiIndex.from_tuples(
            self.columns,
            names=["metric", "subkey", "district"],
        )
        return pd.DataFrame(self.values, columns=columns)

    def to_arrow(self):
        try:
            import pyarrow as pa
        except ImportError as error:
            raise ImportError("ScoreResult.to_arrow() requires pyarrow") from error

        fields = []
        arrays = []
        for index, (metric, subkey, district) in enumerate(self.columns):
            name = json.dumps([metric, subkey, district], separators=(",", ":"))
            fields.append(
                pa.field(
                    name,
                    pa.float64(),
                    metadata={
                        b"metric_key": metric.encode(),
                        b"subkey": json.dumps(subkey).encode(),
                        b"district_id": str(district).encode(),
                    },
                )
            )
            arrays.append(pa.array(self.values[:, index]))
        return pa.Table.from_arrays(arrays, schema=pa.schema(fields))
