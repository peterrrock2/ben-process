//! Graph representation used by every tally mode.
//!
//! The old representation held `Vec<serde_json::Value>` for nodes and a
//! `HashMap<(u64,u64), HashMap<String,f64>>` for edge weights, forcing every
//! per-sample metric to re-walk JSON and re-parse strings on every call. This
//! module pre-parses everything the metrics need at load time into flat,
//! integer-indexed columns. `serde_json::Value` is not held anywhere in `Graph`
//! after [`load_graph`] returns.

use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonGraphData {
    pub directed: bool,
    pub multigraph: bool,
    pub graph: Vec<Value>,
    pub nodes: Vec<Value>,
    pub adjacency: Vec<Value>,
}

/// Pre-parsed graph ready for the hot loop.
#[derive(Debug)]
pub struct Graph {
    /// Numeric node attributes the caller asked for, one column per key.
    /// `attr_columns[col_idx][node_idx]` is the parsed f64 value.
    pub attr_columns: Vec<Vec<f64>>,
    /// Key → column index into `attr_columns`.
    pub attr_index: HashMap<String, usize>,

    /// Region keys the caller asked for, one column per key. Each entry is
    /// `Some(u32)` where the u32 is the interned region id (dense, starting
    /// at 0), or `None` when the node's region value was missing / NaN.
    pub region_columns: Vec<Vec<Option<u32>>>,
    pub region_index: HashMap<String, usize>,
    /// Number of distinct region ids in each column of `region_columns`.
    pub region_id_counts: Vec<u32>,

    /// Deduplicated, sorted (min, max) edges. u32 is plenty for block graphs.
    pub edges: Vec<(u32, u32)>,
    /// When `Some`, aligned with `edges`: `edge_weights[i]` is the weight of
    /// `edges[i]` under the single weight key the caller asked for. Edges
    /// missing the key default to 1.0 (matching the old HashMap lookup fallback).
    pub edge_weights: Option<Vec<f64>>,
}

/// Decide whether a node's value for a region key is meaningful (e.g. a real
/// county id) or missing. Extracted from the old `parse_region_id` — same
/// semantics (Null / empty / "nan" / NaN-number → `None`).
fn parse_region_id(node: &Value, key: &str) -> Option<String> {
    let value = &node[key];
    match value {
        Value::Null => None,
        Value::Number(n) => {
            let v = n.as_f64()?;
            if v.is_nan() {
                None
            } else {
                Some(value.to_string())
            }
        }
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("nan") {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Bool(_) => Some(value.to_string()),
        _ => None,
    }
}

fn parse_numeric(value: &Value, key: &str) -> f64 {
    match value {
        Value::Number(n) => n
            .as_f64()
            .unwrap_or_else(|| panic!("Non-f64 number for key {:?}: {:?}", key, value)),
        Value::String(s) => s.parse::<f64>().unwrap_or_else(|_| {
            panic!(
                "Invalid value type in JSON file. Failed to parse {:?} as f64 for key {:?}",
                value, key
            )
        }),
        _ => panic!(
            "Invalid value type in JSON file. Failed to parse {:?} as f64 for key {:?}",
            value, key
        ),
    }
}

fn parse_numeric_or_zero(value: &Value) -> f64 {
    parse_numeric_opt(value).unwrap_or(0.0)
}

fn parse_numeric_opt(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Load a graph and pre-compute exactly the columns / edge weights the caller
/// will need. Anything not asked for is not parsed.
///
/// * `numeric_keys` — node attribute keys to pre-parse as f64 (tally-keys mode).
/// * `region_keys` — node attribute keys to intern into dense region ids
///   (region-splits / region-pieces modes).
/// * `edge_weight_key` — optional single edge-attribute key to populate
///   `edge_weights` with (cut-edges mode); defaults to 1.0 for edges that
///   don't carry the key.
pub fn load_graph(
    file_path: &str,
    numeric_keys: &[String],
    partial_numeric_keys: &[String],
    region_keys: &[String],
    edge_weight_key: Option<&str>,
    edge_default_value: f64,
) -> Result<Graph> {
    let file =
        File::open(file_path).unwrap_or_else(|_| panic!("File {} not found", file_path));
    let reader = BufReader::new(file);
    let graph_data: JsonGraphData =
        serde_json::from_reader(reader).expect("Unable to parse JSON");

    // --- numeric node attributes ---
    let mut attr_columns: Vec<Vec<f64>> =
        Vec::with_capacity(numeric_keys.len() + partial_numeric_keys.len());
    let mut attr_index: HashMap<String, usize> =
        HashMap::with_capacity(numeric_keys.len() + partial_numeric_keys.len());
    for key in numeric_keys {
        let col: Vec<f64> = graph_data
            .nodes
            .iter()
            .map(|node| parse_numeric(&node[key], key))
            .collect();
        attr_index.insert(key.clone(), attr_columns.len());
        attr_columns.push(col);
    }
    for key in partial_numeric_keys {
        let col: Vec<f64> = graph_data
            .nodes
            .iter()
            .map(|node| {
                node.get(key.as_str())
                    .map(parse_numeric_or_zero)
                    .unwrap_or(0.0)
            })
            .collect();
        attr_index.insert(key.clone(), attr_columns.len());
        attr_columns.push(col);
    }

    // --- interned region ids ---
    let mut region_columns: Vec<Vec<Option<u32>>> = Vec::with_capacity(region_keys.len());
    let mut region_index: HashMap<String, usize> = HashMap::with_capacity(region_keys.len());
    let mut region_id_counts: Vec<u32> = Vec::with_capacity(region_keys.len());
    for key in region_keys {
        let mut interner: HashMap<String, u32> = HashMap::new();
        let col: Vec<Option<u32>> = graph_data
            .nodes
            .iter()
            .map(|node| {
                parse_region_id(node, key).map(|rid| {
                    let next = interner.len() as u32;
                    *interner.entry(rid).or_insert(next)
                })
            })
            .collect();
        region_index.insert(key.clone(), region_columns.len());
        region_id_counts.push(interner.len() as u32);
        region_columns.push(col);
    }

    // --- flat dedup'd edges + (optionally) parallel weight vector ---
    //
    // Semantics match the pre-refactor code exactly:
    //   - `edge_set` tracks which (min, max) pairs exist.
    //   - `edge_weights_map` only holds entries for edges that carry at least
    //     one *numerically parseable* value for `edge_weight_key` on at least
    //     one endpoint. Missing or non-numeric values are not stored; those
    //     edges fall back to 1.0 at lookup time.
    //   - `.insert()` (not `.or_insert()`) — last valid weight wins, matching
    //     the old nested-HashMap `.insert(key, weight)` behavior.
    let mut edge_set: std::collections::HashSet<(u32, u32)> =
        std::collections::HashSet::new();
    let mut edge_weights_map: HashMap<(u32, u32), f64> = HashMap::new();
    for (source_idx, adjacency_val) in graph_data.adjacency.iter().enumerate() {
        let src = source_idx as u32;
        let adj = adjacency_val
            .as_array()
            .expect("Failed to unwrap adjacency");
        for target_data in adj {
            let tgt = target_data["id"]
                .as_u64()
                .expect("Failed to unwrap id") as u32;
            let edge = (src.min(tgt), src.max(tgt));
            edge_set.insert(edge);

            if let Some(wkey) = edge_weight_key {
                if let Some(weight) = target_data.get(wkey).and_then(parse_numeric_opt) {
                    // .insert (overwrite) — matches old last-seen-wins behavior.
                    edge_weights_map.insert(edge, weight);
                }
            }
        }
    }

    let mut edges: Vec<(u32, u32)> = edge_set.into_iter().collect();
    edges.sort_unstable();
    let edge_weights: Option<Vec<f64>> = edge_weight_key.map(|_| {
        edges
            .iter()
            .map(|e| edge_weights_map.get(e).copied().unwrap_or(edge_default_value))
            .collect()
    });

    Ok(Graph {
        attr_columns,
        attr_index,
        region_columns,
        region_index,
        region_id_counts,
        edges,
        edge_weights,
    })
}

#[cfg(test)]
mod tests {
    use super::{load_graph, parse_numeric, parse_numeric_opt, parse_region_id};
    use serde_json::json;
    use tempfile::NamedTempFile;

    fn write_graph(graph_json: serde_json::Value) -> NamedTempFile {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), graph_json.to_string()).unwrap();
        file
    }

    #[test]
    fn parse_region_id_treats_blank_and_nan_strings_as_missing() {
        let node = json!({
            "blank": "   ",
            "nan": " NaN ",
            "null": null,
            "bool": true,
            "number": 7,
            "name": " county-a ",
        });

        assert_eq!(parse_region_id(&node, "blank"), None);
        assert_eq!(parse_region_id(&node, "nan"), None);
        assert_eq!(parse_region_id(&node, "null"), None);
        assert_eq!(parse_region_id(&node, "bool"), Some("true".to_string()));
        assert_eq!(parse_region_id(&node, "number"), Some("7".to_string()));
        assert_eq!(parse_region_id(&node, "name"), Some("county-a".to_string()));
    }

    #[test]
    fn parse_numeric_accepts_numbers_and_numeric_strings() {
        assert_eq!(parse_numeric(&json!(3.5), "pop"), 3.5);
        assert_eq!(parse_numeric(&json!("4.25"), "pop"), 4.25);
    }

    #[test]
    #[should_panic(expected = "Failed to parse")]
    fn parse_numeric_panics_on_invalid_string() {
        let _ = parse_numeric(&json!("not-a-number"), "pop");
    }

    #[test]
    fn parse_numeric_opt_ignores_unparseable_values() {
        assert_eq!(parse_numeric_opt(&json!(2.0)), Some(2.0));
        assert_eq!(parse_numeric_opt(&json!("2.5")), Some(2.5));
        assert_eq!(parse_numeric_opt(&json!("oops")), None);
        assert_eq!(parse_numeric_opt(&json!(false)), None);
    }

    #[test]
    fn load_graph_dedups_sorts_and_populates_requested_columns() {
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                { "pop": 1.0, "region": "A" },
                { "pop": "2.5", "region": "A" },
                { "pop": 3.0, "region": " nan " },
            ],
            "adjacency": [
                [ { "id": 1, "weight": 2.0 }, { "id": 2, "weight": "oops" } ],
                [ { "id": 0, "weight": 4.5 }, { "id": 2 } ],
                [ { "id": 0, "weight": 9.0 }, { "id": 1, "weight": "3.5" } ]
            ]
        });
        let graph_file = write_graph(graph_json);

        let numeric_keys = vec!["pop".to_string()];
        let region_keys = vec!["region".to_string()];
        let graph = load_graph(
            graph_file.path().to_str().unwrap(),
            &numeric_keys,
            &[],
            &region_keys,
            Some("weight"),
            1.0,
        )
        .unwrap();

        assert_eq!(graph.attr_columns, vec![vec![1.0, 2.5, 3.0]]);
        assert_eq!(graph.attr_index["pop"], 0);

        assert_eq!(graph.region_columns, vec![vec![Some(0), Some(0), None]]);
        assert_eq!(graph.region_index["region"], 0);
        assert_eq!(graph.region_id_counts, vec![1]);

        assert_eq!(graph.edges, vec![(0, 1), (0, 2), (1, 2)]);
        assert_eq!(graph.edge_weights, Some(vec![4.5, 9.0, 3.5]));
    }

    #[test]
    fn load_graph_defaults_missing_weights_to_one() {
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                { "pop": 1.0 },
                { "pop": 2.0 },
            ],
            "adjacency": [
                [ { "id": 1 } ],
                [ { "id": 0, "weight": "oops" } ]
            ]
        });
        let graph_file = write_graph(graph_json);

        let graph = load_graph(
            graph_file.path().to_str().unwrap(),
            &[],
            &[],
            &[],
            Some("weight"),
            1.0,
        )
        .unwrap();

        assert_eq!(graph.edges, vec![(0, 1)]);
        assert_eq!(graph.edge_weights, Some(vec![1.0]));
    }

    #[test]
    fn load_graph_partial_numeric_defaults_missing_values_to_zero() {
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                { "area": 1.0, "boundary_perim": 3.0 },
                { "area": 2.0 },
                { "area": 3.0, "boundary_perim": null },
                { "area": 4.0, "boundary_perim": "oops" }
            ],
            "adjacency": [[], [], [], []]
        });
        let graph_file = write_graph(graph_json);

        let graph = load_graph(
            graph_file.path().to_str().unwrap(),
            &["area".to_string()],
            &["boundary_perim".to_string()],
            &[],
            None,
            0.0,
        )
        .unwrap();

        assert_eq!(graph.attr_columns[graph.attr_index["area"]], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            graph.attr_columns[graph.attr_index["boundary_perim"]],
            vec![3.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn load_graph_uses_requested_edge_default_value() {
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                { "pop": 1.0 },
                { "pop": 2.0 },
            ],
            "adjacency": [
                [ { "id": 1 } ],
                [ { "id": 0 } ]
            ]
        });
        let graph_file = write_graph(graph_json);

        let graph = load_graph(
            graph_file.path().to_str().unwrap(),
            &[],
            &[],
            &[],
            Some("shared_perim"),
            0.0,
        )
        .unwrap();

        assert_eq!(graph.edges, vec![(0, 1)]);
        assert_eq!(graph.edge_weights, Some(vec![0.0]));
    }
}
