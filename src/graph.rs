//! Graph representation used by every tally mode.
//!
//! Everything the metrics need is pre-parsed at load time into flat, integer-indexed columns, so
//! the per-sample hot loop never re-walks JSON or re-parses strings. `serde_json::Value` is not
//! held anywhere in `Graph` after [`load_graph`] returns.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader};

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonGraphData {
    pub directed: bool,
    pub multigraph: bool,
    pub graph: Vec<Value>,
    pub nodes: Vec<Value>,
    pub adjacency: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct EdgeWeightRequest {
    pub key: String,
    pub default_value: f64,
}

#[derive(Debug, Clone, Default)]
pub struct GraphLoadRequest {
    pub numeric_keys: Vec<String>,
    pub partial_numeric_keys: Vec<String>,
    pub region_keys: Vec<String>,
    pub edge_weight: Option<EdgeWeightRequest>,
}

/// Pre-parsed graph ready for the hot loop.
#[derive(Debug)]
pub struct Graph {
    /// Number of nodes in the source graph. Held independently of `attr_columns` so the node count
    /// is known even when no numeric keys were requested; used to validate assignment-vector
    /// lengths.
    pub node_count: usize,
    /// Numeric node attributes the caller asked for, one column per key.
    /// `attr_columns[column_index][node_index]` is the parsed f64 value.
    pub attr_columns: Vec<Vec<f64>>,
    /// Key → column index into `attr_columns`.
    pub attr_index: HashMap<String, usize>,

    /// Region keys the caller asked for, one column per key. Each entry is `Some(u32)` where the
    /// u32 is the interned region id (dense, starting at 0), or `None` when the node's region
    /// value was missing / NaN.
    pub region_columns: Vec<Vec<Option<u32>>>,
    pub region_index: HashMap<String, usize>,
    /// Number of distinct region ids in each column of `region_columns`.
    pub region_id_counts: Vec<u32>,

    /// Deduplicated, sorted (min, max) edges. u32 is plenty for block graphs.
    pub edges: Vec<(u32, u32)>,
    /// When `Some`, aligned with `edges`: `edge_weights[i]` is the weight of `edges[i]` under the
    /// single weight key the caller asked for. Edges missing the key fall back to the request's
    /// `default_value`.
    pub edge_weights: Option<Vec<f64>>,
}

impl Graph {
    pub fn numeric_column_index(&self, key: &str) -> Option<usize> {
        self.attr_index.get(key).copied()
    }

    pub fn numeric_column(&self, key: &str) -> Option<&[f64]> {
        self.numeric_column_index(key)
            .map(|column_index| self.attr_columns[column_index].as_slice())
    }

    pub fn region_column_index(&self, key: &str) -> Option<usize> {
        self.region_index.get(key).copied()
    }

    pub fn edge_weight_column(&self) -> Option<&[f64]> {
        self.edge_weights.as_deref()
    }
}

/// Decide whether a node's value for a region key is meaningful (e.g. a real county id) or missing.
/// Returns `None` for a JSON null, an empty or `nan` string, or a NaN number.
fn parse_region_id(node: &Value, key: &str) -> Option<String> {
    let value = &node[key];
    match value {
        Value::Null => None,
        Value::Number(number) => {
            let numeric_value = number.as_f64()?;
            if numeric_value.is_nan() {
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

fn parse_numeric(node: &Value, key: &str) -> io::Result<f64> {
    let extracted_value = node.get(key).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing numeric graph key {:?} on node {:?}", key, node),
        )
    })?;
    let parsed = match extracted_value {
        Value::Number(n) => n.as_f64().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "non-f64 numeric graph value for key {:?}: {:?}",
                    key, extracted_value
                ),
            )
        })?,
        Value::String(s) => s.parse::<f64>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid numeric graph value for key {:?}: {:?} on node {:?}",
                    key, extracted_value, node
                ),
            )
        })?,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid numeric graph value for key {:?}: {:?} on node {:?}",
                    key, extracted_value, node
                ),
            ))
        }
    };

    // Rust's f64 FromStr accepts "NaN"/"inf"/"infinity", and an out-of-range JSON number can parse
    // to ±inf. A single non-finite node value would silently poison its whole district sum to
    // NaN/inf, so reject it here (mirroring parse_region_id, which already drops "nan").
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "non-finite numeric graph value for key {:?}: {:?} on node {:?}",
                key, extracted_value, node
            ),
        ))
    }
}

fn parse_numeric_or_zero(value: &Value) -> f64 {
    parse_numeric_opt(value).unwrap_or(0.0)
}

fn parse_numeric_opt(value: &Value) -> Option<f64> {
    let parsed = match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }?;
    // Drop non-finite values (e.g. the string "NaN"/"inf") so they never reach a tally or edge
    // weight; the lenient path treats them the same as a missing/unparseable value.
    parsed.is_finite().then_some(parsed)
}

/// Resolve the optional node `id` labels into an id → position map.
///
/// Ids are keyed by their JSON text (`Value::to_string`), so integer and string ids both resolve
/// exactly, with no numeric coercion. Returns `None` when no node carries an `id` field, in which
/// case adjacency ids are interpreted as positional indices by the caller.
///
/// Ids on only some nodes, or two nodes sharing an id, make adjacency resolution ambiguous and are
/// hard errors. A labeling that isn't the identity (`id != position`) is allowed but warned: the
/// BEN assignment is assumed to follow `.nodes[]` order, and the warning surfaces that the ids do
/// not.
fn build_node_id_index(nodes: &[Value]) -> io::Result<Option<HashMap<String, u32>>> {
    let nodes_with_id = nodes.iter().filter(|node| node.get("id").is_some()).count();
    if nodes_with_id == 0 {
        return Ok(None);
    }
    if nodes_with_id != nodes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} of {} graph nodes carry an \"id\" field; ids must be present on every node or \
                 on none, otherwise adjacency references cannot be resolved",
                nodes_with_id,
                nodes.len()
            ),
        ));
    }

    let mut id_to_index: HashMap<String, u32> = HashMap::with_capacity(nodes.len());
    let mut first_mismatch: Option<usize> = None;
    for (node_index, node) in nodes.iter().enumerate() {
        let id_value = node.get("id").expect("every node carries an id here");
        if id_to_index
            .insert(id_value.to_string(), node_index as u32)
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "two graph nodes share the id {}; node ids must be unique",
                    id_value
                ),
            ));
        }
        if first_mismatch.is_none() && id_value.as_u64() != Some(node_index as u64) {
            first_mismatch = Some(node_index);
        }
    }

    if let Some(position) = first_mismatch {
        log::warn!(
            "graph node ids do not match their positions in .nodes[] (first mismatch at position \
             {}, id {}); treating .nodes[] order as the true node order: BEN assignment entry i is \
             taken as nodes[i]'s district, and adjacency ids are resolved through the \"id\" field",
            position,
            nodes[position]["id"]
        );
    }

    Ok(Some(id_to_index))
}

/// Load a graph and pre-compute exactly the columns / edge weights the caller will need. Anything
/// not asked for is not parsed.
///
/// `GraphLoadRequest` deliberately supports at most one edge-weight column per load because every
/// current mode needs at most one. Widen the representation only when a mode genuinely needs
/// multiple edge columns at the same time.
pub fn load_graph(file_path: &str, request: GraphLoadRequest) -> crate::error::Result<Graph> {
    let file = File::open(file_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("failed to open graph file {:?}: {}", file_path, e),
        )
    })?;
    let reader = BufReader::new(file);
    let graph_data: JsonGraphData = serde_json::from_reader(reader).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse graph JSON {:?}: {}", file_path, e),
        )
    })?;

    // Edges are symmetrized via `(min, max)` dedup, which silently mangles a directed graph. Reject
    // it rather than producing an undirected reinterpretation the caller never asked for.
    if graph_data.directed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "graph {:?} is marked directed; ben-process only supports undirected dual graphs",
                file_path
            ),
        )
        .into());
    }

    // `.nodes[]` order is the canonical node order everywhere in this tool: attribute columns,
    // adjacency rows, and BEN assignment entries all align by position. Node `id` fields are
    // labels only — networkx's adjacency format references neighbors by id, so when ids are
    // present the edge loop below resolves them back to positions through this map (and a
    // non-identity labeling warns, since BEN assignments follow `.nodes[]` order, not id order).
    // When no node carries an id, adjacency ids are taken as positional indices directly.
    let node_id_index = build_node_id_index(&graph_data.nodes)?;

    // --- numeric node attributes ---
    let mut attr_columns: Vec<Vec<f64>> =
        Vec::with_capacity(request.numeric_keys.len() + request.partial_numeric_keys.len());
    let mut attr_index: HashMap<String, usize> =
        HashMap::with_capacity(request.numeric_keys.len() + request.partial_numeric_keys.len());
    for key in &request.numeric_keys {
        let column: Vec<f64> = graph_data
            .nodes
            .iter()
            .enumerate()
            .map(|(node_index, node)| {
                parse_numeric(node, key).map_err(|e| {
                    io::Error::new(
                        e.kind(),
                        format!(
                            "failed to load numeric graph key {:?} at node {}: {}",
                            key, node_index, e
                        ),
                    )
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        attr_index.insert(key.clone(), attr_columns.len());
        attr_columns.push(column);
    }
    for key in &request.partial_numeric_keys {
        let column: Vec<f64> = graph_data
            .nodes
            .iter()
            .map(|node| {
                node.get(key.as_str())
                    .map(parse_numeric_or_zero)
                    .unwrap_or(0.0)
            })
            .collect();
        attr_index.insert(key.clone(), attr_columns.len());
        attr_columns.push(column);
    }

    // --- interned region ids ---
    let mut region_columns: Vec<Vec<Option<u32>>> = Vec::with_capacity(request.region_keys.len());
    let mut region_index: HashMap<String, usize> =
        HashMap::with_capacity(request.region_keys.len());
    let mut region_id_counts: Vec<u32> = Vec::with_capacity(request.region_keys.len());
    for key in &request.region_keys {
        let mut interner: HashMap<String, u32> = HashMap::new();
        let column: Vec<Option<u32>> = graph_data
            .nodes
            .iter()
            .map(|node| {
                parse_region_id(node, key).map(|region_id| {
                    let next = interner.len() as u32;
                    *interner.entry(region_id).or_insert(next)
                })
            })
            .collect();
        region_index.insert(key.clone(), region_columns.len());
        region_id_counts.push(interner.len() as u32);
        region_columns.push(column);
    }

    // --- flat dedup'd edges + (optionally) parallel weight vector ---
    //
    //   - `edge_set` tracks which (min, max) pairs exist.
    //   - `edge_weights_map` only holds entries for edges that carry at least one *numerically
    //     parseable* value for `edge_weight_key` on at least one endpoint. Missing or non-numeric
    //     values are not stored; those edges fall back to the request's default at lookup time.
    //   - `.insert()` (not `.or_insert()`) so the last valid weight for an edge wins.
    // `node_count` is the contract every assignment is later validated against (one entry per
    // node). The adjacency block must agree with it: one adjacency list per node, and every
    // edge endpoint a real node id. Otherwise an out-of-range id would index `assignment[id]`
    // and panic deep in a metric hot loop mid-run, and a length mismatch would silently
    // under/over-count edges.
    let node_count = graph_data.nodes.len();
    if graph_data.adjacency.len() != node_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "graph has {} nodes but {} adjacency lists; node and adjacency counts must match",
                node_count,
                graph_data.adjacency.len()
            ),
        )
        .into());
    }

    let mut edge_set: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut edge_weights_map: HashMap<(u32, u32), f64> = HashMap::new();
    let edge_weight_key = request.edge_weight.as_ref().map(|edge| edge.key.as_str());
    for (source_index, adjacency_val) in graph_data.adjacency.iter().enumerate() {
        let source = source_index as u32;
        let neighbors = adjacency_val.as_array().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("graph adjacency entry {} is not an array", source_index),
            )
        })?;
        for target_data in neighbors {
            let id_value = target_data.get("id").ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "graph adjacency entry {} has edge without id: {:?}",
                        source_index, target_data
                    ),
                )
            })?;
            let target: u32 = match &node_id_index {
                // Nodes carry id labels: resolve the neighbor's id to its `.nodes[]` position.
                Some(id_to_index) => *id_to_index.get(&id_value.to_string()).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "graph adjacency entry {} references id {} which is not the id of \
                                 any node",
                            source_index, id_value
                        ),
                    )
                })?,
                // No node ids: adjacency ids are positional indices. Validate the full u64 against
                // the node count *before* narrowing to u32, so a huge id can't wrap to a small
                // in-range index and silently create the wrong edge.
                None => {
                    let target_id = id_value.as_u64().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "graph adjacency entry {} has edge without numeric id: {:?}",
                                source_index, target_data
                            ),
                        )
                    })?;
                    if target_id >= node_count as u64 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "graph adjacency entry {} references node id {} but the graph has only {} nodes",
                                source_index, target_id, node_count
                            ),
                        )
                        .into());
                    }
                    target_id as u32
                }
            };
            let edge = (source.min(target), source.max(target));
            edge_set.insert(edge);

            if let Some(weight_key) = edge_weight_key {
                if let Some(weight) = target_data.get(weight_key).and_then(parse_numeric_opt) {
                    // .insert (overwrite) so the last valid weight wins.
                    edge_weights_map.insert(edge, weight);
                }
            }
        }
    }

    let mut edges: Vec<(u32, u32)> = edge_set.into_iter().collect();
    edges.sort_unstable();
    let edge_weights: Option<Vec<f64>> = request.edge_weight.map(|edge_weight_request| {
        edges
            .iter()
            .map(|edge| {
                edge_weights_map
                    .get(edge)
                    .copied()
                    .unwrap_or(edge_weight_request.default_value)
            })
            .collect()
    });

    Ok(Graph {
        node_count,
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
    use super::{
        load_graph, parse_numeric, parse_numeric_opt, parse_region_id, EdgeWeightRequest,
        GraphLoadRequest,
    };
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
        assert_eq!(parse_numeric(&json!({ "pop": 3.5 }), "pop").unwrap(), 3.5);
        assert_eq!(
            parse_numeric(&json!({ "pop": "4.25" }), "pop").unwrap(),
            4.25
        );
    }

    #[test]
    fn parse_numeric_errors_on_invalid_string() {
        let err = parse_numeric(&json!({ "pop": "not-a-number" }), "pop").unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid numeric graph value for key \"pop\""),
            "unexpected error: {err}"
        );
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
            GraphLoadRequest {
                numeric_keys,
                region_keys,
                edge_weight: Some(EdgeWeightRequest {
                    key: "weight".to_string(),
                    default_value: 1.0,
                }),
                ..Default::default()
            },
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
    fn load_graph_errors_when_strict_numeric_key_is_missing_from_nodes() {
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

        let err = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest {
                numeric_keys: vec!["does_not_exist".to_string()],
                ..Default::default()
            },
        )
        .unwrap_err();
        let err = err.to_string();
        assert!(
            err.contains("failed to load numeric graph key \"does_not_exist\" at node 0"),
            "unexpected error: {err}"
        );
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
            GraphLoadRequest {
                edge_weight: Some(EdgeWeightRequest {
                    key: "weight".to_string(),
                    default_value: 1.0,
                }),
                ..Default::default()
            },
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
            GraphLoadRequest {
                numeric_keys: vec!["area".to_string()],
                partial_numeric_keys: vec!["boundary_perim".to_string()],
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            graph.attr_columns[graph.attr_index["area"]],
            vec![1.0, 2.0, 3.0, 4.0]
        );
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
            GraphLoadRequest {
                edge_weight: Some(EdgeWeightRequest {
                    key: "shared_perim".to_string(),
                    default_value: 0.0,
                }),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(graph.edges, vec![(0, 1)]);
        assert_eq!(graph.edge_weights, Some(vec![0.0]));
    }

    #[test]
    fn parse_numeric_rejects_non_finite_strings() {
        // Rust's f64 FromStr accepts "NaN"/"inf"/"infinity". Left unguarded, a single such node
        // value flows straight into a district tally and poisons the whole district sum to NaN/inf.
        // parse_numeric must reject non-finite values the same way parse_region_id rejects "nan".
        assert!(
            parse_numeric(&json!({ "pop": "NaN" }), "pop").is_err(),
            "parse_numeric should reject the string \"NaN\""
        );
        assert!(
            parse_numeric(&json!({ "pop": "inf" }), "pop").is_err(),
            "parse_numeric should reject the string \"inf\""
        );
        assert!(
            parse_numeric(&json!({ "pop": "Infinity" }), "pop").is_err(),
            "parse_numeric should reject the string \"Infinity\""
        );
    }

    #[test]
    fn parse_numeric_opt_rejects_non_finite_strings() {
        // Same hazard on the lenient path used for partial numeric columns and edge weights.
        assert_eq!(parse_numeric_opt(&json!("NaN")), None);
        assert_eq!(parse_numeric_opt(&json!("inf")), None);
        assert_eq!(parse_numeric_opt(&json!("Infinity")), None);
    }

    #[test]
    fn load_graph_rejects_directed_graph() {
        // Edges are symmetrized by (min,max) dedup, so a directed graph would be silently
        // reinterpreted as undirected. Reject it instead.
        let graph_json = json!({
            "directed": true,
            "multigraph": false,
            "graph": [],
            "nodes": [ { "pop": 1.0 }, { "pop": 2.0 } ],
            "adjacency": [ [ { "id": 1 } ], [ { "id": 0 } ] ]
        });
        let graph_file = write_graph(graph_json);

        let err = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("is marked directed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_graph_rejects_out_of_range_adjacency_target() {
        // The graph has 2 nodes, but an adjacency entry references node id 9. The resulting edge
        // (0, 9) would later index `assignment[9]` and panic out-of-bounds deep inside a metric hot
        // loop, mid-run, in a rayon worker. load_graph must reject the inconsistency up front.
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [ { "pop": 1.0 }, { "pop": 2.0 } ],
            "adjacency": [
                [ { "id": 1 } ],
                [ { "id": 0 }, { "id": 9 } ]
            ]
        });
        let graph_file = write_graph(graph_json);

        let result = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest::default(),
        );
        assert!(
            result.is_err(),
            "load_graph should reject an adjacency target id beyond the node count, got Ok"
        );
    }

    #[test]
    fn load_graph_accepts_node_ids_matching_their_position() {
        // Real networkx exports carry an `id` per node; when ids equal positions the graph is
        // aligned and must load.
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [ { "id": 0, "pop": 1.0 }, { "id": 1, "pop": 2.0 } ],
            "adjacency": [ [ { "id": 1 } ], [ { "id": 0 } ] ]
        });
        let graph_file = write_graph(graph_json);

        let graph = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest {
                numeric_keys: vec!["pop".to_string()],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(graph.node_count, 2);
        assert_eq!(graph.edges, vec![(0, 1)]);
    }

    #[test]
    fn load_graph_resolves_permuted_node_ids_to_positions() {
        // `.nodes[]` order is the true node order; ids are labels that adjacency references. This
        // 3-node path (by position: 0 - 1 - 2) carries permuted ids, so resolving adjacency ids
        // positionally would instead produce a self-loop at 0 and the wrong edge set. The loader
        // must map each neighbor id back to the position of the node carrying that id, while
        // attribute columns stay in `.nodes[]` order.
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                { "id": 1, "pop": 10.0 },
                { "id": 0, "pop": 20.0 },
                { "id": 2, "pop": 30.0 },
            ],
            "adjacency": [
                [ { "id": 0 } ],
                [ { "id": 1 }, { "id": 2 } ],
                [ { "id": 0 } ]
            ]
        });
        let graph_file = write_graph(graph_json);

        let graph = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest {
                numeric_keys: vec!["pop".to_string()],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(graph.edges, vec![(0, 1), (1, 2)]);
        assert_eq!(graph.attr_columns, vec![vec![10.0, 20.0, 30.0]]);
    }

    #[test]
    fn load_graph_resolves_string_node_ids_to_positions() {
        // GEOID-indexed graphs have string node ids; adjacency references them by the same
        // strings. The id map resolves them to positions, so the graph loads with positional
        // edges and columns.
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [ { "id": "06037", "pop": 1.0 }, { "id": "06038", "pop": 2.0 } ],
            "adjacency": [ [ { "id": "06038" } ], [ { "id": "06037" } ] ]
        });
        let graph_file = write_graph(graph_json);

        let graph = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest {
                numeric_keys: vec!["pop".to_string()],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(graph.node_count, 2);
        assert_eq!(graph.edges, vec![(0, 1)]);
        assert_eq!(graph.attr_columns, vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn load_graph_rejects_duplicate_node_ids() {
        // Two nodes sharing an id make adjacency references ambiguous — there is no defensible
        // resolution, so this stays a hard error.
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [ { "id": 0, "pop": 1.0 }, { "id": 0, "pop": 2.0 } ],
            "adjacency": [ [ { "id": 0 } ], [ { "id": 0 } ] ]
        });
        let graph_file = write_graph(graph_json);

        let err = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("two graph nodes share the id 0; node ids must be unique"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_graph_rejects_partial_node_ids() {
        // Ids on some-but-not-all nodes leave adjacency references unresolvable for the unlabeled
        // nodes; the all-or-none rule keeps the interpretation unambiguous.
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [ { "id": 0, "pop": 1.0 }, { "pop": 2.0 } ],
            "adjacency": [ [ { "id": 1 } ], [ { "id": 0 } ] ]
        });
        let graph_file = write_graph(graph_json);

        let err = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("ids must be present on every node or on none"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_graph_rejects_adjacency_id_not_matching_any_node() {
        // When nodes carry ids, an adjacency reference to an id no node has is a broken graph —
        // there is no position to resolve it to.
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [ { "id": 0, "pop": 1.0 }, { "id": 1, "pop": 2.0 } ],
            "adjacency": [ [ { "id": 9 } ], [ { "id": 0 } ] ]
        });
        let graph_file = write_graph(graph_json);

        let err = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("references id 9 which is not the id of any node"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_graph_rejects_adjacency_node_count_mismatch() {
        // `node_count` is taken from `nodes` (3 here), so the pipeline validates assignments
        // against 3 nodes — but the edge set was built from a 2-row adjacency. That is an
        // internally inconsistent graph (silent edge undercount → wrong cut-edge /
        // perimeter totals) and should be rejected at load time rather than silently
        // producing wrong tallies.
        let graph_json = json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [ { "pop": 1.0 }, { "pop": 2.0 }, { "pop": 3.0 } ],
            "adjacency": [
                [ { "id": 1 } ],
                [ { "id": 0 } ]
            ]
        });
        let graph_file = write_graph(graph_json);

        let result = load_graph(
            graph_file.path().to_str().unwrap(),
            GraphLoadRequest::default(),
        );
        assert!(
            result.is_err(),
            "load_graph should reject an adjacency list whose length differs from the node count, got Ok"
        );
    }
}
