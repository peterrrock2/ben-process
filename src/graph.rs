use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonGraphData {
    pub directed: bool,
    pub multigraph: bool,
    pub graph: Vec<Value>,
    pub nodes: Vec<Value>,
    pub adjacency: Vec<Value>,
}

/// Basic struct that represents a graph with nodes and edges.
/// This is a minimal struct compared opted for over something like `petgraph::Graph`
/// in order to reduce overhead since all operations in this file are relatively simple.
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Value>,
    pub edges: HashSet<(u64, u64)>,
    pub edge_weights: HashMap<(u64, u64), HashMap<String, f64>>,
}

/// Creates a graph from a JSON file.
///
/// # Arguments
///
/// * `file_path` - A string slice that holds the path to the JSON file.
///
/// # Returns
///
/// * `Result<Graph>` - A result containing the graph if successful, or an error if not.
///
/// # Errors
///
/// This function will return an error if the file cannot be found, read, or parsed.
pub fn make_graph_from_json(file_path: &str) -> Result<Graph> {
    // Read the JSON file
    let mut file = File::open(file_path).expect(format!("File {} not found", file_path).as_str());
    let mut data = String::new();
    file.read_to_string(&mut data).expect("Unable to read file");

    // Parse the JSON data
    let graph_data: JsonGraphData = serde_json::from_str(&data).expect("Unable to parse JSON");

    let mut graph = Graph {
        nodes: graph_data.nodes.clone(),
        edges: HashSet::new(),
        edge_weights: HashMap::new(),
    };

    for (source_idx, target_array) in graph_data.adjacency.iter().enumerate().map(|(x, y)| {
        (
            x as u64,
            y.as_array().expect("Failed to unwrap adjacency").to_vec(),
        )
    }) {
        for target_data in target_array {
            let target_idx: u64 = target_data["id"].as_u64().expect("Failed to unwrap id");
            graph
                .edges
                .insert((source_idx.min(target_idx), source_idx.max(target_idx)));

            // Pre-parse any numeric attributes present on the edge so keyed weighted cut-edges
            // can be computed without re-walking raw JSON later.
            if let Some(edge_obj) = target_data.as_object() {
                let edge_key = (source_idx.min(target_idx), source_idx.max(target_idx));
                for (key, value) in edge_obj {
                    if key == "id" {
                        continue;
                    }
                    let parsed_weight = match value {
                        Value::Number(n) => n.as_f64(),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    };
                    if let Some(weight) = parsed_weight {
                        graph
                            .edge_weights
                            .entry((edge_key.0, edge_key.1))
                            .or_insert_with(HashMap::new)
                            .insert(key.clone(), weight);
                    }
                }
            }
        }
    }

    Ok(graph)
}
