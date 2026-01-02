//! Graph algorithms for pattern detection and analysis.

use crate::graph::{FinancialGraph, NodeId};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

/// Find all triangles (3-cliques) in the graph
///
/// Returns list of (node1, node2, node3) tuples representing triangles
pub fn find_triangles(graph: &FinancialGraph) -> Vec<(String, String, String)> {
    let mut triangles = Vec::new();
    let symbols: Vec<_> = graph.symbols().into_iter().cloned().collect();

    for i in 0..symbols.len() {
        for j in (i + 1)..symbols.len() {
            if !graph.has_edge(&symbols[i], &symbols[j]) {
                continue;
            }

            for k in (j + 1)..symbols.len() {
                if graph.has_edge(&symbols[i], &symbols[k])
                    && graph.has_edge(&symbols[j], &symbols[k])
                {
                    triangles.push((
                        symbols[i].clone(),
                        symbols[j].clone(),
                        symbols[k].clone(),
                    ));
                }
            }
        }
    }

    triangles
}

/// Find all cliques up to specified size using Bron-Kerbosch algorithm
pub fn find_cliques(graph: &FinancialGraph, max_size: usize) -> Vec<Vec<String>> {
    let mut cliques = Vec::new();
    let symbols: Vec<_> = graph.symbols().into_iter().cloned().collect();

    // Build adjacency set for faster lookup
    let mut adj: HashMap<String, HashSet<String>> = HashMap::new();
    for symbol in &symbols {
        adj.insert(symbol.clone(), graph.neighbors(symbol).into_iter().collect());
    }

    // Bron-Kerbosch with pivoting
    let r: HashSet<String> = HashSet::new();
    let p: HashSet<String> = symbols.iter().cloned().collect();
    let x: HashSet<String> = HashSet::new();

    bron_kerbosch(&adj, r, p, x, max_size, &mut cliques);

    cliques
}

/// Recursive Bron-Kerbosch algorithm
fn bron_kerbosch(
    adj: &HashMap<String, HashSet<String>>,
    r: HashSet<String>,
    mut p: HashSet<String>,
    mut x: HashSet<String>,
    max_size: usize,
    cliques: &mut Vec<Vec<String>>,
) {
    if p.is_empty() && x.is_empty() {
        if r.len() >= 2 && r.len() <= max_size {
            let mut clique: Vec<_> = r.into_iter().collect();
            clique.sort();
            cliques.push(clique);
        }
        return;
    }

    if r.len() >= max_size {
        let mut clique: Vec<_> = r.into_iter().collect();
        clique.sort();
        cliques.push(clique);
        return;
    }

    // Choose pivot
    let pivot = p.union(&x).next().cloned();

    let vertices: Vec<_> = if let Some(ref u) = pivot {
        p.difference(adj.get(u).unwrap_or(&HashSet::new()))
            .cloned()
            .collect()
    } else {
        p.iter().cloned().collect()
    };

    for v in vertices {
        let neighbors = adj.get(&v).cloned().unwrap_or_default();

        let mut new_r = r.clone();
        new_r.insert(v.clone());

        let new_p: HashSet<_> = p.intersection(&neighbors).cloned().collect();
        let new_x: HashSet<_> = x.intersection(&neighbors).cloned().collect();

        bron_kerbosch(adj, new_r, new_p, new_x, max_size, cliques);

        p.remove(&v);
        x.insert(v);
    }
}

/// Find star patterns (hub with many connections)
///
/// Returns list of (hub, spokes) tuples where hub has >= min_spokes connections
pub fn find_stars(graph: &FinancialGraph, min_spokes: usize) -> Vec<(String, Vec<String>)> {
    let mut stars = Vec::new();

    for symbol in graph.symbols() {
        let neighbors = graph.neighbors(symbol);
        if neighbors.len() >= min_spokes {
            stars.push((symbol.clone(), neighbors));
        }
    }

    // Sort by number of spokes (most connected first)
    stars.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    stars
}

/// Compute degree centrality for all nodes
///
/// Returns map of symbol -> centrality score (normalized)
pub fn compute_centrality(graph: &FinancialGraph) -> HashMap<String, f64> {
    let mut centrality = HashMap::new();
    let n = graph.node_count();

    if n <= 1 {
        for symbol in graph.symbols() {
            centrality.insert(symbol.clone(), 0.0);
        }
        return centrality;
    }

    let max_degree = (n - 1) as f64;

    for symbol in graph.symbols() {
        let degree = graph.neighbors(symbol).len() as f64;
        centrality.insert(symbol.clone(), degree / max_degree);
    }

    centrality
}

/// Find connected components in the graph
pub fn connected_components(graph: &FinancialGraph) -> Vec<Vec<String>> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut components = Vec::new();

    for symbol in graph.symbols() {
        if visited.contains(symbol) {
            continue;
        }

        let mut component = Vec::new();
        let mut stack = vec![symbol.clone()];

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }

            visited.insert(current.clone());
            component.push(current.clone());

            for neighbor in graph.neighbors(&current) {
                if !visited.contains(&neighbor) {
                    stack.push(neighbor);
                }
            }
        }

        if !component.is_empty() {
            component.sort();
            components.push(component);
        }
    }

    // Sort by size (largest first)
    components.sort_by(|a, b| b.len().cmp(&a.len()));
    components
}

/// Find chains (paths) of specified length
pub fn find_chains(graph: &FinancialGraph, length: usize) -> Vec<Vec<String>> {
    if length < 2 {
        return Vec::new();
    }

    let mut chains = Vec::new();
    let symbols: Vec<_> = graph.symbols().into_iter().cloned().collect();

    for start in &symbols {
        let mut visited = HashSet::new();
        visited.insert(start.clone());
        find_chains_dfs(graph, start.clone(), length, &mut visited, &mut chains);
    }

    // Remove duplicate chains (reverse order is same chain)
    let mut unique_chains = Vec::new();
    let mut seen: HashSet<Vec<String>> = HashSet::new();

    for chain in chains {
        let mut reversed = chain.clone();
        reversed.reverse();

        if !seen.contains(&chain) && !seen.contains(&reversed) {
            seen.insert(chain.clone());
            unique_chains.push(chain);
        }
    }

    unique_chains
}

fn find_chains_dfs(
    graph: &FinancialGraph,
    current: String,
    remaining: usize,
    visited: &mut HashSet<String>,
    chains: &mut Vec<Vec<String>>,
) {
    if remaining == 1 {
        // Reached target length
        let chain: Vec<_> = visited.iter().cloned().collect();
        chains.push(chain);
        return;
    }

    for neighbor in graph.neighbors(&current) {
        if visited.contains(&neighbor) {
            continue;
        }

        visited.insert(neighbor.clone());
        find_chains_dfs(graph, neighbor, remaining - 1, visited, chains);
        visited.remove(&neighbor);
    }
}

/// Graph statistics structure
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub num_triangles: usize,
    pub clustering_coefficient: f64,
    pub num_components: usize,
}

/// Compute comprehensive statistics for the graph
pub fn compute_stats(graph: &FinancialGraph) -> GraphStats {
    let triangles = find_triangles(graph);
    let components = connected_components(graph);

    let degrees: Vec<_> = graph
        .symbols()
        .iter()
        .map(|s| graph.neighbors(s).len())
        .collect();

    let max_degree = degrees.iter().max().copied().unwrap_or(0);

    // Compute clustering coefficient
    let mut total_clustering = 0.0;
    let mut count = 0;

    for symbol in graph.symbols() {
        let neighbors = graph.neighbors(symbol);
        let k = neighbors.len();

        if k < 2 {
            continue;
        }

        // Count edges between neighbors
        let mut edges_between_neighbors = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if graph.has_edge(&neighbors[i], &neighbors[j]) {
                    edges_between_neighbors += 1;
                }
            }
        }

        let max_edges = k * (k - 1) / 2;
        if max_edges > 0 {
            total_clustering += edges_between_neighbors as f64 / max_edges as f64;
            count += 1;
        }
    }

    let clustering_coefficient = if count > 0 {
        total_clustering / count as f64
    } else {
        0.0
    };

    GraphStats {
        node_count: graph.node_count(),
        edge_count: graph.edge_count(),
        density: graph.density(),
        avg_degree: graph.average_degree(),
        max_degree,
        num_triangles: triangles.len(),
        clustering_coefficient,
        num_components: components.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphType, Node};

    fn create_test_graph() -> FinancialGraph {
        let mut graph = FinancialGraph::new(GraphType::Correlation);

        // Create a triangle + one extra node
        graph.add_node(Node::new("A"));
        graph.add_node(Node::new("B"));
        graph.add_node(Node::new("C"));
        graph.add_node(Node::new("D"));

        graph.add_edge("A", "B", 0.9);
        graph.add_edge("B", "C", 0.85);
        graph.add_edge("A", "C", 0.8);
        graph.add_edge("C", "D", 0.75);

        graph
    }

    #[test]
    fn test_find_triangles() {
        let graph = create_test_graph();
        let triangles = find_triangles(&graph);

        assert_eq!(triangles.len(), 1);
        // Should find A-B-C triangle
    }

    #[test]
    fn test_find_stars() {
        let graph = create_test_graph();
        let stars = find_stars(&graph, 2);

        // C has 3 connections (A, B, D), so it's a star
        assert!(!stars.is_empty());
        assert!(stars.iter().any(|(hub, _)| hub == "C"));
    }

    #[test]
    fn test_compute_centrality() {
        let graph = create_test_graph();
        let centrality = compute_centrality(&graph);

        // C should have highest centrality (3 connections)
        assert!(centrality["C"] > centrality["D"]);
    }

    #[test]
    fn test_connected_components() {
        let graph = create_test_graph();
        let components = connected_components(&graph);

        // All nodes are connected
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 4);
    }
}
