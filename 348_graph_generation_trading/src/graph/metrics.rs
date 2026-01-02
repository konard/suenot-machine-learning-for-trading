//! Graph metrics and centrality measures.

use super::MarketGraph;
use std::collections::{HashMap, VecDeque};

/// Graph metrics calculator
#[derive(Debug)]
pub struct GraphMetrics<'a> {
    graph: &'a MarketGraph,
}

impl<'a> GraphMetrics<'a> {
    /// Create a new metrics calculator for a graph
    pub fn new(graph: &'a MarketGraph) -> Self {
        Self { graph }
    }

    /// Calculate degree centrality for all nodes
    ///
    /// Degree centrality = degree / (n - 1)
    pub fn degree_centrality(&self) -> HashMap<String, f64> {
        let n = self.graph.node_count();
        if n <= 1 {
            return HashMap::new();
        }

        let max_degree = (n - 1) as f64;

        self.graph
            .symbols()
            .into_iter()
            .map(|symbol| {
                let degree = self.graph.degree(&symbol) as f64;
                (symbol, degree / max_degree)
            })
            .collect()
    }

    /// Calculate betweenness centrality for all nodes
    ///
    /// Betweenness centrality measures how often a node appears on
    /// shortest paths between other nodes.
    pub fn betweenness_centrality(&self) -> HashMap<String, f64> {
        let symbols = self.graph.symbols();
        let n = symbols.len();

        if n <= 2 {
            return symbols.into_iter().map(|s| (s, 0.0)).collect();
        }

        let mut centrality: HashMap<String, f64> = symbols.iter().map(|s| (s.clone(), 0.0)).collect();

        // Brandes algorithm
        for source in &symbols {
            let (predecessors, distances, num_paths) = self.bfs_predecessors(source);

            // Accumulate dependency
            let mut dependency: HashMap<String, f64> =
                symbols.iter().map(|s| (s.clone(), 0.0)).collect();

            // Process nodes in order of decreasing distance
            let mut nodes_by_distance: Vec<_> = distances.iter().collect();
            nodes_by_distance.sort_by(|a, b| b.1.cmp(a.1));

            for (node, _) in nodes_by_distance {
                if let Some(preds) = predecessors.get(node) {
                    let sigma_w = *num_paths.get(node).unwrap_or(&1) as f64;

                    for pred in preds {
                        let sigma_v = *num_paths.get(pred).unwrap_or(&1) as f64;
                        let delta = (sigma_v / sigma_w) * (1.0 + dependency.get(node).unwrap_or(&0.0));

                        *dependency.get_mut(pred).unwrap() += delta;
                    }
                }

                if node != source {
                    *centrality.get_mut(node).unwrap() += *dependency.get(node).unwrap_or(&0.0);
                }
            }
        }

        // Normalize
        let scale = if n > 2 {
            1.0 / ((n - 1) * (n - 2)) as f64
        } else {
            1.0
        };

        for value in centrality.values_mut() {
            *value *= scale;
        }

        centrality
    }

    /// BFS to find predecessors, distances, and number of shortest paths
    fn bfs_predecessors(
        &self,
        source: &str,
    ) -> (
        HashMap<String, Vec<String>>,
        HashMap<String, usize>,
        HashMap<String, usize>,
    ) {
        let symbols = self.graph.symbols();
        let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();
        let mut distances: HashMap<String, usize> = HashMap::new();
        let mut num_paths: HashMap<String, usize> = HashMap::new();

        for symbol in &symbols {
            predecessors.insert(symbol.clone(), Vec::new());
            distances.insert(symbol.clone(), usize::MAX);
            num_paths.insert(symbol.clone(), 0);
        }

        distances.insert(source.to_string(), 0);
        num_paths.insert(source.to_string(), 1);

        let mut queue = VecDeque::new();
        queue.push_back(source.to_string());

        while let Some(current) = queue.pop_front() {
            let current_dist = *distances.get(&current).unwrap();

            for neighbor in self.graph.neighbors(&current) {
                let neighbor_dist = *distances.get(&neighbor).unwrap();

                if neighbor_dist == usize::MAX {
                    // First visit
                    distances.insert(neighbor.clone(), current_dist + 1);
                    queue.push_back(neighbor.clone());
                }

                if *distances.get(&neighbor).unwrap() == current_dist + 1 {
                    // Shortest path through current
                    let current_paths = *num_paths.get(&current).unwrap();
                    *num_paths.get_mut(&neighbor).unwrap() += current_paths;
                    predecessors.get_mut(&neighbor).unwrap().push(current.clone());
                }
            }
        }

        (predecessors, distances, num_paths)
    }

    /// Calculate closeness centrality for all nodes
    ///
    /// Closeness centrality = (n-1) / sum of shortest path distances
    pub fn closeness_centrality(&self) -> HashMap<String, f64> {
        let symbols = self.graph.symbols();
        let n = symbols.len();

        if n <= 1 {
            return symbols.into_iter().map(|s| (s, 0.0)).collect();
        }

        symbols
            .into_iter()
            .map(|symbol| {
                let distances = self.shortest_path_distances(&symbol);
                let total_distance: usize = distances.values().filter(|&&d| d < usize::MAX).sum();

                let reachable = distances.values().filter(|&&d| d < usize::MAX).count();

                let closeness = if total_distance > 0 && reachable > 1 {
                    (reachable - 1) as f64 / total_distance as f64
                } else {
                    0.0
                };

                (symbol, closeness)
            })
            .collect()
    }

    /// BFS shortest path distances from a source
    fn shortest_path_distances(&self, source: &str) -> HashMap<String, usize> {
        let symbols = self.graph.symbols();
        let mut distances: HashMap<String, usize> =
            symbols.into_iter().map(|s| (s, usize::MAX)).collect();

        distances.insert(source.to_string(), 0);

        let mut queue = VecDeque::new();
        queue.push_back(source.to_string());

        while let Some(current) = queue.pop_front() {
            let current_dist = *distances.get(&current).unwrap();

            for neighbor in self.graph.neighbors(&current) {
                if *distances.get(&neighbor).unwrap() == usize::MAX {
                    distances.insert(neighbor.clone(), current_dist + 1);
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }

    /// Calculate eigenvector centrality (power iteration method)
    pub fn eigenvector_centrality(&self, max_iter: usize, tolerance: f64) -> HashMap<String, f64> {
        let symbols = self.graph.symbols();
        let n = symbols.len();

        if n == 0 {
            return HashMap::new();
        }

        // Initialize with uniform values
        let mut centrality: Vec<f64> = vec![1.0 / n as f64; n];
        let symbol_to_idx: HashMap<_, _> = symbols.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

        for _ in 0..max_iter {
            let mut new_centrality = vec![0.0; n];

            // Power iteration
            for (i, symbol) in symbols.iter().enumerate() {
                for neighbor in self.graph.neighbors(symbol) {
                    if let Some(&j) = symbol_to_idx.get(&neighbor) {
                        new_centrality[i] += centrality[j];
                    }
                }
            }

            // Normalize
            let norm: f64 = new_centrality.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut new_centrality {
                    *x /= norm;
                }
            }

            // Check convergence
            let diff: f64 = centrality
                .iter()
                .zip(new_centrality.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            centrality = new_centrality;

            if diff < tolerance {
                break;
            }
        }

        symbols
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, centrality[i]))
            .collect()
    }

    /// Calculate local clustering coefficient for a node
    pub fn local_clustering(&self, symbol: &str) -> f64 {
        let neighbors: Vec<_> = self.graph.neighbors(symbol);
        let k = neighbors.len();

        if k < 2 {
            return 0.0;
        }

        let mut edges_between_neighbors = 0;

        for i in 0..k {
            for j in (i + 1)..k {
                if self.graph.edge_weight(&neighbors[i], &neighbors[j]).is_some() {
                    edges_between_neighbors += 1;
                }
            }
        }

        let max_edges = k * (k - 1) / 2;
        edges_between_neighbors as f64 / max_edges as f64
    }

    /// Calculate average clustering coefficient
    pub fn average_clustering(&self) -> f64 {
        let symbols = self.graph.symbols();
        if symbols.is_empty() {
            return 0.0;
        }

        let total: f64 = symbols.iter().map(|s| self.local_clustering(s)).sum();
        total / symbols.len() as f64
    }

    /// Calculate graph diameter (longest shortest path)
    pub fn diameter(&self) -> usize {
        let symbols = self.graph.symbols();
        let mut max_distance = 0;

        for symbol in &symbols {
            let distances = self.shortest_path_distances(symbol);
            for &d in distances.values() {
                if d < usize::MAX && d > max_distance {
                    max_distance = d;
                }
            }
        }

        max_distance
    }

    /// Calculate average path length
    pub fn average_path_length(&self) -> f64 {
        let symbols = self.graph.symbols();
        let n = symbols.len();

        if n <= 1 {
            return 0.0;
        }

        let mut total_distance = 0usize;
        let mut count = 0usize;

        for symbol in &symbols {
            let distances = self.shortest_path_distances(symbol);
            for &d in distances.values() {
                if d > 0 && d < usize::MAX {
                    total_distance += d;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_distance as f64 / count as f64
        } else {
            0.0
        }
    }

    /// Detect hub nodes (high centrality)
    pub fn detect_hubs(&self, top_k: usize) -> Vec<(String, f64)> {
        let centrality = self.betweenness_centrality();
        let mut sorted: Vec<_> = centrality.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(top_k);
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> MarketGraph {
        let mut graph = MarketGraph::new();

        // Star topology: A is center
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("A", "C", 1.0);
        graph.add_edge("A", "D", 1.0);
        graph.add_edge("A", "E", 1.0);

        graph
    }

    #[test]
    fn test_degree_centrality() {
        let graph = create_test_graph();
        let metrics = GraphMetrics::new(&graph);
        let centrality = metrics.degree_centrality();

        // A should have highest degree centrality (connected to all others)
        assert!(centrality["A"] > centrality["B"]);
        assert!((centrality["A"] - 1.0).abs() < 1e-10); // 4/(5-1) = 1.0
    }

    #[test]
    fn test_betweenness_centrality() {
        let graph = create_test_graph();
        let metrics = GraphMetrics::new(&graph);
        let centrality = metrics.betweenness_centrality();

        // A should have highest betweenness (all paths go through A)
        for symbol in ["B", "C", "D", "E"] {
            assert!(centrality["A"] > centrality[symbol]);
        }
    }

    #[test]
    fn test_local_clustering() {
        let mut graph = MarketGraph::new();

        // Triangle
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("A", "C", 1.0);

        let metrics = GraphMetrics::new(&graph);

        // All nodes should have clustering coefficient = 1.0 (complete triangle)
        assert!((metrics.local_clustering("A") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diameter() {
        let graph = create_test_graph();
        let metrics = GraphMetrics::new(&graph);

        // In star topology, diameter is 2 (B->A->C)
        assert_eq!(metrics.diameter(), 2);
    }
}
