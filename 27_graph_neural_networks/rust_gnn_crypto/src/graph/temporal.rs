//! Temporal (time-evolving) graph construction.

use super::{CorrelationGraph, CryptoGraph, GraphBuilder};
use std::collections::HashMap;

/// Time-evolving graph with snapshots at different timestamps.
pub struct TemporalGraph {
    /// Graph snapshots indexed by timestamp
    snapshots: HashMap<i64, CryptoGraph>,
    /// Correlation threshold
    threshold: f64,
    /// Window size for correlation
    window: usize,
    /// Update frequency in time units
    update_frequency: usize,
}

impl TemporalGraph {
    /// Create a new temporal graph.
    pub fn new(threshold: f64, window: usize, update_frequency: usize) -> Self {
        Self {
            snapshots: HashMap::new(),
            threshold,
            window,
            update_frequency,
        }
    }

    /// Build snapshots from time series data.
    ///
    /// # Arguments
    /// * `returns` - Returns for each symbol
    /// * `symbols` - Symbol names
    /// * `timestamps` - Timestamps for each observation
    pub fn build_snapshots(
        &mut self,
        returns: &[Vec<f64>],
        symbols: &[String],
        timestamps: &[i64],
    ) {
        let builder = CorrelationGraph::new(self.threshold, self.window);

        // Build snapshot at each update point
        for (i, &ts) in timestamps.iter().enumerate() {
            if i < self.window {
                continue;
            }
            if i % self.update_frequency != 0 {
                continue;
            }

            // Get windowed returns
            let start = i.saturating_sub(self.window);
            let windowed: Vec<Vec<f64>> = returns
                .iter()
                .map(|r| r[start..i].to_vec())
                .collect();

            let graph = builder.build(&windowed, symbols);
            self.snapshots.insert(ts, graph);
        }
    }

    /// Get graph snapshot at a specific timestamp.
    pub fn get_snapshot(&self, timestamp: i64) -> Option<&CryptoGraph> {
        // Find the most recent snapshot before or at timestamp
        let valid_ts: Vec<i64> = self
            .snapshots
            .keys()
            .filter(|&&ts| ts <= timestamp)
            .cloned()
            .collect();

        valid_ts
            .into_iter()
            .max()
            .and_then(|ts| self.snapshots.get(&ts))
    }

    /// Get all timestamps with snapshots.
    pub fn timestamps(&self) -> Vec<i64> {
        let mut ts: Vec<i64> = self.snapshots.keys().cloned().collect();
        ts.sort();
        ts
    }

    /// Get number of snapshots.
    pub fn num_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    /// Analyze graph evolution.
    pub fn analyze_evolution(&self) -> TemporalStats {
        let timestamps = self.timestamps();
        if timestamps.is_empty() {
            return TemporalStats::default();
        }

        let densities: Vec<f64> = timestamps
            .iter()
            .filter_map(|ts| self.snapshots.get(ts).map(|g| g.density()))
            .collect();

        let edge_counts: Vec<usize> = timestamps
            .iter()
            .filter_map(|ts| self.snapshots.get(ts).map(|g| g.edge_count()))
            .collect();

        // Count edge changes
        let mut edge_additions = 0;
        let mut edge_deletions = 0;

        for i in 1..timestamps.len() {
            let prev = self.snapshots.get(&timestamps[i - 1]);
            let curr = self.snapshots.get(&timestamps[i]);

            if let (Some(p), Some(c)) = (prev, curr) {
                let (add, del) = count_edge_changes(p, c);
                edge_additions += add;
                edge_deletions += del;
            }
        }

        TemporalStats {
            num_snapshots: timestamps.len(),
            mean_density: densities.iter().sum::<f64>() / densities.len() as f64,
            mean_edges: edge_counts.iter().sum::<usize>() as f64 / edge_counts.len() as f64,
            total_edge_additions: edge_additions,
            total_edge_deletions: edge_deletions,
            edge_turnover_rate: if timestamps.len() > 1 {
                (edge_additions + edge_deletions) as f64 / (timestamps.len() - 1) as f64
            } else {
                0.0
            },
        }
    }

    /// Get temporal edges for TGN training.
    /// Returns (source, target, timestamp, weight) tuples.
    pub fn get_temporal_edges(&self) -> Vec<(usize, usize, i64, f64)> {
        let mut edges = Vec::new();
        let timestamps = self.timestamps();

        // We need consistent node indexing across snapshots
        let mut symbol_to_idx: HashMap<String, usize> = HashMap::new();
        let mut next_idx = 0;

        for ts in &timestamps {
            if let Some(graph) = self.snapshots.get(ts) {
                // Ensure all symbols have indices
                for symbol in graph.symbols() {
                    if !symbol_to_idx.contains_key(&symbol) {
                        symbol_to_idx.insert(symbol, next_idx);
                        next_idx += 1;
                    }
                }

                // Add edges with timestamp
                for symbol_i in graph.symbols() {
                    for symbol_j in graph.neighbors(&symbol_i) {
                        if let (Some(&i), Some(&j)) =
                            (symbol_to_idx.get(&symbol_i), symbol_to_idx.get(&symbol_j))
                        {
                            if i < j {
                                // Avoid duplicates
                                let weight = graph.edge_weight(&symbol_i, &symbol_j).unwrap_or(1.0);
                                edges.push((i, j, *ts, weight));
                            }
                        }
                    }
                }
            }
        }

        edges
    }
}

/// Count edge additions and deletions between two graphs.
fn count_edge_changes(prev: &CryptoGraph, curr: &CryptoGraph) -> (usize, usize) {
    let prev_edges: std::collections::HashSet<(String, String)> = prev
        .symbols()
        .iter()
        .flat_map(|s| {
            prev.neighbors(s)
                .into_iter()
                .map(move |n| normalize_edge(s.clone(), n))
        })
        .collect();

    let curr_edges: std::collections::HashSet<(String, String)> = curr
        .symbols()
        .iter()
        .flat_map(|s| {
            curr.neighbors(s)
                .into_iter()
                .map(move |n| normalize_edge(s.clone(), n))
        })
        .collect();

    let additions = curr_edges.difference(&prev_edges).count();
    let deletions = prev_edges.difference(&curr_edges).count();

    (additions, deletions)
}

/// Normalize edge to canonical form (smaller symbol first).
fn normalize_edge(a: String, b: String) -> (String, String) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Statistics about temporal graph evolution.
#[derive(Debug, Clone, Default)]
pub struct TemporalStats {
    /// Number of snapshots
    pub num_snapshots: usize,
    /// Mean graph density
    pub mean_density: f64,
    /// Mean number of edges
    pub mean_edges: f64,
    /// Total edge additions across all transitions
    pub total_edge_additions: usize,
    /// Total edge deletions across all transitions
    pub total_edge_deletions: usize,
    /// Average edge turnover per transition
    pub edge_turnover_rate: f64,
}

/// Sliding window graph that maintains recent history.
pub struct SlidingWindowGraph {
    /// Recent graphs
    history: Vec<(i64, CryptoGraph)>,
    /// Maximum history length
    max_history: usize,
    /// Correlation builder
    builder: CorrelationGraph,
}

impl SlidingWindowGraph {
    /// Create a new sliding window graph.
    pub fn new(threshold: f64, window: usize, max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
            builder: CorrelationGraph::new(threshold, window),
        }
    }

    /// Update graph with new data.
    pub fn update(&mut self, returns: &[Vec<f64>], symbols: &[String], timestamp: i64) {
        let graph = self.builder.build(returns, symbols);
        self.history.push((timestamp, graph));

        // Trim old history
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get current graph.
    pub fn current(&self) -> Option<&CryptoGraph> {
        self.history.last().map(|(_, g)| g)
    }

    /// Get graph at specific timestamp.
    pub fn at_timestamp(&self, ts: i64) -> Option<&CryptoGraph> {
        self.history
            .iter()
            .find(|(t, _)| *t == ts)
            .map(|(_, g)| g)
    }

    /// Compute edge stability (how often edge exists in recent history).
    pub fn edge_stability(&self, from: &str, to: &str) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        let count = self
            .history
            .iter()
            .filter(|(_, g)| g.edge_weight(from, to).is_some())
            .count();

        count as f64 / self.history.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_graph() {
        let mut temporal = TemporalGraph::new(0.5, 10, 5);

        // Create sample returns
        let returns = vec![
            (0..20).map(|i| (i as f64 * 0.01)).collect::<Vec<f64>>(),
            (0..20).map(|i| (i as f64 * 0.01)).collect::<Vec<f64>>(),
        ];
        let symbols = vec!["A".to_string(), "B".to_string()];
        let timestamps: Vec<i64> = (0..20).collect();

        temporal.build_snapshots(&returns, &symbols, &timestamps);

        assert!(temporal.num_snapshots() > 0);
    }

    #[test]
    fn test_sliding_window() {
        let mut sliding = SlidingWindowGraph::new(0.5, 5, 3);

        let returns = vec![vec![0.01, 0.02, 0.03, 0.04, 0.05]];
        let symbols = vec!["A".to_string()];

        sliding.update(&returns, &symbols, 1);
        sliding.update(&returns, &symbols, 2);
        sliding.update(&returns, &symbols, 3);
        sliding.update(&returns, &symbols, 4);

        // Should only keep last 3
        assert_eq!(sliding.history.len(), 3);
        assert!(sliding.at_timestamp(1).is_none());
        assert!(sliding.at_timestamp(4).is_some());
    }
}
