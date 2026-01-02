//! Visibility graph construction from time series.
//!
//! A visibility graph connects time points that can "see" each other
//! without being blocked by intermediate points.

use super::MarketGraph;
use crate::data::OHLCV;

/// Visibility graph builder
#[derive(Debug, Clone)]
pub struct VisibilityGraph {
    /// Whether to use natural visibility (vs horizontal)
    natural: bool,
    /// Whether to use weighted edges
    weighted: bool,
}

impl Default for VisibilityGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl VisibilityGraph {
    /// Create a new visibility graph builder
    pub fn new() -> Self {
        Self {
            natural: true,
            weighted: false,
        }
    }

    /// Use natural visibility algorithm
    pub fn natural(mut self) -> Self {
        self.natural = true;
        self
    }

    /// Use horizontal visibility algorithm
    pub fn horizontal(mut self) -> Self {
        self.natural = false;
        self
    }

    /// Use weighted edges (weight = angle or distance)
    pub fn weighted(mut self) -> Self {
        self.weighted = true;
        self
    }

    /// Build visibility graph from price series
    pub fn build(&self, prices: &[f64]) -> MarketGraph {
        let n = prices.len();
        if n < 2 {
            return MarketGraph::new();
        }

        // Create nodes as time indices
        let symbols: Vec<String> = (0..n).map(|i| format!("t{}", i)).collect();
        let mut graph = MarketGraph::with_symbols(&symbols);

        if self.natural {
            self.build_natural(&symbols, prices, &mut graph);
        } else {
            self.build_horizontal(&symbols, prices, &mut graph);
        }

        graph
    }

    /// Build visibility graph from OHLCV data
    pub fn build_from_ohlcv(&self, candles: &[OHLCV]) -> MarketGraph {
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        self.build(&prices)
    }

    /// Natural visibility graph algorithm
    ///
    /// Two points (t_a, y_a) and (t_b, y_b) are connected if for all t_c
    /// between them: y_c < y_a + (y_b - y_a) * (t_c - t_a) / (t_b - t_a)
    fn build_natural(&self, symbols: &[String], prices: &[f64], graph: &mut MarketGraph) {
        let n = prices.len();

        for i in 0..n {
            for j in (i + 1)..n {
                if self.can_see_natural(prices, i, j) {
                    let weight = if self.weighted {
                        self.calculate_angle(prices, i, j)
                    } else {
                        1.0
                    };
                    graph.add_edge(&symbols[i], &symbols[j], weight);
                }
            }
        }
    }

    /// Check if point i can see point j (natural visibility)
    fn can_see_natural(&self, prices: &[f64], i: usize, j: usize) -> bool {
        if j <= i + 1 {
            return j == i + 1; // Adjacent points always visible
        }

        let y_a = prices[i];
        let y_b = prices[j];
        let t_a = i as f64;
        let t_b = j as f64;

        // Check all intermediate points
        for k in (i + 1)..j {
            let t_c = k as f64;
            let y_c = prices[k];

            // Calculate the line height at t_c
            let line_height = y_a + (y_b - y_a) * (t_c - t_a) / (t_b - t_a);

            if y_c >= line_height {
                return false; // Point k blocks visibility
            }
        }

        true
    }

    /// Horizontal visibility graph algorithm
    ///
    /// Two points are connected if there's a horizontal line between them
    /// that doesn't intersect any intermediate points
    fn build_horizontal(&self, symbols: &[String], prices: &[f64], graph: &mut MarketGraph) {
        let n = prices.len();

        for i in 0..n {
            for j in (i + 1)..n {
                if self.can_see_horizontal(prices, i, j) {
                    let weight = if self.weighted {
                        1.0 / (j - i) as f64 // Inverse distance
                    } else {
                        1.0
                    };
                    graph.add_edge(&symbols[i], &symbols[j], weight);
                }
            }
        }
    }

    /// Check if point i can see point j (horizontal visibility)
    fn can_see_horizontal(&self, prices: &[f64], i: usize, j: usize) -> bool {
        if j <= i + 1 {
            return j == i + 1;
        }

        let min_height = prices[i].min(prices[j]);

        // All intermediate points must be below both endpoints
        for k in (i + 1)..j {
            if prices[k] >= min_height {
                return false;
            }
        }

        true
    }

    /// Calculate angle between two points (for weighted edges)
    fn calculate_angle(&self, prices: &[f64], i: usize, j: usize) -> f64 {
        let dy = prices[j] - prices[i];
        let dx = (j - i) as f64;
        dy.atan2(dx)
    }
}

/// Analyze visibility graph properties
pub struct VisibilityAnalysis {
    /// Degree distribution
    pub degree_distribution: Vec<usize>,
    /// Average degree
    pub average_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Average path length
    pub average_path_length: f64,
}

impl VisibilityAnalysis {
    /// Analyze a visibility graph
    pub fn analyze(graph: &MarketGraph) -> Self {
        let n = graph.node_count();
        if n == 0 {
            return Self {
                degree_distribution: Vec::new(),
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                average_path_length: 0.0,
            };
        }

        // Calculate degree distribution
        let symbols = graph.symbols();
        let degrees: Vec<usize> = symbols.iter().map(|s| graph.degree(s)).collect();

        let max_degree = *degrees.iter().max().unwrap_or(&0);
        let mut distribution = vec![0usize; max_degree + 1];
        for &d in &degrees {
            distribution[d] += 1;
        }

        let avg_degree = degrees.iter().sum::<usize>() as f64 / n as f64;

        Self {
            degree_distribution: distribution,
            average_degree: avg_degree,
            clustering_coefficient: 0.0, // TODO: implement
            average_path_length: 0.0,    // TODO: implement
        }
    }
}

/// Detect trend reversals using visibility graph
pub fn detect_reversals(prices: &[f64], min_prominence: f64) -> Vec<usize> {
    let n = prices.len();
    if n < 3 {
        return Vec::new();
    }

    let mut reversals = Vec::new();

    for i in 1..(n - 1) {
        // Local maximum
        if prices[i] > prices[i - 1] && prices[i] > prices[i + 1] {
            let prominence = (prices[i] - prices[i - 1])
                .min(prices[i] - prices[i + 1]);
            if prominence >= min_prominence {
                reversals.push(i);
            }
        }

        // Local minimum
        if prices[i] < prices[i - 1] && prices[i] < prices[i + 1] {
            let prominence = (prices[i - 1] - prices[i])
                .min(prices[i + 1] - prices[i]);
            if prominence >= min_prominence {
                reversals.push(i);
            }
        }
    }

    reversals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_visibility_simple() {
        // Simple increasing sequence - all should be visible
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vg = VisibilityGraph::new();
        let graph = vg.build(&prices);

        // First and last should be visible (no blocking)
        assert!(graph.edge_weight("t0", "t4").is_some());
    }

    #[test]
    fn test_natural_visibility_blocked() {
        // Peak in the middle blocks visibility
        let prices = vec![1.0, 5.0, 1.0];
        let vg = VisibilityGraph::new();
        let graph = vg.build(&prices);

        // t0 and t2 should NOT be visible (blocked by t1)
        assert!(graph.edge_weight("t0", "t2").is_none());

        // But adjacent pairs should be visible
        assert!(graph.edge_weight("t0", "t1").is_some());
        assert!(graph.edge_weight("t1", "t2").is_some());
    }

    #[test]
    fn test_horizontal_visibility() {
        let prices = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let vg = VisibilityGraph::new().horizontal();
        let graph = vg.build(&prices);

        // t0 and t2 should be visible (t1 is below both)
        assert!(graph.edge_weight("t0", "t2").is_some());
    }

    #[test]
    fn test_detect_reversals() {
        let prices = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0];
        let reversals = detect_reversals(&prices, 0.5);

        // Should detect peak at index 2 and trough at index 4
        assert!(reversals.contains(&2));
        assert!(reversals.contains(&4));
    }
}
