//! Correlation-based graph construction.

use super::{CryptoGraph, GraphBuilder};
use crate::data::features::pearson_correlation;

/// Build graphs based on correlation threshold.
pub struct CorrelationGraph {
    /// Correlation threshold for edge creation
    threshold: f64,
    /// Rolling window size for correlation calculation
    window: usize,
}

impl CorrelationGraph {
    /// Create a new correlation graph builder.
    ///
    /// # Arguments
    /// * `threshold` - Minimum correlation for edge creation (0.0 to 1.0)
    /// * `window` - Rolling window size for correlation calculation
    pub fn new(threshold: f64, window: usize) -> Self {
        Self { threshold, window }
    }

    /// Build graph from full return history.
    pub fn build_from_returns(&self, returns: &[Vec<f64>], symbols: &[String]) -> CryptoGraph {
        self.build(returns, symbols)
    }

    /// Build graph from windowed returns (last `window` observations).
    pub fn build_windowed(&self, returns: &[Vec<f64>], symbols: &[String]) -> CryptoGraph {
        let windowed: Vec<Vec<f64>> = returns
            .iter()
            .map(|r| {
                let start = r.len().saturating_sub(self.window);
                r[start..].to_vec()
            })
            .collect();

        self.build(&windowed, symbols)
    }

    /// Get correlation matrix.
    pub fn compute_correlation_matrix(&self, returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = returns.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let corr = pearson_correlation(&returns[i], &returns[j]);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }

        matrix
    }

    /// Analyze graph statistics.
    pub fn analyze(&self, returns: &[Vec<f64>], symbols: &[String]) -> CorrelationStats {
        let matrix = self.compute_correlation_matrix(returns);
        let graph = self.build(returns, symbols);

        let mut correlations = Vec::new();
        for i in 0..matrix.len() {
            for j in (i + 1)..matrix.len() {
                correlations.push(matrix[i][j]);
            }
        }

        correlations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        CorrelationStats {
            num_nodes: graph.node_count(),
            num_edges: graph.edge_count(),
            density: graph.density(),
            mean_correlation: correlations.iter().sum::<f64>() / correlations.len() as f64,
            median_correlation: correlations[correlations.len() / 2],
            min_correlation: *correlations.first().unwrap_or(&0.0),
            max_correlation: *correlations.last().unwrap_or(&0.0),
        }
    }
}

impl GraphBuilder for CorrelationGraph {
    fn build(&self, returns: &[Vec<f64>], symbols: &[String]) -> CryptoGraph {
        let mut graph = CryptoGraph::new();

        // Add all nodes first
        for symbol in symbols {
            graph.add_node(symbol);
        }

        // Compute correlations and add edges
        let n = returns.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let corr = pearson_correlation(&returns[i], &returns[j]);
                if corr.abs() >= self.threshold {
                    graph.add_edge(&symbols[i], &symbols[j], corr);
                }
            }
        }

        graph
    }
}

/// Statistics about correlation graph.
#[derive(Debug, Clone)]
pub struct CorrelationStats {
    /// Number of nodes in graph
    pub num_nodes: usize,
    /// Number of edges in graph
    pub num_edges: usize,
    /// Graph density
    pub density: f64,
    /// Mean correlation
    pub mean_correlation: f64,
    /// Median correlation
    pub median_correlation: f64,
    /// Minimum correlation
    pub min_correlation: f64,
    /// Maximum correlation
    pub max_correlation: f64,
}

/// Builder for sector-based graphs.
pub struct SectorGraph {
    /// Sector assignments
    sectors: std::collections::HashMap<String, String>,
}

impl SectorGraph {
    /// Create a new sector graph builder.
    pub fn new() -> Self {
        Self {
            sectors: std::collections::HashMap::new(),
        }
    }

    /// Add sector assignment.
    pub fn add_sector(&mut self, symbol: &str, sector: &str) {
        self.sectors.insert(symbol.to_string(), sector.to_string());
    }

    /// Build graph with intra-sector edges.
    pub fn build(&self, symbols: &[String]) -> CryptoGraph {
        let mut graph = CryptoGraph::new();

        // Add all nodes
        for symbol in symbols {
            graph.add_node(symbol);
        }

        // Connect nodes in same sector
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let sector_i = self.sectors.get(&symbols[i]);
                let sector_j = self.sectors.get(&symbols[j]);

                if let (Some(si), Some(sj)) = (sector_i, sector_j) {
                    if si == sj {
                        graph.add_edge(&symbols[i], &symbols[j], 1.0);
                    }
                }
            }
        }

        graph
    }

    /// Create default crypto sectors.
    pub fn with_crypto_sectors() -> Self {
        let mut builder = Self::new();

        // Layer 1 blockchains
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT", "ATOMUSDT"] {
            builder.add_sector(symbol, "Layer1");
        }

        // Layer 2 solutions
        for symbol in ["MATICUSDT", "ARBUSDT", "OPUSDT"] {
            builder.add_sector(symbol, "Layer2");
        }

        // DeFi tokens
        for symbol in ["UNIUSDT", "AAVEUSDT", "MKRUSDT", "COMPUSDT", "SUSHIUSDT"] {
            builder.add_sector(symbol, "DeFi");
        }

        // Meme coins
        for symbol in ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"] {
            builder.add_sector(symbol, "Meme");
        }

        // Exchange tokens
        for symbol in ["BNBUSDT", "FTMUSDT"] {
            builder.add_sector(symbol, "Exchange");
        }

        builder
    }
}

impl Default for SectorGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_graph() {
        let returns = vec![
            vec![0.01, 0.02, -0.01, 0.03, 0.01],
            vec![0.01, 0.02, -0.01, 0.03, 0.01], // Same as first - correlation = 1.0
            vec![-0.01, -0.02, 0.01, -0.03, -0.01], // Opposite - correlation = -1.0
        ];
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let builder = CorrelationGraph::new(0.9, 5);
        let graph = builder.build(&returns, &symbols);

        assert_eq!(graph.node_count(), 3);
        // A-B should be connected (corr = 1.0)
        // A-C should be connected (corr = -1.0, abs >= 0.9)
        assert!(graph.edge_count() >= 2);
    }

    #[test]
    fn test_sector_graph() {
        let builder = SectorGraph::with_crypto_sectors();
        let symbols = vec![
            "BTCUSDT".to_string(),
            "ETHUSDT".to_string(),
            "DOGEUSDT".to_string(),
        ];

        let graph = builder.build(&symbols);
        assert_eq!(graph.node_count(), 3);
        // BTC and ETH in same sector, DOGE separate
        assert_eq!(graph.edge_count(), 1);
    }
}
