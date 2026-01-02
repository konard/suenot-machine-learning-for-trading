//! Graph construction methods
//!
//! Build asset graphs from market data using various methods:
//! - Correlation-based
//! - Sector-based
//! - k-Nearest Neighbors
//! - Fully connected

use super::SparseGraph;
use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use ordered_float::OrderedFloat;

/// Asset sectors for cryptocurrency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CryptoSector {
    Layer1,
    Layer2,
    DeFi,
    Meme,
    AI,
    Gaming,
    Exchange,
    Stablecoin,
    Other,
}

impl CryptoSector {
    /// Get sector for a symbol
    pub fn from_symbol(symbol: &str) -> Self {
        let symbol = symbol.to_uppercase();
        let base = symbol.trim_end_matches("USDT").trim_end_matches("USD");

        match base {
            "BTC" | "ETH" | "SOL" | "AVAX" | "NEAR" | "ADA" | "DOT" | "ATOM" => CryptoSector::Layer1,
            "MATIC" | "ARB" | "OP" | "IMX" | "STRK" => CryptoSector::Layer2,
            "UNI" | "AAVE" | "MKR" | "CRV" | "COMP" | "SUSHI" | "DYDX" => CryptoSector::DeFi,
            "DOGE" | "SHIB" | "PEPE" | "BONK" | "WIF" | "FLOKI" => CryptoSector::Meme,
            "FET" | "RNDR" | "AGIX" | "OCEAN" | "TAO" => CryptoSector::AI,
            "AXS" | "SAND" | "MANA" | "GALA" | "IMX" => CryptoSector::Gaming,
            "BNB" | "FTT" | "OKB" | "CRO" | "KCS" => CryptoSector::Exchange,
            "USDT" | "USDC" | "DAI" | "BUSD" => CryptoSector::Stablecoin,
            _ => CryptoSector::Other,
        }
    }
}

/// Graph builder for constructing asset relationship graphs
pub struct GraphBuilder;

impl GraphBuilder {
    /// Build graph from correlation matrix
    ///
    /// Connects assets with absolute correlation above threshold.
    pub fn from_correlation(returns: &Array2<f64>, threshold: f64) -> Result<SparseGraph> {
        let n = returns.ncols();
        let corr_matrix = Self::compute_correlation_matrix(returns);

        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let corr = corr_matrix[[i, j]];
                    if corr.abs() > threshold {
                        edges.push((i, j, corr.abs()));
                    }
                }
            }
        }

        Ok(SparseGraph::from_edges(n, &edges))
    }

    /// Build k-Nearest Neighbors graph
    ///
    /// Each asset is connected to its k most correlated assets.
    pub fn from_knn(returns: &Array2<f64>, k: usize) -> Result<SparseGraph> {
        let n = returns.ncols();
        let corr_matrix = Self::compute_correlation_matrix(returns);

        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            // Get correlations for this asset (excluding self)
            let mut correlations: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, corr_matrix[[i, j]].abs()))
                .collect();

            // Sort by correlation (descending)
            correlations.sort_by_key(|&(_, corr)| std::cmp::Reverse(OrderedFloat(corr)));

            // Connect to top-k neighbors
            for (j, corr) in correlations.into_iter().take(k) {
                edges.push((i, j, corr));
            }
        }

        Ok(SparseGraph::from_edges(n, &edges))
    }

    /// Build sector-based graph
    ///
    /// Connects assets within the same sector.
    pub fn from_sectors(symbols: &[&str]) -> Result<SparseGraph> {
        let n = symbols.len();
        let sectors: Vec<CryptoSector> = symbols.iter().map(|s| CryptoSector::from_symbol(s)).collect();

        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j && sectors[i] == sectors[j] {
                    edges.push((i, j, 1.0));
                }
            }
        }

        let labels: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
        Ok(SparseGraph::from_edges(n, &edges).with_labels(labels))
    }

    /// Build fully connected graph
    ///
    /// All assets are connected (let attention learn sparsity).
    pub fn fully_connected(n: usize) -> SparseGraph {
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges.push((i, j, 1.0));
                }
            }
        }

        SparseGraph::from_edges(n, &edges)
    }

    /// Build hybrid graph (correlation + sector)
    pub fn hybrid(
        returns: &Array2<f64>,
        symbols: &[&str],
        corr_threshold: f64,
        sector_weight: f64,
    ) -> Result<SparseGraph> {
        let n = returns.ncols();
        assert_eq!(n, symbols.len());

        let corr_matrix = Self::compute_correlation_matrix(returns);
        let sectors: Vec<CryptoSector> = symbols.iter().map(|s| CryptoSector::from_symbol(s)).collect();

        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let corr = corr_matrix[[i, j]].abs();
                    let same_sector = if sectors[i] == sectors[j] { 1.0 } else { 0.0 };

                    let weight = corr + sector_weight * same_sector;

                    if weight > corr_threshold {
                        edges.push((i, j, weight));
                    }
                }
            }
        }

        let labels: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
        Ok(SparseGraph::from_edges(n, &edges).with_labels(labels))
    }

    /// Compute correlation matrix from returns
    pub fn compute_correlation_matrix(returns: &Array2<f64>) -> Array2<f64> {
        let n = returns.ncols();
        let mut corr = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let r = Self::pearson_correlation(returns.column(i), returns.column(j));
                corr[[i, j]] = r;
                corr[[j, i]] = r;
            }
        }

        corr
    }

    /// Compute Pearson correlation between two arrays
    fn pearson_correlation(x: ndarray::ArrayView1<f64>, y: ndarray::ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x == 0.0 || var_y == 0.0 {
            return 0.0;
        }

        cov / (var_x.sqrt() * var_y.sqrt())
    }

    /// Create a sample adjacency matrix for testing
    pub fn sample_adjacency(n: usize) -> Array2<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut adj = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Create some random edges with higher probability for nearby indices
                    let distance = (i as i64 - j as i64).abs() as f64;
                    let prob = 0.8 / (1.0 + distance);
                    if rng.gen::<f64>() < prob {
                        adj[[i, j]] = 1.0;
                    }
                }
            }
        }

        adj
    }

    /// Build time-varying graph (rolling correlation)
    pub fn rolling_correlation(
        returns: &Array2<f64>,
        window: usize,
        threshold: f64,
    ) -> Result<Vec<SparseGraph>> {
        let t = returns.nrows();
        let n = returns.ncols();
        let mut graphs = Vec::new();

        for start in 0..=(t - window) {
            let window_returns = returns.slice(ndarray::s![start..start + window, ..]);
            let corr_matrix = Self::compute_correlation_matrix(&window_returns.to_owned());

            let mut edges: Vec<(usize, usize, f64)> = Vec::new();

            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let corr = corr_matrix[[i, j]];
                        if corr.abs() > threshold {
                            edges.push((i, j, corr.abs()));
                        }
                    }
                }
            }

            graphs.push(SparseGraph::from_edges(n, &edges));
        }

        Ok(graphs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_correlation_graph() {
        let returns = Array2::random((100, 5), StandardNormal);
        let graph = GraphBuilder::from_correlation(&returns, 0.0).unwrap();

        assert_eq!(graph.num_nodes(), 5);
        assert!(graph.num_edges() > 0);
    }

    #[test]
    fn test_knn_graph() {
        let returns = Array2::random((100, 5), StandardNormal);
        let graph = GraphBuilder::from_knn(&returns, 2).unwrap();

        assert_eq!(graph.num_nodes(), 5);
        // Each node should have exactly 2 neighbors
        for i in 0..5 {
            assert_eq!(graph.degree(i), 2);
        }
    }

    #[test]
    fn test_sector_graph() {
        let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "UNIUSDT", "AAVEUSDT"];
        let graph = GraphBuilder::from_sectors(&symbols).unwrap();

        // BTC, ETH, SOL are Layer1 - should be connected
        assert!(graph.has_edge(0, 1)); // BTC -> ETH
        assert!(graph.has_edge(1, 2)); // ETH -> SOL

        // UNI, AAVE are DeFi - should be connected
        assert!(graph.has_edge(3, 4)); // UNI -> AAVE
    }

    #[test]
    fn test_fully_connected() {
        let graph = GraphBuilder::fully_connected(4);

        assert_eq!(graph.num_nodes(), 4);
        assert_eq!(graph.num_edges(), 12); // 4 * 3 edges
    }

    #[test]
    fn test_crypto_sectors() {
        assert_eq!(CryptoSector::from_symbol("BTCUSDT"), CryptoSector::Layer1);
        assert_eq!(CryptoSector::from_symbol("UNIUSDT"), CryptoSector::DeFi);
        assert_eq!(CryptoSector::from_symbol("DOGEUSDT"), CryptoSector::Meme);
        assert_eq!(CryptoSector::from_symbol("ARBUSDT"), CryptoSector::Layer2);
    }
}
