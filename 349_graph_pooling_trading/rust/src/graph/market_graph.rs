//! Market graph representation

use crate::api::Kline;
use crate::graph::correlation::{CorrelationGraph, CorrelationMethod};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Features for each node (asset)
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    /// Latest return
    pub return_1: f64,

    /// 5-period momentum
    pub momentum_5: f64,

    /// Volatility (std of returns)
    pub volatility: f64,

    /// Volume ratio (current / average)
    pub volume_ratio: f64,

    /// Win rate (fraction of positive returns)
    pub win_rate: f64,

    /// Cumulative return
    pub cumulative_return: f64,

    /// Average range
    pub avg_range: f64,

    /// Return z-score
    pub return_zscore: f64,
}

impl NodeFeatures {
    /// Convert to feature vector
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.return_1,
            self.momentum_5,
            self.volatility,
            self.volume_ratio,
            self.win_rate,
            self.cumulative_return,
            self.avg_range,
            self.return_zscore,
        ]
    }

    /// Number of features
    pub const N_FEATURES: usize = 8;
}

/// Market graph combining node features and adjacency
pub struct MarketGraph {
    /// Asset symbols
    pub symbols: Vec<String>,

    /// Node features matrix [n_assets, n_features]
    pub features: Array2<f64>,

    /// Adjacency matrix [n_assets, n_assets]
    pub adjacency: Array2<f64>,

    /// Correlation matrix [n_assets, n_assets]
    pub correlations: Array2<f64>,

    /// Raw returns for each asset
    pub returns: HashMap<String, Vec<f64>>,

    /// Latest prices
    pub latest_prices: HashMap<String, f64>,
}

impl MarketGraph {
    /// Build a market graph from kline data
    ///
    /// # Arguments
    /// * `klines` - HashMap of symbol -> Vec<Kline>
    /// * `correlation_threshold` - Minimum correlation to include edge (default 0.3)
    pub fn from_klines(klines: &HashMap<String, Vec<Kline>>) -> anyhow::Result<Self> {
        Self::from_klines_with_threshold(klines, 0.3)
    }

    /// Build a market graph with custom threshold
    pub fn from_klines_with_threshold(
        klines: &HashMap<String, Vec<Kline>>,
        correlation_threshold: f64,
    ) -> anyhow::Result<Self> {
        let symbols: Vec<String> = klines.keys().cloned().collect();
        let n_assets = symbols.len();

        if n_assets == 0 {
            anyhow::bail!("No klines provided");
        }

        // Compute returns for each asset
        let mut returns: HashMap<String, Vec<f64>> = HashMap::new();
        let mut latest_prices: HashMap<String, f64> = HashMap::new();

        for (symbol, kline_data) in klines {
            let rets: Vec<f64> = kline_data
                .windows(2)
                .map(|w| (w[1].close / w[0].close).ln())
                .collect();
            returns.insert(symbol.clone(), rets);

            if let Some(last) = kline_data.last() {
                latest_prices.insert(symbol.clone(), last.close);
            }
        }

        // Build correlation graph
        let corr_graph = CorrelationGraph::from_returns(
            &returns,
            correlation_threshold,
            CorrelationMethod::Pearson,
        );

        // Compute features for each asset
        let mut features = Array2::<f64>::zeros((n_assets, NodeFeatures::N_FEATURES));

        for (i, symbol) in symbols.iter().enumerate() {
            if let Some(kline_data) = klines.get(symbol) {
                let node_features = Self::compute_node_features(kline_data);
                let feat_vec = node_features.to_vec();
                for (j, &f) in feat_vec.iter().enumerate() {
                    features[[i, j]] = f;
                }
            }
        }

        Ok(Self {
            symbols,
            features,
            adjacency: corr_graph.adjacency_matrix,
            correlations: corr_graph.correlation_matrix,
            returns,
            latest_prices,
        })
    }

    /// Compute node features from klines
    fn compute_node_features(klines: &[Kline]) -> NodeFeatures {
        if klines.is_empty() {
            return NodeFeatures {
                return_1: 0.0,
                momentum_5: 0.0,
                volatility: 0.0,
                volume_ratio: 0.0,
                win_rate: 0.0,
                cumulative_return: 0.0,
                avg_range: 0.0,
                return_zscore: 0.0,
            };
        }

        let returns: Vec<f64> = klines
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        if returns.is_empty() {
            return NodeFeatures {
                return_1: 0.0,
                momentum_5: 0.0,
                volatility: 0.0,
                volume_ratio: 0.0,
                win_rate: 0.0,
                cumulative_return: 0.0,
                avg_range: 0.0,
                return_zscore: 0.0,
            };
        }

        let n = returns.len();
        let return_1 = *returns.last().unwrap_or(&0.0);

        // 5-period momentum
        let momentum_5 = if n >= 5 {
            returns[n - 5..].iter().sum::<f64>() / 5.0
        } else {
            returns.iter().sum::<f64>() / n as f64
        };

        // Volatility
        let mean_return = returns.iter().sum::<f64>() / n as f64;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n as f64;
        let volatility = variance.sqrt();

        // Volume ratio
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let volume_ratio = if avg_volume > 0.0 {
            *volumes.last().unwrap_or(&0.0) / avg_volume
        } else {
            1.0
        };

        // Win rate
        let positive_returns = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = positive_returns as f64 / n as f64;

        // Cumulative return
        let cumulative_return = returns.iter().sum::<f64>();

        // Average range
        let ranges: Vec<f64> = klines.iter().map(|k| k.range()).collect();
        let avg_range = ranges.iter().sum::<f64>() / ranges.len() as f64;

        // Return z-score
        let return_zscore = if volatility > 0.0 {
            (return_1 - mean_return) / volatility
        } else {
            0.0
        };

        NodeFeatures {
            return_1,
            momentum_5,
            volatility,
            volume_ratio,
            win_rate,
            cumulative_return,
            avg_range,
            return_zscore,
        }
    }

    /// Get number of assets
    pub fn n_assets(&self) -> usize {
        self.symbols.len()
    }

    /// Get feature dimension
    pub fn n_features(&self) -> usize {
        NodeFeatures::N_FEATURES
    }

    /// Get node degrees (connectivity)
    pub fn degrees(&self) -> Array1<f64> {
        self.adjacency.sum_axis(ndarray::Axis(1))
    }

    /// Get the most connected assets
    pub fn hub_assets(&self, top_k: usize) -> Vec<(String, f64)> {
        let degrees = self.degrees();
        let mut indexed: Vec<(usize, f64)> = degrees.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .into_iter()
            .take(top_k)
            .map(|(i, d)| (self.symbols[i].clone(), d))
            .collect()
    }

    /// Get clusters of highly correlated assets (simple threshold-based)
    pub fn find_clusters(&self, min_correlation: f64) -> Vec<Vec<String>> {
        let n = self.n_assets();
        let mut visited = vec![false; n];
        let mut clusters = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }

            let mut cluster = Vec::new();
            let mut stack = vec![start];

            while let Some(node) = stack.pop() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;
                cluster.push(self.symbols[node].clone());

                // Find connected nodes
                for j in 0..n {
                    if !visited[j] && self.correlations[[node, j]].abs() >= min_correlation {
                        stack.push(j);
                    }
                }
            }

            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }

        clusters
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let degrees = self.degrees();
        let avg_degree = degrees.sum() / self.n_assets() as f64;
        let density = self
            .adjacency
            .iter()
            .filter(|&&x| x > 0.0)
            .count() as f64
            / (self.n_assets() * (self.n_assets() - 1)) as f64;

        let hubs = self.hub_assets(3);
        let hub_str: String = hubs
            .iter()
            .map(|(s, d)| format!("{}: {:.2}", s, d))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "MarketGraph Summary:\n\
             - Assets: {}\n\
             - Features: {}\n\
             - Avg Degree: {:.2}\n\
             - Density: {:.4}\n\
             - Top Hubs: {}",
            self.n_assets(),
            self.n_features(),
            avg_degree,
            density,
            hub_str
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(symbol: &str, n: usize, base_price: f64) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let price = base_price * (1.0 + 0.01 * (i as f64).sin());
                Kline {
                    symbol: symbol.to_string(),
                    open_time: 1000 * i as i64,
                    open: price,
                    high: price * 1.01,
                    low: price * 0.99,
                    close: price * (1.0 + 0.001 * i as f64),
                    volume: 1000.0,
                    turnover: 1000.0 * price,
                }
            })
            .collect()
    }

    #[test]
    fn test_market_graph_creation() {
        let mut klines = HashMap::new();
        klines.insert("BTCUSDT".to_string(), create_test_klines("BTCUSDT", 50, 50000.0));
        klines.insert("ETHUSDT".to_string(), create_test_klines("ETHUSDT", 50, 3000.0));

        let graph = MarketGraph::from_klines(&klines).unwrap();

        assert_eq!(graph.n_assets(), 2);
        assert_eq!(graph.n_features(), 8);
    }
}
