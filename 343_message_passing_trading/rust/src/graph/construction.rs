//! Graph construction utilities.
//!
//! Build market graphs from various data sources including:
//! - Correlation matrices
//! - Sector classifications
//! - Lead-lag relationships

use super::{EdgeType, MarketGraph};
use crate::data::Candle;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during graph construction.
#[derive(Error, Debug)]
pub enum ConstructionError {
    #[error("Insufficient data points: need at least {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Mismatched data lengths for symbols")]
    MismatchedLengths,

    #[error("No data available for symbol: {0}")]
    NoData(String),

    #[error("Invalid threshold: {0}")]
    InvalidThreshold(f64),
}

/// Builder for constructing market graphs.
pub struct GraphBuilder {
    /// Minimum correlation threshold for edges
    correlation_threshold: f64,
    /// Whether to use absolute correlation
    use_absolute_correlation: bool,
    /// Sector classifications
    sectors: HashMap<String, String>,
    /// Minimum data points required
    min_data_points: usize,
}

impl GraphBuilder {
    /// Create a new graph builder with default settings.
    pub fn new() -> Self {
        Self {
            correlation_threshold: 0.5,
            use_absolute_correlation: true,
            sectors: HashMap::new(),
            min_data_points: 30,
        }
    }

    /// Set the correlation threshold for edge creation.
    pub fn correlation_threshold(mut self, threshold: f64) -> Self {
        self.correlation_threshold = threshold;
        self
    }

    /// Set whether to use absolute correlation values.
    pub fn use_absolute_correlation(mut self, use_abs: bool) -> Self {
        self.use_absolute_correlation = use_abs;
        self
    }

    /// Add sector classifications.
    pub fn with_sectors(mut self, sectors: HashMap<String, String>) -> Self {
        self.sectors = sectors;
        self
    }

    /// Set minimum data points required.
    pub fn min_data_points(mut self, min: usize) -> Self {
        self.min_data_points = min;
        self
    }

    /// Build a graph from candle data using correlation.
    pub fn build_from_candles(
        &self,
        candles: &HashMap<String, Vec<Candle>>,
    ) -> Result<MarketGraph, ConstructionError> {
        // Calculate returns for each symbol
        let returns = self.calculate_returns(candles)?;

        // Compute correlation matrix
        let symbols: Vec<&String> = returns.keys().collect();
        let corr_matrix = self.compute_correlation_matrix(&returns, &symbols)?;

        // Build graph
        let mut graph = MarketGraph::new();

        // Add nodes with features
        for symbol in &symbols {
            let symbol_returns = returns.get(*symbol).unwrap();
            let features = self.compute_node_features(symbol_returns);
            graph.add_node(symbol.as_str(), features);
        }

        // Add edges based on correlation
        let n = symbols.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let corr = corr_matrix[[i, j]];
                let corr_value = if self.use_absolute_correlation {
                    corr.abs()
                } else {
                    corr
                };

                if corr_value >= self.correlation_threshold {
                    graph.add_edge(i, j, corr).with_type(EdgeType::Correlation);
                }
            }
        }

        // Add sector edges if available
        self.add_sector_edges(&mut graph, &symbols);

        Ok(graph)
    }

    /// Calculate returns from candle data.
    fn calculate_returns(
        &self,
        candles: &HashMap<String, Vec<Candle>>,
    ) -> Result<HashMap<String, Vec<f64>>, ConstructionError> {
        let mut returns = HashMap::new();

        for (symbol, data) in candles {
            if data.len() < self.min_data_points {
                return Err(ConstructionError::InsufficientData {
                    required: self.min_data_points,
                    actual: data.len(),
                });
            }

            let symbol_returns: Vec<f64> = data
                .windows(2)
                .map(|w| (w[1].close - w[0].close) / w[0].close)
                .collect();

            returns.insert(symbol.clone(), symbol_returns);
        }

        Ok(returns)
    }

    /// Compute correlation matrix from returns.
    fn compute_correlation_matrix(
        &self,
        returns: &HashMap<String, Vec<f64>>,
        symbols: &[&String],
    ) -> Result<Array2<f64>, ConstructionError> {
        let n = symbols.len();
        let mut corr = Array2::zeros((n, n));

        // Find minimum length
        let min_len = symbols
            .iter()
            .map(|s| returns.get(*s).map(|r| r.len()).unwrap_or(0))
            .min()
            .unwrap_or(0);

        if min_len < self.min_data_points - 1 {
            return Err(ConstructionError::InsufficientData {
                required: self.min_data_points - 1,
                actual: min_len,
            });
        }

        for i in 0..n {
            for j in i..n {
                if i == j {
                    corr[[i, j]] = 1.0;
                } else {
                    let r_i = &returns[symbols[i]][..min_len];
                    let r_j = &returns[symbols[j]][..min_len];
                    let c = pearson_correlation(r_i, r_j);
                    corr[[i, j]] = c;
                    corr[[j, i]] = c;
                }
            }
        }

        Ok(corr)
    }

    /// Compute node features from return series.
    fn compute_node_features(&self, returns: &[f64]) -> Array1<f64> {
        let n = returns.len() as f64;
        if n == 0.0 {
            return Array1::zeros(8);
        }

        // Mean return
        let mean: f64 = returns.iter().sum::<f64>() / n;

        // Volatility (std dev)
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let volatility = variance.sqrt();

        // Skewness
        let skewness = if volatility > 0.0 {
            returns.iter().map(|r| ((r - mean) / volatility).powi(3)).sum::<f64>() / n
        } else {
            0.0
        };

        // Kurtosis
        let kurtosis = if volatility > 0.0 {
            returns.iter().map(|r| ((r - mean) / volatility).powi(4)).sum::<f64>() / n - 3.0
        } else {
            0.0
        };

        // Max return
        let max_return = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Min return
        let min_return = returns.iter().cloned().fold(f64::INFINITY, f64::min);

        // Recent momentum (last 5 returns average)
        let recent_n = 5.min(returns.len());
        let momentum: f64 = returns[returns.len() - recent_n..].iter().sum::<f64>() / recent_n as f64;

        // Sharpe-like ratio
        let sharpe = if volatility > 0.0 { mean / volatility } else { 0.0 };

        Array1::from_vec(vec![
            mean,
            volatility,
            skewness,
            kurtosis,
            max_return,
            min_return,
            momentum,
            sharpe,
        ])
    }

    /// Add sector-based edges to the graph.
    fn add_sector_edges(&self, graph: &mut MarketGraph, symbols: &[&String]) {
        if self.sectors.is_empty() {
            return;
        }

        let n = symbols.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let sector_i = self.sectors.get(symbols[i].as_str());
                let sector_j = self.sectors.get(symbols[j].as_str());

                if let (Some(s_i), Some(s_j)) = (sector_i, sector_j) {
                    if s_i == s_j {
                        // Check if edge already exists
                        let edge_exists = graph
                            .edges
                            .iter()
                            .any(|e| (e.source == i && e.target == j) || (e.source == j && e.target == i));

                        if !edge_exists {
                            graph.add_edge(i, j, 0.5).with_type(EdgeType::Sector);
                        }
                    }
                }
            }
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate Pearson correlation coefficient between two series.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len().min(y.len()) {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x <= 0.0 || var_y <= 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Compute lead-lag relationships between return series.
pub fn compute_lead_lag(
    returns: &HashMap<String, Vec<f64>>,
    max_lag: usize,
) -> HashMap<(String, String), (i32, f64)> {
    let mut lead_lag = HashMap::new();
    let symbols: Vec<&String> = returns.keys().collect();

    for i in 0..symbols.len() {
        for j in (i + 1)..symbols.len() {
            let r_i = &returns[symbols[i]];
            let r_j = &returns[symbols[j]];

            let mut best_lag = 0i32;
            let mut best_corr = pearson_correlation(r_i, r_j);

            // Check positive lags (i leads j)
            for lag in 1..=max_lag {
                if lag >= r_i.len() || lag >= r_j.len() {
                    break;
                }
                let corr = pearson_correlation(&r_i[..r_i.len() - lag], &r_j[lag..]);
                if corr.abs() > best_corr.abs() {
                    best_lag = lag as i32;
                    best_corr = corr;
                }
            }

            // Check negative lags (j leads i)
            for lag in 1..=max_lag {
                if lag >= r_i.len() || lag >= r_j.len() {
                    break;
                }
                let corr = pearson_correlation(&r_i[lag..], &r_j[..r_j.len() - lag]);
                if corr.abs() > best_corr.abs() {
                    best_lag = -(lag as i32);
                    best_corr = corr;
                }
            }

            lead_lag.insert(
                (symbols[i].clone(), symbols[j].clone()),
                (best_lag, best_corr),
            );
        }
    }

    lead_lag
}

impl MarketGraph {
    /// Create a graph from correlation data.
    pub fn from_correlations(
        candles: &HashMap<String, Vec<Candle>>,
        threshold: f64,
    ) -> Result<Self, ConstructionError> {
        GraphBuilder::new()
            .correlation_threshold(threshold)
            .build_from_candles(candles)
    }

    /// Create a fully connected graph (useful for attention-based methods).
    pub fn fully_connected(symbols: &[&str], features: &[Array1<f64>]) -> Self {
        let mut graph = MarketGraph::new();

        // Add nodes
        for (symbol, feat) in symbols.iter().zip(features.iter()) {
            graph.add_node(*symbol, feat.clone());
        }

        // Add all edges
        let n = symbols.len();
        for i in 0..n {
            for j in (i + 1)..n {
                graph.add_edge(i, j, 1.0);
            }
        }

        graph
    }

    /// Create a K-nearest neighbors graph.
    pub fn knn(
        candles: &HashMap<String, Vec<Candle>>,
        k: usize,
    ) -> Result<Self, ConstructionError> {
        let builder = GraphBuilder::new().correlation_threshold(0.0);
        let returns = builder.calculate_returns(candles)?;
        let symbols: Vec<&String> = returns.keys().collect();
        let corr_matrix = builder.compute_correlation_matrix(&returns, &symbols)?;

        let mut graph = MarketGraph::new();

        // Add nodes
        for symbol in &symbols {
            let symbol_returns = returns.get(*symbol).unwrap();
            let features = builder.compute_node_features(symbol_returns);
            graph.add_node(symbol.as_str(), features);
        }

        // Add K nearest neighbor edges for each node
        let n = symbols.len();
        for i in 0..n {
            // Get correlations with all other nodes
            let mut correlations: Vec<(usize, f64)> = (0..n)
                .filter(|&j| i != j)
                .map(|j| (j, corr_matrix[[i, j]].abs()))
                .collect();

            // Sort by correlation (descending)
            correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Add top-k edges
            for (j, corr) in correlations.into_iter().take(k) {
                // Avoid duplicate edges
                let edge_exists = graph
                    .edges
                    .iter()
                    .any(|e| (e.source == i && e.target == j) || (e.source == j && e.target == i));

                if !edge_exists {
                    graph.add_edge(i, j, corr);
                }
            }
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candles(prices: &[f64]) -> Vec<Candle> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| Candle {
                timestamp: Utc::now().timestamp() as u64 + i as u64 * 3600,
                open: p,
                high: p * 1.01,
                low: p * 0.99,
                close: p,
                volume: 1000.0,
                symbol: "TEST".to_string(),
            })
            .collect()
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);

        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = pearson_correlation(&x, &z);
        assert!((corr_neg + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_graph_builder() {
        let mut candles = HashMap::new();

        // Create correlated price series
        let base_prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        let corr_prices: Vec<f64> = base_prices.iter().map(|p| p * 1.5 + 10.0).collect();
        let uncorr_prices: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        candles.insert("BTCUSDT".to_string(), make_candles(&base_prices));
        candles.insert("ETHUSDT".to_string(), make_candles(&corr_prices));
        candles.insert("RANDOM".to_string(), make_candles(&uncorr_prices));

        let graph = GraphBuilder::new()
            .correlation_threshold(0.7)
            .min_data_points(10)
            .build_from_candles(&candles)
            .unwrap();

        assert_eq!(graph.node_count(), 3);
        // BTC and ETH should be connected due to high correlation
        assert!(graph.edge_count() >= 1);
    }
}
