//! Graph construction from market data.

use crate::data::{Candle, MarketData};
use crate::graph::{Edge, FinancialGraph, GraphType, Node, NodeId};
use nalgebra::DMatrix;
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during graph building
#[derive(Error, Debug)]
pub enum BuilderError {
    #[error("Insufficient data: need at least {0} periods, got {1}")]
    InsufficientData(usize, usize),
    #[error("No valid returns computed")]
    NoReturns,
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Builder for constructing financial graphs from market data
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    /// Correlation threshold for edge creation
    correlation_threshold: f64,
    /// Window size for rolling calculations
    window_size: usize,
    /// Minimum volume filter
    min_volume: f64,
    /// Graph type to build
    graph_type: GraphType,
    /// Whether to use absolute correlation
    use_absolute_correlation: bool,
}

impl GraphBuilder {
    /// Create a new graph builder with default settings
    pub fn new() -> Self {
        Self {
            correlation_threshold: 0.7,
            window_size: 24,
            min_volume: 0.0,
            graph_type: GraphType::Correlation,
            use_absolute_correlation: false,
        }
    }

    /// Set correlation threshold for edge creation
    pub fn correlation_threshold(mut self, threshold: f64) -> Self {
        self.correlation_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set window size for calculations
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(2);
        self
    }

    /// Set minimum volume filter
    pub fn min_volume(mut self, volume: f64) -> Self {
        self.min_volume = volume.max(0.0);
        self
    }

    /// Set graph type
    pub fn graph_type(mut self, graph_type: GraphType) -> Self {
        self.graph_type = graph_type;
        self
    }

    /// Use absolute correlation values
    pub fn absolute_correlation(mut self, use_absolute: bool) -> Self {
        self.use_absolute_correlation = use_absolute;
        self
    }

    /// Build a correlation network from candle data
    ///
    /// # Arguments
    /// * `data` - Map of symbol to candle data
    ///
    /// # Returns
    /// * `FinancialGraph` - Constructed correlation network
    pub fn build_correlation_network(
        &self,
        data: &HashMap<String, Vec<Candle>>,
    ) -> Result<FinancialGraph, BuilderError> {
        // Validate data
        if data.is_empty() {
            return Err(BuilderError::InsufficientData(1, 0));
        }

        let min_length = data.values().map(|v| v.len()).min().unwrap_or(0);
        if min_length < self.window_size {
            return Err(BuilderError::InsufficientData(self.window_size, min_length));
        }

        // Compute returns for each symbol
        let returns = self.compute_returns(data)?;
        if returns.is_empty() {
            return Err(BuilderError::NoReturns);
        }

        // Compute correlation matrix
        let correlation_matrix = self.compute_correlation_matrix(&returns)?;

        // Build graph from correlation matrix
        let mut graph = FinancialGraph::new(self.graph_type);
        graph.window_size = self.window_size;
        graph.threshold = self.correlation_threshold;

        // Add nodes
        let symbols: Vec<_> = returns.keys().cloned().collect();
        for symbol in &symbols {
            let candles = data.get(symbol).unwrap();
            let last = candles.last().unwrap();

            let node = Node::new(symbol.clone())
                .with_price(last.close)
                .with_volume(last.volume);

            graph.add_node(node);
        }

        // Add edges based on correlation threshold
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let corr = correlation_matrix[(i, j)];
                let effective_corr = if self.use_absolute_correlation {
                    corr.abs()
                } else {
                    corr
                };

                if effective_corr >= self.correlation_threshold {
                    graph.add_edge(&symbols[i], &symbols[j], corr);
                }
            }
        }

        Ok(graph)
    }

    /// Build graph from pre-computed correlation matrix
    pub fn build_from_matrix(
        &self,
        symbols: &[String],
        matrix: &DMatrix<f64>,
        prices: Option<&HashMap<String, f64>>,
    ) -> Result<FinancialGraph, BuilderError> {
        if symbols.len() != matrix.nrows() || matrix.nrows() != matrix.ncols() {
            return Err(BuilderError::ComputationError(
                "Matrix dimensions don't match symbol count".to_string(),
            ));
        }

        let mut graph = FinancialGraph::new(self.graph_type);
        graph.threshold = self.correlation_threshold;

        // Add nodes
        for symbol in symbols {
            let mut node = Node::new(symbol.clone());
            if let Some(prices) = prices {
                if let Some(&price) = prices.get(symbol) {
                    node = node.with_price(price);
                }
            }
            graph.add_node(node);
        }

        // Add edges
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let corr = matrix[(i, j)];
                let effective_corr = if self.use_absolute_correlation {
                    corr.abs()
                } else {
                    corr
                };

                if effective_corr >= self.correlation_threshold {
                    graph.add_edge(&symbols[i], &symbols[j], corr);
                }
            }
        }

        Ok(graph)
    }

    /// Compute log returns for each symbol
    fn compute_returns(
        &self,
        data: &HashMap<String, Vec<Candle>>,
    ) -> Result<HashMap<String, Vec<f64>>, BuilderError> {
        let mut returns = HashMap::new();

        for (symbol, candles) in data {
            if candles.len() < 2 {
                continue;
            }

            // Filter by volume if needed
            let avg_volume: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;
            if avg_volume < self.min_volume {
                continue;
            }

            // Compute log returns
            let symbol_returns: Vec<f64> = candles
                .windows(2)
                .map(|w| (w[1].close / w[0].close).ln())
                .collect();

            // Use only last window_size returns
            let start = symbol_returns.len().saturating_sub(self.window_size);
            returns.insert(symbol.clone(), symbol_returns[start..].to_vec());
        }

        Ok(returns)
    }

    /// Compute Pearson correlation matrix
    fn compute_correlation_matrix(
        &self,
        returns: &HashMap<String, Vec<f64>>,
    ) -> Result<DMatrix<f64>, BuilderError> {
        let symbols: Vec<_> = returns.keys().cloned().collect();
        let n = symbols.len();

        if n == 0 {
            return Err(BuilderError::NoReturns);
        }

        let mut matrix = DMatrix::zeros(n, n);

        for i in 0..n {
            matrix[(i, i)] = 1.0; // Diagonal is 1

            for j in (i + 1)..n {
                let r1 = &returns[&symbols[i]];
                let r2 = &returns[&symbols[j]];

                // Ensure same length
                let len = r1.len().min(r2.len());
                let r1 = &r1[r1.len() - len..];
                let r2 = &r2[r2.len() - len..];

                let corr = pearson_correlation(r1, r2);
                matrix[(i, j)] = corr;
                matrix[(j, i)] = corr; // Symmetric
            }
        }

        Ok(matrix)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < 0.0001);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson_correlation(&x, &y_neg) + 1.0).abs() < 0.0001);

        // No correlation
        let x2 = vec![1.0, 2.0, 3.0, 4.0];
        let y2 = vec![1.0, 3.0, 2.0, 4.0];
        let corr = pearson_correlation(&x2, &y2);
        assert!(corr.abs() < 1.0); // Some correlation but not perfect
    }

    #[test]
    fn test_builder_defaults() {
        let builder = GraphBuilder::new();
        assert_eq!(builder.correlation_threshold, 0.7);
        assert_eq!(builder.window_size, 24);
    }
}
