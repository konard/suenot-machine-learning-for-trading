//! Feature engineering engine

use crate::api::types::{Kline, OrderBook, Ticker};
use crate::features::node_features::NodeFeatures;
use crate::graph::market_graph::MarketGraph;
use ndarray::Array2;
use std::collections::HashMap;

/// Feature engineering engine for Graph Transformer
pub struct FeatureEngine {
    /// Target feature dimension
    pub feature_dim: usize,
    /// Whether to include technical indicators
    pub use_technical: bool,
    /// Whether to include orderbook features
    pub use_orderbook: bool,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new(32)
    }
}

impl FeatureEngine {
    /// Create a new feature engine
    pub fn new(feature_dim: usize) -> Self {
        Self {
            feature_dim,
            use_technical: true,
            use_orderbook: true,
        }
    }

    /// Set whether to use technical indicators
    pub fn with_technical(mut self, use_tech: bool) -> Self {
        self.use_technical = use_tech;
        self
    }

    /// Set whether to use orderbook features
    pub fn with_orderbook(mut self, use_ob: bool) -> Self {
        self.use_orderbook = use_ob;
        self
    }

    /// Compute features for all nodes from tickers
    pub fn compute_from_tickers(&self, tickers: &[Ticker]) -> Array2<f64> {
        let n = tickers.len();
        let mut features = Array2::zeros((n, self.feature_dim));

        for (i, ticker) in tickers.iter().enumerate() {
            let node_features = NodeFeatures::from_ticker(ticker);
            let padded = node_features.to_dim(self.feature_dim);
            features.row_mut(i).assign(&padded);
        }

        features
    }

    /// Compute features from kline history
    pub fn compute_from_klines(
        &self,
        klines_map: &HashMap<String, Vec<Kline>>,
        symbols: &[&str],
    ) -> Array2<f64> {
        let n = symbols.len();
        let mut features = Array2::zeros((n, self.feature_dim));

        for (i, symbol) in symbols.iter().enumerate() {
            if let Some(klines) = klines_map.get(*symbol) {
                let node_features = NodeFeatures::from_klines(klines);
                let padded = node_features.to_dim(self.feature_dim);
                features.row_mut(i).assign(&padded);
            }
        }

        features
    }

    /// Compute features combining tickers, klines, and orderbook
    pub fn compute_full(
        &self,
        tickers: &[Ticker],
        klines_map: Option<&HashMap<String, Vec<Kline>>>,
        orderbooks: Option<&HashMap<String, OrderBook>>,
    ) -> Array2<f64> {
        let n = tickers.len();
        let mut features = Array2::zeros((n, self.feature_dim));

        for (i, ticker) in tickers.iter().enumerate() {
            let mut feature_sets = vec![NodeFeatures::from_ticker(ticker)];

            // Add kline features if available
            if self.use_technical {
                if let Some(klines) = klines_map.and_then(|m| m.get(&ticker.symbol)) {
                    feature_sets.push(NodeFeatures::from_klines(klines));
                }
            }

            // Add orderbook features if available
            if self.use_orderbook {
                if let Some(ob) = orderbooks.and_then(|m| m.get(&ticker.symbol)) {
                    feature_sets.push(NodeFeatures::from_orderbook(ob));
                }
            }

            let combined = NodeFeatures::combine(&feature_sets);
            let padded = combined.to_dim(self.feature_dim);
            features.row_mut(i).assign(&padded);
        }

        features
    }

    /// Update graph node features in-place
    pub fn update_graph_features(&self, graph: &mut MarketGraph, features: &Array2<f64>) {
        for (i, node) in graph.nodes.iter_mut().enumerate() {
            if i < features.nrows() {
                node.features = Some(features.row(i).to_owned());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_engine() {
        let engine = FeatureEngine::new(16);

        let tickers = vec![
            Ticker {
                symbol: "BTCUSDT".to_string(),
                last_price: 45000.0,
                high_24h: 46000.0,
                low_24h: 44000.0,
                price_change_24h: 0.02,
                volume_24h: 1000.0,
                turnover_24h: 45000000.0,
                bid_price: 44990.0,
                ask_price: 45010.0,
            },
            Ticker {
                symbol: "ETHUSDT".to_string(),
                last_price: 2500.0,
                high_24h: 2600.0,
                low_24h: 2400.0,
                price_change_24h: 0.03,
                volume_24h: 5000.0,
                turnover_24h: 12500000.0,
                bid_price: 2499.0,
                ask_price: 2501.0,
            },
        ];

        let features = engine.compute_from_tickers(&tickers);
        assert_eq!(features.shape(), &[2, 16]);
    }
}
