//! Data module for Bybit API client and market data processing
//!
//! This module provides functionality for fetching market data from Bybit
//! and constructing heterogeneous graphs.

mod bybit;
mod types;
mod features;

pub use bybit::{BybitClient, BybitConfig, BybitError};
pub use types::{Kline, OrderBook, OrderBookLevel, Ticker, Trade, FundingRate, OpenInterest};
pub use features::FeatureBuilder;

use crate::graph::{
    HeterogeneousGraph, GraphSchema, NodeType, EdgeType,
    AssetFeatures, ExchangeFeatures, NodeFeatures, EdgeFeatures,
};
use std::collections::HashMap;

/// Build a heterogeneous graph from market data
pub struct GraphBuilder {
    /// Symbol list
    symbols: Vec<String>,
    /// Correlation threshold for edge creation
    correlation_threshold: f64,
    /// Rolling window for correlation (in klines)
    correlation_window: usize,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new(symbols: Vec<String>) -> Self {
        Self {
            symbols,
            correlation_threshold: 0.5,
            correlation_window: 100,
        }
    }

    /// Set correlation threshold
    pub fn with_correlation_threshold(mut self, threshold: f64) -> Self {
        self.correlation_threshold = threshold;
        self
    }

    /// Build graph from ticker data
    pub fn build_from_tickers(
        &self,
        tickers: &HashMap<String, Ticker>,
        price_history: &HashMap<String, Vec<f64>>,
    ) -> HeterogeneousGraph {
        let schema = GraphSchema::trading_schema();
        let mut graph = HeterogeneousGraph::new(schema);

        // Add asset nodes
        for symbol in &self.symbols {
            if let Some(ticker) = tickers.get(symbol) {
                let features = AssetFeatures {
                    price: ticker.last_price,
                    volume_24h: ticker.volume_24h,
                    volatility: self.estimate_volatility(price_history.get(symbol)),
                    market_cap: 0.0,
                    funding_rate: 0.0,
                    open_interest: 0.0,
                    return_1h: 0.0,
                    return_4h: 0.0,
                    return_24h: ticker.price_change_24h / 100.0,
                    spread: (ticker.ask_price - ticker.bid_price) / ticker.last_price,
                    imbalance: 0.0,
                    timestamp: ticker.timestamp,
                };
                graph.add_node(symbol, NodeType::Asset, features.into());
            }
        }

        // Add Bybit exchange node
        let total_volume: f64 = tickers.values().map(|t| t.volume_24h).sum();
        let bybit_features = ExchangeFeatures {
            total_volume,
            num_pairs: self.symbols.len() as u32,
            liquidity_score: 0.95,
            avg_spread: tickers.values().map(|t| {
                (t.ask_price - t.bid_price) / t.last_price
            }).sum::<f64>() / tickers.len() as f64,
            reliability_score: 1.0,
            active_users: 0,
        };
        graph.add_node("Bybit", NodeType::Exchange, bybit_features.into());

        // Add TradesOn edges
        for symbol in &self.symbols {
            if tickers.contains_key(symbol) {
                let volume = tickers.get(symbol).map(|t| t.volume_24h).unwrap_or(0.0);
                graph.add_edge(
                    symbol,
                    "Bybit",
                    EdgeType::TradesOn,
                    EdgeFeatures::with_volume(volume, 0),
                );
            }
        }

        // Add correlation edges
        self.add_correlation_edges(&mut graph, price_history);

        graph
    }

    /// Add correlation edges between assets
    fn add_correlation_edges(
        &self,
        graph: &mut HeterogeneousGraph,
        price_history: &HashMap<String, Vec<f64>>,
    ) {
        let symbols: Vec<&String> = self.symbols.iter().collect();

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                if let (Some(prices_i), Some(prices_j)) = (
                    price_history.get(symbols[i]),
                    price_history.get(symbols[j]),
                ) {
                    if let Some(corr) = self.compute_correlation(prices_i, prices_j) {
                        if corr.abs() >= self.correlation_threshold {
                            graph.add_edge(
                                symbols[i],
                                symbols[j],
                                EdgeType::Correlation,
                                EdgeFeatures::with_correlation(corr, 0),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Compute Pearson correlation between two price series
    fn compute_correlation(&self, prices_a: &[f64], prices_b: &[f64]) -> Option<f64> {
        let n = prices_a.len().min(prices_b.len()).min(self.correlation_window);
        if n < 10 {
            return None;
        }

        let a = &prices_a[prices_a.len().saturating_sub(n)..];
        let b = &prices_b[prices_b.len().saturating_sub(n)..];

        // Compute returns
        let returns_a: Vec<f64> = a.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
        let returns_b: Vec<f64> = b.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

        if returns_a.len() < 5 {
            return None;
        }

        let mean_a: f64 = returns_a.iter().sum::<f64>() / returns_a.len() as f64;
        let mean_b: f64 = returns_b.iter().sum::<f64>() / returns_b.len() as f64;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for (a, b) in returns_a.iter().zip(returns_b.iter()) {
            let da = a - mean_a;
            let db = b - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        if var_a == 0.0 || var_b == 0.0 {
            return None;
        }

        Some(cov / (var_a.sqrt() * var_b.sqrt()))
    }

    /// Estimate volatility from price history
    fn estimate_volatility(&self, prices: Option<&Vec<f64>>) -> f64 {
        match prices {
            Some(p) if p.len() > 10 => {
                let returns: Vec<f64> = p.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                    / returns.len() as f64;
                variance.sqrt()
            }
            _ => 0.02,  // Default volatility
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_builder() {
        let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
        let builder = GraphBuilder::new(symbols);

        let mut tickers = HashMap::new();
        tickers.insert("BTCUSDT".to_string(), Ticker {
            symbol: "BTCUSDT".to_string(),
            last_price: 50000.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            volume_24h: 1_000_000.0,
            turnover_24h: 50_000_000_000.0,
            price_change_24h: 2.0,
            bid_price: 49990.0,
            ask_price: 50010.0,
            timestamp: 0,
        });

        let graph = builder.build_from_tickers(&tickers, &HashMap::new());
        assert!(graph.num_nodes() > 0);
    }
}
