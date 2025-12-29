//! # Feature Engine
//!
//! Comprehensive feature extraction from market data.

use crate::data::orderbook::OrderBook;
use crate::data::snapshot::{FeatureVector, OrderBookSnapshot, TimeWindow};
use crate::data::trade::{Trade, TradeStats};
use crate::orderflow::ofi::OrderFlowCalculator;
use crate::orderflow::vpin::VpinCalculator;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

/// Feature engine for extracting ML features from market data
#[derive(Debug)]
pub struct FeatureEngine {
    /// OFI calculator
    ofi_calculator: OrderFlowCalculator,
    /// VPIN calculator
    vpin_calculator: VpinCalculator,
    /// Recent order books for momentum calculation
    recent_orderbooks: VecDeque<OrderBook>,
    /// Recent trades for trade stats
    recent_trades: VecDeque<Trade>,
    /// Spread history for z-score
    spread_history: VecDeque<f64>,
    /// Configuration
    config: FeatureConfig,
}

/// Feature engine configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Number of order book levels to use
    pub orderbook_levels: usize,
    /// VPIN bucket size
    pub vpin_bucket_size: f64,
    /// VPIN number of buckets
    pub vpin_num_buckets: usize,
    /// History window for rolling calculations
    pub history_window: usize,
    /// Spread history for z-score
    pub spread_window: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            orderbook_levels: 10,
            vpin_bucket_size: 10.0, // 10 BTC per bucket
            vpin_num_buckets: 50,
            history_window: 1000,
            spread_window: 100,
        }
    }
}

impl FeatureEngine {
    /// Create a new feature engine with default config
    pub fn new() -> Self {
        Self::with_config(FeatureConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: FeatureConfig) -> Self {
        Self {
            ofi_calculator: OrderFlowCalculator::with_history_size(config.history_window),
            vpin_calculator: VpinCalculator::new(
                config.vpin_bucket_size,
                config.vpin_num_buckets,
            ),
            recent_orderbooks: VecDeque::with_capacity(100),
            recent_trades: VecDeque::with_capacity(1000),
            spread_history: VecDeque::with_capacity(config.spread_window),
            config,
        }
    }

    /// Update with new order book
    pub fn update_orderbook(&mut self, book: &OrderBook) {
        // Update OFI
        self.ofi_calculator.update(book);

        // Track spread
        if let Some(spread) = book.spread_bps() {
            self.spread_history.push_back(spread);
            if self.spread_history.len() > self.config.spread_window {
                self.spread_history.pop_front();
            }
        }

        // Store recent orderbook
        self.recent_orderbooks.push_back(book.clone());
        if self.recent_orderbooks.len() > 100 {
            self.recent_orderbooks.pop_front();
        }
    }

    /// Update with new trade
    pub fn update_trade(&mut self, trade: &Trade) {
        // Update VPIN
        self.vpin_calculator.add_trade(trade);

        // Store trade
        self.recent_trades.push_back(trade.clone());
        if self.recent_trades.len() > 1000 {
            self.recent_trades.pop_front();
        }
    }

    /// Extract full feature vector
    pub fn extract_features(&self, book: &OrderBook) -> FeatureVector {
        let mut features = FeatureVector::new(book.timestamp);

        // ═══════════════════════════════════════════════════════════════════
        // Order Book Features
        // ═══════════════════════════════════════════════════════════════════
        let ob_snapshot = OrderBookSnapshot::from_orderbook(book);

        features.add("mid_price", ob_snapshot.mid_price);
        features.add("spread_bps", ob_snapshot.spread_bps);
        features.add("spread_zscore", self.spread_zscore());

        features.add("depth_imbalance_l1", ob_snapshot.depth_imbalance_l1);
        features.add("depth_imbalance_l5", ob_snapshot.depth_imbalance_l5);
        features.add("depth_imbalance_l10", ob_snapshot.depth_imbalance_l10);
        features.add("weighted_imbalance", ob_snapshot.weighted_imbalance);

        features.add("bid_depth_l5", ob_snapshot.bid_depth_l5);
        features.add("ask_depth_l5", ob_snapshot.ask_depth_l5);
        features.add("total_depth_l5", ob_snapshot.bid_depth_l5 + ob_snapshot.ask_depth_l5);

        let depth_ratio = if ob_snapshot.ask_depth_l5 > 0.0 {
            ob_snapshot.bid_depth_l5 / ob_snapshot.ask_depth_l5
        } else {
            1.0
        };
        features.add("depth_ratio", depth_ratio);

        features.add("bid_slope", ob_snapshot.bid_slope);
        features.add("ask_slope", ob_snapshot.ask_slope);
        features.add("slope_asymmetry", ob_snapshot.slope_asymmetry());

        // ═══════════════════════════════════════════════════════════════════
        // OFI Features
        // ═══════════════════════════════════════════════════════════════════
        features.add("ofi_1min", self.ofi_calculator.ofi_1min());
        features.add("ofi_5min", self.ofi_calculator.ofi_5min());
        features.add("ofi_15min", self.ofi_calculator.ofi_15min());
        features.add("ofi_cumulative", self.ofi_calculator.cumulative());

        if let Some(zscore) = self.ofi_calculator.z_score(100) {
            features.add("ofi_zscore", zscore);
        } else {
            features.add("ofi_zscore", 0.0);
        }

        // ═══════════════════════════════════════════════════════════════════
        // VPIN Features
        // ═══════════════════════════════════════════════════════════════════
        if let Some(vpin) = self.vpin_calculator.current_vpin() {
            features.add("vpin", vpin);
        } else {
            features.add("vpin", 0.5);
        }

        if let Some(vpin_zscore) = self.vpin_calculator.z_score(50) {
            features.add("vpin_zscore", vpin_zscore);
        } else {
            features.add("vpin_zscore", 0.0);
        }

        // ═══════════════════════════════════════════════════════════════════
        // Trade Features
        // ═══════════════════════════════════════════════════════════════════
        let trade_stats = self.get_trade_stats(TimeWindow::Minutes(1));
        features.add("trade_volume_1min", trade_stats.volume);
        features.add("trade_imbalance", trade_stats.trade_imbalance());
        features.add("buy_volume_1min", trade_stats.buy_volume);
        features.add("sell_volume_1min", trade_stats.sell_volume);
        features.add("trade_count_1min", trade_stats.count as f64);
        features.add("avg_trade_size", trade_stats.avg_size);
        features.add("has_large_trade", if trade_stats.max_size > 2.0 * trade_stats.avg_size { 1.0 } else { 0.0 });

        // ═══════════════════════════════════════════════════════════════════
        // Momentum Features
        // ═══════════════════════════════════════════════════════════════════
        let momentum_1min = self.price_momentum(60);
        let momentum_5min = self.price_momentum(300);

        features.add("momentum_1min", momentum_1min);
        features.add("momentum_5min", momentum_5min);
        features.add("volatility_1min", self.realized_volatility(60));

        // ═══════════════════════════════════════════════════════════════════
        // Time Features
        // ═══════════════════════════════════════════════════════════════════
        let hour = book.timestamp.hour() as f64;
        let minute = book.timestamp.minute() as f64;

        // Cyclical encoding
        features.add("hour_sin", (hour * 2.0 * std::f64::consts::PI / 24.0).sin());
        features.add("hour_cos", (hour * 2.0 * std::f64::consts::PI / 24.0).cos());
        features.add("minute_sin", (minute * 2.0 * std::f64::consts::PI / 60.0).sin());
        features.add("minute_cos", (minute * 2.0 * std::f64::consts::PI / 60.0).cos());

        features
    }

    /// Calculate spread z-score
    fn spread_zscore(&self) -> f64 {
        if self.spread_history.len() < 10 {
            return 0.0;
        }

        let values: Vec<f64> = self.spread_history.iter().cloned().collect();
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        if std > 0.0 {
            let current = values.last().unwrap();
            (current - mean) / std
        } else {
            0.0
        }
    }

    /// Get trade statistics for a time window
    fn get_trade_stats(&self, window: TimeWindow) -> TradeStats {
        if self.recent_trades.is_empty() {
            return TradeStats::default();
        }

        let cutoff = self.recent_trades.back().unwrap().timestamp
            - chrono::Duration::seconds(window.as_seconds() as i64);

        let trades: Vec<Trade> = self
            .recent_trades
            .iter()
            .filter(|t| t.timestamp >= cutoff)
            .cloned()
            .collect();

        TradeStats::from_trades(&trades)
    }

    /// Calculate price momentum over N seconds
    fn price_momentum(&self, seconds: i64) -> f64 {
        if self.recent_orderbooks.len() < 2 {
            return 0.0;
        }

        let latest = self.recent_orderbooks.back().unwrap();
        let cutoff = latest.timestamp - chrono::Duration::seconds(seconds);

        // Find the oldest orderbook in window
        let oldest = self
            .recent_orderbooks
            .iter()
            .find(|ob| ob.timestamp >= cutoff);

        if let (Some(old), Some(old_mid), Some(new_mid)) = (
            oldest,
            oldest.and_then(|ob| ob.mid_price()),
            latest.mid_price(),
        ) {
            (new_mid - old_mid) / old_mid * 10000.0 // in bps
        } else {
            0.0
        }
    }

    /// Calculate realized volatility over N seconds
    fn realized_volatility(&self, seconds: i64) -> f64 {
        if self.recent_orderbooks.len() < 10 {
            return 0.0;
        }

        let latest = self.recent_orderbooks.back().unwrap();
        let cutoff = latest.timestamp - chrono::Duration::seconds(seconds);

        let prices: Vec<f64> = self
            .recent_orderbooks
            .iter()
            .filter(|ob| ob.timestamp >= cutoff)
            .filter_map(|ob| ob.mid_price())
            .collect();

        if prices.len() < 2 {
            return 0.0;
        }

        // Calculate returns
        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Standard deviation of returns
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        variance.sqrt() * 10000.0 // in bps
    }

    /// Reset the engine
    pub fn reset(&mut self) {
        self.ofi_calculator.reset();
        self.vpin_calculator.reset();
        self.recent_orderbooks.clear();
        self.recent_trades.clear();
        self.spread_history.clear();
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<&'static str> {
        vec![
            "mid_price",
            "spread_bps",
            "spread_zscore",
            "depth_imbalance_l1",
            "depth_imbalance_l5",
            "depth_imbalance_l10",
            "weighted_imbalance",
            "bid_depth_l5",
            "ask_depth_l5",
            "total_depth_l5",
            "depth_ratio",
            "bid_slope",
            "ask_slope",
            "slope_asymmetry",
            "ofi_1min",
            "ofi_5min",
            "ofi_15min",
            "ofi_cumulative",
            "ofi_zscore",
            "vpin",
            "vpin_zscore",
            "trade_volume_1min",
            "trade_imbalance",
            "buy_volume_1min",
            "sell_volume_1min",
            "trade_count_1min",
            "avg_trade_size",
            "has_large_trade",
            "momentum_1min",
            "momentum_5min",
            "volatility_1min",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
        ]
    }
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::orderbook::OrderBookLevel;

    fn create_orderbook() -> OrderBook {
        let bids = vec![
            OrderBookLevel::new(100.0, 10.0, 1),
            OrderBookLevel::new(99.5, 20.0, 2),
        ];
        let asks = vec![
            OrderBookLevel::new(100.5, 8.0, 1),
            OrderBookLevel::new(101.0, 15.0, 2),
        ];
        OrderBook::new("BTCUSDT".to_string(), Utc::now(), bids, asks)
    }

    #[test]
    fn test_feature_extraction() {
        let mut engine = FeatureEngine::new();
        let book = create_orderbook();

        engine.update_orderbook(&book);
        let features = engine.extract_features(&book);

        assert!(!features.is_empty());
        assert!(features.get("mid_price").is_some());
        assert!(features.get("spread_bps").is_some());
    }

    #[test]
    fn test_feature_names() {
        let engine = FeatureEngine::new();
        let names = engine.feature_names();

        assert!(!names.is_empty());
        assert!(names.contains(&"ofi_1min"));
        assert!(names.contains(&"vpin"));
    }
}
