//! Feature extraction for market regime classification
//!
//! This module provides feature engineering capabilities to transform
//! raw market data into numerical features suitable for the prototypical network.

use crate::data::types::{Kline, OrderBook, FundingRate, OpenInterest, Trade};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Window sizes for moving averages
    pub ma_windows: Vec<usize>,
    /// Window size for volatility calculation
    pub volatility_window: usize,
    /// Window size for RSI calculation
    pub rsi_window: usize,
    /// Whether to include order book features
    pub include_orderbook: bool,
    /// Whether to include funding rate features
    pub include_funding: bool,
    /// Whether to normalize features
    pub normalize: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            ma_windows: vec![5, 10, 20, 50],
            volatility_window: 20,
            rsi_window: 14,
            include_orderbook: true,
            include_funding: true,
            normalize: true,
        }
    }
}

/// Extracted market features for a single time point
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Feature vector
    pub features: Array1<f64>,
    /// Feature names for interpretability
    pub feature_names: Vec<String>,
}

impl MarketFeatures {
    /// Get feature dimension
    pub fn dim(&self) -> usize {
        self.features.len()
    }

    /// Get feature by name
    pub fn get(&self, name: &str) -> Option<f64> {
        self.feature_names
            .iter()
            .position(|n| n == name)
            .map(|i| self.features[i])
    }
}

/// Feature extractor for market data
pub struct FeatureExtractor {
    config: FeatureConfig,
    /// Running statistics for normalization
    running_mean: Option<Array1<f64>>,
    running_std: Option<Array1<f64>>,
    samples_seen: usize,
}

impl FeatureExtractor {
    /// Create a new feature extractor with default configuration
    pub fn new() -> Self {
        Self::with_config(FeatureConfig::default())
    }

    /// Create a new feature extractor with custom configuration
    pub fn with_config(config: FeatureConfig) -> Self {
        Self {
            config,
            running_mean: None,
            running_std: None,
            samples_seen: 0,
        }
    }

    /// Extract features from kline data
    pub fn extract_from_klines(&self, klines: &[Kline]) -> MarketFeatures {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if klines.is_empty() {
            return MarketFeatures {
                features: Array1::zeros(0),
                feature_names: names,
            };
        }

        // Basic price features
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Returns
        let returns = self.calculate_returns(&closes);
        if let Some(last_return) = returns.last() {
            features.push(*last_return);
            names.push("return_1".to_string());
        }

        // Cumulative returns over different windows
        for &window in &[5, 10, 20] {
            if returns.len() >= window {
                let cum_return: f64 = returns[returns.len() - window..].iter().sum();
                features.push(cum_return);
                names.push(format!("return_{}", window));
            }
        }

        // Moving averages and price position relative to MAs
        let last_close = *closes.last().unwrap_or(&0.0);
        for &window in &self.config.ma_windows {
            if closes.len() >= window {
                let ma = self.simple_moving_average(&closes, window);
                if let Some(&ma_val) = ma.last() {
                    // Price relative to MA
                    let rel_ma = if ma_val > 0.0 {
                        (last_close / ma_val - 1.0)
                    } else {
                        0.0
                    };
                    features.push(rel_ma);
                    names.push(format!("price_rel_ma_{}", window));
                }
            }
        }

        // MA crossover signals
        if closes.len() >= 20 {
            let ma_5 = self.simple_moving_average(&closes, 5);
            let ma_20 = self.simple_moving_average(&closes, 20);
            if let (Some(&ma5), Some(&ma20)) = (ma_5.last(), ma_20.last()) {
                let crossover = if ma20 > 0.0 { ma5 / ma20 - 1.0 } else { 0.0 };
                features.push(crossover);
                names.push("ma_5_20_crossover".to_string());
            }
        }

        // Volatility (standard deviation of returns)
        if returns.len() >= self.config.volatility_window {
            let volatility = self.calculate_std(&returns[returns.len() - self.config.volatility_window..]);
            features.push(volatility);
            names.push("volatility".to_string());

            // Volatility change
            if returns.len() >= self.config.volatility_window * 2 {
                let prev_vol = self.calculate_std(
                    &returns[returns.len() - self.config.volatility_window * 2
                        ..returns.len() - self.config.volatility_window]
                );
                let vol_change = if prev_vol > 0.0 { volatility / prev_vol - 1.0 } else { 0.0 };
                features.push(vol_change);
                names.push("volatility_change".to_string());
            }
        }

        // RSI
        if returns.len() >= self.config.rsi_window {
            let rsi = self.calculate_rsi(&returns, self.config.rsi_window);
            features.push(rsi / 100.0 - 0.5); // Normalize to [-0.5, 0.5]
            names.push("rsi".to_string());
        }

        // ATR (Average True Range)
        if klines.len() >= 14 {
            let atr = self.calculate_atr(&klines[klines.len() - 14..]);
            let atr_pct = if last_close > 0.0 { atr / last_close } else { 0.0 };
            features.push(atr_pct);
            names.push("atr_pct".to_string());
        }

        // Volume features
        if volumes.len() >= 20 {
            let vol_ma = self.simple_moving_average(&volumes, 20);
            if let Some(&vol_ma_val) = vol_ma.last() {
                let last_vol = *volumes.last().unwrap_or(&0.0);
                let rel_vol = if vol_ma_val > 0.0 { last_vol / vol_ma_val } else { 1.0 };
                features.push(rel_vol - 1.0);
                names.push("relative_volume".to_string());
            }
        }

        // Price position within range (0 = at low, 1 = at high)
        if klines.len() >= 20 {
            let recent_high = highs[highs.len() - 20..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let recent_low = lows[lows.len() - 20..].iter().cloned().fold(f64::INFINITY, f64::min);
            let range = recent_high - recent_low;
            let position = if range > 0.0 {
                (last_close - recent_low) / range
            } else {
                0.5
            };
            features.push(position - 0.5); // Normalize to [-0.5, 0.5]
            names.push("price_position".to_string());
        }

        // Candle patterns (last few candles)
        if klines.len() >= 3 {
            let recent_klines = &klines[klines.len() - 3..];
            let bullish_count = recent_klines.iter().filter(|k| k.is_bullish()).count() as f64;
            features.push(bullish_count / 3.0 - 0.5);
            names.push("recent_bullish_ratio".to_string());
        }

        // Momentum (rate of change)
        if closes.len() >= 10 {
            let roc = (last_close / closes[closes.len() - 10] - 1.0);
            features.push(roc);
            names.push("momentum_10".to_string());
        }

        MarketFeatures {
            features: Array1::from_vec(features),
            feature_names: names,
        }
    }

    /// Extract features including order book data
    pub fn extract_with_orderbook(
        &self,
        klines: &[Kline],
        order_book: &OrderBook,
    ) -> MarketFeatures {
        let mut base_features = self.extract_from_klines(klines);

        if !self.config.include_orderbook {
            return base_features;
        }

        // Order book imbalance
        let imbalance_5 = order_book.imbalance(5);
        let imbalance_10 = order_book.imbalance(10);

        base_features.features = ndarray::concatenate![
            ndarray::Axis(0),
            base_features.features,
            Array1::from_vec(vec![imbalance_5, imbalance_10])
        ];
        base_features.feature_names.push("orderbook_imbalance_5".to_string());
        base_features.feature_names.push("orderbook_imbalance_10".to_string());

        // Spread
        if let Some(spread_pct) = order_book.spread_pct() {
            base_features.features = ndarray::concatenate![
                ndarray::Axis(0),
                base_features.features,
                Array1::from_vec(vec![spread_pct])
            ];
            base_features.feature_names.push("spread_pct".to_string());
        }

        base_features
    }

    /// Extract features including funding rate
    pub fn extract_with_funding(
        &self,
        klines: &[Kline],
        funding_rate: &FundingRate,
    ) -> MarketFeatures {
        let mut base_features = self.extract_from_klines(klines);

        if !self.config.include_funding {
            return base_features;
        }

        // Funding rate features
        let funding = funding_rate.funding_rate * 100.0; // Convert to percentage
        let funding_signal = funding_rate.sentiment_signal();

        base_features.features = ndarray::concatenate![
            ndarray::Axis(0),
            base_features.features,
            Array1::from_vec(vec![funding, funding_signal])
        ];
        base_features.feature_names.push("funding_rate".to_string());
        base_features.feature_names.push("funding_signal".to_string());

        base_features
    }

    /// Extract comprehensive features from all available data
    pub fn extract_full(
        &self,
        klines: &[Kline],
        order_book: Option<&OrderBook>,
        funding_rate: Option<&FundingRate>,
        open_interest: Option<&OpenInterest>,
        prev_open_interest: Option<&OpenInterest>,
    ) -> MarketFeatures {
        let mut features = self.extract_from_klines(klines);

        // Add order book features
        if let Some(ob) = order_book {
            if self.config.include_orderbook {
                let imbalance_5 = ob.imbalance(5);
                let imbalance_10 = ob.imbalance(10);
                let spread = ob.spread_pct().unwrap_or(0.0);

                features.features = ndarray::concatenate![
                    ndarray::Axis(0),
                    features.features,
                    Array1::from_vec(vec![imbalance_5, imbalance_10, spread])
                ];
                features.feature_names.push("orderbook_imbalance_5".to_string());
                features.feature_names.push("orderbook_imbalance_10".to_string());
                features.feature_names.push("spread_pct".to_string());
            }
        }

        // Add funding rate features
        if let Some(fr) = funding_rate {
            if self.config.include_funding {
                let funding = fr.funding_rate * 100.0;
                let signal = fr.sentiment_signal();

                features.features = ndarray::concatenate![
                    ndarray::Axis(0),
                    features.features,
                    Array1::from_vec(vec![funding, signal])
                ];
                features.feature_names.push("funding_rate".to_string());
                features.feature_names.push("funding_signal".to_string());
            }
        }

        // Add open interest features
        if let (Some(oi), Some(prev_oi)) = (open_interest, prev_open_interest) {
            let oi_change = oi.change_pct(prev_oi) / 100.0;
            features.features = ndarray::concatenate![
                ndarray::Axis(0),
                features.features,
                Array1::from_vec(vec![oi_change])
            ];
            features.feature_names.push("oi_change".to_string());
        }

        features
    }

    /// Normalize features using running statistics
    pub fn normalize(&mut self, features: &mut MarketFeatures) {
        if !self.config.normalize {
            return;
        }

        let n = features.features.len();

        // Initialize running statistics if needed
        if self.running_mean.is_none() {
            self.running_mean = Some(Array1::zeros(n));
            self.running_std = Some(Array1::ones(n));
        }

        // Update running statistics (Welford's algorithm)
        self.samples_seen += 1;
        let mean = self.running_mean.as_mut().unwrap();
        let std = self.running_std.as_mut().unwrap();

        for i in 0..n.min(mean.len()) {
            let delta = features.features[i] - mean[i];
            mean[i] += delta / self.samples_seen as f64;

            if self.samples_seen > 1 {
                // Running variance update
                let delta2 = features.features[i] - mean[i];
                let new_var = ((self.samples_seen - 1) as f64 * std[i].powi(2) + delta * delta2)
                    / self.samples_seen as f64;
                std[i] = new_var.sqrt().max(1e-8);
            }
        }

        // Apply normalization
        for i in 0..n.min(mean.len()) {
            features.features[i] = (features.features[i] - mean[i]) / std[i];
        }
    }

    /// Extract features for multiple time windows (for time series)
    pub fn extract_sequence(&self, klines: &[Kline], window_size: usize) -> Array2<f64> {
        let mut sequence = Vec::new();

        for i in window_size..=klines.len() {
            let window = &klines[i - window_size..i];
            let features = self.extract_from_klines(window);
            sequence.push(features.features.to_vec());
        }

        if sequence.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_features = sequence[0].len();
        let n_samples = sequence.len();

        Array2::from_shape_vec(
            (n_samples, n_features),
            sequence.into_iter().flatten().collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((n_samples, n_features)))
    }

    // Helper functions

    fn calculate_returns(&self, prices: &[f64]) -> Vec<f64> {
        prices
            .windows(2)
            .map(|w| {
                if w[0] > 0.0 {
                    w[1] / w[0] - 1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn simple_moving_average(&self, data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![];
        }

        data.windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    fn calculate_std(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    fn calculate_rsi(&self, returns: &[f64], window: usize) -> f64 {
        if returns.len() < window {
            return 50.0;
        }

        let recent = &returns[returns.len() - window..];
        let gains: f64 = recent.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = recent.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();

        if losses == 0.0 {
            return 100.0;
        }
        if gains == 0.0 {
            return 0.0;
        }

        let rs = gains / losses;
        100.0 - 100.0 / (1.0 + rs)
    }

    fn calculate_atr(&self, klines: &[Kline]) -> f64 {
        if klines.len() < 2 {
            return 0.0;
        }

        let trs: Vec<f64> = klines
            .windows(2)
            .map(|w| {
                let high_low = w[1].high - w[1].low;
                let high_close = (w[1].high - w[0].close).abs();
                let low_close = (w[1].low - w[0].close).abs();
                high_low.max(high_close).max(low_close)
            })
            .collect();

        trs.iter().sum::<f64>() / trs.len() as f64
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let base_price = 100.0 + (i as f64 * 0.5);
                // Calculate day and hour to avoid invalid time (hour must be 0-23)
                let day = 1 + (i / 24) as u32;
                let hour = (i % 24) as u32;
                Kline::new(
                    Utc.with_ymd_and_hms(2024, 1, day, hour, 0, 0).unwrap(),
                    base_price,
                    base_price + 1.0,
                    base_price - 0.5,
                    base_price + 0.3,
                    1000.0 + i as f64 * 10.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_feature_extraction() {
        let klines = create_test_klines(50);
        let extractor = FeatureExtractor::new();

        let features = extractor.extract_from_klines(&klines);

        assert!(features.dim() > 0);
        assert!(!features.feature_names.is_empty());
        assert_eq!(features.dim(), features.feature_names.len());
    }

    #[test]
    fn test_returns_calculation() {
        let extractor = FeatureExtractor::new();
        let prices = vec![100.0, 105.0, 102.0];

        let returns = extractor.calculate_returns(&prices);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.05).abs() < 1e-10);
        assert!((returns[1] - (-0.0285714286)).abs() < 1e-6);
    }

    #[test]
    fn test_rsi_calculation() {
        let extractor = FeatureExtractor::new();
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, 0.01, -0.005, 0.01, 0.015, -0.01, 0.02, -0.015, 0.01, 0.005];

        let rsi = extractor.calculate_rsi(&returns, 14);

        assert!(rsi >= 0.0 && rsi <= 100.0);
    }

    #[test]
    fn test_sequence_extraction() {
        let klines = create_test_klines(30);
        let extractor = FeatureExtractor::new();

        let sequence = extractor.extract_sequence(&klines, 20);

        assert_eq!(sequence.nrows(), 11); // 30 - 20 + 1 = 11
        assert!(sequence.ncols() > 0);
    }
}
