//! Feature extraction from raw market data

use super::types::{Kline, OrderBook};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Window size for moving averages
    pub ma_windows: Vec<usize>,
    /// RSI period
    pub rsi_period: usize,
    /// ATR period
    pub atr_period: usize,
    /// Order book depth for features
    pub orderbook_depth: usize,
    /// Whether to normalize features
    pub normalize: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            ma_windows: vec![5, 10, 20],
            rsi_period: 14,
            atr_period: 14,
            orderbook_depth: 10,
            normalize: true,
        }
    }
}

/// Extracted market features
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
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_extractor() -> Self {
        Self::new(FeatureConfig::default())
    }

    /// Extract features from klines
    pub fn extract_from_klines(&self, klines: &[Kline]) -> MarketFeatures {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if klines.is_empty() {
            return MarketFeatures {
                features: Array1::zeros(0),
                feature_names: vec![],
            };
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let _lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Latest price info
        let last = klines.last().unwrap();
        let last_return = last.return_pct();
        features.push(last_return);
        names.push("return".to_string());

        // Moving averages and their ratios
        for &window in &self.config.ma_windows {
            if closes.len() >= window {
                let ma = moving_average(&closes, window);
                let ratio = last.close / ma - 1.0;
                features.push(ratio);
                names.push(format!("ma{}_ratio", window));
            }
        }

        // RSI
        if closes.len() >= self.config.rsi_period + 1 {
            let rsi = compute_rsi(&closes, self.config.rsi_period);
            features.push((rsi - 50.0) / 50.0); // Normalize to -1 to 1
            names.push("rsi".to_string());
        }

        // ATR (Average True Range) - normalized by price
        if klines.len() >= self.config.atr_period {
            let atr = compute_atr(klines, self.config.atr_period);
            features.push(atr / last.close);
            names.push("atr_ratio".to_string());
        }

        // Volume features
        if volumes.len() >= 20 {
            let vol_ma = moving_average(&volumes, 20);
            let vol_ratio = last.volume / vol_ma;
            features.push(vol_ratio - 1.0);
            names.push("volume_ratio".to_string());
        }

        // Volatility (std of returns)
        if closes.len() >= 20 {
            let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
            let volatility = std_dev(&returns);
            features.push(volatility);
            names.push("volatility".to_string());
        }

        // Momentum (rate of change)
        for period in [5, 10, 20] {
            if closes.len() > period {
                let roc = (last.close - closes[closes.len() - 1 - period])
                    / closes[closes.len() - 1 - period];
                features.push(roc);
                names.push(format!("momentum_{}", period));
            }
        }

        // High-low range ratio
        if !highs.is_empty() {
            let range = last.high - last.low;
            let avg_range: f64 = klines.iter().map(|k| k.high - k.low).sum::<f64>() / klines.len() as f64;
            if avg_range > 0.0 {
                features.push(range / avg_range - 1.0);
                names.push("range_ratio".to_string());
            }
        }

        // Body ratio (body / range)
        let body_ratio = if last.range() > 0.0 {
            last.body_size() / last.range()
        } else {
            0.5
        };
        features.push(body_ratio);
        names.push("body_ratio".to_string());

        // Bullish/bearish candle
        features.push(if last.is_bullish() { 1.0 } else { -1.0 });
        names.push("direction".to_string());

        // Normalize if configured
        let features = if self.config.normalize {
            normalize_features(&features)
        } else {
            features
        };

        MarketFeatures {
            features: Array1::from_vec(features),
            feature_names: names,
        }
    }

    /// Extract features from order book
    pub fn extract_from_orderbook(&self, orderbook: &OrderBook) -> MarketFeatures {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // Order book imbalance at different depths
        for depth in [1, 5, 10] {
            let imbalance = orderbook.imbalance(depth);
            features.push(imbalance);
            names.push(format!("imbalance_{}", depth));
        }

        // Spread
        if let Some(spread_pct) = orderbook.spread_pct() {
            features.push(spread_pct);
            names.push("spread_pct".to_string());
        }

        // Bid/ask depth ratio
        let bid_depth: f64 = orderbook.bids.iter()
            .take(self.config.orderbook_depth)
            .map(|l| l.quantity)
            .sum();
        let ask_depth: f64 = orderbook.asks.iter()
            .take(self.config.orderbook_depth)
            .map(|l| l.quantity)
            .sum();

        if bid_depth + ask_depth > 0.0 {
            features.push((bid_depth - ask_depth) / (bid_depth + ask_depth));
            names.push("depth_imbalance".to_string());
        }

        MarketFeatures {
            features: Array1::from_vec(features),
            feature_names: names,
        }
    }

    /// Combine features from multiple sources
    pub fn combine_features(&self, kline_features: &MarketFeatures, ob_features: &MarketFeatures) -> MarketFeatures {
        let mut combined = kline_features.features.to_vec();
        combined.extend(ob_features.features.iter());

        let mut names = kline_features.feature_names.clone();
        names.extend(ob_features.feature_names.iter().cloned());

        MarketFeatures {
            features: Array1::from_vec(combined),
            feature_names: names,
        }
    }

    /// Get expected feature dimension for klines only
    pub fn kline_feature_dim(&self) -> usize {
        // Base features + MA features + RSI + ATR + volume + volatility + momentum + range + body + direction
        let base = 1; // return
        let ma = self.config.ma_windows.len();
        let rsi = 1;
        let atr = 1;
        let volume = 1;
        let volatility = 1;
        let momentum = 3; // 5, 10, 20
        let range = 1;
        let body = 1;
        let direction = 1;
        base + ma + rsi + atr + volume + volatility + momentum + range + body + direction
    }

    /// Get config
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }
}

/// Compute simple moving average
fn moving_average(data: &[f64], window: usize) -> f64 {
    if data.len() < window {
        return data.iter().sum::<f64>() / data.len() as f64;
    }
    data[data.len() - window..].iter().sum::<f64>() / window as f64
}

/// Compute RSI (Relative Strength Index)
fn compute_rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 {
        return 50.0;
    }

    let changes: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();
    let recent = &changes[changes.len() - period..];

    let gains: f64 = recent.iter().filter(|&&c| c > 0.0).sum();
    let losses: f64 = recent.iter().filter(|&&c| c < 0.0).map(|c| c.abs()).sum();

    if losses == 0.0 {
        return 100.0;
    }
    if gains == 0.0 {
        return 0.0;
    }

    let rs = gains / losses;
    100.0 - 100.0 / (1.0 + rs)
}

/// Compute ATR (Average True Range)
fn compute_atr(klines: &[Kline], period: usize) -> f64 {
    if klines.len() < 2 {
        return 0.0;
    }

    let mut tr_values = Vec::with_capacity(klines.len() - 1);
    for i in 1..klines.len() {
        let prev_close = klines[i - 1].close;
        let high = klines[i].high;
        let low = klines[i].low;

        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        tr_values.push(tr);
    }

    moving_average(&tr_values, period.min(tr_values.len()))
}

/// Compute standard deviation
fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Normalize features to reasonable ranges
fn normalize_features(features: &[f64]) -> Vec<f64> {
    features.iter().map(|&f| {
        // Clip extreme values and apply soft normalization
        let clipped = f.max(-5.0).min(5.0);
        clipped.tanh()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n).map(|i| Kline {
            timestamp: Utc::now(),
            open: 100.0 + i as f64,
            high: 102.0 + i as f64,
            low: 99.0 + i as f64,
            close: 101.0 + i as f64,
            volume: 1000.0 + (i * 10) as f64,
            turnover: 100000.0,
        }).collect()
    }

    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor::default_extractor();
        let klines = create_test_klines(30);
        let features = extractor.extract_from_klines(&klines);

        assert!(features.dim() > 0);
        assert_eq!(features.features.len(), features.feature_names.len());
    }

    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((moving_average(&data, 3) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        // Strong uptrend should have high RSI
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = compute_rsi(&closes, 14);
        assert!(rsi > 70.0);
    }
}
