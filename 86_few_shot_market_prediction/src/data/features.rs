//! Feature extraction from market data

use super::types::{Kline, OrderBook};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Window sizes for moving averages
    pub window_sizes: Vec<usize>,
    /// Whether to include volume features
    pub include_volume: bool,
    /// Whether to include technical indicators
    pub include_technical: bool,
    /// Whether to include order book features
    pub include_orderbook: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            window_sizes: vec![5, 10, 20, 50],
            include_volume: true,
            include_technical: true,
            include_orderbook: true,
        }
    }
}

/// Extracted market features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFeatures {
    /// Price-based features
    pub price_features: Vec<f64>,
    /// Volume-based features
    pub volume_features: Vec<f64>,
    /// Technical indicator features
    pub technical_features: Vec<f64>,
    /// Order book features
    pub orderbook_features: Vec<f64>,
}

impl MarketFeatures {
    /// Create empty features
    pub fn empty() -> Self {
        Self {
            price_features: Vec::new(),
            volume_features: Vec::new(),
            technical_features: Vec::new(),
            orderbook_features: Vec::new(),
        }
    }

    /// Convert to a flat feature vector
    pub fn to_array(&self) -> Array1<f64> {
        let mut features = Vec::new();
        features.extend(&self.price_features);
        features.extend(&self.volume_features);
        features.extend(&self.technical_features);
        features.extend(&self.orderbook_features);
        Array1::from_vec(features)
    }

    /// Get total feature dimension
    pub fn dim(&self) -> usize {
        self.price_features.len()
            + self.volume_features.len()
            + self.technical_features.len()
            + self.orderbook_features.len()
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

    /// Extract features from kline data
    pub fn extract_from_klines(&self, klines: &[Kline]) -> MarketFeatures {
        if klines.is_empty() {
            return MarketFeatures::empty();
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        let mut features = MarketFeatures::empty();

        // Price features
        features.price_features = self.extract_price_features(&closes);

        // Volume features
        if self.config.include_volume {
            features.volume_features = self.extract_volume_features(&volumes);
        }

        // Technical indicators
        if self.config.include_technical {
            features.technical_features = self.extract_technical_features(&closes, &klines);
        }

        features
    }

    /// Extract features from klines and order book
    pub fn extract_with_orderbook(&self, klines: &[Kline], orderbook: &OrderBook) -> MarketFeatures {
        let mut features = self.extract_from_klines(klines);

        if self.config.include_orderbook {
            features.orderbook_features = self.extract_orderbook_features(orderbook);
        }

        features
    }

    /// Extract price-based features
    fn extract_price_features(&self, closes: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();

        if closes.len() < 2 {
            return vec![0.0; 10];
        }

        // Returns
        let returns: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Latest return
        features.push(*returns.last().unwrap_or(&0.0));

        // Cumulative returns over different windows
        for &window in &self.config.window_sizes {
            if returns.len() >= window {
                let cum_return: f64 = returns[returns.len() - window..].iter().sum();
                features.push(cum_return);
            } else {
                features.push(0.0);
            }
        }

        // Moving average ratios
        let current_price = *closes.last().unwrap();
        for &window in &self.config.window_sizes {
            if closes.len() >= window {
                let ma: f64 = closes[closes.len() - window..].iter().sum::<f64>() / window as f64;
                features.push(current_price / ma - 1.0);
            } else {
                features.push(0.0);
            }
        }

        features
    }

    /// Extract volume-based features
    fn extract_volume_features(&self, volumes: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();

        if volumes.len() < 20 {
            return vec![0.0; 4];
        }

        // Volume ratio to 20-period average
        let vol_ma: f64 = volumes[volumes.len() - 20..].iter().sum::<f64>() / 20.0;
        if vol_ma > 0.0 {
            features.push(volumes.last().unwrap() / vol_ma - 1.0);
        } else {
            features.push(0.0);
        }

        // Volume volatility
        let vol_std = std_dev(&volumes[volumes.len() - 20..]);
        features.push(vol_std / vol_ma.max(1e-10));

        // Volume trend
        if volumes.len() >= 10 {
            let recent_avg: f64 = volumes[volumes.len() - 5..].iter().sum::<f64>() / 5.0;
            let prev_avg: f64 = volumes[volumes.len() - 10..volumes.len() - 5].iter().sum::<f64>() / 5.0;
            if prev_avg > 0.0 {
                features.push(recent_avg / prev_avg - 1.0);
            } else {
                features.push(0.0);
            }
        } else {
            features.push(0.0);
        }

        // Current volume rank
        let current_vol = *volumes.last().unwrap();
        let rank = volumes.iter().filter(|&&v| v < current_vol).count() as f64 / volumes.len() as f64;
        features.push(rank);

        features
    }

    /// Extract technical indicator features
    fn extract_technical_features(&self, closes: &[f64], klines: &[Kline]) -> Vec<f64> {
        let mut features = Vec::new();

        if closes.len() < 20 {
            return vec![0.0; 6];
        }

        // RSI (14-period)
        let rsi = calculate_rsi(closes, 14);
        features.push(rsi / 100.0 - 0.5);  // Normalize to [-0.5, 0.5]

        // Volatility (20-period)
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
        if returns.len() >= 20 {
            let volatility = std_dev(&returns[returns.len() - 20..]);
            features.push(volatility);
        } else {
            features.push(0.0);
        }

        // Price momentum (5-period)
        if closes.len() >= 6 {
            let momentum = closes.last().unwrap() / closes[closes.len() - 6] - 1.0;
            features.push(momentum);
        } else {
            features.push(0.0);
        }

        // Average True Range (ATR) normalized
        if klines.len() >= 14 {
            let atr = calculate_atr(klines, 14);
            let current_price = *closes.last().unwrap();
            features.push(atr / current_price);
        } else {
            features.push(0.0);
        }

        // High-Low range ratio
        if !klines.is_empty() {
            let last = klines.last().unwrap();
            if last.open > 0.0 {
                features.push(last.range() / last.open);
            } else {
                features.push(0.0);
            }
        } else {
            features.push(0.0);
        }

        // Bullish candle ratio (recent 10)
        let bullish_count = klines.iter().rev().take(10).filter(|k| k.is_bullish()).count();
        features.push(bullish_count as f64 / 10.0 - 0.5);

        features
    }

    /// Extract order book features
    fn extract_orderbook_features(&self, orderbook: &OrderBook) -> Vec<f64> {
        let mut features = Vec::new();

        // Spread percentage
        if let Some(spread_pct) = orderbook.spread_pct() {
            features.push(spread_pct);
        } else {
            features.push(0.0);
        }

        // Imbalance at different depths
        for depth in [5, 10, 20] {
            features.push(orderbook.imbalance(depth));
        }

        features
    }

    /// Get expected feature dimension
    pub fn expected_dim(&self) -> usize {
        let mut dim = 0;

        // Price features: 1 + window_sizes.len() * 2
        dim += 1 + self.config.window_sizes.len() * 2;

        // Volume features
        if self.config.include_volume {
            dim += 4;
        }

        // Technical features
        if self.config.include_technical {
            dim += 6;
        }

        // Order book features
        if self.config.include_orderbook {
            dim += 4;
        }

        dim
    }
}

/// Calculate standard deviation
fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Calculate RSI
fn calculate_rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 {
        return 50.0;
    }

    let returns: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();
    let recent_returns = &returns[returns.len() - period..];

    let gains: f64 = recent_returns.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = recent_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;

    if avg_loss == 0.0 {
        return 100.0;
    }

    let rs = avg_gain / avg_loss;
    100.0 - 100.0 / (1.0 + rs)
}

/// Calculate Average True Range
fn calculate_atr(klines: &[Kline], period: usize) -> f64 {
    if klines.len() < period + 1 {
        return 0.0;
    }

    let mut true_ranges = Vec::new();
    for i in 1..klines.len() {
        let high_low = klines[i].high - klines[i].low;
        let high_prev_close = (klines[i].high - klines[i - 1].close).abs();
        let low_prev_close = (klines[i].low - klines[i - 1].close).abs();
        let tr = high_low.max(high_prev_close).max(low_prev_close);
        true_ranges.push(tr);
    }

    if true_ranges.len() < period {
        return 0.0;
    }

    true_ranges[true_ranges.len() - period..].iter().sum::<f64>() / period as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                timestamp: Utc::now(),
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());
        let klines = create_test_klines(50);

        let features = extractor.extract_from_klines(&klines);
        assert!(features.dim() > 0);
    }

    #[test]
    fn test_rsi_calculation() {
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = calculate_rsi(&closes, 14);
        assert!(rsi >= 0.0 && rsi <= 100.0);
    }

    #[test]
    fn test_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = std_dev(&values);
        assert!((std - 1.4142).abs() < 0.01);
    }
}
