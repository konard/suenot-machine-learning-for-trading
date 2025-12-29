//! # Feature Engineering
//!
//! Technical indicators and feature extraction for Neural ODE models.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::candles::{Candle, CandleData};

/// Trading symbol with configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub base_asset: String,
    pub quote_asset: String,
    pub min_qty: f64,
    pub tick_size: f64,
}

impl Symbol {
    pub fn new(name: &str, base: &str, quote: &str) -> Self {
        Self {
            name: name.to_string(),
            base_asset: base.to_string(),
            quote_asset: quote.to_string(),
            min_qty: 0.001,
            tick_size: 0.01,
        }
    }

    /// Common crypto pairs
    pub fn btc_usdt() -> Self {
        Self::new("BTCUSDT", "BTC", "USDT")
    }

    pub fn eth_usdt() -> Self {
        Self::new("ETHUSDT", "ETH", "USDT")
    }

    pub fn sol_usdt() -> Self {
        Self::new("SOLUSDT", "SOL", "USDT")
    }
}

/// Collection of features for model input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Features {
    /// Number of assets
    pub n_assets: usize,
    /// Number of features per asset
    pub n_features: usize,
    /// Feature matrix (n_assets x n_features)
    pub data: Vec<Vec<f64>>,
    /// Feature names
    pub names: Vec<String>,
}

impl Features {
    /// Create empty features
    pub fn new(n_assets: usize, n_features: usize) -> Self {
        Self {
            n_assets,
            n_features,
            data: vec![vec![0.0; n_features]; n_assets],
            names: Vec::new(),
        }
    }

    /// Create from ndarray
    pub fn from_array(arr: Array2<f64>, names: Vec<String>) -> Self {
        let (n_assets, n_features) = arr.dim();
        let data: Vec<Vec<f64>> = arr
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        Self {
            n_assets,
            n_features,
            data,
            names,
        }
    }

    /// Convert to ndarray
    pub fn to_array(&self) -> Array2<f64> {
        let flat: Vec<f64> = self.data.iter().flatten().copied().collect();
        Array2::from_shape_vec((self.n_assets, self.n_features), flat)
            .expect("Invalid feature dimensions")
    }

    /// Get features for a specific asset
    pub fn get_asset(&self, idx: usize) -> Option<&Vec<f64>> {
        self.data.get(idx)
    }

    /// Flatten to 1D array
    pub fn flatten(&self) -> Vec<f64> {
        self.data.iter().flatten().copied().collect()
    }
}

/// Technical indicators calculator
#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    /// RSI period
    pub rsi_period: usize,
    /// MACD fast period
    pub macd_fast: usize,
    /// MACD slow period
    pub macd_slow: usize,
    /// MACD signal period
    pub macd_signal: usize,
    /// Bollinger Bands period
    pub bb_period: usize,
    /// Bollinger Bands standard deviations
    pub bb_std: f64,
    /// ATR period
    pub atr_period: usize,
}

impl Default for TechnicalIndicators {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            bb_period: 20,
            bb_std: 2.0,
            atr_period: 14,
        }
    }
}

impl TechnicalIndicators {
    /// Calculate all features for candle data
    pub fn calculate_all(&self, candles: &CandleData) -> Features {
        let closes = candles.close_prices();
        let highs: Vec<f64> = candles.candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.candles.iter().map(|c| c.volume).collect();

        let n = closes.len();
        if n == 0 {
            return Features::new(1, 0);
        }

        let mut features = Vec::new();
        let mut names = Vec::new();

        // 1. Returns (last period)
        let returns = if n > 1 {
            (closes[n - 1] - closes[n - 2]) / closes[n - 2]
        } else {
            0.0
        };
        features.push(returns);
        names.push("returns".to_string());

        // 2. Log returns
        let log_returns = if n > 1 && closes[n - 2] > 0.0 {
            (closes[n - 1] / closes[n - 2]).ln()
        } else {
            0.0
        };
        features.push(log_returns);
        names.push("log_returns".to_string());

        // 3. RSI
        let rsi = self.calculate_rsi(&closes);
        features.push(rsi / 100.0); // Normalize to [0, 1]
        names.push("rsi".to_string());

        // 4. MACD
        let (macd, signal, hist) = self.calculate_macd(&closes);
        features.push(macd);
        names.push("macd".to_string());
        features.push(signal);
        names.push("macd_signal".to_string());
        features.push(hist);
        names.push("macd_hist".to_string());

        // 5. Bollinger Bands
        let (bb_upper, bb_middle, bb_lower) = self.calculate_bollinger(&closes);
        let bb_position = if bb_upper != bb_lower {
            (closes[n - 1] - bb_lower) / (bb_upper - bb_lower)
        } else {
            0.5
        };
        features.push(bb_position);
        names.push("bb_position".to_string());

        // 6. ATR (normalized)
        let atr = self.calculate_atr(&highs, &lows, &closes);
        let atr_norm = if closes[n - 1] > 0.0 {
            atr / closes[n - 1]
        } else {
            0.0
        };
        features.push(atr_norm);
        names.push("atr_norm".to_string());

        // 7. Volume features
        let vol_sma = self.simple_moving_average(&volumes, 20);
        let vol_ratio = if vol_sma > 0.0 {
            volumes[n - 1] / vol_sma
        } else {
            1.0
        };
        features.push(vol_ratio.min(5.0) / 5.0); // Normalize, cap at 5x
        names.push("volume_ratio".to_string());

        // 8. Price momentum (multiple periods)
        for period in [5, 10, 20] {
            if n > period {
                let momentum = (closes[n - 1] - closes[n - 1 - period]) / closes[n - 1 - period];
                features.push(momentum);
                names.push(format!("momentum_{}", period));
            } else {
                features.push(0.0);
                names.push(format!("momentum_{}", period));
            }
        }

        // 9. Volatility (rolling std of returns)
        let volatility = self.calculate_volatility(&closes, 20);
        features.push(volatility.min(0.1) / 0.1); // Normalize, cap at 10%
        names.push("volatility".to_string());

        // 10. Price position relative to recent high/low
        let recent_high = highs.iter().rev().take(20).cloned().fold(f64::MIN, f64::max);
        let recent_low = lows.iter().rev().take(20).cloned().fold(f64::MAX, f64::min);
        let price_position = if recent_high != recent_low {
            (closes[n - 1] - recent_low) / (recent_high - recent_low)
        } else {
            0.5
        };
        features.push(price_position);
        names.push("price_position".to_string());

        Features {
            n_assets: 1,
            n_features: features.len(),
            data: vec![features],
            names,
        }
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn calculate_rsi(&self, prices: &[f64]) -> f64 {
        if prices.len() < self.rsi_period + 1 {
            return 50.0; // Neutral if not enough data
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in (prices.len() - self.rsi_period)..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        let avg_gain = gains / self.rsi_period as f64;
        let avg_loss = losses / self.rsi_period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn calculate_macd(&self, prices: &[f64]) -> (f64, f64, f64) {
        if prices.len() < self.macd_slow {
            return (0.0, 0.0, 0.0);
        }

        let ema_fast = self.exponential_moving_average(prices, self.macd_fast);
        let ema_slow = self.exponential_moving_average(prices, self.macd_slow);

        let macd = ema_fast - ema_slow;

        // For signal line, we'd need historical MACD values
        // Simplified: use EMA of recent price differences
        let signal = macd * 0.9; // Approximation

        let histogram = macd - signal;

        // Normalize by price
        let last_price = prices.last().unwrap_or(&1.0);
        let norm_factor = if *last_price > 0.0 {
            100.0 / last_price
        } else {
            1.0
        };

        (macd * norm_factor, signal * norm_factor, histogram * norm_factor)
    }

    /// Calculate Bollinger Bands
    pub fn calculate_bollinger(&self, prices: &[f64]) -> (f64, f64, f64) {
        if prices.len() < self.bb_period {
            let last = prices.last().copied().unwrap_or(0.0);
            return (last, last, last);
        }

        let recent: Vec<f64> = prices.iter().rev().take(self.bb_period).copied().collect();
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;

        let variance: f64 = recent
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / recent.len() as f64;

        let std_dev = variance.sqrt();

        let upper = mean + self.bb_std * std_dev;
        let lower = mean - self.bb_std * std_dev;

        (upper, mean, lower)
    }

    /// Calculate ATR (Average True Range)
    pub fn calculate_atr(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> f64 {
        if highs.len() < 2 || highs.len() != lows.len() || highs.len() != closes.len() {
            return 0.0;
        }

        let period = self.atr_period.min(highs.len() - 1);
        let mut tr_sum = 0.0;

        let start = highs.len() - period;
        for i in start..highs.len() {
            let high_low = highs[i] - lows[i];
            let high_close = (highs[i] - closes[i - 1]).abs();
            let low_close = (lows[i] - closes[i - 1]).abs();

            let tr = high_low.max(high_close).max(low_close);
            tr_sum += tr;
        }

        tr_sum / period as f64
    }

    /// Calculate volatility (std dev of returns)
    pub fn calculate_volatility(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 0.0;
        }

        let returns: Vec<f64> = prices
            .windows(2)
            .rev()
            .take(period)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        variance.sqrt()
    }

    /// Simple Moving Average
    fn simple_moving_average(&self, data: &[f64], period: usize) -> f64 {
        if data.len() < period {
            return data.iter().sum::<f64>() / data.len().max(1) as f64;
        }

        let sum: f64 = data.iter().rev().take(period).sum();
        sum / period as f64
    }

    /// Exponential Moving Average
    fn exponential_moving_average(&self, data: &[f64], period: usize) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = data[0];

        for price in data.iter().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }

        ema
    }
}

/// Calculate features for multiple assets
pub fn calculate_multi_asset_features(
    data: &[CandleData],
    indicators: &TechnicalIndicators,
) -> Features {
    let n_assets = data.len();
    if n_assets == 0 {
        return Features::new(0, 0);
    }

    let mut all_features: Vec<Features> = data
        .iter()
        .map(|d| indicators.calculate_all(d))
        .collect();

    // Find max features (should be same for all)
    let n_features = all_features.first().map(|f| f.n_features).unwrap_or(0);

    // Combine into single Features struct
    let combined_data: Vec<Vec<f64>> = all_features
        .iter()
        .map(|f| f.data.first().cloned().unwrap_or_default())
        .collect();

    let names = all_features
        .first()
        .map(|f| f.names.clone())
        .unwrap_or_default();

    Features {
        n_assets,
        n_features,
        data: combined_data,
        names,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_prices() -> Vec<f64> {
        vec![
            100.0, 102.0, 101.0, 103.0, 104.0, 102.0, 105.0, 106.0, 104.0, 107.0,
            108.0, 106.0, 109.0, 110.0, 108.0, 111.0, 112.0, 110.0, 113.0, 115.0,
        ]
    }

    #[test]
    fn test_rsi_calculation() {
        let prices = create_test_prices();
        let indicators = TechnicalIndicators::default();
        let rsi = indicators.calculate_rsi(&prices);

        // RSI should be between 0 and 100
        assert!(rsi >= 0.0 && rsi <= 100.0);
        // Given upward trend, RSI should be above 50
        assert!(rsi > 50.0);
    }

    #[test]
    fn test_bollinger_bands() {
        let prices = create_test_prices();
        let indicators = TechnicalIndicators::default();
        let (upper, middle, lower) = indicators.calculate_bollinger(&prices);

        assert!(upper > middle);
        assert!(middle > lower);
        // Last price should be within bands (usually)
        let last = prices.last().unwrap();
        assert!(*last >= lower && *last <= upper);
    }

    #[test]
    fn test_volatility() {
        let prices = create_test_prices();
        let indicators = TechnicalIndicators::default();
        let vol = indicators.calculate_volatility(&prices, 10);

        // Volatility should be positive
        assert!(vol > 0.0);
        // Should be reasonable (less than 100%)
        assert!(vol < 1.0);
    }

    #[test]
    fn test_features_creation() {
        let features = Features::new(3, 10);
        assert_eq!(features.n_assets, 3);
        assert_eq!(features.n_features, 10);
        assert_eq!(features.data.len(), 3);
        assert_eq!(features.data[0].len(), 10);
    }
}
