//! Feature engineering utilities for graph nodes and edges.

use crate::data::Candle;
use ndarray::Array1;
use std::collections::HashMap;

/// Technical indicators for node features.
pub struct TechnicalFeatures;

impl TechnicalFeatures {
    /// Compute comprehensive technical features from candle data.
    pub fn compute(candles: &[Candle], lookback: usize) -> Array1<f64> {
        let n = candles.len();
        if n < lookback {
            return Array1::zeros(20);
        }

        let recent = &candles[n - lookback..];
        let closes: Vec<f64> = recent.iter().map(|c| c.close).collect();
        let highs: Vec<f64> = recent.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = recent.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = recent.iter().map(|c| c.volume).collect();

        let mut features = Vec::with_capacity(20);

        // 1. Returns
        let returns = Self::compute_returns(&closes);
        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        features.push(mean_return);

        // 2. Volatility
        let volatility = Self::std_dev(&returns);
        features.push(volatility);

        // 3. Momentum (rate of change)
        let momentum = if closes.len() >= 2 {
            (closes.last().unwrap() - closes.first().unwrap()) / closes.first().unwrap()
        } else {
            0.0
        };
        features.push(momentum);

        // 4-5. RSI
        let rsi = Self::compute_rsi(&closes, 14.min(closes.len() - 1));
        features.push(rsi);
        features.push((rsi - 50.0) / 50.0); // Normalized RSI

        // 6-8. Moving averages
        let sma_short = Self::sma(&closes, 5.min(closes.len()));
        let sma_long = Self::sma(&closes, 20.min(closes.len()));
        let current_price = *closes.last().unwrap_or(&0.0);

        features.push((current_price - sma_short) / sma_short.max(1e-10)); // Price relative to SMA5
        features.push((current_price - sma_long) / sma_long.max(1e-10));   // Price relative to SMA20
        features.push((sma_short - sma_long) / sma_long.max(1e-10));       // SMA crossover

        // 9-10. Bollinger Bands
        let (bb_upper, bb_lower) = Self::bollinger_bands(&closes, 20.min(closes.len()), 2.0);
        let bb_width = (bb_upper - bb_lower) / current_price.max(1e-10);
        let bb_position = (current_price - bb_lower) / (bb_upper - bb_lower).max(1e-10);
        features.push(bb_width);
        features.push(bb_position);

        // 11-12. Volume features
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len().max(1) as f64;
        let current_volume = *volumes.last().unwrap_or(&0.0);
        let volume_ratio = current_volume / avg_volume.max(1e-10);
        features.push(volume_ratio);
        features.push(Self::std_dev(&volumes) / avg_volume.max(1e-10)); // Volume volatility

        // 13-14. High-Low range
        let avg_range: f64 = highs.iter().zip(lows.iter())
            .map(|(h, l)| (h - l) / l.max(1e-10))
            .sum::<f64>() / highs.len().max(1) as f64;
        let current_range = (highs.last().unwrap_or(&0.0) - lows.last().unwrap_or(&0.0))
            / lows.last().unwrap_or(&1.0).max(1e-10);
        features.push(avg_range);
        features.push(current_range / avg_range.max(1e-10));

        // 15-16. Trend strength
        let trend = Self::compute_trend_strength(&closes);
        features.push(trend);
        features.push(trend.abs()); // Absolute trend strength

        // 17-18. MACD-like features
        let ema_12 = Self::ema(&closes, 12.min(closes.len()));
        let ema_26 = Self::ema(&closes, 26.min(closes.len()));
        let macd = ema_12 - ema_26;
        features.push(macd / current_price.max(1e-10));
        features.push((macd / current_price.max(1e-10)).signum());

        // 19-20. Skewness and Kurtosis of returns
        features.push(Self::skewness(&returns));
        features.push(Self::kurtosis(&returns));

        Array1::from_vec(features)
    }

    /// Compute simple returns.
    fn compute_returns(prices: &[f64]) -> Vec<f64> {
        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0].max(1e-10))
            .collect()
    }

    /// Compute standard deviation.
    fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Compute RSI (Relative Strength Index).
    fn compute_rsi(prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0;
        }

        let changes: Vec<f64> = prices
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let gains: f64 = changes.iter().filter(|&&c| c > 0.0).sum();
        let losses: f64 = changes.iter().filter(|&&c| c < 0.0).map(|c| -c).sum();

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss < 1e-10 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Compute Simple Moving Average.
    fn sma(prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() || period == 0 {
            return 0.0;
        }
        let n = prices.len().min(period);
        prices[prices.len() - n..].iter().sum::<f64>() / n as f64
    }

    /// Compute Exponential Moving Average.
    fn ema(prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() || period == 0 {
            return 0.0;
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];

        for &price in &prices[1..] {
            ema = alpha * price + (1.0 - alpha) * ema;
        }

        ema
    }

    /// Compute Bollinger Bands.
    fn bollinger_bands(prices: &[f64], period: usize, num_std: f64) -> (f64, f64) {
        let sma = Self::sma(prices, period);
        let n = prices.len().min(period);
        let recent = &prices[prices.len() - n..];

        let variance = recent.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        (sma + num_std * std, sma - num_std * std)
    }

    /// Compute trend strength using linear regression.
    fn compute_trend_strength(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let n = prices.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = prices.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in prices.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator < 1e-10 {
            return 0.0;
        }

        // Normalize slope by price level
        let slope = numerator / denominator;
        slope / y_mean.max(1e-10)
    }

    /// Compute skewness.
    fn skewness(values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = Self::std_dev(values);

        if std < 1e-10 {
            return 0.0;
        }

        values.iter()
            .map(|v| ((v - mean) / std).powi(3))
            .sum::<f64>() / n
    }

    /// Compute excess kurtosis.
    fn kurtosis(values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = Self::std_dev(values);

        if std < 1e-10 {
            return 0.0;
        }

        values.iter()
            .map(|v| ((v - mean) / std).powi(4))
            .sum::<f64>() / n - 3.0
    }
}

/// Edge feature computation.
pub struct EdgeFeatures;

impl EdgeFeatures {
    /// Compute edge features between two assets.
    pub fn compute(
        candles_a: &[Candle],
        candles_b: &[Candle],
        lookback: usize,
    ) -> Array1<f64> {
        let n_a = candles_a.len();
        let n_b = candles_b.len();

        if n_a < lookback || n_b < lookback {
            return Array1::zeros(6);
        }

        let recent_a = &candles_a[n_a - lookback..];
        let recent_b = &candles_b[n_b - lookback..];

        let returns_a: Vec<f64> = recent_a
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();
        let returns_b: Vec<f64> = recent_b
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        let mut features = Vec::with_capacity(6);

        // 1. Correlation
        let corr = Self::correlation(&returns_a, &returns_b);
        features.push(corr);

        // 2. Lead-lag correlation (A leads B)
        let lead_corr = if returns_a.len() > 1 && returns_b.len() > 1 {
            Self::correlation(&returns_a[..returns_a.len() - 1], &returns_b[1..])
        } else {
            0.0
        };
        features.push(lead_corr);

        // 3. Lag correlation (B leads A)
        let lag_corr = if returns_a.len() > 1 && returns_b.len() > 1 {
            Self::correlation(&returns_a[1..], &returns_b[..returns_b.len() - 1])
        } else {
            0.0
        };
        features.push(lag_corr);

        // 4. Volume correlation
        let vol_a: Vec<f64> = recent_a.iter().map(|c| c.volume).collect();
        let vol_b: Vec<f64> = recent_b.iter().map(|c| c.volume).collect();
        features.push(Self::correlation(&vol_a, &vol_b));

        // 5. Relative volatility
        let std_a = TechnicalFeatures::std_dev(&returns_a);
        let std_b = TechnicalFeatures::std_dev(&returns_b);
        features.push(std_a / std_b.max(1e-10));

        // 6. Beta (covariance / variance)
        let cov = Self::covariance(&returns_a, &returns_b);
        let var_b = returns_b.iter().map(|r| r.powi(2)).sum::<f64>() / returns_b.len().max(1) as f64;
        features.push(cov / var_b.max(1e-10));

        Array1::from_vec(features)
    }

    /// Compute Pearson correlation.
    fn correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

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

    /// Compute covariance.
    fn covariance(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>() / n
    }
}

/// Crypto-specific sector classifications.
pub fn get_crypto_sectors() -> HashMap<String, String> {
    let mut sectors = HashMap::new();

    // Layer 1 blockchains
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT"] {
        sectors.insert(symbol.to_string(), "Layer1".to_string());
    }

    // DeFi tokens
    for symbol in ["UNIUSDT", "AAVEUSDT", "LINKUSDT", "MKRUSDT", "COMPUSDT", "SUSHIUSDT"] {
        sectors.insert(symbol.to_string(), "DeFi".to_string());
    }

    // Exchange tokens
    for symbol in ["BNBUSDT", "OKBUSDT", "FTMUSDT"] {
        sectors.insert(symbol.to_string(), "Exchange".to_string());
    }

    // Meme coins
    for symbol in ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"] {
        sectors.insert(symbol.to_string(), "Meme".to_string());
    }

    // Layer 2 / Scaling
    for symbol in ["MATICUSDT", "ARBUSDT", "OPUSDT"] {
        sectors.insert(symbol.to_string(), "Layer2".to_string());
    }

    sectors
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(close: f64, volume: f64) -> Candle {
        Candle {
            timestamp: Utc::now().timestamp() as u64,
            open: close * 0.99,
            high: close * 1.01,
            low: close * 0.98,
            close,
            volume,
            symbol: "TEST".to_string(),
        }
    }

    #[test]
    fn test_technical_features() {
        let candles: Vec<Candle> = (0..50)
            .map(|i| make_candle(100.0 + i as f64 * 0.5, 1000.0 + i as f64 * 10.0))
            .collect();

        let features = TechnicalFeatures::compute(&candles, 20);
        assert_eq!(features.len(), 20);

        // Check that features are reasonable (not NaN or Inf)
        for &f in features.iter() {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_edge_features() {
        let candles_a: Vec<Candle> = (0..50)
            .map(|i| make_candle(100.0 + i as f64, 1000.0))
            .collect();
        let candles_b: Vec<Candle> = (0..50)
            .map(|i| make_candle(50.0 + i as f64 * 0.5, 500.0))
            .collect();

        let features = EdgeFeatures::compute(&candles_a, &candles_b, 20);
        assert_eq!(features.len(), 6);

        // Correlation should be positive for trending prices
        assert!(features[0] > 0.5);
    }
}
