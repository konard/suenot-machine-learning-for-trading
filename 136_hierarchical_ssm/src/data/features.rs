//! Feature engineering for financial time series.
//!
//! Computes technical indicators from OHLCV data to serve as
//! input features for the Hierarchical SSM.

use super::bybit::Candle;

/// A single row of computed features.
#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub log_return: f64,
    pub volatility_20: f64,
    pub rsi_14: f64,
    pub macd_signal: f64,
    pub volume_ratio: f64,
    pub hl_range: f64,
    pub co_range: f64,
    pub price_position: f64,
}

impl FeatureRow {
    /// Convert to a fixed-size vector for model input.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.log_return,
            self.volatility_20,
            self.rsi_14,
            self.macd_signal,
            self.volume_ratio,
            self.hl_range,
            self.co_range,
            self.price_position,
        ]
    }
}

/// Compute features from a slice of candles.
///
/// Requires at least 27 candles (for MACD with 26-period EMA + 1 for diff).
/// Returns features for each valid timestep (after warmup period).
pub fn compute_features(candles: &[Candle]) -> Vec<FeatureRow> {
    let n = candles.len();
    if n < 27 {
        return Vec::new();
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    // Log returns
    let log_returns: Vec<f64> = (1..n)
        .map(|i| (closes[i] / closes[i - 1]).ln())
        .collect();

    // Rolling volatility (20-period)
    let volatility = rolling_std(&log_returns, 20);

    // RSI (14-period)
    let rsi = compute_rsi(&closes, 14);

    // MACD
    let ema12 = ema(&closes, 12);
    let ema26 = ema(&closes, 26);
    let macd: Vec<f64> = ema12.iter().zip(ema26.iter()).map(|(a, b)| a - b).collect();
    let macd_signal_line = ema(&macd, 9);
    let macd_hist: Vec<f64> = macd
        .iter()
        .zip(macd_signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    // Volume ratio (20-period average)
    let vol_avg = rolling_mean(&volumes, 20);

    // Find the start index where all indicators are valid
    let start = 26; // Max warmup period

    let mut features = Vec::new();

    for i in start..n {
        let ret_idx = i - 1; // log_returns is offset by 1
        if ret_idx >= log_returns.len() {
            break;
        }

        let vol_20 = if ret_idx < volatility.len() {
            volatility[ret_idx]
        } else {
            continue;
        };

        if vol_20.is_nan() {
            continue;
        }

        let rsi_val = if i < rsi.len() { rsi[i] } else { continue };
        if rsi_val.is_nan() {
            continue;
        }

        let macd_val = if i < macd_hist.len() {
            macd_hist[i]
        } else {
            continue;
        };

        let vol_ratio = if i < vol_avg.len() && vol_avg[i] > 0.0 {
            volumes[i] / vol_avg[i]
        } else {
            continue;
        };

        let range = highs[i] - lows[i];
        let hl_range = if closes[i] != 0.0 {
            range / closes[i]
        } else {
            0.0
        };
        let co_range = if closes[i] != 0.0 {
            (closes[i] - opens[i]) / closes[i]
        } else {
            0.0
        };
        let price_pos = if range > 0.0 {
            (closes[i] - lows[i]) / range
        } else {
            0.5
        };

        features.push(FeatureRow {
            log_return: log_returns[ret_idx],
            volatility_20: vol_20,
            rsi_14: rsi_val / 100.0, // Normalize to [0, 1]
            macd_signal: macd_val,
            volume_ratio: vol_ratio,
            hl_range,
            co_range,
            price_position: price_pos,
        });
    }

    features
}

/// Normalize feature vectors to zero mean and unit variance.
pub fn normalize_features(features: &[FeatureRow]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new();
    }

    let vecs: Vec<Vec<f64>> = features.iter().map(|f| f.to_vec()).collect();
    let dim = vecs[0].len();
    let n = vecs.len() as f64;

    // Compute mean and std per feature
    let mut means = vec![0.0; dim];
    let mut stds = vec![0.0; dim];

    for v in &vecs {
        for d in 0..dim {
            means[d] += v[d];
        }
    }
    for d in 0..dim {
        means[d] /= n;
    }

    for v in &vecs {
        for d in 0..dim {
            stds[d] += (v[d] - means[d]).powi(2);
        }
    }
    for d in 0..dim {
        stds[d] = (stds[d] / n).sqrt().max(1e-8);
    }

    // Normalize
    vecs.iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .map(|(d, &val)| (val - means[d]) / stds[d])
                .collect()
        })
        .collect()
}

// --- Helper functions ---

fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];
    if data.len() < window {
        return result;
    }
    let mut sum: f64 = data[..window].iter().sum();
    result[window - 1] = sum / window as f64;
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result[i] = sum / window as f64;
    }
    result
}

fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];
    if data.len() < window {
        return result;
    }
    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window as f64;
        result[i] = var.sqrt();
    }
    result
}

fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];
    if data.is_empty() {
        return result;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    result[0] = data[0];
    for i in 1..data.len() {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }
    result
}

fn compute_rsi(closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    let mut rsi = vec![f64::NAN; n];
    if n <= period {
        return rsi;
    }

    let deltas: Vec<f64> = (1..n).map(|i| closes[i] - closes[i - 1]).collect();

    let mut avg_gain: f64 = deltas[..period]
        .iter()
        .map(|d| d.max(0.0))
        .sum::<f64>()
        / period as f64;
    let mut avg_loss: f64 = deltas[..period]
        .iter()
        .map(|d| (-d).max(0.0))
        .sum::<f64>()
        / period as f64;

    if avg_loss > 0.0 {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - 100.0 / (1.0 + rs);
    } else {
        rsi[period] = 100.0;
    }

    for i in period..deltas.len() {
        let gain = deltas[i].max(0.0);
        let loss = (-deltas[i]).max(0.0);
        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs);
        } else {
            rsi[i + 1] = 100.0;
        }
    }

    rsi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_candles;

    #[test]
    fn test_compute_features() {
        let candles = generate_synthetic_candles(100, 50000.0);
        let features = compute_features(&candles);
        assert!(!features.is_empty());

        for f in &features {
            assert!(!f.log_return.is_nan());
            assert!(!f.volatility_20.is_nan());
            assert!(f.rsi_14 >= 0.0 && f.rsi_14 <= 1.0);
        }
    }

    #[test]
    fn test_normalize_features() {
        let candles = generate_synthetic_candles(100, 50000.0);
        let features = compute_features(&candles);
        let normalized = normalize_features(&features);
        assert_eq!(normalized.len(), features.len());

        if !normalized.is_empty() {
            assert_eq!(normalized[0].len(), 8);
        }
    }
}
