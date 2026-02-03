//! Feature engineering for trading data.
//!
//! Computes technical indicators from OHLCV data:
//! - Log returns
//! - RSI (Relative Strength Index)
//! - MACD (Moving Average Convergence Divergence)
//! - Bollinger Bands
//! - ATR (Average True Range)
//! - OBV (On-Balance Volume)
//! - SMA ratios

use crate::data::bybit::Kline;

/// Feature vector for a single timestep.
#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub log_return: f64,
    pub rsi: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_hist: f64,
    pub bb_upper: f64,
    pub bb_lower: f64,
    pub bb_width: f64,
    pub atr: f64,
    pub obv: f64,
    pub volume_ratio: f64,
    pub sma_5_ratio: f64,
    pub sma_10_ratio: f64,
    pub sma_20_ratio: f64,
    pub sma_50_ratio: f64,
}

impl FeatureRow {
    /// Convert to Vec<f64> for model input.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.log_return,
            self.rsi,
            self.macd,
            self.macd_signal,
            self.macd_hist,
            self.bb_upper,
            self.bb_lower,
            self.bb_width,
            self.atr,
            self.obv,
            self.volume_ratio,
            self.sma_5_ratio,
            self.sma_10_ratio,
            self.sma_20_ratio,
            self.sma_50_ratio,
        ]
    }

    /// Number of features.
    pub fn feature_count() -> usize {
        15
    }
}

/// Feature generator that computes technical indicators from kline data.
#[derive(Debug, Clone)]
pub struct FeatureGenerator {
    pub rsi_period: usize,
    pub macd_fast: usize,
    pub macd_slow: usize,
    pub macd_signal_period: usize,
    pub bb_period: usize,
    pub bb_std: f64,
    pub atr_period: usize,
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal_period: 9,
            bb_period: 20,
            bb_std: 2.0,
            atr_period: 14,
        }
    }
}

impl FeatureGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate features from kline data.
    ///
    /// Returns None for the first ~50 rows due to warmup period.
    pub fn generate(&self, klines: &[Kline]) -> Vec<Option<FeatureRow>> {
        let n = klines.len();
        if n < 51 {
            return vec![None; n];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Compute all indicators
        let log_returns = self.compute_log_returns(&closes);
        let rsi = self.compute_rsi(&closes);
        let (macd_line, macd_sig, macd_hist) = self.compute_macd(&closes);
        let (bb_upper, bb_lower, bb_width) = self.compute_bollinger(&closes);
        let atr = self.compute_atr(&highs, &lows, &closes);
        let obv = self.compute_obv(&closes, &volumes);
        let volume_ratio = self.compute_volume_ratio(&volumes);
        let sma_5 = self.compute_sma_ratio(&closes, 5);
        let sma_10 = self.compute_sma_ratio(&closes, 10);
        let sma_20 = self.compute_sma_ratio(&closes, 20);
        let sma_50 = self.compute_sma_ratio(&closes, 50);

        (0..n)
            .map(|i| {
                if i < 50 {
                    return None;
                }
                Some(FeatureRow {
                    log_return: log_returns[i],
                    rsi: rsi[i],
                    macd: macd_line[i],
                    macd_signal: macd_sig[i],
                    macd_hist: macd_hist[i],
                    bb_upper: bb_upper[i],
                    bb_lower: bb_lower[i],
                    bb_width: bb_width[i],
                    atr: atr[i],
                    obv: obv[i],
                    volume_ratio: volume_ratio[i],
                    sma_5_ratio: sma_5[i],
                    sma_10_ratio: sma_10[i],
                    sma_20_ratio: sma_20[i],
                    sma_50_ratio: sma_50[i],
                })
            })
            .collect()
    }

    fn compute_log_returns(&self, closes: &[f64]) -> Vec<f64> {
        let mut ret = vec![0.0; closes.len()];
        for i in 1..closes.len() {
            if closes[i - 1] > 0.0 {
                ret[i] = (closes[i] / closes[i - 1]).ln();
            }
        }
        ret
    }

    fn compute_rsi(&self, closes: &[f64]) -> Vec<f64> {
        let n = closes.len();
        let mut rsi = vec![0.5; n];
        let period = self.rsi_period;

        if n <= period {
            return rsi;
        }

        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        for i in 1..=period {
            let delta = closes[i] - closes[i - 1];
            if delta > 0.0 {
                avg_gain += delta;
            } else {
                avg_loss -= delta;
            }
        }
        avg_gain /= period as f64;
        avg_loss /= period as f64;

        for i in period..n {
            if i > period {
                let delta = closes[i] - closes[i - 1];
                let (gain, loss) = if delta > 0.0 { (delta, 0.0) } else { (0.0, -delta) };
                avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
                avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
            }

            if avg_loss > 1e-10 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 1.0 - 1.0 / (1.0 + rs); // normalized to 0-1
            } else {
                rsi[i] = 1.0;
            }
        }
        rsi
    }

    fn compute_macd(&self, closes: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = ema(closes, self.macd_fast);
        let ema_slow = ema(closes, self.macd_slow);

        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .zip(closes.iter())
            .map(|((f, s), c)| if *c > 1e-10 { (f - s) / c } else { 0.0 })
            .collect();

        let signal = ema(&macd_line, self.macd_signal_period);
        let hist: Vec<f64> = macd_line.iter().zip(signal.iter()).map(|(m, s)| m - s).collect();

        (macd_line, signal, hist)
    }

    fn compute_bollinger(&self, closes: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = closes.len();
        let period = self.bb_period;
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];
        let mut width = vec![0.0; n];

        for i in period..n {
            let window = &closes[i + 1 - period..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let var: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = var.sqrt();

            let c = closes[i];
            if c > 1e-10 {
                upper[i] = (mean + self.bb_std * std - c) / c;
                lower[i] = (c - mean + self.bb_std * std) / c;
                width[i] = (2.0 * self.bb_std * std) / c;
            }
        }

        (upper, lower, width)
    }

    fn compute_atr(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
        let n = highs.len();
        let period = self.atr_period;
        let mut atr = vec![0.0; n];

        let mut tr_vals = Vec::with_capacity(n);
        tr_vals.push(highs[0] - lows[0]);

        for i in 1..n {
            let tr = (highs[i] - lows[i])
                .max((highs[i] - closes[i - 1]).abs())
                .max((lows[i] - closes[i - 1]).abs());
            tr_vals.push(tr);
        }

        if n > period {
            let mut avg: f64 = tr_vals[..period].iter().sum::<f64>() / period as f64;
            atr[period - 1] = if closes[period - 1] > 1e-10 {
                avg / closes[period - 1]
            } else {
                0.0
            };

            for i in period..n {
                avg = (avg * (period as f64 - 1.0) + tr_vals[i]) / period as f64;
                atr[i] = if closes[i] > 1e-10 { avg / closes[i] } else { 0.0 };
            }
        }

        atr
    }

    fn compute_obv(&self, closes: &[f64], volumes: &[f64]) -> Vec<f64> {
        let n = closes.len();
        let mut obv = vec![0.0; n];
        let mut cumulative = 0.0;

        for i in 1..n {
            if closes[i] > closes[i - 1] {
                cumulative += volumes[i];
            } else if closes[i] < closes[i - 1] {
                cumulative -= volumes[i];
            }
            obv[i] = cumulative;
        }

        // Normalize using rolling z-score (window=50)
        let window = 50;
        let mut normalized = vec![0.0; n];
        for i in window..n {
            let w = &obv[i + 1 - window..=i];
            let mean: f64 = w.iter().sum::<f64>() / window as f64;
            let var: f64 = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            let std = var.sqrt();
            normalized[i] = if std > 1e-10 { (obv[i] - mean) / std } else { 0.0 };
        }

        normalized
    }

    fn compute_volume_ratio(&self, volumes: &[f64]) -> Vec<f64> {
        let n = volumes.len();
        let window = 20;
        let mut ratio = vec![1.0; n];

        for i in window..n {
            let avg: f64 = volumes[i + 1 - window..=i].iter().sum::<f64>() / window as f64;
            ratio[i] = if avg > 1e-10 { volumes[i] / avg } else { 1.0 };
        }

        ratio
    }

    fn compute_sma_ratio(&self, closes: &[f64], period: usize) -> Vec<f64> {
        let n = closes.len();
        let mut ratio = vec![0.0; n];

        for i in period..n {
            let sma: f64 = closes[i + 1 - period..=i].iter().sum::<f64>() / period as f64;
            ratio[i] = if sma > 1e-10 { (closes[i] - sma) / sma } else { 0.0 };
        }

        ratio
    }
}

/// Compute Exponential Moving Average.
fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    if n == 0 || period == 0 {
        return result;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    result[0] = data[0];

    for i in 1..n {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_klines;

    #[test]
    fn test_feature_generation() {
        let klines = generate_synthetic_klines(200);
        let gen = FeatureGenerator::new();
        let features = gen.generate(&klines);

        assert_eq!(features.len(), 200);
        // First 50 should be None (warmup)
        assert!(features[0].is_none());
        assert!(features[49].is_none());
        // After warmup should be Some
        assert!(features[50].is_some());

        let row = features[100].as_ref().unwrap();
        assert_eq!(row.to_vec().len(), FeatureRow::feature_count());
    }

    #[test]
    fn test_rsi_range() {
        let klines = generate_synthetic_klines(100);
        let gen = FeatureGenerator::new();
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let rsi = gen.compute_rsi(&closes);

        for &r in &rsi[14..] {
            assert!(r >= 0.0 && r <= 1.0, "RSI out of range: {}", r);
        }
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 3);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 0.001);
    }
}
