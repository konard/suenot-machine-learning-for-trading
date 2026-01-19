//! Feature engineering for financial time series

use crate::api::Kline;
use ndarray::{Array1, Array2};

/// Feature names for the model
pub const FEATURE_NAMES: &[&str] = &[
    "log_return",
    "volatility_20",
    "rsi_14",
    "ma_ratio_20",
    "ma_ratio_50",
    "volume_ratio",
    "high_low_range",
    "close_open_ratio",
];

/// Number of features
pub const NUM_FEATURES: usize = FEATURE_NAMES.len();

/// Computed features for model input
#[derive(Debug, Clone)]
pub struct Features {
    /// Feature matrix [time, features]
    pub values: Array2<f64>,
    /// Target values (future returns)
    pub targets: Option<Array1<f64>>,
    /// Timestamps
    pub timestamps: Vec<u64>,
    /// Close prices (for backtesting)
    pub close_prices: Vec<f64>,
}

impl Features {
    /// Create empty features
    pub fn empty() -> Self {
        Self {
            values: Array2::zeros((0, NUM_FEATURES)),
            targets: None,
            timestamps: Vec::new(),
            close_prices: Vec::new(),
        }
    }

    /// Compute features from klines
    pub fn from_klines(klines: &[Kline], target_horizon: usize) -> Self {
        if klines.len() < 51 {
            return Self::empty();
        }

        let n = klines.len();
        let valid_len = n.saturating_sub(target_horizon);

        let mut values = Array2::zeros((valid_len, NUM_FEATURES));
        let mut timestamps = Vec::with_capacity(valid_len);
        let mut close_prices = Vec::with_capacity(valid_len);

        // Compute features for each time step
        for i in 50..valid_len {
            let kline = &klines[i];
            timestamps.push(kline.timestamp);
            close_prices.push(kline.close);

            // Log return
            let log_return = if klines[i - 1].close > 0.0 {
                (kline.close / klines[i - 1].close).ln()
            } else {
                0.0
            };
            values[[i - 50, 0]] = log_return;

            // 20-period volatility
            let returns_20: Vec<f64> = (i - 19..=i)
                .filter_map(|j| {
                    if j > 0 && klines[j - 1].close > 0.0 {
                        Some((klines[j].close / klines[j - 1].close).ln())
                    } else {
                        None
                    }
                })
                .collect();
            let volatility = std_dev(&returns_20);
            values[[i - 50, 1]] = volatility;

            // RSI 14
            let rsi = compute_rsi(&klines[i - 14..=i]);
            values[[i - 50, 2]] = (rsi - 50.0) / 50.0; // Normalize to [-1, 1]

            // MA ratio 20
            let ma_20 = mean(&klines[i - 19..=i].iter().map(|k| k.close).collect::<Vec<_>>());
            let ma_ratio_20 = if ma_20 > 0.0 {
                kline.close / ma_20 - 1.0
            } else {
                0.0
            };
            values[[i - 50, 3]] = ma_ratio_20;

            // MA ratio 50
            let ma_50 = mean(&klines[i - 49..=i].iter().map(|k| k.close).collect::<Vec<_>>());
            let ma_ratio_50 = if ma_50 > 0.0 {
                kline.close / ma_50 - 1.0
            } else {
                0.0
            };
            values[[i - 50, 4]] = ma_ratio_50;

            // Volume ratio (current vs 20-period average)
            let avg_volume = mean(&klines[i - 19..=i].iter().map(|k| k.volume).collect::<Vec<_>>());
            let volume_ratio = if avg_volume > 0.0 {
                (kline.volume / avg_volume).ln()
            } else {
                0.0
            };
            values[[i - 50, 5]] = volume_ratio.clamp(-3.0, 3.0);

            // High-low range
            let range = if kline.close > 0.0 {
                (kline.high - kline.low) / kline.close
            } else {
                0.0
            };
            values[[i - 50, 6]] = range;

            // Close-open ratio
            let co_ratio = if kline.open > 0.0 {
                (kline.close - kline.open) / kline.open
            } else {
                0.0
            };
            values[[i - 50, 7]] = co_ratio;
        }

        // Compute targets (future returns)
        let mut targets = Array1::zeros(valid_len - 50);
        for i in 50..valid_len {
            let future_idx = (i + target_horizon).min(n - 1);
            let future_return = if klines[i].close > 0.0 {
                (klines[future_idx].close / klines[i].close).ln()
            } else {
                0.0
            };
            targets[i - 50] = future_return;
        }

        // Slice values to match
        let values = values.slice(ndarray::s![..valid_len - 50, ..]).to_owned();
        let timestamps = timestamps[..valid_len - 50].to_vec();
        let close_prices = close_prices[..valid_len - 50].to_vec();

        Self {
            values,
            targets: Some(targets),
            timestamps,
            close_prices,
        }
    }

    /// Number of time steps
    pub fn len(&self) -> usize {
        self.values.nrows()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Normalize features using z-score
    pub fn normalize(&mut self) {
        for col in 0..self.values.ncols() {
            let column = self.values.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let std = std_dev(&column.to_vec());

            if std > 1e-8 {
                for row in 0..self.values.nrows() {
                    self.values[[row, col]] = (self.values[[row, col]] - mean) / std;
                }
            }
        }
    }

    /// Split into train and test sets
    pub fn train_test_split(&self, train_ratio: f64) -> (Features, Features) {
        let n = self.len();
        let train_size = ((n as f64) * train_ratio) as usize;

        let train = Features {
            values: self.values.slice(ndarray::s![..train_size, ..]).to_owned(),
            targets: self.targets.as_ref().map(|t| t.slice(ndarray::s![..train_size]).to_owned()),
            timestamps: self.timestamps[..train_size].to_vec(),
            close_prices: self.close_prices[..train_size].to_vec(),
        };

        let test = Features {
            values: self.values.slice(ndarray::s![train_size.., ..]).to_owned(),
            targets: self.targets.as_ref().map(|t| t.slice(ndarray::s![train_size..]).to_owned()),
            timestamps: self.timestamps[train_size..].to_vec(),
            close_prices: self.close_prices[train_size..].to_vec(),
        };

        (train, test)
    }
}

/// Compute RSI (Relative Strength Index)
fn compute_rsi(klines: &[Kline]) -> f64 {
    if klines.len() < 2 {
        return 50.0;
    }

    let mut gains = 0.0;
    let mut losses = 0.0;
    let mut gain_count = 0;
    let mut loss_count = 0;

    for i in 1..klines.len() {
        let change = klines[i].close - klines[i - 1].close;
        if change > 0.0 {
            gains += change;
            gain_count += 1;
        } else {
            losses -= change;
            loss_count += 1;
        }
    }

    let avg_gain = if gain_count > 0 { gains / gain_count as f64 } else { 0.0 };
    let avg_loss = if loss_count > 0 { losses / loss_count as f64 } else { 0.0 };

    if avg_loss == 0.0 {
        100.0
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

/// Compute mean of a slice
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute standard deviation of a slice
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let base_price = 100.0 + (i as f64 * 0.1);
                Kline {
                    timestamp: i as u64 * 3600000,
                    open: base_price,
                    high: base_price * 1.01,
                    low: base_price * 0.99,
                    close: base_price * (1.0 + (i % 2) as f64 * 0.005),
                    volume: 1000.0 + (i % 10) as f64 * 100.0,
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_features_from_klines() {
        let klines = create_test_klines(200);
        let features = Features::from_klines(&klines, 24);

        assert!(features.len() > 0);
        assert_eq!(features.values.ncols(), NUM_FEATURES);
        assert!(features.targets.is_some());
    }

    #[test]
    fn test_features_normalize() {
        let klines = create_test_klines(200);
        let mut features = Features::from_klines(&klines, 24);
        features.normalize();

        // Check that mean is approximately 0 for each column
        for col in 0..features.values.ncols() {
            let column_mean = features.values.column(col).mean().unwrap_or(0.0);
            assert!(column_mean.abs() < 0.1, "Column {} mean: {}", col, column_mean);
        }
    }

    #[test]
    fn test_train_test_split() {
        let klines = create_test_klines(200);
        let features = Features::from_klines(&klines, 24);

        let (train, test) = features.train_test_split(0.8);

        let total = train.len() + test.len();
        assert_eq!(total, features.len());
        assert!(train.len() > test.len());
    }

    #[test]
    fn test_rsi() {
        let klines = create_test_klines(15);
        let rsi = compute_rsi(&klines);

        assert!(rsi >= 0.0 && rsi <= 100.0);
    }

    #[test]
    fn test_mean_std() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((mean(&values) - 3.0).abs() < 0.001);
        assert!((std_dev(&values) - 1.5811).abs() < 0.001);
    }
}
