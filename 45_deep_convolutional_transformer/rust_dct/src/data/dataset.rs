//! Dataset preparation for DCT model

use super::features::{compute_features, create_movement_labels};
use super::loader::OHLCV;
use ndarray::{Array2, Array3, Axis};

/// Configuration for dataset preparation
pub struct DatasetConfig {
    /// Lookback window size
    pub lookback: usize,
    /// Prediction horizon
    pub horizon: usize,
    /// Movement classification threshold
    pub threshold: f64,
    /// Train/validation/test split ratios
    pub train_ratio: f64,
    pub val_ratio: f64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            lookback: 30,
            horizon: 1,
            threshold: 0.005,
            train_ratio: 0.7,
            val_ratio: 0.15,
        }
    }
}

/// Prepared dataset for training
pub struct PreparedDataset {
    pub x_train: Array3<f64>,
    pub y_train: Vec<i32>,
    pub x_val: Array3<f64>,
    pub y_val: Vec<i32>,
    pub x_test: Array3<f64>,
    pub y_test: Vec<i32>,
    pub feature_names: Vec<String>,
}

/// Create sliding window sequences from features
fn create_sequences(
    features: &Array2<f64>,
    labels: &[i32],
    lookback: usize,
    horizon: usize,
) -> (Array3<f64>, Vec<i32>) {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    if n_samples < lookback + horizon {
        return (Array3::zeros((0, lookback, n_features)), Vec::new());
    }

    let n_sequences = n_samples - lookback - horizon + 1;
    let mut x = Array3::zeros((n_sequences, lookback, n_features));
    let mut y = Vec::with_capacity(n_sequences);

    for i in 0..n_sequences {
        // Copy lookback window
        for j in 0..lookback {
            for k in 0..n_features {
                x[[i, j, k]] = features[[i + j, k]];
            }
        }
        // Label is at the end of lookback window
        y.push(labels[i + lookback - 1]);
    }

    (x, y)
}

/// Z-score normalization along axis 0
fn normalize_features(data: &mut Array3<f64>) {
    let shape = data.dim();

    for k in 0..shape.2 {
        // Compute mean and std for each feature across all samples and time steps
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let val = data[[i, j, k]];
                if val.is_finite() {
                    sum += val;
                    count += 1;
                }
            }
        }

        let mean = if count > 0 { sum / count as f64 } else { 0.0 };

        let mut var_sum = 0.0;
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let val = data[[i, j, k]];
                if val.is_finite() {
                    var_sum += (val - mean).powi(2);
                }
            }
        }

        let std = if count > 1 {
            (var_sum / (count - 1) as f64).sqrt()
        } else {
            1.0
        };

        let std = if std < 1e-8 { 1.0 } else { std };

        // Apply normalization
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let val = data[[i, j, k]];
                if val.is_finite() {
                    data[[i, j, k]] = (val - mean) / std;
                } else {
                    data[[i, j, k]] = 0.0;
                }
            }
        }
    }
}

/// Prepare dataset from OHLCV data
pub fn prepare_dataset(ohlcv: &OHLCV, config: &DatasetConfig) -> Option<PreparedDataset> {
    // Compute features
    let features = compute_features(ohlcv);
    if features.nrows() == 0 {
        return None;
    }

    // Create labels
    let labels = create_movement_labels(&ohlcv.close, config.threshold, config.horizon);

    // Create sequences
    let (mut x, y) = create_sequences(&features, labels.as_slice().unwrap(), config.lookback, config.horizon);

    if x.dim().0 == 0 {
        return None;
    }

    // Normalize features
    normalize_features(&mut x);

    // Split into train/val/test
    let n = x.dim().0;
    let train_end = (n as f64 * config.train_ratio) as usize;
    let val_end = (n as f64 * (config.train_ratio + config.val_ratio)) as usize;

    let x_train = x.slice(ndarray::s![0..train_end, .., ..]).to_owned();
    let y_train = y[0..train_end].to_vec();

    let x_val = x.slice(ndarray::s![train_end..val_end, .., ..]).to_owned();
    let y_val = y[train_end..val_end].to_vec();

    let x_test = x.slice(ndarray::s![val_end.., .., ..]).to_owned();
    let y_test = y[val_end..].to_vec();

    let feature_names = vec![
        "log_returns".to_string(),
        "hl_ratio".to_string(),
        "oc_ratio".to_string(),
        "ma_ratio_5".to_string(),
        "ma_ratio_10".to_string(),
        "ma_ratio_20".to_string(),
        "volatility_5".to_string(),
        "volatility_20".to_string(),
        "rsi_normalized".to_string(),
        "macd".to_string(),
        "macd_signal".to_string(),
        "volume_ratio".to_string(),
        "bollinger_position".to_string(),
    ];

    Some(PreparedDataset {
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        feature_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_prepare_dataset() {
        // Create synthetic OHLCV data
        let n = 100;
        let base_price = 100.0;
        let mut close = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let price = base_price * (1.0 + 0.001 * i as f64);
            close.push(price);
            open.push(price * 0.99);
            high.push(price * 1.01);
            low.push(price * 0.98);
            volume.push(1000000.0);
        }

        let ohlcv = OHLCV {
            timestamps: (0..n as i64).collect(),
            open: Array1::from_vec(open),
            high: Array1::from_vec(high),
            low: Array1::from_vec(low),
            close: Array1::from_vec(close),
            volume: Array1::from_vec(volume),
        };

        let config = DatasetConfig::default();
        let dataset = prepare_dataset(&ohlcv, &config);

        assert!(dataset.is_some());
        let ds = dataset.unwrap();
        assert!(ds.x_train.dim().0 > 0);
    }
}
