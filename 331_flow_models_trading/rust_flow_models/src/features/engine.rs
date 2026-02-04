//! Feature engineering engine

use crate::api::types::Kline;
use crate::features::indicators::TechnicalIndicators;
use ndarray::{Array1, Array2};

/// Feature engineering engine for market data
pub struct FeatureEngine {
    /// Names of computed features
    pub feature_names: Vec<String>,
}

impl FeatureEngine {
    /// Create new feature engine
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
        }
    }

    /// Compute features from OHLCV data
    pub fn compute_features(&mut self, klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();

        // Extract price and volume arrays
        let close: Array1<f64> = klines.iter().map(|k| k.close).collect();
        let high: Array1<f64> = klines.iter().map(|k| k.high).collect();
        let low: Array1<f64> = klines.iter().map(|k| k.low).collect();
        let volume: Array1<f64> = klines.iter().map(|k| k.volume).collect();

        // Compute features
        let return_1 = TechnicalIndicators::returns(&close, 1);
        let return_5 = TechnicalIndicators::returns(&close, 5);
        let return_10 = TechnicalIndicators::returns(&close, 10);

        let volatility_20 = TechnicalIndicators::rolling_std(&TechnicalIndicators::log_returns(&close, 1), 20);
        let volatility_50 = TechnicalIndicators::rolling_std(&TechnicalIndicators::log_returns(&close, 1), 50);

        let volume_ratio = TechnicalIndicators::volume_ratio(&volume, 20);

        let rsi = TechnicalIndicators::rsi(&close, 14);
        let bb_position = TechnicalIndicators::bollinger_position(&close, 20, 2.0);

        let (_, _, macd_hist) = TechnicalIndicators::macd(&close, 12, 26, 9);

        // Set feature names
        self.feature_names = vec![
            "return_1".to_string(),
            "return_5".to_string(),
            "return_10".to_string(),
            "volatility_20".to_string(),
            "volatility_50".to_string(),
            "volume_ratio".to_string(),
            "rsi".to_string(),
            "bb_position".to_string(),
            "macd_hist".to_string(),
        ];

        // Combine into feature matrix
        let num_features = self.feature_names.len();
        let mut features = Array2::zeros((n, num_features));

        features.column_mut(0).assign(&return_1);
        features.column_mut(1).assign(&return_5);
        features.column_mut(2).assign(&return_10);
        features.column_mut(3).assign(&volatility_20);
        features.column_mut(4).assign(&volatility_50);
        features.column_mut(5).assign(&volume_ratio);
        features.column_mut(6).assign(&rsi);
        features.column_mut(7).assign(&bb_position);
        features.column_mut(8).assign(&macd_hist);

        // Normalize features
        self.normalize(&mut features);

        features
    }

    /// Compute minimal features for flow model
    pub fn compute_minimal_features(&mut self, klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        let close: Array1<f64> = klines.iter().map(|k| k.close).collect();
        let volume: Array1<f64> = klines.iter().map(|k| k.volume).collect();

        let return_1 = TechnicalIndicators::returns(&close, 1);
        let return_5 = TechnicalIndicators::returns(&close, 5);
        let volatility = TechnicalIndicators::rolling_std(&TechnicalIndicators::log_returns(&close, 1), 20);
        let volume_ratio = TechnicalIndicators::volume_ratio(&volume, 20);

        self.feature_names = vec![
            "return_1".to_string(),
            "return_5".to_string(),
            "volatility".to_string(),
            "volume_ratio".to_string(),
        ];

        let mut features = Array2::zeros((n, 4));
        features.column_mut(0).assign(&return_1);
        features.column_mut(1).assign(&return_5);
        features.column_mut(2).assign(&volatility);
        features.column_mut(3).assign(&volume_ratio);

        self.normalize(&mut features);
        features
    }

    /// Normalize features to zero mean and unit variance
    fn normalize(&self, features: &mut Array2<f64>) {
        let n_features = features.ncols();

        for j in 0..n_features {
            let col = features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(0.0);

            let std = if std > 1e-10 { std } else { 1.0 };

            features.column_mut(j).mapv_inplace(|x| (x - mean) / std);
        }
    }

    /// Remove rows with NaN/Inf values
    pub fn clean_features(features: &Array2<f64>) -> Array2<f64> {
        let valid_rows: Vec<usize> = (0..features.nrows())
            .filter(|&i| {
                features.row(i).iter().all(|x| x.is_finite())
            })
            .collect();

        let n_valid = valid_rows.len();
        let n_features = features.ncols();

        let mut clean = Array2::zeros((n_valid, n_features));
        for (new_i, &old_i) in valid_rows.iter().enumerate() {
            clean.row_mut(new_i).assign(&features.row(old_i));
        }

        clean
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.feature_names.len()
    }

    /// Get feature names
    pub fn get_feature_names(&self) -> &[String] {
        &self.feature_names
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

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                start_time: i as i64 * 60000,
                open: 100.0 + (i as f64) * 0.1,
                high: 101.0 + (i as f64) * 0.1,
                low: 99.0 + (i as f64) * 0.1,
                close: 100.5 + (i as f64) * 0.1,
                volume: 1000.0 + (i as f64) * 10.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_compute_features() {
        let klines = create_test_klines(100);
        let mut engine = FeatureEngine::new();
        let features = engine.compute_features(&klines);

        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), 9);
        assert_eq!(engine.feature_names.len(), 9);
    }

    #[test]
    fn test_clean_features() {
        let mut features = Array2::zeros((5, 3));
        features[[2, 1]] = f64::NAN;

        let clean = FeatureEngine::clean_features(&features);
        assert_eq!(clean.nrows(), 4); // One row removed
    }
}
