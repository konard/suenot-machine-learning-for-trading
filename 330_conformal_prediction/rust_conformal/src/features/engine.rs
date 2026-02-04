//! Feature engineering engine

use crate::api::types::Kline;
use super::indicators::TechnicalIndicators;
use ndarray::Array2;

/// Feature vector for a single time point
#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub timestamp: i64,
    pub features: Vec<f64>,
    pub target: Option<f64>,
}

/// Feature engineering engine
pub struct FeatureEngine {
    /// Prediction horizon (periods ahead to predict)
    prediction_horizon: usize,
}

impl FeatureEngine {
    /// Create new feature engine
    pub fn new(prediction_horizon: usize) -> Self {
        Self {
            prediction_horizon: prediction_horizon.max(1),
        }
    }

    /// Get feature names
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "returns_1", "returns_5", "returns_10", "returns_20",
            "log_returns_1", "log_returns_5",
            "volatility_10", "volatility_20",
            "rsi_14", "rsi_7",
            "macd_norm", "macd_hist_norm",
            "momentum_10", "momentum_20",
            "sma_ratio_10", "sma_ratio_20",
            "bb_position",
            "stoch_k", "stoch_d",
            "atr_norm",
        ]
    }

    /// Create features from klines
    pub fn create_features(&self, klines: &[Kline]) -> Vec<FeatureRow> {
        if klines.len() < 50 {
            return vec![];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        // Calculate all indicators
        let returns_1 = TechnicalIndicators::returns(&closes, 1);
        let returns_5 = TechnicalIndicators::returns(&closes, 5);
        let returns_10 = TechnicalIndicators::returns(&closes, 10);
        let returns_20 = TechnicalIndicators::returns(&closes, 20);
        let log_returns_1 = TechnicalIndicators::log_returns(&closes, 1);
        let log_returns_5 = TechnicalIndicators::log_returns(&closes, 5);

        let volatility_10 = TechnicalIndicators::rolling_std(&returns_1, 10);
        let volatility_20 = TechnicalIndicators::rolling_std(&returns_1, 20);

        let rsi_14 = TechnicalIndicators::rsi(&closes, 14);
        let rsi_7 = TechnicalIndicators::rsi(&closes, 7);

        let (macd, _signal, macd_hist) = TechnicalIndicators::macd(&closes, 12, 26, 9);

        let momentum_10 = TechnicalIndicators::momentum(&closes, 10);
        let momentum_20 = TechnicalIndicators::momentum(&closes, 20);

        let sma_10 = TechnicalIndicators::sma(&closes, 10);
        let sma_20 = TechnicalIndicators::sma(&closes, 20);

        let (bb_upper, bb_mid, bb_lower) = TechnicalIndicators::bollinger_bands(&closes, 20, 2.0);

        let (stoch_k, stoch_d) = TechnicalIndicators::stochastic(klines, 14);

        let atr = TechnicalIndicators::atr(klines, 14);

        // Target: future returns
        let target_returns = TechnicalIndicators::returns(&closes, self.prediction_horizon);

        // Build feature rows
        let mut rows = Vec::new();

        for i in 0..klines.len() {
            // Check if all features are valid
            let features = vec![
                returns_1[i],
                returns_5[i],
                returns_10[i],
                returns_20[i],
                log_returns_1[i],
                log_returns_5[i],
                volatility_10[i],
                volatility_20[i],
                rsi_14[i] / 100.0, // Normalize to 0-1
                rsi_7[i] / 100.0,
                macd[i] / closes[i].max(1.0), // Normalize
                macd_hist[i] / closes[i].max(1.0),
                momentum_10[i] / closes[i].max(1.0),
                momentum_20[i] / closes[i].max(1.0),
                if sma_10[i].is_nan() { f64::NAN } else { closes[i] / sma_10[i] - 1.0 },
                if sma_20[i].is_nan() { f64::NAN } else { closes[i] / sma_20[i] - 1.0 },
                Self::bb_position(closes[i], bb_lower[i], bb_upper[i]),
                stoch_k[i] / 100.0,
                stoch_d[i] / 100.0,
                atr[i] / closes[i].max(1.0),
            ];

            // Skip if any feature is NaN
            if features.iter().any(|f| f.is_nan()) {
                continue;
            }

            // Target (shifted forward)
            let target = if i + self.prediction_horizon < target_returns.len() {
                let t = target_returns[i + self.prediction_horizon];
                if t.is_nan() { None } else { Some(t) }
            } else {
                None
            };

            rows.push(FeatureRow {
                timestamp: klines[i].timestamp,
                features,
                target,
            });
        }

        rows
    }

    /// Convert feature rows to ndarray matrix
    pub fn to_matrix(&self, rows: &[FeatureRow]) -> (Array2<f64>, Vec<f64>) {
        let n_features = Self::feature_names().len();
        let n_samples = rows.len();

        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Vec::with_capacity(n_samples);

        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.features.iter().enumerate() {
                x[[i, j]] = val;
            }
            y.push(row.target.unwrap_or(0.0));
        }

        (x, y)
    }

    /// Split data into train/calibration/test sets
    pub fn split_data(
        &self,
        rows: &[FeatureRow],
        train_ratio: f64,
        cal_ratio: f64,
    ) -> (Vec<FeatureRow>, Vec<FeatureRow>, Vec<FeatureRow>) {
        let n = rows.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let cal_end = (n as f64 * (train_ratio + cal_ratio)) as usize;

        let train = rows[..train_end].to_vec();
        let cal = rows[train_end..cal_end].to_vec();
        let test = rows[cal_end..].to_vec();

        (train, cal, test)
    }

    /// Calculate Bollinger Band position
    fn bb_position(price: f64, lower: f64, upper: f64) -> f64 {
        if lower.is_nan() || upper.is_nan() || (upper - lower).abs() < 1e-10 {
            return f64::NAN;
        }
        (price - lower) / (upper - lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let price = 100.0 + (i as f64).sin() * 5.0;
                Kline::new(
                    i as i64 * 3600000,
                    price,
                    price + 1.0,
                    price - 1.0,
                    price + 0.5,
                    1000.0,
                    100000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_create_features() {
        let klines = create_test_klines(100);
        let engine = FeatureEngine::new(5);
        let rows = engine.create_features(&klines);

        assert!(!rows.is_empty());
        assert_eq!(rows[0].features.len(), FeatureEngine::feature_names().len());
    }

    #[test]
    fn test_split_data() {
        let klines = create_test_klines(100);
        let engine = FeatureEngine::new(5);
        let rows = engine.create_features(&klines);

        let (train, cal, test) = engine.split_data(&rows, 0.6, 0.2);

        assert!(train.len() > cal.len());
        assert!(cal.len() > 0);
        assert!(test.len() > 0);
    }
}
