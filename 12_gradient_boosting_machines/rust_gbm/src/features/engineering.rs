//! Feature engineering for machine learning models
//!
//! This module transforms raw market data into features suitable for ML models.

use crate::data::{Candle, Dataset};
use crate::features::technical::*;

/// Feature engineering configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Periods for moving averages
    pub ma_periods: Vec<usize>,
    /// Period for RSI
    pub rsi_period: usize,
    /// MACD parameters (fast, slow, signal)
    pub macd_params: (usize, usize, usize),
    /// Period for Bollinger Bands
    pub bb_period: usize,
    /// Standard deviations for Bollinger Bands
    pub bb_std: f64,
    /// Period for ATR
    pub atr_period: usize,
    /// Period for Stochastic
    pub stoch_k_period: usize,
    /// Period for Stochastic %D
    pub stoch_d_period: usize,
    /// Periods for ROC
    pub roc_periods: Vec<usize>,
    /// Periods for volatility
    pub volatility_periods: Vec<usize>,
    /// Number of lagged returns to include
    pub lag_periods: Vec<usize>,
    /// Forward periods for target (prediction horizon)
    pub target_period: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            ma_periods: vec![5, 10, 20, 50],
            rsi_period: 14,
            macd_params: (12, 26, 9),
            bb_period: 20,
            bb_std: 2.0,
            atr_period: 14,
            stoch_k_period: 14,
            stoch_d_period: 3,
            roc_periods: vec![1, 5, 10, 20],
            volatility_periods: vec![5, 10, 20],
            lag_periods: vec![1, 2, 3, 5, 10],
            target_period: 1,
        }
    }
}

/// Feature engineer that creates ML features from candle data
pub struct FeatureEngineer {
    config: FeatureConfig,
}

impl FeatureEngineer {
    /// Create a new feature engineer with default configuration
    pub fn new() -> Self {
        Self {
            config: FeatureConfig::default(),
        }
    }

    /// Create a new feature engineer with custom configuration
    pub fn with_config(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Get feature names based on configuration
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // Price-based features
        names.push("close".to_string());
        names.push("open_close_ratio".to_string());
        names.push("high_low_ratio".to_string());
        names.push("upper_shadow_ratio".to_string());
        names.push("lower_shadow_ratio".to_string());
        names.push("body_ratio".to_string());

        // Volume features
        names.push("volume".to_string());
        names.push("volume_ma_ratio".to_string());
        names.push("turnover".to_string());

        // Moving averages
        for period in &self.config.ma_periods {
            names.push(format!("sma_{}", period));
            names.push(format!("ema_{}", period));
            names.push(format!("price_sma_{}_ratio", period));
            names.push(format!("price_ema_{}_ratio", period));
        }

        // RSI
        names.push("rsi".to_string());

        // MACD
        names.push("macd".to_string());
        names.push("macd_signal".to_string());
        names.push("macd_histogram".to_string());

        // Bollinger Bands
        names.push("bb_upper".to_string());
        names.push("bb_lower".to_string());
        names.push("bb_bandwidth".to_string());
        names.push("bb_position".to_string());

        // ATR
        names.push("atr".to_string());
        names.push("atr_ratio".to_string());

        // Stochastic
        names.push("stoch_k".to_string());
        names.push("stoch_d".to_string());

        // OBV
        names.push("obv".to_string());
        names.push("obv_change".to_string());

        // ROC
        for period in &self.config.roc_periods {
            names.push(format!("roc_{}", period));
        }

        // Volatility
        for period in &self.config.volatility_periods {
            names.push(format!("volatility_{}", period));
        }

        // Lagged returns
        for period in &self.config.lag_periods {
            names.push(format!("return_lag_{}", period));
        }

        // Time features
        names.push("hour".to_string());
        names.push("day_of_week".to_string());

        names
    }

    /// Build features from candle data
    pub fn build_features(&self, candles: &[Candle]) -> Dataset {
        let n = candles.len();
        if n == 0 {
            return Dataset::new(String::new(), self.feature_names());
        }

        let symbol = candles[0].symbol.clone();
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        // Calculate all indicators
        let mut smas: Vec<Vec<f64>> = Vec::new();
        let mut emas: Vec<Vec<f64>> = Vec::new();

        for period in &self.config.ma_periods {
            smas.push(sma(&closes, *period));
            emas.push(ema(&closes, *period));
        }

        let rsi_values = rsi(&closes, self.config.rsi_period);

        let macd_result = macd(
            &closes,
            self.config.macd_params.0,
            self.config.macd_params.1,
            self.config.macd_params.2,
        );

        let bb = bollinger_bands(&closes, self.config.bb_period, self.config.bb_std);
        let atr_values = atr(candles, self.config.atr_period);
        let stoch = stochastic(candles, self.config.stoch_k_period, self.config.stoch_d_period);
        let obv_values = obv(candles);
        let volume_sma = sma(&volumes, 20);

        let mut rocs: Vec<Vec<f64>> = Vec::new();
        for period in &self.config.roc_periods {
            rocs.push(roc(&closes, *period));
        }

        let mut volatilities: Vec<Vec<f64>> = Vec::new();
        for period in &self.config.volatility_periods {
            volatilities.push(volatility(&closes, *period));
        }

        let rets = returns(&closes);

        // Calculate targets (forward returns)
        let mut targets = vec![f64::NAN; n];
        for i in 0..(n - self.config.target_period) {
            if closes[i] != 0.0 {
                targets[i] = (closes[i + self.config.target_period] - closes[i]) / closes[i] * 100.0;
            }
        }

        // Build dataset
        let feature_names = self.feature_names();
        let mut dataset = Dataset::new(symbol, feature_names.clone());

        // Determine the minimum valid index (where all indicators have values)
        let min_period = self.config.ma_periods.iter().max().unwrap_or(&50).max(&50);
        let start_idx = *min_period;

        for i in start_idx..(n - self.config.target_period) {
            let mut features = Vec::with_capacity(feature_names.len());

            // Price-based features
            features.push(closes[i]);
            features.push(if candles[i].open != 0.0 {
                candles[i].close / candles[i].open
            } else {
                1.0
            });
            features.push(if lows[i] != 0.0 {
                highs[i] / lows[i]
            } else {
                1.0
            });

            // Shadow ratios
            let body_size = candles[i].body_size();
            let total_range = highs[i] - lows[i];
            features.push(if total_range > 0.0 {
                candles[i].upper_shadow() / total_range
            } else {
                0.0
            });
            features.push(if total_range > 0.0 {
                candles[i].lower_shadow() / total_range
            } else {
                0.0
            });
            features.push(if total_range > 0.0 {
                body_size / total_range
            } else {
                0.0
            });

            // Volume features
            features.push(volumes[i]);
            features.push(if !volume_sma[i].is_nan() && volume_sma[i] != 0.0 {
                volumes[i] / volume_sma[i]
            } else {
                1.0
            });
            features.push(candles[i].turnover);

            // Moving averages
            for (j, _period) in self.config.ma_periods.iter().enumerate() {
                let sma_val = smas[j][i];
                let ema_val = emas[j][i];
                features.push(sma_val);
                features.push(ema_val);
                features.push(if !sma_val.is_nan() && sma_val != 0.0 {
                    closes[i] / sma_val
                } else {
                    1.0
                });
                features.push(if !ema_val.is_nan() && ema_val != 0.0 {
                    closes[i] / ema_val
                } else {
                    1.0
                });
            }

            // RSI
            features.push(if rsi_values[i].is_nan() {
                50.0
            } else {
                rsi_values[i]
            });

            // MACD
            features.push(if macd_result.macd_line[i].is_nan() {
                0.0
            } else {
                macd_result.macd_line[i]
            });
            features.push(if macd_result.signal_line[i].is_nan() {
                0.0
            } else {
                macd_result.signal_line[i]
            });
            features.push(if macd_result.histogram[i].is_nan() {
                0.0
            } else {
                macd_result.histogram[i]
            });

            // Bollinger Bands
            features.push(if bb.upper[i].is_nan() { closes[i] } else { bb.upper[i] });
            features.push(if bb.lower[i].is_nan() { closes[i] } else { bb.lower[i] });
            features.push(if bb.bandwidth[i].is_nan() { 0.0 } else { bb.bandwidth[i] });

            // BB position (where price is within bands)
            let bb_position = if !bb.upper[i].is_nan() && !bb.lower[i].is_nan() && bb.upper[i] != bb.lower[i] {
                (closes[i] - bb.lower[i]) / (bb.upper[i] - bb.lower[i])
            } else {
                0.5
            };
            features.push(bb_position);

            // ATR
            features.push(if atr_values[i].is_nan() { 0.0 } else { atr_values[i] });
            features.push(if !atr_values[i].is_nan() && closes[i] != 0.0 {
                atr_values[i] / closes[i] * 100.0
            } else {
                0.0
            });

            // Stochastic
            features.push(if stoch.k[i].is_nan() { 50.0 } else { stoch.k[i] });
            features.push(if stoch.d[i].is_nan() { 50.0 } else { stoch.d[i] });

            // OBV
            features.push(obv_values[i]);
            features.push(if i > 0 && obv_values[i - 1] != 0.0 {
                (obv_values[i] - obv_values[i - 1]) / obv_values[i - 1].abs() * 100.0
            } else {
                0.0
            });

            // ROC
            for roc_vals in &rocs {
                features.push(if roc_vals[i].is_nan() { 0.0 } else { roc_vals[i] });
            }

            // Volatility
            for vol_vals in &volatilities {
                features.push(if vol_vals[i].is_nan() { 0.0 } else { vol_vals[i] });
            }

            // Lagged returns
            for period in &self.config.lag_periods {
                let lag_idx = i.saturating_sub(*period);
                features.push(if !rets[lag_idx].is_nan() {
                    rets[lag_idx]
                } else {
                    0.0
                });
            }

            // Time features
            let hour = candles[i].timestamp.format("%H").to_string().parse::<f64>().unwrap_or(0.0);
            let day_of_week = candles[i].timestamp.format("%u").to_string().parse::<f64>().unwrap_or(0.0);
            features.push(hour);
            features.push(day_of_week);

            // Add sample to dataset
            dataset.add_sample(features, targets[i], candles[i].timestamp);
        }

        dataset
    }

    /// Build features and filter out samples with NaN values
    pub fn build_clean_features(&self, candles: &[Candle]) -> Dataset {
        let dataset = self.build_features(candles);

        let feature_names = dataset.feature_names.clone();
        let symbol = dataset.symbol.clone();
        let mut clean_dataset = Dataset::new(symbol, feature_names);

        for i in 0..dataset.len() {
            let features = &dataset.features[i];
            let target = dataset.targets[i];

            // Skip samples with NaN values
            if target.is_nan() {
                continue;
            }

            let has_nan = features.iter().any(|f| f.is_nan() || f.is_infinite());
            if has_nan {
                continue;
            }

            clean_dataset.add_sample(
                features.clone(),
                target,
                dataset.timestamps[i],
            );
        }

        clean_dataset
    }
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        let mut candles = Vec::new();
        let base_price = 100.0;

        for i in 0..n {
            let price = base_price + (i as f64 * 0.1).sin() * 10.0;
            candles.push(Candle {
                timestamp: Utc::now(),
                symbol: "BTCUSDT".to_string(),
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price + 0.5,
                volume: 1000.0 + i as f64 * 10.0,
                turnover: (price + 0.5) * (1000.0 + i as f64 * 10.0),
            });
        }

        candles
    }

    #[test]
    fn test_feature_engineer() {
        let candles = create_test_candles(200);
        let engineer = FeatureEngineer::new();

        let dataset = engineer.build_clean_features(&candles);

        assert!(!dataset.is_empty());
        assert_eq!(dataset.num_features(), engineer.feature_names().len());
    }
}
