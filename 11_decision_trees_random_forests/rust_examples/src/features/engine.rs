//! Feature engineering engine

use super::indicators::*;
use crate::data::{Candle, Dataset};

/// Feature types that can be computed
#[derive(Debug, Clone)]
pub enum Feature {
    /// Simple Moving Average
    SMA(usize),
    /// Exponential Moving Average
    EMA(usize),
    /// Relative Strength Index
    RSI(usize),
    /// MACD histogram
    MACDHistogram { fast: usize, slow: usize, signal: usize },
    /// Bollinger Band %B
    BollingerPercentB { period: usize, std_dev: f64 },
    /// Bollinger Band width
    BollingerWidth { period: usize, std_dev: f64 },
    /// Average True Range
    ATR(usize),
    /// Stochastic %K
    StochasticK { k_period: usize, d_period: usize },
    /// Stochastic %D
    StochasticD { k_period: usize, d_period: usize },
    /// Price momentum (ROC)
    Momentum(usize),
    /// Volatility
    Volatility(usize),
    /// Volume ratio (current / SMA)
    VolumeRatio(usize),
    /// Price distance from VWAP
    VWAPDistance,
    /// Candle body ratio
    BodyRatio,
    /// Upper shadow ratio
    UpperShadowRatio,
    /// Lower shadow ratio
    LowerShadowRatio,
    /// Returns over N periods
    Returns(usize),
    /// Hour of day (0-23)
    HourOfDay,
    /// Day of week (0-6)
    DayOfWeek,
}

impl Feature {
    pub fn name(&self) -> String {
        match self {
            Feature::SMA(p) => format!("sma_{}", p),
            Feature::EMA(p) => format!("ema_{}", p),
            Feature::RSI(p) => format!("rsi_{}", p),
            Feature::MACDHistogram { fast, slow, signal } => {
                format!("macd_hist_{}_{}_{}",fast, slow, signal)
            }
            Feature::BollingerPercentB { period, .. } => format!("bb_pct_b_{}", period),
            Feature::BollingerWidth { period, .. } => format!("bb_width_{}", period),
            Feature::ATR(p) => format!("atr_{}", p),
            Feature::StochasticK { k_period, .. } => format!("stoch_k_{}", k_period),
            Feature::StochasticD { k_period, d_period } => {
                format!("stoch_d_{}_{}", k_period, d_period)
            }
            Feature::Momentum(p) => format!("momentum_{}", p),
            Feature::Volatility(p) => format!("volatility_{}", p),
            Feature::VolumeRatio(p) => format!("vol_ratio_{}", p),
            Feature::VWAPDistance => "vwap_distance".to_string(),
            Feature::BodyRatio => "body_ratio".to_string(),
            Feature::UpperShadowRatio => "upper_shadow_ratio".to_string(),
            Feature::LowerShadowRatio => "lower_shadow_ratio".to_string(),
            Feature::Returns(p) => format!("returns_{}", p),
            Feature::HourOfDay => "hour_of_day".to_string(),
            Feature::DayOfWeek => "day_of_week".to_string(),
        }
    }
}

/// Feature engineering engine
pub struct FeatureEngine {
    features: Vec<Feature>,
    target_horizon: usize,
}

impl FeatureEngine {
    /// Create a new feature engine with default features
    pub fn new() -> Self {
        Self {
            features: Self::default_features(),
            target_horizon: 1,
        }
    }

    /// Create engine with custom features
    pub fn with_features(features: Vec<Feature>) -> Self {
        Self {
            features,
            target_horizon: 1,
        }
    }

    /// Set target prediction horizon
    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.target_horizon = horizon;
        self
    }

    /// Default feature set for crypto trading
    pub fn default_features() -> Vec<Feature> {
        vec![
            // Trend indicators
            Feature::SMA(7),
            Feature::SMA(21),
            Feature::SMA(50),
            Feature::EMA(12),
            Feature::EMA(26),
            // Momentum
            Feature::RSI(14),
            Feature::MACDHistogram {
                fast: 12,
                slow: 26,
                signal: 9,
            },
            Feature::Momentum(5),
            Feature::Momentum(10),
            // Volatility
            Feature::BollingerPercentB {
                period: 20,
                std_dev: 2.0,
            },
            Feature::BollingerWidth {
                period: 20,
                std_dev: 2.0,
            },
            Feature::ATR(14),
            Feature::Volatility(14),
            // Stochastic
            Feature::StochasticK {
                k_period: 14,
                d_period: 3,
            },
            Feature::StochasticD {
                k_period: 14,
                d_period: 3,
            },
            // Volume
            Feature::VolumeRatio(20),
            Feature::VWAPDistance,
            // Candle patterns
            Feature::BodyRatio,
            Feature::UpperShadowRatio,
            Feature::LowerShadowRatio,
            // Returns
            Feature::Returns(1),
            Feature::Returns(5),
            Feature::Returns(10),
        ]
    }

    /// Generate features from candles
    pub fn generate(&self, candles: &[Candle]) -> Dataset {
        let n = candles.len();
        let feature_names: Vec<String> = self.features.iter().map(|f| f.name()).collect();

        let mut all_features: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; self.features.len()];

        // Pre-compute common values
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        // Compute each feature
        for (idx, feature) in self.features.iter().enumerate() {
            let values = self.compute_feature(feature, candles, &closes, &volumes);
            all_features[idx] = values;
        }

        // Transpose and create samples
        let mut features: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();
        let mut timestamps: Vec<i64> = Vec::new();

        // Skip initial NaN values and ensure we have target
        let lookback = self.required_lookback();

        for i in lookback..(n - self.target_horizon) {
            let row: Vec<f64> = all_features.iter().map(|f| f[i]).collect();

            // Skip if any feature is NaN
            if row.iter().any(|v| v.is_nan()) {
                continue;
            }

            // Calculate forward return as target
            let future_return = if closes[i] != 0.0 {
                (closes[i + self.target_horizon] - closes[i]) / closes[i]
            } else {
                0.0
            };

            features.push(row);
            labels.push(future_return);
            timestamps.push(candles[i].timestamp);
        }

        Dataset::from_data(features, labels, feature_names, timestamps)
    }

    /// Compute a single feature
    fn compute_feature(
        &self,
        feature: &Feature,
        candles: &[Candle],
        closes: &[f64],
        volumes: &[f64],
    ) -> Vec<f64> {
        match feature {
            Feature::SMA(period) => {
                let sma_vals = sma(closes, *period);
                closes
                    .iter()
                    .zip(sma_vals.iter())
                    .map(|(c, s)| if *s != 0.0 && !s.is_nan() { c / s - 1.0 } else { f64::NAN })
                    .collect()
            }
            Feature::EMA(period) => {
                let ema_vals = ema(closes, *period);
                closes
                    .iter()
                    .zip(ema_vals.iter())
                    .map(|(c, e)| if *e != 0.0 && !e.is_nan() { c / e - 1.0 } else { f64::NAN })
                    .collect()
            }
            Feature::RSI(period) => {
                let rsi_vals = rsi(candles, *period);
                rsi_vals.iter().map(|v| v / 100.0).collect() // Normalize to 0-1
            }
            Feature::MACDHistogram { fast, slow, signal } => {
                let macd_result = macd(candles, *fast, *slow, *signal);
                macd_result.histogram
            }
            Feature::BollingerPercentB { period, std_dev } => {
                let bb = bollinger_bands(candles, *period, *std_dev);
                bb.percent_b
            }
            Feature::BollingerWidth { period, std_dev } => {
                let bb = bollinger_bands(candles, *period, *std_dev);
                bb.bandwidth
            }
            Feature::ATR(period) => {
                let atr_vals = atr(candles, *period);
                // Normalize by price
                atr_vals
                    .iter()
                    .zip(closes.iter())
                    .map(|(a, c)| if *c != 0.0 { a / c } else { f64::NAN })
                    .collect()
            }
            Feature::StochasticK { k_period, d_period } => {
                let stoch = stochastic(candles, *k_period, *d_period);
                stoch.k.iter().map(|v| v / 100.0).collect()
            }
            Feature::StochasticD { k_period, d_period } => {
                let stoch = stochastic(candles, *k_period, *d_period);
                stoch.d.iter().map(|v| v / 100.0).collect()
            }
            Feature::Momentum(period) => momentum(closes, *period),
            Feature::Volatility(period) => volatility(candles, *period),
            Feature::VolumeRatio(period) => {
                let vol_sma = sma(volumes, *period);
                volumes
                    .iter()
                    .zip(vol_sma.iter())
                    .map(|(v, s)| if *s != 0.0 && !s.is_nan() { v / s } else { f64::NAN })
                    .collect()
            }
            Feature::VWAPDistance => {
                let vwap_vals = vwap(candles);
                closes
                    .iter()
                    .zip(vwap_vals.iter())
                    .map(|(c, v)| if *v != 0.0 && !v.is_nan() { c / v - 1.0 } else { f64::NAN })
                    .collect()
            }
            Feature::BodyRatio => candles
                .iter()
                .map(|c| {
                    let range = c.range();
                    if range != 0.0 {
                        c.body() / range
                    } else {
                        0.0
                    }
                })
                .collect(),
            Feature::UpperShadowRatio => candles
                .iter()
                .map(|c| {
                    let range = c.range();
                    if range != 0.0 {
                        c.upper_shadow() / range
                    } else {
                        0.0
                    }
                })
                .collect(),
            Feature::LowerShadowRatio => candles
                .iter()
                .map(|c| {
                    let range = c.range();
                    if range != 0.0 {
                        c.lower_shadow() / range
                    } else {
                        0.0
                    }
                })
                .collect(),
            Feature::Returns(period) => {
                let mut result = vec![f64::NAN; closes.len()];
                for i in *period..closes.len() {
                    if closes[i - period] != 0.0 {
                        result[i] = (closes[i] - closes[i - period]) / closes[i - period];
                    }
                }
                result
            }
            Feature::HourOfDay => candles
                .iter()
                .map(|c| {
                    c.datetime()
                        .map(|dt| dt.format("%H").to_string().parse::<f64>().unwrap_or(0.0) / 24.0)
                        .unwrap_or(0.0)
                })
                .collect(),
            Feature::DayOfWeek => candles
                .iter()
                .map(|c| {
                    c.datetime()
                        .map(|dt| dt.format("%u").to_string().parse::<f64>().unwrap_or(0.0) / 7.0)
                        .unwrap_or(0.0)
                })
                .collect(),
        }
    }

    /// Calculate required lookback period
    fn required_lookback(&self) -> usize {
        self.features
            .iter()
            .map(|f| match f {
                Feature::SMA(p) | Feature::EMA(p) | Feature::RSI(p) => *p,
                Feature::MACDHistogram { slow, signal, .. } => slow + signal,
                Feature::BollingerPercentB { period, .. } => *period,
                Feature::BollingerWidth { period, .. } => *period,
                Feature::ATR(p) | Feature::Volatility(p) => *p,
                Feature::StochasticK { k_period, .. } => *k_period,
                Feature::StochasticD { k_period, d_period } => k_period + d_period,
                Feature::Momentum(p) | Feature::VolumeRatio(p) | Feature::Returns(p) => *p,
                _ => 1,
            })
            .max()
            .unwrap_or(50)
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

    #[test]
    fn test_feature_generation() {
        // Create sample candles
        let candles: Vec<Candle> = (0..100)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                Candle::new(
                    i * 60000,
                    price,
                    price + 1.0,
                    price - 1.0,
                    price + 0.5,
                    1000.0 + i as f64,
                )
            })
            .collect();

        let engine = FeatureEngine::new();
        let dataset = engine.generate(&candles);

        assert!(dataset.n_samples() > 0);
        assert!(dataset.n_features() > 0);
    }
}
