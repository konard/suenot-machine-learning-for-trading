//! Simple prediction model for surprise events
//!
//! Uses historical patterns to predict future surprises.
//! This is a baseline model - in practice, more sophisticated ML would be used.

use crate::data::types::Candle;
use crate::features::{SurpriseCalculator, SurpriseMetrics};
use serde::{Deserialize, Serialize};

/// Simple predictor based on historical patterns
pub struct SimplePredictor {
    /// Lookback for features
    lookback: usize,
    /// Threshold for significant events
    threshold: f64,
}

impl Default for SimplePredictor {
    fn default() -> Self {
        Self {
            lookback: 20,
            threshold: 2.0,
        }
    }
}

impl SimplePredictor {
    /// Create a new predictor
    pub fn new(lookback: usize, threshold: f64) -> Self {
        Self { lookback, threshold }
    }

    /// Extract features for prediction
    pub fn extract_features(&self, candles: &[Candle]) -> Option<PredictionFeatures> {
        if candles.len() < self.lookback + 1 {
            return None;
        }

        let n = candles.len();
        let recent = &candles[(n - self.lookback - 1)..n];

        // Calculate basic statistics
        let returns: Vec<f64> = recent.iter().map(|c| c.return_pct()).collect();
        let volumes: Vec<f64> = recent.iter().map(|c| c.volume).collect();

        let return_mean = mean(&returns);
        let return_std = std(&returns);
        let volume_mean = mean(&volumes);
        let volume_std = std(&volumes);

        // Recent trends
        let last_3_returns: Vec<f64> = returns.iter().rev().take(3).cloned().collect();
        let return_trend = last_3_returns.iter().sum::<f64>() / 3.0;

        let last_3_volumes: Vec<f64> = volumes.iter().rev().take(3).cloned().collect();
        let volume_trend = if volume_mean > 0.0 {
            last_3_volumes.iter().sum::<f64>() / 3.0 / volume_mean - 1.0
        } else {
            0.0
        };

        // Momentum
        let momentum = if n >= 5 {
            (candles[n - 1].close - candles[n - 5].close) / candles[n - 5].close
        } else {
            0.0
        };

        // Volatility regime
        let volatility = return_std;
        let volatility_percentile = self.calculate_volatility_percentile(candles);

        Some(PredictionFeatures {
            return_mean,
            return_std,
            return_trend,
            volume_mean,
            volume_std,
            volume_trend,
            momentum,
            volatility,
            volatility_percentile,
            last_return: returns.last().copied().unwrap_or(0.0),
            last_volume_zscore: if volume_std > 0.0 {
                (volumes.last().unwrap_or(&0.0) - volume_mean) / volume_std
            } else {
                0.0
            },
        })
    }

    /// Make a prediction based on features
    pub fn predict(&self, features: &PredictionFeatures) -> Prediction {
        // Simple rule-based prediction (baseline)
        // In practice, this would be a trained ML model

        let mut score = 0.0;

        // Trend following
        if features.return_trend > 0.0 && features.momentum > 0.0 {
            score += 0.5;
        } else if features.return_trend < 0.0 && features.momentum < 0.0 {
            score -= 0.5;
        }

        // Volume confirmation
        if features.volume_trend > 0.2 {
            score *= 1.2;
        }

        // Mean reversion in high volatility
        if features.volatility_percentile > 0.8 {
            score *= -0.5; // Expect reversal
        }

        // Strong recent move may continue
        if features.last_return.abs() > 2.0 * features.return_std {
            score += 0.3 * features.last_return.signum();
        }

        // Volume spike often precedes moves
        if features.last_volume_zscore > 2.0 {
            score *= 1.3;
        }

        // Convert to probability-like score
        let probability = 0.5 + score.tanh() * 0.3;

        Prediction {
            direction: if score > 0.0 {
                PredictionDirection::Bullish
            } else if score < 0.0 {
                PredictionDirection::Bearish
            } else {
                PredictionDirection::Neutral
            },
            confidence: score.abs().min(1.0),
            probability,
            features: features.clone(),
        }
    }

    /// Predict using candles directly
    pub fn predict_from_candles(&self, candles: &[Candle]) -> Option<Prediction> {
        self.extract_features(candles).map(|f| self.predict(&f))
    }

    /// Backtest predictions on historical data
    pub fn backtest(&self, candles: &[Candle]) -> Vec<PredictionResult> {
        if candles.len() < self.lookback + 2 {
            return vec![];
        }

        let mut results = Vec::new();

        for i in (self.lookback + 1)..(candles.len() - 1) {
            let historical = &candles[..=i];
            if let Some(prediction) = self.predict_from_candles(historical) {
                let actual_return = candles[i + 1].return_pct();
                let was_correct = match prediction.direction {
                    PredictionDirection::Bullish => actual_return > 0.0,
                    PredictionDirection::Bearish => actual_return < 0.0,
                    PredictionDirection::Neutral => actual_return.abs() < 0.01,
                };

                results.push(PredictionResult {
                    timestamp: candles[i].timestamp,
                    prediction,
                    actual_return,
                    was_correct,
                });
            }
        }

        results
    }

    /// Calculate backtest metrics
    pub fn calculate_metrics(&self, results: &[PredictionResult]) -> BacktestMetrics {
        if results.is_empty() {
            return BacktestMetrics::default();
        }

        let total = results.len();
        let correct = results.iter().filter(|r| r.was_correct).count();
        let accuracy = correct as f64 / total as f64;

        let bullish_results: Vec<_> = results
            .iter()
            .filter(|r| matches!(r.prediction.direction, PredictionDirection::Bullish))
            .collect();
        let bearish_results: Vec<_> = results
            .iter()
            .filter(|r| matches!(r.prediction.direction, PredictionDirection::Bearish))
            .collect();

        let bullish_accuracy = if !bullish_results.is_empty() {
            bullish_results.iter().filter(|r| r.was_correct).count() as f64
                / bullish_results.len() as f64
        } else {
            0.0
        };

        let bearish_accuracy = if !bearish_results.is_empty() {
            bearish_results.iter().filter(|r| r.was_correct).count() as f64
                / bearish_results.len() as f64
        } else {
            0.0
        };

        // Calculate returns if following predictions
        let mut cumulative_return = 1.0;
        for result in results {
            let position = match result.prediction.direction {
                PredictionDirection::Bullish => 1.0,
                PredictionDirection::Bearish => -1.0,
                PredictionDirection::Neutral => 0.0,
            };
            cumulative_return *= 1.0 + position * result.actual_return;
        }

        BacktestMetrics {
            total_predictions: total,
            correct_predictions: correct,
            accuracy,
            bullish_accuracy,
            bearish_accuracy,
            cumulative_return: cumulative_return - 1.0,
        }
    }

    fn calculate_volatility_percentile(&self, candles: &[Candle]) -> f64 {
        if candles.len() < self.lookback * 2 {
            return 0.5;
        }

        let returns: Vec<f64> = candles.iter().map(|c| c.return_pct()).collect();

        // Calculate rolling volatilities
        let mut volatilities = Vec::new();
        for i in self.lookback..returns.len() {
            let slice = &returns[(i - self.lookback)..i];
            volatilities.push(std(slice));
        }

        if volatilities.is_empty() {
            return 0.5;
        }

        let current_vol = *volatilities.last().unwrap();
        let count_below = volatilities.iter().filter(|&&v| v < current_vol).count();
        count_below as f64 / volatilities.len() as f64
    }
}

/// Features used for prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionFeatures {
    pub return_mean: f64,
    pub return_std: f64,
    pub return_trend: f64,
    pub volume_mean: f64,
    pub volume_std: f64,
    pub volume_trend: f64,
    pub momentum: f64,
    pub volatility: f64,
    pub volatility_percentile: f64,
    pub last_return: f64,
    pub last_volume_zscore: f64,
}

/// Prediction direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionDirection {
    Bullish,
    Bearish,
    Neutral,
}

impl std::fmt::Display for PredictionDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictionDirection::Bullish => write!(f, "Bullish"),
            PredictionDirection::Bearish => write!(f, "Bearish"),
            PredictionDirection::Neutral => write!(f, "Neutral"),
        }
    }
}

/// A single prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub direction: PredictionDirection,
    pub confidence: f64,
    pub probability: f64,
    pub features: PredictionFeatures,
}

/// Result of a prediction (with actual outcome)
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub timestamp: u64,
    pub prediction: Prediction,
    pub actual_return: f64,
    pub was_correct: bool,
}

/// Backtest metrics
#[derive(Debug, Clone, Default)]
pub struct BacktestMetrics {
    pub total_predictions: usize,
    pub correct_predictions: usize,
    pub accuracy: f64,
    pub bullish_accuracy: f64,
    pub bearish_accuracy: f64,
    pub cumulative_return: f64,
}

impl std::fmt::Display for BacktestMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Accuracy: {:.2}% ({}/{}) | Bullish: {:.2}% | Bearish: {:.2}% | Return: {:.2}%",
            self.accuracy * 100.0,
            self.correct_predictions,
            self.total_predictions,
            self.bullish_accuracy * 100.0,
            self.bearish_accuracy * 100.0,
            self.cumulative_return * 100.0
        )
    }
}

// Helper functions
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance: f64 = values.iter().map(|x| (x - m).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles() -> Vec<Candle> {
        (0..50)
            .map(|i| {
                let trend = i as f64 * 0.1;
                Candle {
                    timestamp: i * 1000,
                    open: 100.0 + trend,
                    high: 102.0 + trend,
                    low: 98.0 + trend,
                    close: 101.0 + trend,
                    volume: 1000.0,
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_feature_extraction() {
        let predictor = SimplePredictor::new(10, 2.0);
        let candles = make_candles();
        let features = predictor.extract_features(&candles);

        assert!(features.is_some());
        let f = features.unwrap();
        assert!(f.return_mean.is_finite());
        assert!(f.volatility >= 0.0);
    }

    #[test]
    fn test_prediction() {
        let predictor = SimplePredictor::new(10, 2.0);
        let candles = make_candles();
        let prediction = predictor.predict_from_candles(&candles);

        assert!(prediction.is_some());
        let p = prediction.unwrap();
        assert!(p.confidence >= 0.0 && p.confidence <= 1.0);
        assert!(p.probability >= 0.0 && p.probability <= 1.0);
    }

    #[test]
    fn test_backtest() {
        let predictor = SimplePredictor::new(10, 2.0);
        let candles = make_candles();
        let results = predictor.backtest(&candles);

        assert!(!results.is_empty());

        let metrics = predictor.calculate_metrics(&results);
        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
    }
}
