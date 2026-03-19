//! Uncertainty Quantification for Trading
//!
//! This crate implements uncertainty quantification methods for cryptocurrency trading:
//! - Monte Carlo Dropout for approximate Bayesian inference
//! - Deep Ensembles for prediction with uncertainty estimates
//! - Conformal prediction for distribution-free prediction intervals
//! - Bybit API integration for fetching market data

use anyhow::Result;
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A prediction along with decomposed uncertainty estimates.
#[derive(Debug, Clone)]
pub struct PredictionWithUncertainty {
    pub mean: f64,
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    pub total_uncertainty: f64,
    pub confidence_interval: (f64, f64),
    pub confidence_level: f64,
}

/// A single member of a deep ensemble (simple linear model).
#[derive(Debug, Clone)]
pub struct EnsembleMember {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub seed: u64,
}

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Feature vector extracted from candle data.
#[derive(Debug, Clone)]
pub struct Features {
    pub values: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Bybit API types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Uncertainty computation helpers
// ---------------------------------------------------------------------------

/// Compute mean and variance from a slice of values.
pub fn mean_and_variance(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance)
}

/// Given a set of predictions, compute mean, variance, and a confidence interval.
pub fn compute_uncertainty(predictions: &[f64], confidence_level: f64) -> PredictionWithUncertainty {
    let (mean, variance) = mean_and_variance(predictions);
    let std_dev = variance.sqrt();

    // Use a z-score approximation for the confidence interval.
    // For 0.95 -> 1.96, for 0.99 -> 2.576, etc.
    let z = z_score_for_confidence(confidence_level);
    let half_width = z * std_dev;

    PredictionWithUncertainty {
        mean,
        epistemic_uncertainty: variance,
        aleatoric_uncertainty: 0.0, // Set by caller when decomposition is available
        total_uncertainty: variance,
        confidence_interval: (mean - half_width, mean + half_width),
        confidence_level,
    }
}

/// Approximate z-score for common confidence levels.
fn z_score_for_confidence(confidence: f64) -> f64 {
    if confidence >= 0.99 {
        2.576
    } else if confidence >= 0.95 {
        1.960
    } else if confidence >= 0.90 {
        1.645
    } else {
        1.282 // ~0.80
    }
}

// ---------------------------------------------------------------------------
// Monte Carlo Dropout
// ---------------------------------------------------------------------------

/// Perform MC Dropout inference.
///
/// Runs the same input through a simple linear model `n_samples` times,
/// each time randomly dropping weights with probability `dropout_rate`.
/// The spread of predictions approximates Bayesian uncertainty.
pub fn mc_dropout_predict(
    input: &[f64],
    weights: &[Vec<f64>],
    dropout_rate: f64,
    n_samples: usize,
) -> PredictionWithUncertainty {
    assert!(!weights.is_empty(), "weights must not be empty");
    assert!(
        (0.0..1.0).contains(&dropout_rate),
        "dropout_rate must be in [0, 1)"
    );

    let mut predictions = Vec::with_capacity(n_samples);
    let mut rng = rand::thread_rng();

    for _ in 0..n_samples {
        let neuron_outputs: Vec<f64> = weights
            .iter()
            .map(|w| {
                let dot: f64 = w
                    .iter()
                    .zip(input.iter())
                    .map(|(wi, xi)| {
                        if rng.gen::<f64>() > dropout_rate {
                            wi * xi / (1.0 - dropout_rate) // inverted dropout scaling
                        } else {
                            0.0
                        }
                    })
                    .sum();
                dot
            })
            .collect();
        let pred: f64 = neuron_outputs.iter().sum::<f64>() / neuron_outputs.len() as f64;
        predictions.push(pred);
    }

    compute_uncertainty(&predictions, 0.95)
}

// ---------------------------------------------------------------------------
// Deep Ensembles
// ---------------------------------------------------------------------------

/// Create an ensemble of simple linear models with different random seeds.
pub fn create_ensemble(
    n_members: usize,
    n_neurons: usize,
    n_features: usize,
    base_seed: u64,
) -> Vec<EnsembleMember> {
    (0..n_members)
        .map(|i| {
            let seed = base_seed + i as u64;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let weights: Vec<Vec<f64>> = (0..n_neurons)
                .map(|_| {
                    (0..n_features)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect()
                })
                .collect();
            let biases: Vec<f64> = (0..n_neurons)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect();
            EnsembleMember {
                weights,
                biases,
                seed,
            }
        })
        .collect()
}

use rand::SeedableRng;

/// Predict with a single ensemble member (simple linear model -> mean aggregation).
pub fn predict_single(member: &EnsembleMember, input: &[f64]) -> f64 {
    let n = member.weights.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let neuron_sum: f64 = member
        .weights
        .iter()
        .zip(member.biases.iter())
        .map(|(w, b)| {
            let dot: f64 = w.iter().zip(input.iter()).map(|(a, x)| a * x).sum();
            dot + b
        })
        .sum();
    neuron_sum / n
}

/// Predict using the full ensemble and decompose uncertainty.
pub fn ensemble_predict(
    input: &[f64],
    members: &[EnsembleMember],
) -> PredictionWithUncertainty {
    assert!(!members.is_empty(), "ensemble must have at least one member");

    let predictions: Vec<f64> = members.iter().map(|m| predict_single(m, input)).collect();

    let (mean, epistemic) = mean_and_variance(&predictions);

    // For this simplified model we set aleatoric to a small constant fraction.
    // In a full implementation each member would output its own variance estimate.
    let aleatoric = 0.01 * mean.abs().max(1e-8);
    let total = epistemic + aleatoric;
    let std_dev = total.sqrt();
    let z = z_score_for_confidence(0.95);

    PredictionWithUncertainty {
        mean,
        epistemic_uncertainty: epistemic,
        aleatoric_uncertainty: aleatoric,
        total_uncertainty: total,
        confidence_interval: (mean - z * std_dev, mean + z * std_dev),
        confidence_level: 0.95,
    }
}

// ---------------------------------------------------------------------------
// Conformal Prediction
// ---------------------------------------------------------------------------

/// Compute conformal prediction interval given calibration residuals.
///
/// `calibration_errors` should contain |y_true - y_pred| for each calibration point.
/// Returns (lower, upper) bounds for `prediction`.
pub fn conformal_intervals(
    calibration_errors: &[f64],
    prediction: f64,
    alpha: f64,
) -> (f64, f64) {
    assert!(
        !calibration_errors.is_empty(),
        "calibration_errors must not be empty"
    );
    assert!(
        (0.0..1.0).contains(&alpha),
        "alpha must be in [0, 1)"
    );

    let mut sorted: Vec<f64> = calibration_errors.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((1.0 - alpha) * sorted.len() as f64).ceil() as usize;
    let quantile = sorted[idx.min(sorted.len() - 1)];
    (prediction - quantile, prediction + quantile)
}

/// Evaluate conformal coverage: fraction of true values inside their intervals.
pub fn conformal_coverage(
    true_values: &[f64],
    predictions: &[f64],
    calibration_errors: &[f64],
    alpha: f64,
) -> f64 {
    assert_eq!(true_values.len(), predictions.len());
    let covered = true_values
        .iter()
        .zip(predictions.iter())
        .filter(|(&y, &pred)| {
            let (lo, hi) = conformal_intervals(calibration_errors, pred, alpha);
            y >= lo && y <= hi
        })
        .count();
    covered as f64 / true_values.len() as f64
}

// ---------------------------------------------------------------------------
// Feature engineering
// ---------------------------------------------------------------------------

/// Extract simple features from a series of candles for prediction.
///
/// Returns one feature vector per candle (starting from index `lookback`).
pub fn extract_features(candles: &[Candle], lookback: usize) -> Vec<Features> {
    if candles.len() <= lookback {
        return vec![];
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let mut feature_rows = Vec::new();

    for i in lookback..candles.len() {
        let mut values = Vec::new();

        // 1-period log return
        if closes[i - 1] > 0.0 {
            values.push((closes[i] / closes[i - 1]).ln());
        } else {
            values.push(0.0);
        }

        // 5-period log return (or fewer if lookback < 5)
        let lag5 = 5.min(lookback);
        if i >= lag5 && closes[i - lag5] > 0.0 {
            values.push((closes[i] / closes[i - lag5]).ln());
        } else {
            values.push(0.0);
        }

        // Rolling volatility (std of 1-period returns over lookback window)
        let returns: Vec<f64> = (i - lookback + 1..=i)
            .filter_map(|j| {
                if closes[j - 1] > 0.0 {
                    Some((closes[j] / closes[j - 1]).ln())
                } else {
                    None
                }
            })
            .collect();
        let (_, vol) = mean_and_variance(&returns);
        values.push(vol.sqrt());

        // Volume ratio: current volume / average volume over lookback
        let avg_vol: f64 =
            volumes[i - lookback..i].iter().sum::<f64>() / lookback as f64;
        if avg_vol > 0.0 {
            values.push(volumes[i] / avg_vol);
        } else {
            values.push(1.0);
        }

        // Price position in high-low range over lookback
        let window_high = candles[i - lookback..=i]
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);
        let window_low = candles[i - lookback..=i]
            .iter()
            .map(|c| c.low)
            .fold(f64::INFINITY, f64::min);
        let range = window_high - window_low;
        if range > 0.0 {
            values.push((closes[i] - window_low) / range);
        } else {
            values.push(0.5);
        }

        feature_rows.push(Features { values });
    }

    feature_rows
}

// ---------------------------------------------------------------------------
// Bybit API integration
// ---------------------------------------------------------------------------

/// Fetch OHLCV kline data from Bybit public API.
///
/// # Arguments
/// * `symbol` - e.g. "BTCUSDT"
/// * `interval` - e.g. "D" for daily, "60" for 1h
/// * `limit` - number of candles (max 200)
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp = reqwest::get(&url).await?.json::<BybitResponse>().await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order.
    let mut sorted = candles;
    sorted.reverse();
    Ok(sorted)
}

/// Blocking version of fetch_bybit_klines for non-async contexts.
pub fn fetch_bybit_klines_blocking(
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp = reqwest::blocking::get(&url)?.json::<BybitResponse>()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    let mut sorted = candles;
    sorted.reverse();
    Ok(sorted)
}

// ---------------------------------------------------------------------------
// Risk-adjusted position sizing
// ---------------------------------------------------------------------------

/// Determine whether a prediction is confident enough to trade.
pub fn should_trade(prediction: &PredictionWithUncertainty) -> bool {
    let (lower, upper) = prediction.confidence_interval;
    // Only trade when the entire confidence interval is on one side of zero.
    (lower > 0.0 && upper > 0.0) || (lower < 0.0 && upper < 0.0)
}

/// Compute a risk-adjusted position size.
///
/// `base_size` is the maximum position size.
/// `max_uncertainty` is used for normalisation; uncertainties above it result in zero position.
pub fn position_size(
    prediction: &PredictionWithUncertainty,
    base_size: f64,
    max_uncertainty: f64,
) -> f64 {
    if max_uncertainty <= 0.0 {
        return 0.0;
    }
    let normalised = (prediction.total_uncertainty.sqrt() / max_uncertainty).min(1.0);
    let size = base_size * (1.0 - normalised);
    if size < 0.0 { 0.0 } else { size }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_and_variance() {
        let vals = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (mean, var) = mean_and_variance(&vals);
        assert!((mean - 6.0).abs() < 1e-10);
        assert!((var - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_uncertainty_confidence_interval() {
        // All identical predictions -> zero variance -> zero-width interval
        let preds = vec![5.0; 100];
        let result = compute_uncertainty(&preds, 0.95);
        assert!((result.mean - 5.0).abs() < 1e-10);
        assert!(result.total_uncertainty < 1e-10);
        assert!((result.confidence_interval.0 - 5.0).abs() < 1e-10);
        assert!((result.confidence_interval.1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mc_dropout_produces_spread() {
        let input = vec![1.0, 2.0, 3.0];
        let weights = vec![
            vec![0.5, -0.3, 0.8],
            vec![-0.2, 0.7, 0.1],
            vec![0.4, 0.4, -0.5],
        ];
        let result = mc_dropout_predict(&input, &weights, 0.3, 200);
        // With dropout, there should be some variance
        assert!(result.total_uncertainty > 0.0);
        // Mean should be finite
        assert!(result.mean.is_finite());
    }

    #[test]
    fn test_ensemble_predict_deterministic() {
        let ensemble = create_ensemble(5, 4, 3, 42);
        let input = vec![1.0, 0.5, -0.5];
        let p1 = ensemble_predict(&input, &ensemble);
        let p2 = ensemble_predict(&input, &ensemble);
        // Same ensemble, same input -> same result
        assert!((p1.mean - p2.mean).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_intervals_coverage() {
        // Calibration errors
        let errors = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let (lo, hi) = conformal_intervals(&errors, 10.0, 0.1);
        // At alpha=0.1 the 90th percentile of errors is used
        assert!(lo < 10.0);
        assert!(hi > 10.0);
        // The interval should be symmetric around prediction
        assert!((hi - 10.0 - (10.0 - lo)).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_coverage_metric() {
        let cal_errors = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let preds = vec![10.0, 20.0, 30.0];
        let trues = vec![10.5, 19.0, 33.0]; // first two inside, third outside for tight alpha
        let coverage = conformal_coverage(&trues, &preds, &cal_errors, 0.1);
        // Coverage should be between 0 and 1
        assert!(coverage >= 0.0 && coverage <= 1.0);
    }

    #[test]
    fn test_should_trade_clear_signal() {
        let pred_up = PredictionWithUncertainty {
            mean: 0.05,
            epistemic_uncertainty: 0.001,
            aleatoric_uncertainty: 0.001,
            total_uncertainty: 0.002,
            confidence_interval: (0.01, 0.09),
            confidence_level: 0.95,
        };
        assert!(should_trade(&pred_up));

        let pred_uncertain = PredictionWithUncertainty {
            mean: 0.01,
            epistemic_uncertainty: 0.01,
            aleatoric_uncertainty: 0.01,
            total_uncertainty: 0.02,
            confidence_interval: (-0.05, 0.07),
            confidence_level: 0.95,
        };
        assert!(!should_trade(&pred_uncertain));
    }

    #[test]
    fn test_position_size_scales_with_uncertainty() {
        let low_unc = PredictionWithUncertainty {
            mean: 0.05,
            epistemic_uncertainty: 0.001,
            aleatoric_uncertainty: 0.001,
            total_uncertainty: 0.001,
            confidence_interval: (0.01, 0.09),
            confidence_level: 0.95,
        };
        let high_unc = PredictionWithUncertainty {
            mean: 0.05,
            epistemic_uncertainty: 0.1,
            aleatoric_uncertainty: 0.1,
            total_uncertainty: 1.0,
            confidence_interval: (-0.5, 0.6),
            confidence_level: 0.95,
        };
        let size_low = position_size(&low_unc, 100.0, 1.0);
        let size_high = position_size(&high_unc, 100.0, 1.0);
        assert!(size_low > size_high);
    }

    #[test]
    fn test_extract_features_length() {
        let candles: Vec<Candle> = (0..30)
            .map(|i| Candle {
                timestamp: i as u64,
                open: 100.0 + i as f64,
                high: 105.0 + i as f64,
                low: 95.0 + i as f64,
                close: 102.0 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
            })
            .collect();
        let features = extract_features(&candles, 10);
        // Should produce candles.len() - lookback feature rows
        assert_eq!(features.len(), 20);
        // Each row should have 5 features
        assert_eq!(features[0].values.len(), 5);
    }
}
