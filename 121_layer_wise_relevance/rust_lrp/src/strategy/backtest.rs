//! Backtesting engine with LRP insights

use ndarray::Array1;
use crate::model::LRPNetwork;
use crate::data::Dataset;
use super::signals::{Signal, SignalGenerator, calculate_coherence};

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (fraction)
    pub transaction_cost: f64,
    /// Confidence threshold for trading
    pub confidence_threshold: f64,
    /// Maximum position size (fraction of capital)
    pub max_position: f64,
    /// Use LRP coherence for position sizing
    pub use_lrp_sizing: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            transaction_cost: 0.001,
            confidence_threshold: 0.55,
            max_position: 1.0,
            use_lrp_sizing: true,
        }
    }
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResults {
    /// Total return
    pub total_return: f64,
    /// Final capital
    pub final_capital: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Average coherence
    pub avg_coherence: f64,
    /// Number of trades
    pub n_trades: usize,
    /// Feature importance (aggregated)
    pub feature_importance: Vec<(String, f64)>,
}

/// Run backtest with LRP-based position sizing
pub fn backtest(
    model: &mut LRPNetwork,
    dataset: &Dataset,
    config: BacktestConfig,
) -> BacktestResults {
    let mut capital = config.initial_capital;
    let mut position = 0.0;

    let mut returns: Vec<f64> = Vec::new();
    let mut confidences: Vec<f64> = Vec::new();
    let mut coherences: Vec<f64> = Vec::new();
    let mut capital_history = vec![capital];
    let mut n_trades = 0;
    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    let mut feature_relevance_sum = vec![0.0; dataset.feature_names.len()];

    let signal_gen = SignalGenerator::new(
        dataset.feature_names.clone(),
        config.confidence_threshold,
    );

    for sample in &dataset.samples {
        // Flatten input
        let input = sample.x.view()
            .into_shape(dataset.input_dim())
            .unwrap()
            .to_owned();

        // Get prediction and explanation
        let probs = model.predict_proba(&input);
        let pred_class = if probs[1] > probs[0] { 1 } else { 0 };
        let confidence = probs[pred_class].max(probs[1 - pred_class]);

        let relevance = model.explain(&input, Some(pred_class));
        let coherence = calculate_coherence(&relevance);

        // Aggregate feature relevances
        let n_features = dataset.feature_names.len();
        let n_timesteps = relevance.len() / n_features;
        for t in 0..n_timesteps {
            for f in 0..n_features {
                let idx = t * n_features + f;
                if idx < relevance.len() {
                    feature_relevance_sum[f] += relevance[idx].abs();
                }
            }
        }

        // Determine target position
        let target_position = if confidence >= config.confidence_threshold {
            let base = if pred_class == 1 { 1.0 } else { -1.0 };

            if config.use_lrp_sizing {
                let scaling = confidence * (0.5 + 0.5 * coherence);
                base * scaling.min(config.max_position)
            } else {
                base * config.max_position
            }
        } else {
            0.0
        };

        // Calculate trading costs
        let position_change = (target_position - position).abs();
        let costs = position_change * config.transaction_cost * capital;

        if position_change > 0.01 {
            n_trades += 1;
        }

        // Simulate return
        let actual_direction = sample.y;
        let correct = (pred_class == 1) == (actual_direction == 1);
        if confidence >= config.confidence_threshold {
            total_predictions += 1;
            if correct {
                correct_predictions += 1;
            }
        }

        let actual_return = if actual_direction == 1 { 0.01 } else { -0.01 };
        let pnl = position * actual_return * capital - costs;
        capital += pnl;

        position = target_position;

        // Record
        if capital > 0.0 {
            returns.push(pnl / (capital - pnl).max(1.0));
        }
        confidences.push(confidence);
        coherences.push(coherence);
        capital_history.push(capital);
    }

    // Calculate metrics
    let total_return = (capital - config.initial_capital) / config.initial_capital;
    let sharpe = calculate_sharpe(&returns);
    let sortino = calculate_sortino(&returns);
    let max_dd = calculate_max_drawdown(&capital_history);
    let win_rate = if total_predictions > 0 {
        correct_predictions as f64 / total_predictions as f64
    } else {
        0.0
    };
    let profit_factor = calculate_profit_factor(&returns);

    // Normalize feature importance
    let total_rel: f64 = feature_relevance_sum.iter().sum();
    let feature_importance: Vec<(String, f64)> = dataset.feature_names
        .iter()
        .zip(feature_relevance_sum.iter())
        .map(|(name, &rel)| (name.clone(), rel / total_rel.max(1e-9)))
        .collect();

    BacktestResults {
        total_return,
        final_capital: capital,
        sharpe_ratio: sharpe,
        sortino_ratio: sortino,
        max_drawdown: max_dd,
        win_rate,
        profit_factor,
        avg_confidence: confidences.iter().sum::<f64>() / confidences.len().max(1) as f64,
        avg_coherence: coherences.iter().sum::<f64>() / coherences.len().max(1) as f64,
        n_trades,
        feature_importance,
    }
}

/// Calculate annualized Sharpe ratio
fn calculate_sharpe(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();

    if std < 1e-9 {
        return 0.0;
    }

    252.0_f64.sqrt() * mean / std
}

/// Calculate Sortino ratio
fn calculate_sortino(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside: Vec<f64> = returns.iter()
        .filter(|&&r| r < 0.0)
        .copied()
        .collect();

    if downside.is_empty() {
        return f64::INFINITY;
    }

    let downside_std = (downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64).sqrt();

    if downside_std < 1e-9 {
        return 0.0;
    }

    252.0_f64.sqrt() * mean / downside_std
}

/// Calculate maximum drawdown
fn calculate_max_drawdown(capital_history: &[f64]) -> f64 {
    let mut peak = capital_history[0];
    let mut max_dd = 0.0;

    for &capital in capital_history {
        if capital > peak {
            peak = capital;
        }
        let dd = (peak - capital) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Calculate profit factor
fn calculate_profit_factor(returns: &[f64]) -> f64 {
    let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if losses < 1e-9 {
        return f64::INFINITY;
    }

    gains / losses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LRPNetwork, LRPConfig};
    use crate::data::{prepare_data, generate_sample_data};

    #[test]
    fn test_backtest() {
        let config = LRPConfig::default();
        let mut model = LRPNetwork::new(960, vec![64, 32], 2, config);

        let features = generate_sample_data(200);
        let dataset = prepare_data(&features, 60, 1);

        let backtest_config = BacktestConfig::default();
        let results = backtest(&mut model, &dataset, backtest_config);

        assert!(results.final_capital > 0.0);
        assert!(!results.feature_importance.is_empty());
    }
}
