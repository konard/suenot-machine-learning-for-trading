//! Evaluation metrics for fine-tuned financial LLMs.
//!
//! This module provides comprehensive evaluation for:
//! - Classification metrics (accuracy, F1, precision, recall)
//! - Trading metrics (Sharpe, Sortino, drawdown)
//! - Signal quality metrics

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metrics for classification evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Per-class precision
    pub precision: HashMap<String, f64>,
    /// Per-class recall
    pub recall: HashMap<String, f64>,
    /// Per-class F1 score
    pub f1: HashMap<String, f64>,
    /// Macro-averaged F1 score
    pub macro_f1: f64,
    /// Weighted F1 score
    pub weighted_f1: f64,
    /// Support (count) per class
    pub support: HashMap<String, usize>,
}

impl SentimentMetrics {
    /// Convert to a simple dictionary representation
    pub fn to_dict(&self) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        result.insert("accuracy".to_string(), self.accuracy);
        result.insert("macro_f1".to_string(), self.macro_f1);
        result.insert("weighted_f1".to_string(), self.weighted_f1);
        result
    }
}

/// Metrics for trading strategy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    /// Total return over the period
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (risk-adjusted return)
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside risk-adjusted return)
    pub sortino_ratio: f64,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Maximum drawdown duration in periods
    pub max_drawdown_duration: usize,
    /// Win rate (fraction of winning trades)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average winning return
    pub avg_win: f64,
    /// Average losing return
    pub avg_loss: f64,
    /// Expected value per trade
    pub expectancy: f64,
    /// Annualized volatility
    pub volatility: f64,
}

impl TradingMetrics {
    /// Convert to a simple dictionary representation
    pub fn to_dict(&self) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        result.insert("total_return".to_string(), self.total_return);
        result.insert("annualized_return".to_string(), self.annualized_return);
        result.insert("sharpe_ratio".to_string(), self.sharpe_ratio);
        result.insert("sortino_ratio".to_string(), self.sortino_ratio);
        result.insert("calmar_ratio".to_string(), self.calmar_ratio);
        result.insert("max_drawdown".to_string(), self.max_drawdown);
        result.insert("max_drawdown_duration".to_string(), self.max_drawdown_duration as f64);
        result.insert("win_rate".to_string(), self.win_rate);
        result.insert("profit_factor".to_string(), self.profit_factor);
        result.insert("avg_win".to_string(), self.avg_win);
        result.insert("avg_loss".to_string(), self.avg_loss);
        result.insert("expectancy".to_string(), self.expectancy);
        result.insert("volatility".to_string(), self.volatility);
        result
    }
}

/// Signal quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQualityMetrics {
    /// Directional accuracy of signals
    pub directional_accuracy: f64,
    /// Total return from following signals
    pub total_signal_return: f64,
    /// Average return per signal
    pub avg_signal_return: f64,
    /// Information coefficient (correlation between signals and returns)
    pub information_coefficient: f64,
    /// Hit rate for buy signals
    pub buy_hit_rate: f64,
    /// Hit rate for sell signals
    pub sell_hit_rate: f64,
    /// Number of signals generated
    pub num_signals: usize,
    /// Number of actual trades (non-hold signals)
    pub num_trades: usize,
    /// Ratio of trades to total signals
    pub trade_ratio: f64,
}

/// Evaluate sentiment classification performance.
///
/// # Arguments
/// * `predictions` - Predicted class labels
/// * `labels` - True class labels
/// * `label_names` - Optional names for each class
///
/// # Returns
/// Comprehensive classification metrics
///
/// # Example
/// ```
/// use llm_finance::evaluate::evaluate_sentiment;
///
/// let predictions = vec![0, 1, 2, 1, 0];
/// let labels = vec![0, 1, 2, 1, 1];
/// let names = vec!["Bearish", "Neutral", "Bullish"];
///
/// let metrics = evaluate_sentiment(&predictions, &labels, Some(&names));
/// println!("Accuracy: {:.4}", metrics.accuracy);
/// ```
pub fn evaluate_sentiment(
    predictions: &[i32],
    labels: &[i32],
    label_names: Option<&[&str]>,
) -> SentimentMetrics {
    let default_names = vec!["Bearish", "Neutral", "Bullish"];
    let names = label_names.unwrap_or(&default_names);

    // Calculate accuracy
    let correct: usize = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| *p == *l)
        .count();
    let accuracy = correct as f64 / predictions.len() as f64;

    // Get unique labels
    let mut unique_labels: Vec<i32> = predictions.iter().chain(labels.iter()).cloned().collect();
    unique_labels.sort();
    unique_labels.dedup();

    let mut precision = HashMap::new();
    let mut recall = HashMap::new();
    let mut f1 = HashMap::new();
    let mut support = HashMap::new();

    for &label in &unique_labels {
        let name = if (label as usize) < names.len() {
            names[label as usize].to_string()
        } else {
            label.to_string()
        };

        // True positives, false positives, false negatives
        let tp: usize = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| **p == label && **l == label)
            .count();
        let fp: usize = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| **p == label && **l != label)
            .count();
        let fn_count: usize = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| **p != label && **l == label)
            .count();

        // Precision
        let prec = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        precision.insert(name.clone(), prec);

        // Recall
        let rec = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        recall.insert(name.clone(), rec);

        // F1
        let f1_score = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };
        f1.insert(name.clone(), f1_score);

        // Support
        let sup: usize = labels.iter().filter(|&&l| l == label).count();
        support.insert(name, sup);
    }

    // Macro F1
    let macro_f1: f64 = f1.values().sum::<f64>() / f1.len() as f64;

    // Weighted F1
    let total_support: usize = support.values().sum();
    let weighted_f1: f64 = f1
        .iter()
        .map(|(name, f1_val)| f1_val * *support.get(name).unwrap_or(&0) as f64)
        .sum::<f64>()
        / total_support as f64;

    SentimentMetrics {
        accuracy,
        precision,
        recall,
        f1,
        macro_f1,
        weighted_f1,
        support,
    }
}

/// Evaluate trading strategy performance.
///
/// # Arguments
/// * `returns` - Vector of period returns
/// * `risk_free_rate` - Annual risk-free rate (default 0.02)
/// * `periods_per_year` - Trading periods per year (default 252 for daily)
///
/// # Returns
/// Comprehensive trading metrics
///
/// # Example
/// ```
/// use llm_finance::evaluate::evaluate_trading;
///
/// let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01];
/// let metrics = evaluate_trading(&returns, 0.02, 252);
/// println!("Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
/// ```
pub fn evaluate_trading(returns: &[f64], risk_free_rate: f64, periods_per_year: usize) -> TradingMetrics {
    if returns.is_empty() {
        return TradingMetrics {
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            max_drawdown_duration: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            expectancy: 0.0,
            volatility: 0.0,
        };
    }

    // Total return
    let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;

    // Mean and std
    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_return = variance.sqrt();

    // Annualized metrics
    let annualized_return = (1.0 + total_return).powf(periods_per_year as f64 / returns.len() as f64) - 1.0;
    let volatility = std_return * (periods_per_year as f64).sqrt();

    // Risk-adjusted returns
    let excess_return = mean_return - risk_free_rate / periods_per_year as f64;

    // Sharpe ratio
    let sharpe_ratio = if volatility > 0.0 {
        (excess_return * periods_per_year as f64) / volatility
    } else {
        0.0
    };

    // Sortino ratio
    let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
    let downside_std = if !negative_returns.is_empty() {
        let neg_mean: f64 = negative_returns.iter().sum::<f64>() / negative_returns.len() as f64;
        let neg_var: f64 = negative_returns.iter().map(|r| (r - neg_mean).powi(2)).sum::<f64>() / negative_returns.len() as f64;
        neg_var.sqrt()
    } else {
        std_return
    };
    let sortino_ratio = if downside_std > 0.0 {
        (excess_return * periods_per_year as f64) / (downside_std * (periods_per_year as f64).sqrt())
    } else {
        0.0
    };

    // Drawdown analysis
    let mut cumulative = vec![1.0];
    for r in returns {
        cumulative.push(cumulative.last().unwrap() * (1.0 + r));
    }

    let mut running_max = vec![cumulative[0]];
    for &c in &cumulative[1..] {
        running_max.push(running_max.last().unwrap().max(c));
    }

    let drawdowns: Vec<f64> = cumulative
        .iter()
        .zip(running_max.iter())
        .map(|(c, m)| (c - m) / m)
        .collect();

    let max_drawdown = drawdowns.iter().map(|d| d.abs()).fold(0.0, f64::max);

    // Max drawdown duration
    let mut max_dd_duration = 0;
    let mut current_duration = 0;
    for d in &drawdowns {
        if *d < 0.0 {
            current_duration += 1;
            max_dd_duration = max_dd_duration.max(current_duration);
        } else {
            current_duration = 0;
        }
    }

    // Calmar ratio
    let calmar_ratio = if max_drawdown > 0.0 {
        annualized_return / max_drawdown
    } else {
        0.0
    };

    // Win/loss stats
    let winners: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).cloned().collect();
    let losers: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

    let win_rate = winners.len() as f64 / returns.len() as f64;
    let avg_win = if !winners.is_empty() {
        winners.iter().sum::<f64>() / winners.len() as f64
    } else {
        0.0
    };
    let avg_loss = if !losers.is_empty() {
        losers.iter().sum::<f64>() / losers.len() as f64
    } else {
        0.0
    };

    // Profit factor
    let gross_profit: f64 = winners.iter().sum();
    let gross_loss: f64 = losers.iter().sum::<f64>().abs();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        f64::INFINITY
    };

    // Expectancy
    let expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss;

    TradingMetrics {
        total_return,
        annualized_return,
        sharpe_ratio,
        sortino_ratio,
        calmar_ratio,
        max_drawdown,
        max_drawdown_duration: max_dd_duration,
        win_rate,
        profit_factor,
        avg_win,
        avg_loss,
        expectancy,
        volatility,
    }
}

/// Evaluate signal quality against actual returns.
///
/// # Arguments
/// * `signals` - Trading signals (-1=sell, 0=hold, 1=buy)
/// * `returns` - Actual returns following signals
///
/// # Returns
/// Signal quality metrics
pub fn evaluate_signal_quality(signals: &[i32], returns: &[f64]) -> SignalQualityMetrics {
    if signals.is_empty() || returns.is_empty() {
        return SignalQualityMetrics {
            directional_accuracy: 0.0,
            total_signal_return: 0.0,
            avg_signal_return: 0.0,
            information_coefficient: 0.0,
            buy_hit_rate: 0.0,
            sell_hit_rate: 0.0,
            num_signals: 0,
            num_trades: 0,
            trade_ratio: 0.0,
        };
    }

    let len = signals.len().min(returns.len());
    let signals = &signals[..len];
    let returns = &returns[..len];

    // Directional accuracy (excluding holds)
    let non_hold: Vec<(i32, f64)> = signals
        .iter()
        .zip(returns.iter())
        .filter(|(s, _)| **s != 0)
        .map(|(s, r)| (*s, *r))
        .collect();

    let directional_accuracy = if !non_hold.is_empty() {
        non_hold
            .iter()
            .filter(|(s, r)| (*s > 0 && *r > 0.0) || (*s < 0 && *r < 0.0))
            .count() as f64
            / non_hold.len() as f64
    } else {
        0.0
    };

    // Signal returns
    let signal_returns: Vec<f64> = signals
        .iter()
        .zip(returns.iter())
        .map(|(s, r)| *s as f64 * r)
        .collect();

    let total_signal_return: f64 = signal_returns.iter().sum();
    let avg_signal_return = if !non_hold.is_empty() {
        non_hold.iter().map(|(s, r)| *s as f64 * r).sum::<f64>() / non_hold.len() as f64
    } else {
        0.0
    };

    // Information coefficient (correlation)
    let signal_mean: f64 = signals.iter().map(|s| *s as f64).sum::<f64>() / len as f64;
    let return_mean: f64 = returns.iter().sum::<f64>() / len as f64;

    let cov: f64 = signals
        .iter()
        .zip(returns.iter())
        .map(|(s, r)| (*s as f64 - signal_mean) * (r - return_mean))
        .sum::<f64>()
        / len as f64;

    let signal_std = (signals.iter().map(|s| (*s as f64 - signal_mean).powi(2)).sum::<f64>() / len as f64).sqrt();
    let return_std = (returns.iter().map(|r| (r - return_mean).powi(2)).sum::<f64>() / len as f64).sqrt();

    let information_coefficient = if signal_std > 0.0 && return_std > 0.0 {
        cov / (signal_std * return_std)
    } else {
        0.0
    };

    // Hit rates by signal type
    let buy_signals: Vec<(i32, f64)> = non_hold.iter().filter(|(s, _)| *s > 0).cloned().collect();
    let sell_signals: Vec<(i32, f64)> = non_hold.iter().filter(|(s, _)| *s < 0).cloned().collect();

    let buy_hit_rate = if !buy_signals.is_empty() {
        buy_signals.iter().filter(|(_, r)| *r > 0.0).count() as f64 / buy_signals.len() as f64
    } else {
        0.0
    };

    let sell_hit_rate = if !sell_signals.is_empty() {
        sell_signals.iter().filter(|(_, r)| *r < 0.0).count() as f64 / sell_signals.len() as f64
    } else {
        0.0
    };

    SignalQualityMetrics {
        directional_accuracy,
        total_signal_return,
        avg_signal_return,
        information_coefficient,
        buy_hit_rate,
        sell_hit_rate,
        num_signals: len,
        num_trades: non_hold.len(),
        trade_ratio: non_hold.len() as f64 / len as f64,
    }
}

/// Generate comprehensive evaluation report.
pub fn generate_evaluation_report(
    sentiment_metrics: &SentimentMetrics,
    trading_metrics: &TradingMetrics,
    signal_quality: &SignalQualityMetrics,
) -> String {
    let mut lines = vec![
        "=".repeat(60),
        "FINE-TUNED LLM EVALUATION REPORT".to_string(),
        "=".repeat(60),
        String::new(),
        "CLASSIFICATION METRICS".to_string(),
        "-".repeat(40),
        format!("Accuracy: {:.4}", sentiment_metrics.accuracy),
        format!("Macro F1: {:.4}", sentiment_metrics.macro_f1),
        format!("Weighted F1: {:.4}", sentiment_metrics.weighted_f1),
        String::new(),
        "Per-Class Performance:".to_string(),
    ];

    for (name, f1_val) in &sentiment_metrics.f1 {
        let prec = sentiment_metrics.precision.get(name).unwrap_or(&0.0);
        let rec = sentiment_metrics.recall.get(name).unwrap_or(&0.0);
        let sup = sentiment_metrics.support.get(name).unwrap_or(&0);
        lines.push(format!(
            "  {}: P={:.3}, R={:.3}, F1={:.3} (n={})",
            name, prec, rec, f1_val, sup
        ));
    }

    lines.extend(vec![
        String::new(),
        "TRADING METRICS".to_string(),
        "-".repeat(40),
        format!("Total Return: {:.2}%", trading_metrics.total_return * 100.0),
        format!("Annualized Return: {:.2}%", trading_metrics.annualized_return * 100.0),
        format!("Sharpe Ratio: {:.2}", trading_metrics.sharpe_ratio),
        format!("Sortino Ratio: {:.2}", trading_metrics.sortino_ratio),
        format!("Calmar Ratio: {:.2}", trading_metrics.calmar_ratio),
        format!("Max Drawdown: {:.2}%", trading_metrics.max_drawdown * 100.0),
        format!("Win Rate: {:.2}%", trading_metrics.win_rate * 100.0),
        format!("Profit Factor: {:.2}", trading_metrics.profit_factor),
        format!("Expectancy: {:.4}", trading_metrics.expectancy),
        String::new(),
        "SIGNAL QUALITY".to_string(),
        "-".repeat(40),
        format!("Directional Accuracy: {:.2}%", signal_quality.directional_accuracy * 100.0),
        format!("Information Coefficient: {:.4}", signal_quality.information_coefficient),
        format!("Buy Hit Rate: {:.2}%", signal_quality.buy_hit_rate * 100.0),
        format!("Sell Hit Rate: {:.2}%", signal_quality.sell_hit_rate * 100.0),
        format!("Trade Ratio: {:.2}%", signal_quality.trade_ratio * 100.0),
        String::new(),
        "=".repeat(60),
    ]);

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_sentiment() {
        let predictions = vec![0, 1, 2, 1, 0, 2, 1, 2, 0, 1];
        let labels = vec![0, 1, 2, 1, 1, 2, 1, 2, 0, 0];

        let metrics = evaluate_sentiment(&predictions, &labels, None);

        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        assert!(metrics.macro_f1 >= 0.0 && metrics.macro_f1 <= 1.0);
    }

    #[test]
    fn test_evaluate_trading() {
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005];

        let metrics = evaluate_trading(&returns, 0.02, 252);

        assert!(!metrics.sharpe_ratio.is_nan());
        assert!(metrics.max_drawdown >= 0.0);
    }

    #[test]
    fn test_evaluate_signal_quality() {
        let signals = vec![1, -1, 1, 0, -1, 1, 0, -1, 1, 1];
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.01, -0.01, 0.03];

        let metrics = evaluate_signal_quality(&signals, &returns);

        assert!(metrics.directional_accuracy >= 0.0 && metrics.directional_accuracy <= 1.0);
        assert_eq!(metrics.num_signals, 10);
    }
}
