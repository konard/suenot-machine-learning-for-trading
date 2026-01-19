//! Backtesting and strategy evaluation for multi-scale attention.
//!
//! This module provides utilities for:
//! - Backtesting trading strategies
//! - Performance metrics calculation
//! - Strategy comparison

use crate::model::Predictions;
use std::collections::HashMap;

/// Configuration for backtesting
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (as fraction, e.g., 0.001 = 0.1%)
    pub transaction_cost: f64,
    /// Confidence threshold for trading
    pub confidence_threshold: f64,
    /// Maximum position size (as fraction of capital)
    pub max_position_size: f64,
    /// Stop loss threshold (as fraction)
    pub stop_loss: Option<f64>,
    /// Take profit threshold (as fraction)
    pub take_profit: Option<f64>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            confidence_threshold: 0.55,
            max_position_size: 1.0,
            stop_loss: None,
            take_profit: None,
        }
    }
}

/// Result of backtesting
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Equity curve over time
    pub equity_curve: Vec<f64>,
    /// Returns for each period
    pub returns: Vec<f64>,
    /// Positions taken (-1, 0, 1)
    pub positions: Vec<i32>,
    /// Trade signals
    pub signals: Vec<i32>,
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Number of trades
    pub n_trades: usize,
    /// Profit factor
    pub profit_factor: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
}

impl BacktestResult {
    /// Generate a performance report string
    pub fn report(&self) -> String {
        format!(
            r#"
================================================================================
                        MULTI-SCALE STRATEGY BACKTEST RESULTS
================================================================================

PERFORMANCE METRICS
------------------
Total Return:        {:>10.2}%
Annualized Return:   {:>10.2}%
Sharpe Ratio:        {:>10.3}
Sortino Ratio:       {:>10.3}
Calmar Ratio:        {:>10.3}
Max Drawdown:        {:>10.2}%

TRADING STATISTICS
------------------
Number of Trades:    {:>10}
Win Rate:            {:>10.2}%
Profit Factor:       {:>10.3}

EQUITY
------
Initial Capital:     {:>10.2}
Final Equity:        {:>10.2}

================================================================================
"#,
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.calmar_ratio,
            self.max_drawdown * 100.0,
            self.n_trades,
            self.win_rate * 100.0,
            self.profit_factor,
            self.equity_curve.first().unwrap_or(&0.0),
            self.equity_curve.last().unwrap_or(&0.0),
        )
    }
}

/// Backtest a multi-scale strategy
///
/// # Arguments
/// * `predictions` - Model predictions
/// * `prices` - Price series for backtesting
/// * `config` - Backtest configuration
///
/// # Returns
/// BacktestResult with performance metrics
pub fn backtest_multi_scale_strategy(
    predictions: &Predictions,
    prices: &[f64],
    config: &BacktestConfig,
) -> BacktestResult {
    let n = predictions.direction.nrows().min(prices.len());

    if n == 0 {
        return BacktestResult {
            equity_curve: vec![config.initial_capital],
            returns: vec![],
            positions: vec![],
            signals: vec![],
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            n_trades: 0,
            profit_factor: 0.0,
            calmar_ratio: 0.0,
        };
    }

    let mut equity = config.initial_capital;
    let mut equity_curve = vec![equity];
    let mut returns = Vec::with_capacity(n);
    let mut positions = Vec::with_capacity(n);
    let mut signals = Vec::with_capacity(n);
    let mut current_position = 0i32;
    let mut n_trades = 0;
    let mut wins = 0;
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;

    for i in 0..n - 1 {
        // Get prediction confidence
        let direction_prob = predictions.direction[[i, 0]];
        let uncertainty = predictions.uncertainty[[i, 0]];

        // Generate signal based on confidence
        let signal = if direction_prob > config.confidence_threshold && uncertainty < 0.5 {
            1 // Long
        } else if direction_prob < (1.0 - config.confidence_threshold) && uncertainty < 0.5 {
            -1 // Short
        } else {
            0 // Neutral
        };

        signals.push(signal);

        // Calculate position change
        let position_change = signal - current_position;

        // Apply transaction costs for position changes
        if position_change != 0 {
            let cost = equity * config.transaction_cost * position_change.abs() as f64;
            equity -= cost;
            n_trades += 1;
        }

        // Calculate return for this period
        let price_return = if i + 1 < prices.len() && prices[i] > 0.0 {
            (prices[i + 1] - prices[i]) / prices[i]
        } else {
            0.0
        };

        // Apply position return
        let position_return = current_position as f64 * price_return * config.max_position_size;
        let period_pnl = equity * position_return;

        // Track wins/losses
        if position_return > 0.0 {
            wins += 1;
            gross_profit += period_pnl;
        } else if position_return < 0.0 {
            gross_loss += period_pnl.abs();
        }

        equity += period_pnl;
        returns.push(position_return);
        positions.push(current_position);
        equity_curve.push(equity);

        current_position = signal;
    }

    // Add final position
    positions.push(current_position);
    signals.push(0);

    // Calculate metrics
    let total_return = (equity - config.initial_capital) / config.initial_capital;

    // Annualized return (assuming daily data, 252 trading days)
    let n_periods = returns.len() as f64;
    let annualized_return = if n_periods > 0.0 {
        ((1.0 + total_return).powf(252.0 / n_periods)) - 1.0
    } else {
        0.0
    };

    // Calculate Sharpe ratio
    let mean_return = if !returns.is_empty() {
        returns.iter().sum::<f64>() / returns.len() as f64
    } else {
        0.0
    };

    let std_return = if returns.len() > 1 {
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    let sharpe_ratio = if std_return > 0.0 {
        (mean_return * 252.0_f64.sqrt()) / std_return
    } else {
        0.0
    };

    // Calculate Sortino ratio (downside deviation)
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    let downside_std = if downside_returns.len() > 1 {
        let variance = downside_returns
            .iter()
            .map(|r| r.powi(2))
            .sum::<f64>()
            / downside_returns.len() as f64;
        variance.sqrt()
    } else {
        0.0
    };

    let sortino_ratio = if downside_std > 0.0 {
        (mean_return * 252.0_f64.sqrt()) / downside_std
    } else {
        0.0
    };

    // Calculate maximum drawdown
    let mut peak = equity_curve[0];
    let mut max_drawdown = 0.0;
    for &equity in &equity_curve {
        if equity > peak {
            peak = equity;
        }
        let drawdown = (peak - equity) / peak;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    // Win rate
    let win_rate = if n_trades > 0 {
        wins as f64 / n_trades as f64
    } else {
        0.0
    };

    // Profit factor
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // Calmar ratio
    let calmar_ratio = if max_drawdown > 0.0 {
        annualized_return / max_drawdown
    } else if annualized_return > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    BacktestResult {
        equity_curve,
        returns,
        positions,
        signals,
        total_return,
        annualized_return,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        win_rate,
        n_trades,
        profit_factor,
        calmar_ratio,
    }
}

/// Compare strategies using different scales
pub fn compare_scale_strategies(
    predictions: &Predictions,
    prices: &[f64],
    config: &BacktestConfig,
) -> HashMap<String, BacktestResult> {
    let mut results = HashMap::new();

    // Full multi-scale strategy
    let multi_scale_result = backtest_multi_scale_strategy(predictions, prices, config);
    results.insert("multi_scale".to_string(), multi_scale_result);

    // Short-term only strategy (using short-term predictions for direction)
    let short_term_predictions = Predictions {
        short_term: predictions.short_term.clone(),
        medium_term: predictions.short_term.clone(),
        long_term: predictions.short_term.clone(),
        direction: predictions.short_term.mapv(|v| if v > 0.0 { 0.7 } else { 0.3 }),
        uncertainty: predictions.uncertainty.clone(),
    };
    let short_term_result = backtest_multi_scale_strategy(&short_term_predictions, prices, config);
    results.insert("short_term_only".to_string(), short_term_result);

    // Long-term only strategy
    let long_term_predictions = Predictions {
        short_term: predictions.long_term.clone(),
        medium_term: predictions.long_term.clone(),
        long_term: predictions.long_term.clone(),
        direction: predictions.long_term.mapv(|v| if v > 0.0 { 0.7 } else { 0.3 }),
        uncertainty: predictions.uncertainty.clone(),
    };
    let long_term_result = backtest_multi_scale_strategy(&long_term_predictions, prices, config);
    results.insert("long_term_only".to_string(), long_term_result);

    results
}

/// Calculate Value at Risk (VaR) at given confidence level
pub fn calculate_var(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted_returns: Vec<f64> = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
    let idx = idx.min(sorted_returns.len() - 1);

    -sorted_returns[idx]
}

/// Calculate Expected Shortfall (CVaR) at given confidence level
pub fn calculate_expected_shortfall(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted_returns: Vec<f64> = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let cutoff_idx = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
    let cutoff_idx = cutoff_idx.max(1);

    let tail: Vec<f64> = sorted_returns.iter().take(cutoff_idx).copied().collect();
    let es: f64 = tail.iter().sum::<f64>() / tail.len() as f64;

    -es
}

/// Generate comparison report for multiple strategies
pub fn generate_comparison_report(results: &HashMap<String, BacktestResult>) -> String {
    let mut report = String::new();
    report.push_str("\n================================================================================\n");
    report.push_str("                     STRATEGY COMPARISON REPORT\n");
    report.push_str("================================================================================\n\n");

    report.push_str(&format!(
        "{:<20} {:>12} {:>12} {:>12} {:>12}\n",
        "Strategy", "Return %", "Sharpe", "Sortino", "Max DD %"
    ));
    report.push_str(&format!("{}\n", "-".repeat(72)));

    for (name, result) in results {
        report.push_str(&format!(
            "{:<20} {:>12.2} {:>12.3} {:>12.3} {:>12.2}\n",
            name,
            result.total_return * 100.0,
            result.sharpe_ratio,
            result.sortino_ratio,
            result.max_drawdown * 100.0
        ));
    }

    report.push_str(&format!("{}\n", "-".repeat(72)));
    report.push_str("\n================================================================================\n");

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_predictions(n: usize) -> Predictions {
        Predictions {
            short_term: Array2::from_elem((n, 1), 0.01),
            medium_term: Array2::from_elem((n, 1), 0.02),
            long_term: Array2::from_elem((n, 1), 0.03),
            direction: Array2::from_elem((n, 1), 0.6),
            uncertainty: Array2::from_elem((n, 1), 0.1),
        }
    }

    #[test]
    fn test_backtest_basic() {
        let predictions = create_test_predictions(100);
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 * (1.0 + 0.001 * i as f64))
            .collect();

        let config = BacktestConfig::default();
        let result = backtest_multi_scale_strategy(&predictions, &prices, &config);

        assert!(!result.equity_curve.is_empty());
        assert_eq!(result.positions.len(), 100);
    }

    #[test]
    fn test_var_calculation() {
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        let var_95 = calculate_var(&returns, 0.95);
        assert!(var_95 > 0.0);
    }

    #[test]
    fn test_expected_shortfall() {
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        let es = calculate_expected_shortfall(&returns, 0.95);
        assert!(es > 0.0);
    }

    #[test]
    fn test_compare_strategies() {
        let predictions = create_test_predictions(50);
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 * (1.0 + 0.001 * i as f64))
            .collect();

        let config = BacktestConfig::default();
        let comparison = compare_scale_strategies(&predictions, &prices, &config);

        assert!(comparison.contains_key("multi_scale"));
        assert!(comparison.contains_key("short_term_only"));
        assert!(comparison.contains_key("long_term_only"));
    }
}
