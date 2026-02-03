//! Trading strategy implementation using extracted rules.
//!
//! This module provides:
//! - Rule-based trading strategy
//! - Backtesting framework
//! - Performance metrics calculation

use crate::rules::{Rule, RuleSet};
use serde::{Deserialize, Serialize};

/// A single trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Index in the data series
    pub idx: usize,
    /// Trade type: "BUY" or "SELL"
    pub trade_type: String,
    /// Execution price
    pub price: f64,
    /// Position size
    pub size: f64,
    /// Explanation of why this trade was made
    pub explanation: Vec<String>,
}

/// Results from backtesting a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return (as decimal, e.g., 0.15 for 15%)
    pub total_return: f64,
    /// Annualized Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown (as decimal)
    pub max_drawdown: f64,
    /// Total number of trades
    pub n_trades: usize,
    /// Win rate (fraction of profitable trades)
    pub win_rate: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// List of all trades
    pub trades: Vec<Trade>,
    /// Equity curve (portfolio value over time)
    pub equity_curve: Vec<f64>,
}

/// Rule-based trading strategy.
pub struct RuleBasedStrategy {
    /// Rule set for signal generation
    ruleset: RuleSet,
    /// Threshold for signal generation
    signal_threshold: f64,
}

impl RuleBasedStrategy {
    /// Create a new rule-based strategy.
    pub fn new(rules: Vec<Rule>, feature_names: Vec<String>) -> Self {
        Self {
            ruleset: RuleSet::new(rules, feature_names),
            signal_threshold: 0.5,
        }
    }

    /// Create from an existing RuleSet.
    pub fn from_ruleset(ruleset: RuleSet) -> Self {
        Self {
            ruleset,
            signal_threshold: 0.5,
        }
    }

    /// Set the signal threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.signal_threshold = threshold;
        self
    }

    /// Generate a trading signal for the given features.
    ///
    /// # Returns
    /// * `1` for BUY
    /// * `-1` for SELL
    /// * `0` for HOLD
    pub fn generate_signal(&self, features: &[f64]) -> i32 {
        self.ruleset.predict(features)
    }

    /// Get explanation for a signal.
    pub fn explain_signal(&self, features: &[f64]) -> Vec<String> {
        self.ruleset.explain(features)
    }

    /// Get the underlying ruleset.
    pub fn ruleset(&self) -> &RuleSet {
        &self.ruleset
    }
}

/// Backtest a rule-based strategy.
///
/// # Arguments
/// * `strategy` - The trading strategy to test
/// * `features` - Feature vectors for each time step
/// * `prices` - Closing prices for each time step
/// * `initial_capital` - Starting capital
///
/// # Returns
/// Backtest results including returns, Sharpe ratio, and trade history.
pub fn backtest(
    strategy: &RuleBasedStrategy,
    features: &[Vec<f64>],
    prices: &[f64],
    initial_capital: f64,
) -> BacktestResult {
    assert_eq!(features.len(), prices.len(), "Features and prices must have same length");

    let mut capital = initial_capital;
    let mut position: f64 = 0.0; // Positive for long, negative for short
    let mut returns: Vec<f64> = Vec::new();
    let mut trades: Vec<Trade> = Vec::new();
    let mut equity_curve: Vec<f64> = vec![initial_capital];
    let mut trade_returns: Vec<f64> = Vec::new();
    let mut entry_price: f64 = 0.0;

    for i in 0..features.len().saturating_sub(1) {
        let signal = strategy.generate_signal(&features[i]);
        let price = prices[i];
        let next_price = prices[i + 1];

        // Execute trades based on signal
        match (signal, position.signum() as i32) {
            // BUY signal and not already long
            (1, p) if p <= 0 => {
                // Close short position if any
                if position < 0.0 {
                    let pnl = position * (entry_price - price);
                    capital += pnl;
                    trade_returns.push(pnl / (position.abs() * entry_price));
                }

                // Open long position
                let shares = capital / price;
                position = shares;
                entry_price = price;
                capital = 0.0;

                trades.push(Trade {
                    idx: i,
                    trade_type: "BUY".to_string(),
                    price,
                    size: shares,
                    explanation: strategy.explain_signal(&features[i]),
                });
            }
            // SELL signal and not already short
            (-1, p) if p >= 0 => {
                // Close long position if any
                if position > 0.0 {
                    capital = position * price;
                    let pnl = position * (price - entry_price);
                    trade_returns.push(pnl / (position * entry_price));
                    position = 0.0;
                }

                // Open short position
                let shares = capital / price;
                position = -shares;
                entry_price = price;
                capital = 0.0;

                trades.push(Trade {
                    idx: i,
                    trade_type: "SELL".to_string(),
                    price,
                    size: shares,
                    explanation: strategy.explain_signal(&features[i]),
                });
            }
            _ => {}
        }

        // Calculate period return
        let period_return = if position > 0.0 {
            (next_price - price) / price
        } else if position < 0.0 {
            (price - next_price) / price
        } else {
            0.0
        };

        returns.push(period_return);

        // Update equity curve
        let current_equity = if position > 0.0 {
            position * next_price
        } else if position < 0.0 {
            capital + position * (entry_price - next_price)
        } else {
            capital
        };
        equity_curve.push(current_equity.max(0.0));
    }

    // Close any remaining position
    if position != 0.0 {
        let final_price = *prices.last().unwrap_or(&0.0);
        if position > 0.0 {
            capital = position * final_price;
        } else {
            capital += position * (entry_price - final_price);
        }
    }

    // Calculate metrics
    let total_return = if initial_capital > 0.0 {
        (equity_curve.last().unwrap_or(&initial_capital) / initial_capital) - 1.0
    } else {
        0.0
    };

    let sharpe_ratio = calculate_sharpe_ratio(&returns, 252.0);
    let max_drawdown = calculate_max_drawdown(&equity_curve);

    let win_rate = if trade_returns.is_empty() {
        0.0
    } else {
        trade_returns.iter().filter(|&&r| r > 0.0).count() as f64
            / trade_returns.len() as f64
    };

    let avg_trade_return = if trade_returns.is_empty() {
        0.0
    } else {
        trade_returns.iter().sum::<f64>() / trade_returns.len() as f64
    };

    BacktestResult {
        total_return,
        sharpe_ratio,
        max_drawdown,
        n_trades: trades.len(),
        win_rate,
        avg_trade_return,
        trades,
        equity_curve,
    }
}

/// Calculate annualized Sharpe ratio.
fn calculate_sharpe_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

    let variance: f64 = returns
        .iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    (mean_return / std_dev) * periods_per_year.sqrt()
}

/// Calculate maximum drawdown from equity curve.
fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut max_dd = 0.0;
    let mut peak = equity_curve[0];

    for &equity in equity_curve {
        if equity > peak {
            peak = equity;
        }

        let dd = if peak > 0.0 {
            (peak - equity) / peak
        } else {
            0.0
        };

        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Simple buy-and-hold benchmark strategy.
pub fn buy_and_hold_benchmark(prices: &[f64], initial_capital: f64) -> BacktestResult {
    if prices.is_empty() {
        return BacktestResult {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            n_trades: 0,
            win_rate: 0.0,
            avg_trade_return: 0.0,
            trades: vec![],
            equity_curve: vec![initial_capital],
        };
    }

    let shares = initial_capital / prices[0];
    let equity_curve: Vec<f64> = prices.iter().map(|&p| shares * p).collect();

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let total_return = (equity_curve.last().unwrap() / initial_capital) - 1.0;
    let sharpe_ratio = calculate_sharpe_ratio(&returns, 252.0);
    let max_drawdown = calculate_max_drawdown(&equity_curve);

    BacktestResult {
        total_return,
        sharpe_ratio,
        max_drawdown,
        n_trades: 1,
        win_rate: if total_return > 0.0 { 1.0 } else { 0.0 },
        avg_trade_return: total_return,
        trades: vec![Trade {
            idx: 0,
            trade_type: "BUY".to_string(),
            price: prices[0],
            size: shares,
            explanation: vec!["Buy and hold benchmark".to_string()],
        }],
        equity_curve,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{Condition, Operator};

    #[test]
    fn test_backtest_basic() {
        // Create simple rules
        let rules = vec![
            Rule::new(
                vec![Condition::new(0, Operator::LessOrEqual, 30.0)],
                1, // BUY when RSI <= 30
                0.8,
            ),
            Rule::new(
                vec![Condition::new(0, Operator::GreaterThan, 70.0)],
                -1, // SELL when RSI > 70
                0.8,
            ),
        ];

        let feature_names = vec!["RSI".to_string()];
        let strategy = RuleBasedStrategy::new(rules, feature_names);

        // Simulate: RSI goes low (buy), then high (sell)
        let features = vec![
            vec![25.0], // BUY signal
            vec![50.0], // HOLD
            vec![50.0], // HOLD
            vec![75.0], // SELL signal
            vec![50.0], // HOLD
        ];

        let prices = vec![100.0, 105.0, 110.0, 115.0, 110.0];

        let result = backtest(&strategy, &features, &prices, 10000.0);

        assert!(result.n_trades >= 1);
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_sharpe_ratio() {
        // Constant positive returns should have high Sharpe
        let returns = vec![0.01; 100];
        let sharpe = calculate_sharpe_ratio(&returns, 252.0);
        assert!(sharpe > 10.0); // Very high because no variance

        // Zero returns
        let zero_returns = vec![0.0; 100];
        let zero_sharpe = calculate_sharpe_ratio(&zero_returns, 252.0);
        assert_eq!(zero_sharpe, 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        // Equity goes up then down
        let equity = vec![100.0, 110.0, 120.0, 100.0, 90.0, 95.0];
        let max_dd = calculate_max_drawdown(&equity);

        // Max drawdown is from peak 120 to trough 90 = 25%
        assert!((max_dd - 0.25).abs() < 0.0001);
    }

    #[test]
    fn test_buy_and_hold_benchmark() {
        let prices = vec![100.0, 110.0, 105.0, 120.0];
        let result = buy_and_hold_benchmark(&prices, 10000.0);

        // Total return should be 20%
        assert!((result.total_return - 0.20).abs() < 0.0001);
        assert_eq!(result.n_trades, 1);
    }
}
