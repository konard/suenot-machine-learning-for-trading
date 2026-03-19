//! Backtesting Module
//!
//! This module provides backtesting capabilities for comparing
//! teacher and student (distilled) trading models.

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// A single trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_time: DateTime<Utc>,
    pub exit_time: Option<DateTime<Utc>>,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub direction: TradeDirection,
    pub size: f64,
    pub pnl: Option<f64>,
    pub confidence: f64,
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    Long,
    Short,
}

impl Trade {
    /// Create a new trade
    pub fn new(
        entry_time: DateTime<Utc>,
        entry_price: f64,
        direction: TradeDirection,
        size: f64,
        confidence: f64,
    ) -> Self {
        Self {
            entry_time,
            exit_time: None,
            entry_price,
            exit_price: None,
            direction,
            size,
            pnl: None,
            confidence,
        }
    }

    /// Close the trade and calculate PnL
    pub fn close(&mut self, exit_time: DateTime<Utc>, exit_price: f64) -> f64 {
        self.exit_time = Some(exit_time);
        self.exit_price = Some(exit_price);

        let pnl = match self.direction {
            TradeDirection::Long => (exit_price - self.entry_price) * self.size,
            TradeDirection::Short => (self.entry_price - exit_price) * self.size,
        };

        self.pnl = Some(pnl);
        pnl
    }

    /// Check if trade is open
    pub fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }
}

/// Backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub total_trades: usize,
    pub avg_trade_pnl: f64,
    pub profit_factor: f64,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
}

impl BacktestResult {
    /// Print a summary of results
    pub fn summary(&self) -> String {
        format!(
            r#"
Backtest Results:
  Total Return:    {:.2}%
  Sharpe Ratio:    {:.2}
  Max Drawdown:    {:.2}%
  Win Rate:        {:.2}%
  Total Trades:    {}
  Avg Trade PnL:   ${:.2}
  Profit Factor:   {:.2}
"#,
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.total_trades,
            self.avg_trade_pnl,
            self.profit_factor
        )
    }
}

/// Comparison results between teacher and student
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub teacher_result: BacktestResult,
    pub student_result: BacktestResult,
    pub agreement_rate: f64,
    pub return_correlation: f64,
}

impl ComparisonResult {
    /// Print comparison summary
    pub fn summary(&self) -> String {
        format!(
            r#"
Model Comparison Results:
{}

TEACHER MODEL:
{}

STUDENT MODEL:
{}

COMPARISON METRICS:
  Agreement Rate:     {:.2}%
  Return Correlation: {:.2}
  Return Difference:  {:+.2}%
  Sharpe Difference:  {:+.2}
"#,
            "=".repeat(50),
            self.teacher_result.summary(),
            self.student_result.summary(),
            self.agreement_rate * 100.0,
            self.return_correlation,
            (self.student_result.total_return - self.teacher_result.total_return) * 100.0,
            self.student_result.sharpe_ratio - self.teacher_result.sharpe_ratio
        )
    }
}

/// Backtesting configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub position_size: f64,
    pub transaction_cost: f64,
    pub slippage: f64,
    pub risk_free_rate: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            transaction_cost: 0.001,
            slippage: 0.0005,
            risk_free_rate: 0.02,
        }
    }
}

/// Backtester for distilled models
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with predictions
    pub fn run(
        &self,
        predictions: &Array1<f64>,
        prices: &[f64],
        timestamps: &[DateTime<Utc>],
    ) -> BacktestResult {
        let mut equity = self.config.initial_capital;
        let mut equity_history = vec![equity];
        let mut trades: Vec<Trade> = Vec::new();
        let mut current_trade: Option<Trade> = None;

        for i in 1..predictions.len().min(prices.len()) {
            let signal = predictions[i];
            let price = prices[i];
            let time = timestamps[i];
            let _prev_price = prices[i - 1];

            // Update equity with open position
            if let Some(ref trade) = current_trade {
                let unrealized_pnl = match trade.direction {
                    TradeDirection::Long => (price - trade.entry_price) * trade.size,
                    TradeDirection::Short => (trade.entry_price - price) * trade.size,
                };
                equity_history.push(
                    self.config.initial_capital
                        + trades.iter().filter_map(|t| t.pnl).sum::<f64>()
                        + unrealized_pnl,
                );
            } else {
                equity_history.push(equity);
            }

            // Check if we should close position
            if let Some(ref mut trade) = current_trade {
                let should_close = match trade.direction {
                    TradeDirection::Long => signal < 0.5,
                    TradeDirection::Short => signal > 0.5,
                };

                if should_close {
                    let exit_price = match trade.direction {
                        TradeDirection::Long => price * (1.0 - self.config.slippage),
                        TradeDirection::Short => price * (1.0 + self.config.slippage),
                    };
                    let pnl = trade.close(time, exit_price);
                    equity += pnl - trade.size * exit_price * self.config.transaction_cost;
                    trades.push(trade.clone());
                    current_trade = None;
                }
            }

            // Check if we should open position
            if current_trade.is_none() {
                let should_open_long = signal > 0.6;
                let should_open_short = signal < 0.4;

                if should_open_long || should_open_short {
                    let direction = if should_open_long {
                        TradeDirection::Long
                    } else {
                        TradeDirection::Short
                    };

                    let entry_price = match direction {
                        TradeDirection::Long => price * (1.0 + self.config.slippage),
                        TradeDirection::Short => price * (1.0 - self.config.slippage),
                    };

                    let size = (equity * self.config.position_size) / entry_price;
                    equity -= size * entry_price * self.config.transaction_cost;

                    current_trade = Some(Trade::new(
                        time,
                        entry_price,
                        direction,
                        size,
                        signal.abs(),
                    ));
                }
            }
        }

        // Close any remaining position
        if let Some(ref mut trade) = current_trade {
            let last_idx = prices.len() - 1;
            let exit_price = match trade.direction {
                TradeDirection::Long => prices[last_idx] * (1.0 - self.config.slippage),
                TradeDirection::Short => prices[last_idx] * (1.0 + self.config.slippage),
            };
            let pnl = trade.close(timestamps[last_idx], exit_price);
            equity += pnl;
            trades.push(trade.clone());
        }

        // Calculate metrics
        let total_return = (equity - self.config.initial_capital) / self.config.initial_capital;
        let sharpe_ratio = self.calculate_sharpe(&equity_history);
        let max_drawdown = self.calculate_max_drawdown(&equity_history);
        let win_rate = self.calculate_win_rate(&trades);
        let avg_trade_pnl = if trades.is_empty() {
            0.0
        } else {
            trades.iter().filter_map(|t| t.pnl).sum::<f64>() / trades.len() as f64
        };
        let profit_factor = self.calculate_profit_factor(&trades);

        BacktestResult {
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            total_trades: trades.len(),
            avg_trade_pnl,
            profit_factor,
            trades,
            equity_curve: equity_history,
        }
    }

    /// Compare teacher and student models
    pub fn compare(
        &self,
        teacher_predictions: &Array1<f64>,
        student_predictions: &Array1<f64>,
        prices: &[f64],
        timestamps: &[DateTime<Utc>],
    ) -> ComparisonResult {
        let teacher_result = self.run(teacher_predictions, prices, timestamps);
        let student_result = self.run(student_predictions, prices, timestamps);

        // Calculate agreement rate
        let agreements: usize = teacher_predictions
            .iter()
            .zip(student_predictions.iter())
            .map(|(t, s)| {
                let t_signal = if *t > 0.5 { 1 } else { 0 };
                let s_signal = if *s > 0.5 { 1 } else { 0 };
                if t_signal == s_signal {
                    1
                } else {
                    0
                }
            })
            .sum();

        let agreement_rate = agreements as f64 / teacher_predictions.len() as f64;

        // Calculate return correlation
        let return_correlation =
            self.calculate_correlation(&teacher_result.equity_curve, &student_result.equity_curve);

        ComparisonResult {
            teacher_result,
            student_result,
            agreement_rate,
            return_correlation,
        }
    }

    fn calculate_sharpe(&self, equity: &[f64]) -> f64 {
        if equity.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = equity
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        let daily_rf = self.config.risk_free_rate / 252.0;
        (252.0_f64).sqrt() * (mean_return - daily_rf) / std_dev
    }

    fn calculate_max_drawdown(&self, equity: &[f64]) -> f64 {
        let mut max_value = equity[0];
        let mut max_drawdown: f64 = 0.0;

        for &value in equity {
            max_value = max_value.max(value);
            let drawdown = (max_value - value) / max_value;
            max_drawdown = max_drawdown.max(drawdown);
        }

        max_drawdown
    }

    fn calculate_win_rate(&self, trades: &[Trade]) -> f64 {
        if trades.is_empty() {
            return 0.0;
        }

        let wins = trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).count();
        wins as f64 / trades.len() as f64
    }

    fn calculate_profit_factor(&self, trades: &[Trade]) -> f64 {
        let gross_profit: f64 = trades
            .iter()
            .filter_map(|t| t.pnl)
            .filter(|&pnl| pnl > 0.0)
            .sum();

        let gross_loss: f64 = trades
            .iter()
            .filter_map(|t| t.pnl)
            .filter(|&pnl| pnl < 0.0)
            .map(|pnl| pnl.abs())
            .sum();

        if gross_loss == 0.0 {
            if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            gross_profit / gross_loss
        }
    }

    fn calculate_correlation(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n < 2 {
            return 0.0;
        }

        let mean_a: f64 = a.iter().take(n).sum::<f64>() / n as f64;
        let mean_b: f64 = b.iter().take(n).sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for i in 0..n {
            let diff_a = a[i] - mean_a;
            let diff_b = b[i] - mean_b;
            cov += diff_a * diff_b;
            var_a += diff_a * diff_a;
            var_b += diff_b * diff_b;
        }

        let denom = (var_a * var_b).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            cov / denom
        }
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_trade_creation() {
        let trade = Trade::new(
            Utc::now(),
            100.0,
            TradeDirection::Long,
            10.0,
            0.75,
        );
        assert!(trade.is_open());
    }

    #[test]
    fn test_trade_close() {
        let mut trade = Trade::new(
            Utc::now(),
            100.0,
            TradeDirection::Long,
            10.0,
            0.75,
        );
        let pnl = trade.close(Utc::now(), 110.0);
        assert_eq!(pnl, 100.0); // (110 - 100) * 10
        assert!(!trade.is_open());
    }

    #[test]
    fn test_backtester() {
        let backtester = Backtester::default();

        let predictions = array![0.3, 0.7, 0.8, 0.6, 0.3, 0.2, 0.4, 0.7, 0.9, 0.5];
        let prices: Vec<f64> = vec![100.0, 101.0, 102.0, 101.5, 100.0, 99.0, 100.0, 102.0, 104.0, 103.0];
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc::now() + chrono::Duration::hours(i))
            .collect();

        let result = backtester.run(&predictions, &prices, &timestamps);

        assert!(!result.equity_curve.is_empty());
        println!("{}", result.summary());
    }
}
