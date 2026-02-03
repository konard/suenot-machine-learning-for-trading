//! Backtesting engine for RDD strategies.

use crate::trading::signals::TradingSignal;
use crate::trading::strategy::RDDStrategy;
use crate::{RDDError, Result};

/// Trade record.
#[derive(Debug, Clone)]
pub struct Trade {
    /// Entry bar index
    pub entry_idx: usize,
    /// Exit bar index
    pub exit_idx: usize,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size (positive = long, negative = short)
    pub size: f64,
    /// Gross return
    pub gross_return: f64,
    /// Net return (after costs)
    pub net_return: f64,
    /// Signal that triggered the trade
    pub signal_strength: f64,
}

impl Trade {
    /// Calculate PnL.
    pub fn pnl(&self) -> f64 {
        if self.size > 0.0 {
            (self.exit_price - self.entry_price) * self.size
        } else {
            (self.entry_price - self.exit_price) * self.size.abs()
        }
    }

    /// Check if profitable.
    pub fn is_winner(&self) -> bool {
        self.net_return > 0.0
    }

    /// Get holding period.
    pub fn holding_period(&self) -> usize {
        self.exit_idx - self.entry_idx
    }
}

/// Backtest results.
#[derive(Debug, Clone)]
pub struct BacktestResults {
    /// Total return
    pub total_return: f64,
    /// Annualized return (assuming 252 trading days)
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Average holding period
    pub avg_holding_period: f64,
    /// List of trades
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<f64>,
}

impl BacktestResults {
    /// Print summary.
    pub fn summary(&self) -> String {
        format!(
            "=== Backtest Results ===\n\
             Total Return: {:.2}%\n\
             Annualized Return: {:.2}%\n\
             Sharpe Ratio: {:.3}\n\
             Sortino Ratio: {:.3}\n\
             Max Drawdown: {:.2}%\n\
             Win Rate: {:.2}%\n\
             Profit Factor: {:.2}\n\
             Number of Trades: {}\n\
             Avg Trade Return: {:.4}%\n\
             Avg Holding Period: {:.1} bars",
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.num_trades,
            self.avg_trade_return * 100.0,
            self.avg_holding_period
        )
    }
}

/// Backtest engine.
#[derive(Debug, Clone)]
pub struct BacktestEngine {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (as fraction)
    pub transaction_cost: f64,
    /// Slippage (as fraction)
    pub slippage: f64,
    /// Maximum position size (as fraction of capital)
    pub max_position_size: f64,
    /// Number of bars per year (for annualization)
    pub bars_per_year: f64,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(initial_capital: f64, transaction_cost: f64) -> Self {
        Self {
            initial_capital,
            transaction_cost,
            slippage: 0.0001,
            max_position_size: 0.1,
            bars_per_year: 252.0 * 24.0, // Hourly bars, 252 trading days
        }
    }

    /// Set slippage.
    pub fn with_slippage(mut self, slippage: f64) -> Self {
        self.slippage = slippage;
        self
    }

    /// Set maximum position size.
    pub fn with_max_position(mut self, max_size: f64) -> Self {
        self.max_position_size = max_size;
        self
    }

    /// Set bars per year for annualization.
    pub fn with_bars_per_year(mut self, bars: f64) -> Self {
        self.bars_per_year = bars;
        self
    }

    /// Run backtest on price data with signals.
    pub fn run(
        &self,
        prices: &[f64],
        signals: &[TradingSignal],
        holding_period: usize,
    ) -> Result<BacktestResults> {
        if prices.len() != signals.len() {
            return Err(RDDError::InvalidParameter(
                "prices and signals must have same length".to_string(),
            ));
        }

        let n = prices.len();
        if n < holding_period + 1 {
            return Err(RDDError::InsufficientData {
                needed: holding_period + 1,
                got: n,
            });
        }

        let mut trades = Vec::new();
        let mut equity = self.initial_capital;
        let mut equity_curve = vec![equity];
        let mut in_position = false;
        let mut position_entry_idx = 0;
        let mut position_size: f64 = 0.0;
        let mut position_entry_price = 0.0;

        for i in 0..n {
            // Check for exit
            if in_position && i >= position_entry_idx + holding_period {
                // Exit position
                let exit_price = prices[i] * (1.0 - self.slippage * position_size.signum());
                let gross_return = if position_size > 0.0 {
                    (exit_price - position_entry_price) / position_entry_price
                } else {
                    (position_entry_price - exit_price) / position_entry_price
                };
                let net_return = gross_return - 2.0 * self.transaction_cost;

                let trade = Trade {
                    entry_idx: position_entry_idx,
                    exit_idx: i,
                    entry_price: position_entry_price,
                    exit_price,
                    size: position_size,
                    gross_return,
                    net_return,
                    signal_strength: signals[position_entry_idx].suggested_size,
                };

                equity *= 1.0 + net_return * position_size.abs();
                trades.push(trade);
                in_position = false;
            }

            // Check for entry
            if !in_position && i + holding_period < n {
                let signal = &signals[i];
                if signal.is_actionable() {
                    let size = (signal.suggested_size * self.max_position_size)
                        .min(self.max_position_size);

                    if size > 0.001 {
                        in_position = true;
                        position_entry_idx = i;
                        position_size = if signal.is_long() { size } else { -size };
                        position_entry_price =
                            prices[i] * (1.0 + self.slippage * position_size.signum());
                    }
                }
            }

            equity_curve.push(equity);
        }

        // Calculate statistics
        let results = self.calculate_statistics(trades, equity_curve);
        Ok(results)
    }

    /// Run backtest using an RDD strategy.
    pub fn run_strategy(
        &self,
        prices: &[f64],
        indicator_values: &[f64],
        strategy: &RDDStrategy,
    ) -> Result<BacktestResults> {
        if !strategy.is_fitted() {
            return Err(RDDError::NotFitted);
        }

        let signals = strategy.generate_signals(indicator_values);
        self.run(prices, &signals, strategy.holding_period)
    }

    /// Calculate backtest statistics.
    fn calculate_statistics(&self, trades: Vec<Trade>, equity_curve: Vec<f64>) -> BacktestResults {
        let num_trades = trades.len();

        if num_trades == 0 {
            return BacktestResults {
                total_return: 0.0,
                annualized_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                num_trades: 0,
                avg_trade_return: 0.0,
                avg_holding_period: 0.0,
                trades,
                equity_curve,
            };
        }

        // Total return
        let initial = equity_curve.first().unwrap_or(&1.0);
        let final_equity = equity_curve.last().unwrap_or(&1.0);
        let total_return = (final_equity - initial) / initial;

        // Annualized return
        let n_bars = equity_curve.len() as f64;
        let years = n_bars / self.bars_per_year;
        let annualized_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Returns for Sharpe/Sortino
        let returns: Vec<f64> = trades.iter().map(|t| t.net_return).collect();
        let mean_return = returns.iter().sum::<f64>() / num_trades as f64;
        let std_return = if num_trades > 1 {
            let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
                / (num_trades - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Sharpe ratio (assuming 0 risk-free rate)
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (self.bars_per_year / trades.len() as f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if !downside_returns.is_empty() {
            let variance: f64 = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (self.bars_per_year / trades.len() as f64).sqrt()
        } else {
            0.0
        };

        // Max drawdown
        let mut max_equity = equity_curve[0];
        let mut max_drawdown: f64 = 0.0;
        for &eq in &equity_curve {
            max_equity = max_equity.max(eq);
            let drawdown = (max_equity - eq) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Win rate
        let winners = trades.iter().filter(|t| t.is_winner()).count();
        let win_rate = winners as f64 / num_trades as f64;

        // Profit factor
        let gross_profit: f64 = trades
            .iter()
            .filter(|t| t.net_return > 0.0)
            .map(|t| t.net_return)
            .sum();
        let gross_loss: f64 = trades
            .iter()
            .filter(|t| t.net_return < 0.0)
            .map(|t| t.net_return.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Average holding period
        let total_holding: usize = trades.iter().map(|t| t.holding_period()).sum();
        let avg_holding_period = total_holding as f64 / num_trades as f64;

        BacktestResults {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades,
            avg_trade_return: mean_return,
            avg_holding_period,
            trades,
            equity_curve,
        }
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(100_000.0, 0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::signals::SignalStrength;

    #[test]
    fn test_backtest_engine() {
        // Generate simple price series
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        // Generate alternating signals
        let signals: Vec<TradingSignal> = prices
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i % 20 == 0 {
                    TradingSignal::long(0.02, SignalStrength::Medium)
                } else {
                    TradingSignal::neutral()
                }
            })
            .collect();

        let engine = BacktestEngine::new(100_000.0, 0.001);
        let results = engine.run(&prices, &signals, 10).unwrap();

        assert!(results.num_trades > 0);
        println!("{}", results.summary());
    }
}
