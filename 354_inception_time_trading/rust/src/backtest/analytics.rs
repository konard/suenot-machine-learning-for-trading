//! Performance analytics
//!
//! This module provides performance metrics calculation for backtests.

use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;

use crate::strategy::Signal;

/// Individual trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_time: i64,
    pub entry_price: f64,
    pub exit_time: i64,
    pub exit_price: f64,
    pub size: f64,
    pub side: Signal,
    pub pnl: f64,
    pub commission: f64,
}

impl Trade {
    /// Calculate return percentage
    pub fn return_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            match self.side {
                Signal::Buy => (self.exit_price - self.entry_price) / self.entry_price * 100.0,
                Signal::Sell => (self.entry_price - self.exit_price) / self.entry_price * 100.0,
                Signal::Hold => 0.0,
            }
        } else {
            0.0
        }
    }

    /// Check if profitable
    pub fn is_profitable(&self) -> bool {
        self.pnl > 0.0
    }

    /// Get holding period in milliseconds
    pub fn holding_period(&self) -> i64 {
        self.exit_time - self.entry_time
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return percentage
    pub total_return_pct: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Average win
    pub avg_win: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Total trades
    pub total_trades: usize,
    /// Winning trades
    pub winning_trades: usize,
    /// Losing trades
    pub losing_trades: usize,
    /// Maximum consecutive wins
    pub max_consecutive_wins: usize,
    /// Maximum consecutive losses
    pub max_consecutive_losses: usize,
    /// Average trade duration (hours)
    pub avg_trade_duration_hours: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from trades and equity curve
    pub fn from_trades(trades: &[Trade], equity_curve: &[f64], initial_capital: f64) -> Self {
        let total_trades = trades.len();

        if total_trades == 0 {
            return Self::default();
        }

        // Basic metrics
        let final_equity = *equity_curve.last().unwrap_or(&initial_capital);
        let total_return_pct = (final_equity - initial_capital) / initial_capital * 100.0;

        // Win/Loss metrics
        let winning_trades: Vec<_> = trades.iter().filter(|t| t.is_profitable()).collect();
        let losing_trades: Vec<_> = trades.iter().filter(|t| !t.is_profitable()).collect();

        let win_count = winning_trades.len();
        let loss_count = losing_trades.len();
        let win_rate = win_count as f64 / total_trades as f64;

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_win = if win_count > 0 {
            gross_profit / win_count as f64
        } else {
            0.0
        };

        let avg_loss = if loss_count > 0 {
            gross_loss / loss_count as f64
        } else {
            0.0
        };

        // Consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) = Self::consecutive_stats(trades);

        // Average trade duration
        let total_duration: i64 = trades.iter().map(|t| t.holding_period()).sum();
        let avg_trade_duration_hours = if total_trades > 0 {
            (total_duration / total_trades as i64) as f64 / 3600000.0
        } else {
            0.0
        };

        // Calculate returns for Sharpe/Sortino
        let returns: Vec<f64> = if equity_curve.len() > 1 {
            equity_curve
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect()
        } else {
            vec![]
        };

        // Sharpe ratio (assuming risk-free rate of 0)
        let sharpe_ratio = if !returns.is_empty() {
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let std_return = returns.std_dev();
            if std_return > 0.0 {
                mean_return / std_return * (252.0_f64).sqrt() // Annualized
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Sortino ratio (downside deviation only)
        let sortino_ratio = if !returns.is_empty() {
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
            let downside_dev = if !downside_returns.is_empty() {
                (downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                    / downside_returns.len() as f64)
                    .sqrt()
            } else {
                0.0
            };
            if downside_dev > 0.0 {
                mean_return / downside_dev * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown = Self::calculate_max_drawdown(equity_curve);

        // Annualized return (assuming 252 trading days)
        let num_periods = equity_curve.len().max(1) as f64;
        let annualized_return = ((final_equity / initial_capital).powf(252.0 / num_periods) - 1.0) * 100.0;

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / (max_drawdown * 100.0)
        } else if annualized_return > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        Self {
            total_return_pct,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            avg_win,
            avg_loss,
            total_trades,
            winning_trades: win_count,
            losing_trades: loss_count,
            max_consecutive_wins,
            max_consecutive_losses,
            avg_trade_duration_hours,
        }
    }

    /// Calculate maximum drawdown from equity curve
    fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_drawdown = 0.0;
        let mut peak = equity_curve[0];

        for &equity in equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Calculate consecutive wins and losses
    fn consecutive_stats(trades: &[Trade]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.is_profitable() {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }

        (max_wins, max_losses)
    }

    /// Format metrics as string
    pub fn to_string(&self) -> String {
        format!(
            "Total Return: {:.2}%\n\
             Annualized Return: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             Sortino Ratio: {:.2}\n\
             Max Drawdown: {:.2}%\n\
             Calmar Ratio: {:.2}\n\
             Win Rate: {:.1}%\n\
             Profit Factor: {:.2}\n\
             Total Trades: {}\n\
             Winning Trades: {}\n\
             Losing Trades: {}",
            self.total_return_pct,
            self.annualized_return,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.win_rate * 100.0,
            self.profit_factor,
            self.total_trades,
            self.winning_trades,
            self.losing_trades
        )
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return_pct: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            avg_trade_duration_hours: 0.0,
        }
    }
}

/// Complete backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
    pub metrics: PerformanceMetrics,
}

impl BacktestResult {
    /// Print summary
    pub fn print_summary(&self) {
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("                      BACKTEST RESULTS");
        println!("═══════════════════════════════════════════════════════════════\n");
        println!("{}", self.metrics.to_string());
        println!("\n═══════════════════════════════════════════════════════════════\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 95.0, 100.0, 120.0];
        let dd = PerformanceMetrics::calculate_max_drawdown(&equity);
        // Peak was 110, trough was 95, drawdown = (110-95)/110 ≈ 0.136
        assert!((dd - 0.136).abs() < 0.01);
    }

    #[test]
    fn test_consecutive_stats() {
        let trades = vec![
            Trade {
                entry_time: 0, entry_price: 100.0, exit_time: 1, exit_price: 0.0,
                size: 1.0, side: Signal::Buy, pnl: 10.0, commission: 0.0,
            },
            Trade {
                entry_time: 0, entry_price: 100.0, exit_time: 1, exit_price: 0.0,
                size: 1.0, side: Signal::Buy, pnl: 5.0, commission: 0.0,
            },
            Trade {
                entry_time: 0, entry_price: 100.0, exit_time: 1, exit_price: 0.0,
                size: 1.0, side: Signal::Buy, pnl: -3.0, commission: 0.0,
            },
        ];

        let (wins, losses) = PerformanceMetrics::consecutive_stats(&trades);
        assert_eq!(wins, 2);
        assert_eq!(losses, 1);
    }
}
