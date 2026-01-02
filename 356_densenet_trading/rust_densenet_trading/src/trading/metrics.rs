//! Performance Metrics
//!
//! Trading performance measurement and analysis.

use serde::{Deserialize, Serialize};
use super::position::ClosedPosition;

/// Trading performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of trades
    pub num_trades: usize,

    /// Number of winning trades
    pub num_wins: usize,

    /// Number of losing trades
    pub num_losses: usize,

    /// Win rate
    pub win_rate: f64,

    /// Average win
    pub avg_win: f64,

    /// Average loss
    pub avg_loss: f64,

    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,

    /// Average trade P&L
    pub avg_trade: f64,

    /// Expectancy (expected value per trade)
    pub expectancy: f64,

    /// Largest winning trade
    pub largest_win: f64,

    /// Largest losing trade
    pub largest_loss: f64,

    /// Maximum consecutive wins
    pub max_consec_wins: usize,

    /// Maximum consecutive losses
    pub max_consec_losses: usize,

    /// Average holding period (in milliseconds)
    pub avg_holding_period: u64,

    /// Total fees paid
    pub total_fees: f64,

    /// Sharpe ratio (if equity curve available)
    pub sharpe_ratio: Option<f64>,

    /// Sortino ratio
    pub sortino_ratio: Option<f64>,

    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: Option<f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            num_trades: 0,
            num_wins: 0,
            num_losses: 0,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            avg_trade: 0.0,
            expectancy: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            max_consec_wins: 0,
            max_consec_losses: 0,
            avg_holding_period: 0,
            total_fees: 0.0,
            sharpe_ratio: None,
            sortino_ratio: None,
            calmar_ratio: None,
        }
    }
}

impl PerformanceMetrics {
    /// Calculate metrics from closed trades
    pub fn from_trades(trades: &[ClosedPosition], initial_capital: f64) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        let num_trades = trades.len();

        // Separate wins and losses
        let wins: Vec<&ClosedPosition> = trades.iter().filter(|t| t.is_winner()).collect();
        let losses: Vec<&ClosedPosition> = trades.iter().filter(|t| !t.is_winner()).collect();

        let num_wins = wins.len();
        let num_losses = losses.len();

        // Win rate
        let win_rate = num_wins as f64 / num_trades as f64;

        // Average win/loss
        let gross_profit: f64 = wins.iter().map(|t| t.realized_pnl).sum();
        let gross_loss: f64 = losses.iter().map(|t| t.realized_pnl.abs()).sum();

        let avg_win = if num_wins > 0 { gross_profit / num_wins as f64 } else { 0.0 };
        let avg_loss = if num_losses > 0 { gross_loss / num_losses as f64 } else { 0.0 };

        // Profit factor
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };

        // Average trade
        let total_pnl: f64 = trades.iter().map(|t| t.realized_pnl).sum();
        let avg_trade = total_pnl / num_trades as f64;

        // Expectancy
        let expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss;

        // Largest win/loss
        let largest_win = wins.iter().map(|t| t.realized_pnl).fold(0.0, f64::max);
        let largest_loss = losses.iter().map(|t| t.realized_pnl.abs()).fold(0.0, f64::max);

        // Consecutive wins/losses
        let (max_consec_wins, max_consec_losses) = Self::calculate_streaks(trades);

        // Average holding period
        let total_holding: u64 = trades.iter().map(|t| t.holding_period()).sum();
        let avg_holding_period = total_holding / num_trades as u64;

        // Total fees
        let total_fees: f64 = trades.iter().map(|t| t.fees).sum();

        // Calculate ratios from returns
        let returns: Vec<f64> = trades.iter().map(|t| t.realized_pnl_percent / 100.0).collect();
        let (sharpe, sortino) = Self::calculate_ratios(&returns);

        Self {
            num_trades,
            num_wins,
            num_losses,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            avg_trade,
            expectancy,
            largest_win,
            largest_loss,
            max_consec_wins,
            max_consec_losses,
            avg_holding_period,
            total_fees,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            calmar_ratio: None,
        }
    }

    /// Calculate consecutive win/loss streaks
    fn calculate_streaks(trades: &[ClosedPosition]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.is_winner() {
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

    /// Calculate Sharpe and Sortino ratios
    fn calculate_ratios(returns: &[f64]) -> (Option<f64>, Option<f64>) {
        if returns.is_empty() {
            return (None, None);
        }

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;

        // Standard deviation
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Downside deviation
        let downside_variance: f64 = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>() / n;
        let downside_dev = downside_variance.sqrt();

        // Annualization factor (assuming daily trades)
        let annual_factor = (252.0_f64).sqrt();

        let sharpe = if std_dev > 0.0 {
            Some(mean / std_dev * annual_factor)
        } else {
            None
        };

        let sortino = if downside_dev > 0.0 {
            Some(mean / downside_dev * annual_factor)
        } else {
            None
        };

        (sharpe, sortino)
    }

    /// Set Calmar ratio (requires max drawdown)
    pub fn with_calmar(mut self, total_return: f64, max_drawdown: f64) -> Self {
        if max_drawdown > 0.0 {
            self.calmar_ratio = Some(total_return / max_drawdown);
        }
        self
    }

    /// Check if strategy is profitable
    pub fn is_profitable(&self) -> bool {
        self.avg_trade > 0.0
    }

    /// Get edge (expected return per unit of risk)
    pub fn edge(&self) -> f64 {
        if self.avg_loss > 0.0 {
            self.expectancy / self.avg_loss
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Metrics")?;
        writeln!(f, "===================")?;
        writeln!(f, "Trades: {} (W: {}, L: {})", self.num_trades, self.num_wins, self.num_losses)?;
        writeln!(f, "Win Rate: {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Avg Win: ${:.2}", self.avg_win)?;
        writeln!(f, "Avg Loss: ${:.2}", self.avg_loss)?;
        writeln!(f, "Profit Factor: {:.2}", self.profit_factor)?;
        writeln!(f, "Expectancy: ${:.2}", self.expectancy)?;
        writeln!(f, "Largest Win: ${:.2}", self.largest_win)?;
        writeln!(f, "Largest Loss: ${:.2}", self.largest_loss)?;
        writeln!(f, "Max Consec Wins: {}", self.max_consec_wins)?;
        writeln!(f, "Max Consec Losses: {}", self.max_consec_losses)?;
        writeln!(f, "Total Fees: ${:.2}", self.total_fees)?;

        if let Some(sharpe) = self.sharpe_ratio {
            writeln!(f, "Sharpe Ratio: {:.2}", sharpe)?;
        }
        if let Some(sortino) = self.sortino_ratio {
            writeln!(f, "Sortino Ratio: {:.2}", sortino)?;
        }
        if let Some(calmar) = self.calmar_ratio {
            writeln!(f, "Calmar Ratio: {:.2}", calmar)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::position::{Position, ExitReason};

    fn create_test_trades() -> Vec<ClosedPosition> {
        vec![
            ClosedPosition::close(
                Position::long(100.0, 1.0, 0),
                110.0, 1000, 0.001, ExitReason::Signal
            ),
            ClosedPosition::close(
                Position::long(110.0, 1.0, 1000),
                105.0, 2000, 0.001, ExitReason::StopLoss
            ),
            ClosedPosition::close(
                Position::short(105.0, 1.0, 2000),
                100.0, 3000, 0.001, ExitReason::TakeProfit
            ),
        ]
    }

    #[test]
    fn test_metrics_calculation() {
        let trades = create_test_trades();
        let metrics = PerformanceMetrics::from_trades(&trades, 10000.0);

        assert_eq!(metrics.num_trades, 3);
        assert_eq!(metrics.num_wins, 2);
        assert_eq!(metrics.num_losses, 1);
        assert!((metrics.win_rate - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_profit_factor() {
        let trades = create_test_trades();
        let metrics = PerformanceMetrics::from_trades(&trades, 10000.0);

        assert!(metrics.profit_factor > 1.0); // More profit than loss
    }

    #[test]
    fn test_empty_trades() {
        let metrics = PerformanceMetrics::from_trades(&[], 10000.0);
        assert_eq!(metrics.num_trades, 0);
        assert_eq!(metrics.win_rate, 0.0);
    }

    #[test]
    fn test_streaks() {
        let trades = create_test_trades();
        let metrics = PerformanceMetrics::from_trades(&trades, 10000.0);

        assert!(metrics.max_consec_wins >= 1);
        assert!(metrics.max_consec_losses >= 1);
    }
}
