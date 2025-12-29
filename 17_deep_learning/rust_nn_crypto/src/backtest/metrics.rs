//! Backtest Performance Metrics
//!
//! Calculate various performance metrics for trading strategies

use serde::{Deserialize, Serialize};

/// Backtest performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestMetrics {
    /// Total return (percentage)
    pub total_return: f64,
    /// Annualized return (percentage)
    pub annual_return: f64,
    /// Annualized volatility (percentage)
    pub annual_volatility: f64,
    /// Sharpe ratio (assuming risk-free rate of 0)
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside deviation)
    pub sortino_ratio: f64,
    /// Maximum drawdown (percentage)
    pub max_drawdown: f64,
    /// Calmar ratio (annual return / max drawdown)
    pub calmar_ratio: f64,
    /// Win rate (percentage)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Average trade return (percentage)
    pub avg_trade_return: f64,
    /// Average winning trade (percentage)
    pub avg_win: f64,
    /// Average losing trade (percentage)
    pub avg_loss: f64,
    /// Largest winning trade (percentage)
    pub largest_win: f64,
    /// Largest losing trade (percentage)
    pub largest_loss: f64,
    /// Average holding period (in bars/candles)
    pub avg_holding_period: f64,
    /// Final capital
    pub final_capital: f64,
}

impl BacktestMetrics {
    /// Calculate metrics from equity curve and trade returns
    pub fn calculate(
        equity_curve: &[f64],
        trade_returns: &[f64],
        initial_capital: f64,
        periods_per_year: f64,
    ) -> Self {
        let n = equity_curve.len();

        // Total return
        let final_capital = *equity_curve.last().unwrap_or(&initial_capital);
        let total_return = (final_capital - initial_capital) / initial_capital * 100.0;

        // Calculate returns
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Annualized return
        let years = n as f64 / periods_per_year;
        let annual_return = if years > 0.0 {
            ((final_capital / initial_capital).powf(1.0 / years) - 1.0) * 100.0
        } else {
            0.0
        };

        // Volatility
        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len().max(1) as f64;
        let daily_volatility = variance.sqrt();
        let annual_volatility = daily_volatility * periods_per_year.sqrt() * 100.0;

        // Sharpe ratio
        let sharpe_ratio = if annual_volatility > 0.0 {
            annual_return / annual_volatility
        } else {
            0.0
        };

        // Sortino ratio (only downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_deviation = downside_variance.sqrt() * periods_per_year.sqrt() * 100.0;
        let sortino_ratio = if downside_deviation > 0.0 {
            annual_return / downside_deviation
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown = Self::calculate_max_drawdown(equity_curve);

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let total_trades = trade_returns.len();
        let winning_trades = trade_returns.iter().filter(|&&r| r > 0.0).count();
        let losing_trades = trade_returns.iter().filter(|&&r| r < 0.0).count();

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };

        // Profit factor
        let gross_profit: f64 = trade_returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = trade_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Average returns
        let avg_trade_return = if total_trades > 0 {
            trade_returns.iter().sum::<f64>() / total_trades as f64 * 100.0
        } else {
            0.0
        };

        let winning_returns: Vec<f64> = trade_returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let losing_returns: Vec<f64> = trade_returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let avg_win = if !winning_returns.is_empty() {
            winning_returns.iter().sum::<f64>() / winning_returns.len() as f64 * 100.0
        } else {
            0.0
        };

        let avg_loss = if !losing_returns.is_empty() {
            losing_returns.iter().sum::<f64>() / losing_returns.len() as f64 * 100.0
        } else {
            0.0
        };

        let largest_win = trade_returns
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max) * 100.0;

        let largest_loss = trade_returns
            .iter()
            .cloned()
            .fold(0.0_f64, f64::min) * 100.0;

        Self {
            total_return,
            annual_return,
            annual_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
            avg_trade_return,
            avg_win,
            avg_loss,
            largest_win,
            largest_loss,
            avg_holding_period: 0.0, // Would need trade duration data
            final_capital,
        }
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_equity = equity_curve[0];
        let mut max_drawdown = 0.0;

        for &equity in equity_curve {
            if equity > max_equity {
                max_equity = equity;
            }
            let drawdown = (max_equity - equity) / max_equity * 100.0;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Print summary report
    pub fn print_report(&self) {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                    BACKTEST RESULTS                          ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Performance Metrics                                          ║");
        println!("╟──────────────────────────────────────────────────────────────╢");
        println!("║ Total Return:         {:>12.2}%                          ║", self.total_return);
        println!("║ Annual Return:        {:>12.2}%                          ║", self.annual_return);
        println!("║ Annual Volatility:    {:>12.2}%                          ║", self.annual_volatility);
        println!("║ Sharpe Ratio:         {:>12.2}                           ║", self.sharpe_ratio);
        println!("║ Sortino Ratio:        {:>12.2}                           ║", self.sortino_ratio);
        println!("║ Max Drawdown:         {:>12.2}%                          ║", self.max_drawdown);
        println!("║ Calmar Ratio:         {:>12.2}                           ║", self.calmar_ratio);
        println!("╟──────────────────────────────────────────────────────────────╢");
        println!("║ Trade Statistics                                             ║");
        println!("╟──────────────────────────────────────────────────────────────╢");
        println!("║ Total Trades:         {:>12}                            ║", self.total_trades);
        println!("║ Winning Trades:       {:>12}                            ║", self.winning_trades);
        println!("║ Losing Trades:        {:>12}                            ║", self.losing_trades);
        println!("║ Win Rate:             {:>12.2}%                          ║", self.win_rate);
        println!("║ Profit Factor:        {:>12.2}                           ║", self.profit_factor);
        println!("║ Avg Trade Return:     {:>12.2}%                          ║", self.avg_trade_return);
        println!("║ Avg Win:              {:>12.2}%                          ║", self.avg_win);
        println!("║ Avg Loss:             {:>12.2}%                          ║", self.avg_loss);
        println!("║ Largest Win:          {:>12.2}%                          ║", self.largest_win);
        println!("║ Largest Loss:         {:>12.2}%                          ║", self.largest_loss);
        println!("╟──────────────────────────────────────────────────────────────╢");
        println!("║ Final Capital:        {:>12.2}                           ║", self.final_capital);
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 95.0, 100.0, 90.0, 110.0];
        let dd = BacktestMetrics::calculate_max_drawdown(&equity);
        // Max drawdown should be from 110 to 90 = 18.18%
        assert!((dd - 18.18).abs() < 0.1);
    }

    #[test]
    fn test_metrics_calculation() {
        let equity = vec![10000.0, 10100.0, 10200.0, 10150.0, 10300.0];
        let trade_returns = vec![0.01, 0.01, -0.005, 0.015];

        let metrics = BacktestMetrics::calculate(&equity, &trade_returns, 10000.0, 365.0);

        assert!(metrics.total_return > 0.0);
        assert_eq!(metrics.total_trades, 4);
        assert_eq!(metrics.winning_trades, 3);
        assert_eq!(metrics.losing_trades, 1);
    }
}
