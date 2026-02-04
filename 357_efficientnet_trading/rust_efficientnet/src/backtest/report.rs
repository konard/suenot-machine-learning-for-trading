//! Backtest reporting

use chrono::{DateTime, Utc};

use super::engine::Trade;

/// Backtest report with all metrics
#[derive(Debug, Clone)]
pub struct BacktestReport {
    pub total_return_pct: f64,
    pub annual_return_pct: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown_pct: f64,
    pub win_rate_pct: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub avg_trade_pnl: f64,
    pub avg_holding_hours: f64,
    pub final_equity: f64,
    pub initial_capital: f64,
}

impl BacktestReport {
    /// Create report from backtest data
    pub fn from_backtest(
        trades: &[Trade],
        equity_curve: &[f64],
        timestamps: &[DateTime<Utc>],
        initial_capital: f64,
    ) -> Self {
        if equity_curve.is_empty() {
            return Self::empty(initial_capital);
        }

        let final_equity = *equity_curve.last().unwrap_or(&initial_capital);
        let total_return_pct = (final_equity / initial_capital - 1.0) * 100.0;

        // Calculate returns
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Annualized return (assuming hourly data)
        let n_periods = equity_curve.len();
        let periods_per_year = 365.0 * 24.0;
        let annual_return_pct = if n_periods > 1 {
            ((final_equity / initial_capital).powf(periods_per_year / n_periods as f64) - 1.0) * 100.0
        } else {
            0.0
        };

        // Sharpe ratio
        let excess_returns: Vec<f64> = returns
            .iter()
            .map(|r| r - 0.02 / periods_per_year)
            .collect();
        let sharpe_ratio = calculate_sharpe(&excess_returns, periods_per_year);

        // Sortino ratio
        let sortino_ratio = calculate_sortino(&returns, periods_per_year);

        // Max drawdown
        let max_drawdown_pct = calculate_max_drawdown(equity_curve) * 100.0;

        // Trade statistics
        let completed_trades: Vec<&Trade> = trades
            .iter()
            .filter(|t| t.pnl.is_some())
            .collect();

        let total_trades = completed_trades.len();
        let winning_trades = completed_trades
            .iter()
            .filter(|t| t.pnl.unwrap_or(0.0) > 0.0)
            .count();

        let win_rate_pct = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };

        let total_profits: f64 = completed_trades
            .iter()
            .map(|t| t.pnl.unwrap_or(0.0))
            .filter(|&p| p > 0.0)
            .sum();

        let total_losses: f64 = completed_trades
            .iter()
            .map(|t| t.pnl.unwrap_or(0.0).abs())
            .filter(|&p| completed_trades.iter().any(|t| t.pnl.unwrap_or(0.0) < 0.0))
            .sum();

        let profit_factor = if total_losses > 0.0 {
            total_profits / total_losses
        } else {
            f64::INFINITY
        };

        let avg_trade_pnl = if total_trades > 0 {
            completed_trades
                .iter()
                .map(|t| t.pnl.unwrap_or(0.0))
                .sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        // Average holding period
        let holding_hours: Vec<f64> = completed_trades
            .iter()
            .filter_map(|t| {
                if let (Some(exit), entry) = (t.exit_time, t.entry_time) {
                    Some((exit - entry).num_seconds() as f64 / 3600.0)
                } else {
                    None
                }
            })
            .collect();

        let avg_holding_hours = if !holding_hours.is_empty() {
            holding_hours.iter().sum::<f64>() / holding_hours.len() as f64
        } else {
            0.0
        };

        Self {
            total_return_pct,
            annual_return_pct,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown_pct,
            win_rate_pct,
            profit_factor,
            total_trades,
            avg_trade_pnl,
            avg_holding_hours,
            final_equity,
            initial_capital,
        }
    }

    /// Create an empty report
    pub fn empty(initial_capital: f64) -> Self {
        Self {
            total_return_pct: 0.0,
            annual_return_pct: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown_pct: 0.0,
            win_rate_pct: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            avg_trade_pnl: 0.0,
            avg_holding_hours: 0.0,
            final_equity: initial_capital,
            initial_capital,
        }
    }

    /// Print a formatted summary
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(50));
        println!("EFFICIENTNET BACKTEST RESULTS");
        println!("{}", "=".repeat(50));
        println!("Total Return:         {:>10.2}%", self.total_return_pct);
        println!("Annual Return:        {:>10.2}%", self.annual_return_pct);
        println!("Sharpe Ratio:         {:>10.2}", self.sharpe_ratio);
        println!("Sortino Ratio:        {:>10.2}", self.sortino_ratio);
        println!("Max Drawdown:         {:>10.2}%", self.max_drawdown_pct);
        println!("{}", "-".repeat(50));
        println!("Total Trades:         {:>10}", self.total_trades);
        println!("Win Rate:             {:>10.2}%", self.win_rate_pct);
        println!("Profit Factor:        {:>10.2}", self.profit_factor);
        println!("Avg Trade P&L:        ${:>9.2}", self.avg_trade_pnl);
        println!("Avg Holding (hours):  {:>10.2}", self.avg_holding_hours);
        println!("{}", "=".repeat(50));
        println!("Initial Capital:      ${:>9.2}", self.initial_capital);
        println!("Final Equity:         ${:>9.2}", self.final_equity);
        println!("{}\n", "=".repeat(50));
    }
}

/// Calculate Sharpe ratio
fn calculate_sharpe(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();

    if std < 1e-10 {
        return 0.0;
    }

    periods_per_year.sqrt() * mean / std
}

/// Calculate Sortino ratio
fn calculate_sortino(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

    if downside_returns.is_empty() {
        return f64::INFINITY;
    }

    let downside_variance = downside_returns
        .iter()
        .map(|r| r.powi(2))
        .sum::<f64>() / downside_returns.len() as f64;
    let downside_std = downside_variance.sqrt();

    if downside_std < 1e-10 {
        return 0.0;
    }

    periods_per_year.sqrt() * mean / downside_std
}

/// Calculate maximum drawdown
fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut peak = equity_curve[0];
    let mut max_dd = 0.0;

    for &equity in equity_curve {
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_report() {
        let report = BacktestReport::empty(100000.0);
        assert_eq!(report.total_trades, 0);
        assert_eq!(report.total_return_pct, 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 120.0, 100.0, 130.0];
        let dd = calculate_max_drawdown(&equity);
        // Max drawdown is from 120 to 100 = 16.67%
        assert!((dd - 0.1667).abs() < 0.01);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sharpe = calculate_sharpe(&returns, 252.0);
        assert!(sharpe > 0.0);
    }
}
