//! Performance metrics calculation

use crate::backtest::engine::TradeRecord;
use crate::strategy::PositionSide;

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annual_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub volatility: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from equity curve
    pub fn from_equity_curve(equity: &[f64], initial_capital: f64) -> Self {
        if equity.len() < 2 {
            return Self::default();
        }

        let total_return = (equity.last().unwrap() / initial_capital - 1.0) * 100.0;

        // Calculate returns
        let returns: Vec<f64> = equity
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = std(&returns);

        // Annualize (assuming daily data, 252 trading days)
        let annual_return = mean_return * 252.0 * 100.0;
        let annual_volatility = volatility * (252.0_f64).sqrt();

        // Sharpe ratio (assuming 0 risk-free rate)
        let sharpe_ratio = if annual_volatility > 0.0 {
            annual_return / annual_volatility
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        let downside_std = std(&downside_returns);
        let sortino_ratio = if downside_std > 0.0 {
            annual_return / (downside_std * (252.0_f64).sqrt())
        } else {
            0.0
        };

        // Max drawdown
        let max_drawdown = calculate_max_drawdown(equity);

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        Self {
            total_return,
            annual_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            volatility: annual_volatility,
            win_rate: 0.0,
            profit_factor: 0.0,
        }
    }

    /// Combine with trade statistics
    pub fn with_trade_stats(mut self, stats: &TradeStats) -> Self {
        self.win_rate = stats.win_rate;
        self.profit_factor = stats.profit_factor;
        self
    }
}

/// Trade statistics
#[derive(Debug, Clone, Default)]
pub struct TradeStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub profit_factor: f64,
    pub avg_trade: f64,
    pub avg_holding_time: f64,
    pub long_trades: usize,
    pub short_trades: usize,
}

impl TradeStats {
    /// Calculate statistics from trade records
    pub fn from_trades(trades: &[TradeRecord]) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        let total_trades = trades.len();

        let winning_trades: Vec<&TradeRecord> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&TradeRecord> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_count = winning_trades.len();
        let lose_count = losing_trades.len();

        let win_rate = win_count as f64 / total_trades as f64 * 100.0;

        let total_wins: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let total_losses: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();

        let avg_win = if win_count > 0 {
            total_wins / win_count as f64
        } else {
            0.0
        };

        let avg_loss = if lose_count > 0 {
            total_losses / lose_count as f64
        } else {
            0.0
        };

        let largest_win = winning_trades
            .iter()
            .map(|t| t.pnl)
            .fold(0.0_f64, f64::max);

        let largest_loss = losing_trades
            .iter()
            .map(|t| t.pnl.abs())
            .fold(0.0_f64, f64::max);

        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else if total_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade: f64 = trades.iter().map(|t| t.pnl).sum::<f64>() / total_trades as f64;

        let avg_holding_time: f64 = trades
            .iter()
            .map(|t| (t.exit_time - t.entry_time) as f64)
            .sum::<f64>()
            / total_trades as f64;

        let long_trades = trades
            .iter()
            .filter(|t| t.side == PositionSide::Long)
            .count();
        let short_trades = trades
            .iter()
            .filter(|t| t.side == PositionSide::Short)
            .count();

        Self {
            total_trades,
            winning_trades: win_count,
            losing_trades: lose_count,
            win_rate,
            avg_win,
            avg_loss,
            largest_win,
            largest_loss,
            profit_factor,
            avg_trade,
            avg_holding_time,
            long_trades,
            short_trades,
        }
    }
}

/// Calculate standard deviation
fn std(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

/// Calculate maximum drawdown
fn calculate_max_drawdown(equity: &[f64]) -> f64 {
    if equity.is_empty() {
        return 0.0;
    }

    let mut peak = equity[0];
    let mut max_dd = 0.0;

    for &value in equity {
        if value > peak {
            peak = value;
        }
        let drawdown = (peak - value) / peak * 100.0;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 120.0, 90.0, 100.0];
        let dd = calculate_max_drawdown(&equity);

        // Max drawdown is from 120 to 90 = 25%
        assert!((dd - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_metrics_calculation() {
        let equity: Vec<f64> = (0..100)
            .map(|i| 10000.0 * (1.0 + 0.001 * i as f64))
            .collect();

        let metrics = PerformanceMetrics::from_equity_curve(&equity, 10000.0);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.sharpe_ratio >= 0.0);
    }

    #[test]
    fn test_trade_stats() {
        let trades = vec![
            TradeRecord {
                entry_time: 0,
                exit_time: 100,
                side: PositionSide::Long,
                entry_price: 100.0,
                exit_price: 110.0,
                size: 1.0,
                pnl: 10.0,
                pnl_percent: 10.0,
                exit_reason: crate::backtest::engine::ExitReason::Signal,
            },
            TradeRecord {
                entry_time: 200,
                exit_time: 300,
                side: PositionSide::Long,
                entry_price: 110.0,
                exit_price: 105.0,
                size: 1.0,
                pnl: -5.0,
                pnl_percent: -4.5,
                exit_reason: crate::backtest::engine::ExitReason::StopLoss,
            },
        ];

        let stats = TradeStats::from_trades(&trades);

        assert_eq!(stats.total_trades, 2);
        assert_eq!(stats.winning_trades, 1);
        assert_eq!(stats.losing_trades, 1);
        assert_eq!(stats.win_rate, 50.0);
    }
}
