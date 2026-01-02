//! Performance metrics for trading strategies

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Trade record for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub entry_time: u64,
    pub exit_time: u64,
    pub holding_time_secs: u64,
}

impl TradeRecord {
    pub fn new(
        symbol: impl Into<String>,
        side: impl Into<String>,
        entry_price: f64,
        exit_price: f64,
        size: f64,
        entry_time: u64,
        exit_time: u64,
    ) -> Self {
        let side_str = side.into();
        let pnl = if side_str == "BUY" {
            (exit_price - entry_price) * size
        } else {
            (entry_price - exit_price) * size
        };
        let pnl_pct = pnl / (entry_price * size);

        Self {
            symbol: symbol.into(),
            side: side_str,
            entry_price,
            exit_price,
            size,
            pnl,
            pnl_pct,
            entry_time,
            exit_time,
            holding_time_secs: (exit_time - entry_time) / 1000,
        }
    }

    pub fn is_winner(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_pnl: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub avg_holding_time: f64,
    pub expectancy: f64,
}

impl Metrics {
    /// Calculate metrics from trade records
    pub fn from_trades(trades: &[TradeRecord]) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        let total = trades.len();
        let winners: Vec<_> = trades.iter().filter(|t| t.is_winner()).collect();
        let losers: Vec<_> = trades.iter().filter(|t| !t.is_winner()).collect();

        let winning = winners.len();
        let losing = losers.len();
        let win_rate = winning as f64 / total as f64;

        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let avg_pnl = total_pnl / total as f64;

        let avg_win = if !winners.is_empty() {
            winners.iter().map(|t| t.pnl).sum::<f64>() / winning as f64
        } else {
            0.0
        };

        let avg_loss = if !losers.is_empty() {
            losers.iter().map(|t| t.pnl.abs()).sum::<f64>() / losing as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winners.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losers.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Calculate Sharpe ratio (simplified, assuming daily returns)
        let returns: Vec<f64> = trades.iter().map(|t| t.pnl_pct).collect();
        let sharpe_ratio = calculate_sharpe(&returns);
        let sortino_ratio = calculate_sortino(&returns);

        // Calculate drawdown
        let equity_curve = cumulative_equity(trades);
        let max_drawdown = max_drawdown(&equity_curve);

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            (total_pnl / equity_curve.first().unwrap_or(&1.0)) / max_drawdown
        } else {
            0.0
        };

        let avg_holding_time =
            trades.iter().map(|t| t.holding_time_secs as f64).sum::<f64>() / total as f64;

        // Expectancy
        let expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss;

        Self {
            total_trades: total,
            winning_trades: winning,
            losing_trades: losing,
            win_rate,
            total_pnl,
            avg_pnl,
            avg_win,
            avg_loss,
            profit_factor,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            avg_holding_time,
            expectancy,
        }
    }
}

/// Calculate Sharpe ratio
fn calculate_sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
        / (returns.len() - 1) as f64;
    let std = variance.sqrt();

    if std > 0.0 {
        mean / std * (252.0_f64).sqrt() // Annualized
    } else {
        0.0
    }
}

/// Calculate Sortino ratio (downside deviation)
fn calculate_sortino(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    if downside.is_empty() {
        return f64::INFINITY;
    }

    let downside_variance: f64 =
        downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
    let downside_std = downside_variance.sqrt();

    if downside_std > 0.0 {
        mean / downside_std * (252.0_f64).sqrt()
    } else {
        0.0
    }
}

/// Calculate cumulative equity from trades
fn cumulative_equity(trades: &[TradeRecord]) -> Vec<f64> {
    let mut equity = vec![1.0];
    let mut current = 1.0;

    for trade in trades {
        current *= 1.0 + trade.pnl_pct;
        equity.push(current);
    }

    equity
}

/// Calculate maximum drawdown
fn max_drawdown(equity: &[f64]) -> f64 {
    if equity.is_empty() {
        return 0.0;
    }

    let mut peak = equity[0];
    let mut max_dd = 0.0;

    for &val in equity {
        if val > peak {
            peak = val;
        }
        let dd = (peak - val) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Performance tracker for real-time tracking
#[derive(Debug)]
pub struct PerformanceTracker {
    trades: VecDeque<TradeRecord>,
    max_trades: usize,
    equity_curve: VecDeque<f64>,
    starting_equity: f64,
    current_equity: f64,
}

impl PerformanceTracker {
    pub fn new(starting_equity: f64) -> Self {
        Self {
            trades: VecDeque::new(),
            max_trades: 10000,
            equity_curve: VecDeque::from([starting_equity]),
            starting_equity,
            current_equity: starting_equity,
        }
    }

    pub fn record_trade(&mut self, trade: TradeRecord) {
        self.current_equity += trade.pnl;
        self.equity_curve.push_back(self.current_equity);

        self.trades.push_back(trade);

        if self.trades.len() > self.max_trades {
            self.trades.pop_front();
        }
        if self.equity_curve.len() > self.max_trades {
            self.equity_curve.pop_front();
        }
    }

    pub fn metrics(&self) -> Metrics {
        let trades: Vec<_> = self.trades.iter().cloned().collect();
        Metrics::from_trades(&trades)
    }

    pub fn current_equity(&self) -> f64 {
        self.current_equity
    }

    pub fn total_return(&self) -> f64 {
        (self.current_equity - self.starting_equity) / self.starting_equity
    }

    pub fn current_drawdown(&self) -> f64 {
        let peak = self.equity_curve.iter().cloned().fold(0.0, f64::max);
        if peak > 0.0 {
            (peak - self.current_equity) / peak
        } else {
            0.0
        }
    }

    pub fn trade_count(&self) -> usize {
        self.trades.len()
    }

    pub fn recent_trades(&self, n: usize) -> Vec<&TradeRecord> {
        self.trades.iter().rev().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trades() -> Vec<TradeRecord> {
        vec![
            TradeRecord::new("BTC", "BUY", 100.0, 110.0, 1.0, 1000, 2000),
            TradeRecord::new("BTC", "BUY", 110.0, 105.0, 1.0, 3000, 4000),
            TradeRecord::new("BTC", "SELL", 105.0, 95.0, 1.0, 5000, 6000),
            TradeRecord::new("BTC", "BUY", 95.0, 100.0, 1.0, 7000, 8000),
        ]
    }

    #[test]
    fn test_trade_record() {
        let trade = TradeRecord::new("BTC", "BUY", 100.0, 110.0, 1.0, 1000, 2000);
        assert!(trade.is_winner());
        assert_eq!(trade.pnl, 10.0);
        assert_eq!(trade.pnl_pct, 0.1);
    }

    #[test]
    fn test_metrics() {
        let trades = create_test_trades();
        let metrics = Metrics::from_trades(&trades);

        assert_eq!(metrics.total_trades, 4);
        assert_eq!(metrics.winning_trades, 3);
        assert_eq!(metrics.losing_trades, 1);
        assert!(metrics.win_rate > 0.7);
        assert!(metrics.total_pnl > 0.0);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(1000.0);

        tracker.record_trade(TradeRecord::new("BTC", "BUY", 100.0, 110.0, 1.0, 1000, 2000));
        tracker.record_trade(TradeRecord::new("BTC", "BUY", 110.0, 105.0, 1.0, 3000, 4000));

        assert_eq!(tracker.trade_count(), 2);
        assert_eq!(tracker.current_equity(), 1005.0); // 1000 + 10 - 5
    }
}
