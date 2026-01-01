//! Performance Metrics for Trading Strategies
//!
//! This module provides functions for calculating trading performance metrics
//! such as Sharpe ratio, maximum drawdown, win rate, etc.

use std::collections::VecDeque;

/// Trade record for performance tracking
#[derive(Debug, Clone)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: u64,
    /// Exit timestamp
    pub exit_time: u64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size
    pub size: f64,
    /// Is long trade
    pub is_long: bool,
    /// Realized PnL
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
}

impl Trade {
    /// Create a new trade record
    pub fn new(
        entry_time: u64,
        exit_time: u64,
        entry_price: f64,
        exit_price: f64,
        size: f64,
        is_long: bool,
    ) -> Self {
        let direction = if is_long { 1.0 } else { -1.0 };
        let return_pct = direction * (exit_price - entry_price) / entry_price * 100.0;
        let pnl = direction * (exit_price - entry_price) * size;

        Self {
            entry_time,
            exit_time,
            entry_price,
            exit_price,
            size,
            is_long,
            pnl,
            return_pct,
        }
    }

    /// Duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.exit_time - self.entry_time
    }

    /// Check if trade was profitable
    pub fn is_winner(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Performance metrics for a trading strategy
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// List of completed trades
    trades: Vec<Trade>,
    /// Equity curve (cumulative PnL)
    equity_curve: Vec<f64>,
    /// Initial capital
    initial_capital: f64,
}

impl PerformanceMetrics {
    /// Create a new performance tracker
    pub fn new(initial_capital: f64) -> Self {
        Self {
            trades: Vec::new(),
            equity_curve: vec![initial_capital],
            initial_capital,
        }
    }

    /// Record a completed trade
    pub fn record_trade(&mut self, trade: Trade) {
        let last_equity = *self.equity_curve.last().unwrap_or(&self.initial_capital);
        self.equity_curve.push(last_equity + trade.pnl);
        self.trades.push(trade);
    }

    /// Total number of trades
    pub fn total_trades(&self) -> usize {
        self.trades.len()
    }

    /// Number of winning trades
    pub fn winning_trades(&self) -> usize {
        self.trades.iter().filter(|t| t.is_winner()).count()
    }

    /// Number of losing trades
    pub fn losing_trades(&self) -> usize {
        self.trades.iter().filter(|t| !t.is_winner()).count()
    }

    /// Win rate (percentage of winning trades)
    pub fn win_rate(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        self.winning_trades() as f64 / self.trades.len() as f64 * 100.0
    }

    /// Total PnL
    pub fn total_pnl(&self) -> f64 {
        self.trades.iter().map(|t| t.pnl).sum()
    }

    /// Total return percentage
    pub fn total_return_pct(&self) -> f64 {
        self.total_pnl() / self.initial_capital * 100.0
    }

    /// Average PnL per trade
    pub fn avg_pnl(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        self.total_pnl() / self.trades.len() as f64
    }

    /// Average winning trade PnL
    pub fn avg_win(&self) -> f64 {
        let winners: Vec<&Trade> = self.trades.iter().filter(|t| t.is_winner()).collect();
        if winners.is_empty() {
            return 0.0;
        }
        winners.iter().map(|t| t.pnl).sum::<f64>() / winners.len() as f64
    }

    /// Average losing trade PnL
    pub fn avg_loss(&self) -> f64 {
        let losers: Vec<&Trade> = self.trades.iter().filter(|t| !t.is_winner()).collect();
        if losers.is_empty() {
            return 0.0;
        }
        losers.iter().map(|t| t.pnl).sum::<f64>() / losers.len() as f64
    }

    /// Profit factor (gross profit / gross loss)
    pub fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self.trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = self.trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();

        if gross_loss == 0.0 {
            return if gross_profit > 0.0 { f64::INFINITY } else { 0.0 };
        }
        gross_profit / gross_loss
    }

    /// Maximum drawdown (percentage)
    pub fn max_drawdown(&self) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        let mut max_equity = self.equity_curve[0];
        let mut max_drawdown = 0.0;

        for &equity in &self.equity_curve {
            max_equity = max_equity.max(equity);
            let drawdown = (max_equity - equity) / max_equity * 100.0;
            max_drawdown = max_drawdown.max(drawdown);
        }

        max_drawdown
    }

    /// Calculate returns for Sharpe ratio
    fn calculate_returns(&self) -> Vec<f64> {
        self.trades.iter().map(|t| t.return_pct / 100.0).collect()
    }

    /// Sharpe ratio (assuming risk-free rate = 0)
    pub fn sharpe_ratio(&self, periods_per_year: f64) -> f64 {
        let returns = self.calculate_returns();
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std_return = variance.sqrt();

        if std_return == 0.0 {
            return 0.0;
        }

        let annualized_return = mean_return * periods_per_year;
        let annualized_std = std_return * periods_per_year.sqrt();

        annualized_return / annualized_std
    }

    /// Sortino ratio (downside deviation only)
    pub fn sortino_ratio(&self, periods_per_year: f64) -> f64 {
        let returns = self.calculate_returns();
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        // Calculate downside deviation
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if negative_returns.is_empty() {
            return if mean_return > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
        }

        let downside_variance = negative_returns.iter().map(|r| r.powi(2)).sum::<f64>()
            / negative_returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std == 0.0 {
            return 0.0;
        }

        let annualized_return = mean_return * periods_per_year;
        let annualized_downside = downside_std * periods_per_year.sqrt();

        annualized_return / annualized_downside
    }

    /// Calmar ratio (annualized return / max drawdown)
    pub fn calmar_ratio(&self, periods_per_year: f64) -> f64 {
        let max_dd = self.max_drawdown();
        if max_dd == 0.0 {
            return 0.0;
        }

        let returns = self.calculate_returns();
        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let annualized_return = mean_return * periods_per_year * 100.0;

        annualized_return / max_dd
    }

    /// Expectancy (expected value per trade)
    pub fn expectancy(&self) -> f64 {
        let win_rate = self.win_rate() / 100.0;
        let avg_win = self.avg_win();
        let avg_loss = self.avg_loss().abs();

        win_rate * avg_win - (1.0 - win_rate) * avg_loss
    }

    /// Get summary statistics
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_trades: self.total_trades(),
            winning_trades: self.winning_trades(),
            losing_trades: self.losing_trades(),
            win_rate: self.win_rate(),
            total_pnl: self.total_pnl(),
            total_return_pct: self.total_return_pct(),
            avg_pnl: self.avg_pnl(),
            avg_win: self.avg_win(),
            avg_loss: self.avg_loss(),
            profit_factor: self.profit_factor(),
            max_drawdown: self.max_drawdown(),
            sharpe_ratio: self.sharpe_ratio(252.0), // Daily trading
            sortino_ratio: self.sortino_ratio(252.0),
            expectancy: self.expectancy(),
        }
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    /// Get all trades
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }
}

/// Summary of performance metrics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub total_return_pct: f64,
    pub avg_pnl: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub expectancy: f64,
}

impl MetricsSummary {
    /// Format as a table string
    pub fn to_table(&self) -> String {
        format!(
            r#"
╔══════════════════════════════════════════════════╗
║           PERFORMANCE SUMMARY                     ║
╠══════════════════════════════════════════════════╣
║ Total Trades:     {:>10}                        ║
║ Winners:          {:>10}                        ║
║ Losers:           {:>10}                        ║
║ Win Rate:         {:>10.2}%                       ║
╠══════════════════════════════════════════════════╣
║ Total PnL:        {:>10.2}                        ║
║ Total Return:     {:>10.2}%                       ║
║ Avg PnL/Trade:    {:>10.2}                        ║
║ Avg Win:          {:>10.2}                        ║
║ Avg Loss:         {:>10.2}                        ║
╠══════════════════════════════════════════════════╣
║ Profit Factor:    {:>10.2}                        ║
║ Max Drawdown:     {:>10.2}%                       ║
║ Sharpe Ratio:     {:>10.2}                        ║
║ Sortino Ratio:    {:>10.2}                        ║
║ Expectancy:       {:>10.2}                        ║
╚══════════════════════════════════════════════════╝
"#,
            self.total_trades,
            self.winning_trades,
            self.losing_trades,
            self.win_rate,
            self.total_pnl,
            self.total_return_pct,
            self.avg_pnl,
            self.avg_win,
            self.avg_loss,
            self.profit_factor,
            self.max_drawdown,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.expectancy
        )
    }
}

/// Rolling metrics for online performance tracking
#[derive(Debug)]
pub struct RollingMetrics {
    /// Window size
    window_size: usize,
    /// Recent returns
    returns: VecDeque<f64>,
    /// Recent equity values
    equity: VecDeque<f64>,
}

impl RollingMetrics {
    /// Create a new rolling metrics tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            returns: VecDeque::with_capacity(window_size),
            equity: VecDeque::with_capacity(window_size),
        }
    }

    /// Add a new return observation
    pub fn add_return(&mut self, return_value: f64, equity_value: f64) {
        if self.returns.len() >= self.window_size {
            self.returns.pop_front();
            self.equity.pop_front();
        }
        self.returns.push_back(return_value);
        self.equity.push_back(equity_value);
    }

    /// Rolling Sharpe ratio
    pub fn rolling_sharpe(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance: f64 = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / self.returns.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            0.0
        } else {
            mean / std * (252.0_f64).sqrt()
        }
    }

    /// Rolling max drawdown
    pub fn rolling_max_drawdown(&self) -> f64 {
        if self.equity.len() < 2 {
            return 0.0;
        }

        let mut max_equity = self.equity[0];
        let mut max_dd = 0.0;

        for &eq in &self.equity {
            max_equity = max_equity.max(eq);
            let dd = (max_equity - eq) / max_equity * 100.0;
            max_dd = max_dd.max(dd);
        }

        max_dd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_creation() {
        let trade = Trade::new(0, 1000, 100.0, 110.0, 1.0, true);
        assert!(trade.is_winner());
        assert_eq!(trade.return_pct, 10.0);
        assert_eq!(trade.pnl, 10.0);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new(10000.0);

        // Add winning trade
        metrics.record_trade(Trade::new(0, 1000, 100.0, 110.0, 1.0, true));
        // Add losing trade
        metrics.record_trade(Trade::new(1000, 2000, 110.0, 105.0, 1.0, true));

        assert_eq!(metrics.total_trades(), 2);
        assert_eq!(metrics.winning_trades(), 1);
        assert_eq!(metrics.win_rate(), 50.0);
    }

    #[test]
    fn test_max_drawdown() {
        let mut metrics = PerformanceMetrics::new(10000.0);

        metrics.record_trade(Trade::new(0, 1, 100.0, 110.0, 100.0, true)); // +1000
        metrics.record_trade(Trade::new(1, 2, 110.0, 100.0, 100.0, true)); // -1000
        metrics.record_trade(Trade::new(2, 3, 100.0, 95.0, 100.0, true));  // -500

        let max_dd = metrics.max_drawdown();
        assert!(max_dd > 0.0);
    }

    #[test]
    fn test_profit_factor() {
        let mut metrics = PerformanceMetrics::new(10000.0);

        metrics.record_trade(Trade::new(0, 1, 100.0, 120.0, 1.0, true)); // +20
        metrics.record_trade(Trade::new(1, 2, 100.0, 90.0, 1.0, true));  // -10

        assert_eq!(metrics.profit_factor(), 2.0);
    }
}
