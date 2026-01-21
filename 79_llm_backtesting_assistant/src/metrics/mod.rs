//! Performance metrics calculation module
//!
//! This module provides functionality for calculating various trading performance metrics
//! including risk-adjusted returns, drawdown analysis, and trade statistics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents a single trade in the backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: DateTime<Utc>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size (positive for long, negative for short)
    pub quantity: f64,
    /// Profit/loss from the trade
    pub pnl: f64,
    /// Trade direction
    pub side: TradeSide,
    /// Trading symbol
    pub symbol: String,
}

impl Trade {
    /// Create a new trade
    pub fn new(
        entry_time: DateTime<Utc>,
        exit_time: DateTime<Utc>,
        entry_price: f64,
        exit_price: f64,
        quantity: f64,
        side: TradeSide,
        symbol: String,
    ) -> Self {
        let pnl = match side {
            TradeSide::Long => (exit_price - entry_price) * quantity,
            TradeSide::Short => (entry_price - exit_price) * quantity.abs(),
        };

        Self {
            entry_time,
            exit_time,
            entry_price,
            exit_price,
            quantity,
            pnl,
            side,
            symbol,
        }
    }

    /// Calculate the return percentage of this trade
    pub fn return_pct(&self) -> f64 {
        match self.side {
            TradeSide::Long => (self.exit_price - self.entry_price) / self.entry_price,
            TradeSide::Short => (self.entry_price - self.exit_price) / self.entry_price,
        }
    }

    /// Check if this trade was profitable
    pub fn is_profitable(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradeSide {
    Long,
    Short,
}

/// Comprehensive performance metrics for a backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return percentage
    pub annualized_return: f64,
    /// Sharpe ratio (risk-adjusted return)
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside risk-adjusted return)
    pub sortino_ratio: f64,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Maximum drawdown duration in days
    pub max_drawdown_duration_days: u32,
    /// Volatility (annualized standard deviation)
    pub volatility: f64,
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total number of trades
    pub total_trades: u32,
    /// Number of winning trades
    pub winning_trades: u32,
    /// Number of losing trades
    pub losing_trades: u32,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Average winning trade return
    pub avg_win: f64,
    /// Average losing trade return
    pub avg_loss: f64,
    /// Largest winning trade
    pub largest_win: f64,
    /// Largest losing trade
    pub largest_loss: f64,
    /// Risk-free rate used in calculations
    pub risk_free_rate: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            max_drawdown_duration_days: 0,
            volatility: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            avg_trade_return: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            risk_free_rate: 0.0,
        }
    }
}

/// Calculator for performance metrics
pub struct MetricsCalculator {
    /// Risk-free rate for Sharpe/Sortino calculations
    risk_free_rate: f64,
    /// Trading days per year (252 for stocks, 365 for crypto)
    trading_days_per_year: u32,
}

impl MetricsCalculator {
    /// Create a new calculator with default parameters
    pub fn new() -> Self {
        Self {
            risk_free_rate: 0.02, // 2% default risk-free rate
            trading_days_per_year: 252,
        }
    }

    /// Create a calculator for cryptocurrency markets (365 trading days)
    pub fn for_crypto() -> Self {
        Self {
            risk_free_rate: 0.02,
            trading_days_per_year: 365,
        }
    }

    /// Set the risk-free rate
    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Set the number of trading days per year
    pub fn with_trading_days(mut self, days: u32) -> Self {
        self.trading_days_per_year = days;
        self
    }

    /// Calculate all performance metrics from a list of trades and equity curve
    pub fn calculate(&self, trades: &[Trade], equity_curve: &[f64]) -> PerformanceMetrics {
        if trades.is_empty() || equity_curve.len() < 2 {
            return PerformanceMetrics::default();
        }

        let returns = self.calculate_returns(equity_curve);
        let total_return = self.calculate_total_return(equity_curve);
        let annualized_return = self.calculate_annualized_return(total_return, equity_curve.len());
        let volatility = self.calculate_volatility(&returns);
        let sharpe_ratio = self.calculate_sharpe_ratio(annualized_return, volatility);
        let sortino_ratio = self.calculate_sortino_ratio(&returns, annualized_return);
        let (max_drawdown, max_dd_duration) = self.calculate_max_drawdown(equity_curve);
        let calmar_ratio = self.calculate_calmar_ratio(annualized_return, max_drawdown);

        let trade_stats = self.calculate_trade_statistics(trades);

        PerformanceMetrics {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            max_drawdown_duration_days: max_dd_duration,
            volatility,
            win_rate: trade_stats.win_rate,
            profit_factor: trade_stats.profit_factor,
            total_trades: trades.len() as u32,
            winning_trades: trade_stats.winning_trades,
            losing_trades: trade_stats.losing_trades,
            avg_trade_return: trade_stats.avg_trade_return,
            avg_win: trade_stats.avg_win,
            avg_loss: trade_stats.avg_loss,
            largest_win: trade_stats.largest_win,
            largest_loss: trade_stats.largest_loss,
            risk_free_rate: self.risk_free_rate,
        }
    }

    /// Calculate daily returns from equity curve
    fn calculate_returns(&self, equity_curve: &[f64]) -> Vec<f64> {
        equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate total return
    fn calculate_total_return(&self, equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }
        let first = equity_curve.first().unwrap();
        let last = equity_curve.last().unwrap();
        (last - first) / first
    }

    /// Calculate annualized return
    fn calculate_annualized_return(&self, total_return: f64, num_periods: usize) -> f64 {
        let years = num_periods as f64 / self.trading_days_per_year as f64;
        if years <= 0.0 {
            return 0.0;
        }
        (1.0 + total_return).powf(1.0 / years) - 1.0
    }

    /// Calculate annualized volatility
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt() * (self.trading_days_per_year as f64).sqrt()
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self, annualized_return: f64, volatility: f64) -> f64 {
        if volatility == 0.0 {
            return 0.0;
        }
        (annualized_return - self.risk_free_rate) / volatility
    }

    /// Calculate Sortino ratio (uses downside deviation)
    fn calculate_sortino_ratio(&self, returns: &[f64], annualized_return: f64) -> f64 {
        let daily_rf = self.risk_free_rate / self.trading_days_per_year as f64;
        let downside_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < daily_rf)
            .map(|&r| (r - daily_rf).powi(2))
            .collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_deviation =
            (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt()
                * (self.trading_days_per_year as f64).sqrt();

        if downside_deviation == 0.0 {
            return f64::INFINITY;
        }

        (annualized_return - self.risk_free_rate) / downside_deviation
    }

    /// Calculate maximum drawdown and duration
    fn calculate_max_drawdown(&self, equity_curve: &[f64]) -> (f64, u32) {
        if equity_curve.is_empty() {
            return (0.0, 0);
        }

        let mut peak = equity_curve[0];
        let mut max_dd = 0.0;
        let mut max_dd_duration = 0u32;
        let mut current_dd_start = 0usize;

        for (i, &value) in equity_curve.iter().enumerate() {
            if value > peak {
                peak = value;
                current_dd_start = i;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
                max_dd_duration = (i - current_dd_start) as u32;
            }
        }

        (max_dd, max_dd_duration)
    }

    /// Calculate Calmar ratio
    fn calculate_calmar_ratio(&self, annualized_return: f64, max_drawdown: f64) -> f64 {
        if max_drawdown == 0.0 {
            return f64::INFINITY;
        }
        annualized_return / max_drawdown
    }

    /// Calculate trade-level statistics
    fn calculate_trade_statistics(&self, trades: &[Trade]) -> TradeStatistics {
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_win = if !winning_trades.is_empty() {
            gross_profit / winning_trades.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing_trades.is_empty() {
            gross_loss / losing_trades.len() as f64
        } else {
            0.0
        };

        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let avg_trade_return = total_pnl / trades.len() as f64;

        let largest_win = trades
            .iter()
            .map(|t| t.pnl)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let largest_loss = trades
            .iter()
            .map(|t| t.pnl)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
            .abs();

        TradeStatistics {
            winning_trades: winning_trades.len() as u32,
            losing_trades: losing_trades.len() as u32,
            win_rate: winning_trades.len() as f64 / trades.len() as f64,
            profit_factor,
            avg_trade_return,
            avg_win,
            avg_loss,
            largest_win,
            largest_loss,
        }
    }
}

impl Default for MetricsCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal struct for trade statistics
struct TradeStatistics {
    winning_trades: u32,
    losing_trades: u32,
    win_rate: f64,
    profit_factor: f64,
    avg_trade_return: f64,
    avg_win: f64,
    avg_loss: f64,
    largest_win: f64,
    largest_loss: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_return_calculation() {
        let calculator = MetricsCalculator::new();
        let equity = vec![10000.0, 10500.0, 11000.0, 10800.0, 11500.0];
        let total_return = calculator.calculate_total_return(&equity);
        assert!((total_return - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_max_drawdown() {
        let calculator = MetricsCalculator::new();
        let equity = vec![10000.0, 11000.0, 9000.0, 9500.0, 10500.0];
        let (max_dd, _) = calculator.calculate_max_drawdown(&equity);
        // Max DD should be (11000 - 9000) / 11000 = 0.1818...
        assert!((max_dd - 0.1818).abs() < 0.01);
    }

    #[test]
    fn test_trade_return_pct() {
        let trade = Trade::new(
            Utc::now(),
            Utc::now(),
            100.0,
            110.0,
            1.0,
            TradeSide::Long,
            "TEST".to_string(),
        );
        assert!((trade.return_pct() - 0.1).abs() < 0.001);
    }
}
