//! Backtesting report and metrics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::strategy::signal::SignalType;

/// Individual trade record.
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
    /// Position size
    pub position_size: f64,
    /// Signal type (long/short)
    pub signal_type: SignalType,
    /// Profit/loss
    pub pnl: f64,
    /// Profit/loss percentage
    pub pnl_pct: f64,
    /// Holding period in hours
    pub holding_period: u32,
    /// Entry confidence
    pub entry_confidence: f64,
    /// Entry uncertainty
    pub entry_uncertainty: f64,
    /// Exit reason
    pub exit_reason: String,
}

impl Trade {
    /// Check if trade was profitable.
    pub fn is_profitable(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Backtest performance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    /// List of trades
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Daily returns
    pub returns: Vec<f64>,

    // Performance metrics
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Number of trades
    pub n_trades: usize,

    // Uncertainty analysis
    /// Win rate on high confidence trades
    pub high_conf_win_rate: f64,
    /// Win rate on low confidence trades
    pub low_conf_win_rate: f64,
}

impl BacktestReport {
    /// Create a new backtest report.
    pub fn new(
        trades: Vec<Trade>,
        equity_curve: Vec<f64>,
        returns: Vec<f64>,
        initial_capital: f64,
    ) -> Self {
        let final_value = equity_curve.last().copied().unwrap_or(initial_capital);
        let total_return = (final_value - initial_capital) / initial_capital;

        Self {
            trades,
            equity_curve,
            returns,
            total_return,
            annual_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_trade_return: 0.0,
            n_trades: 0,
            high_conf_win_rate: 0.0,
            low_conf_win_rate: 0.0,
        }
    }

    /// Calculate all performance metrics.
    pub fn calculate_metrics(&mut self) {
        self.n_trades = self.trades.len();

        // Sharpe ratio
        if !self.returns.is_empty() {
            let mean_return = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
            let variance = self.returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / self.returns.len() as f64;
            let std = variance.sqrt();

            if std > 0.0 {
                self.sharpe_ratio = mean_return / std * (252.0_f64).sqrt();
            }

            // Sortino ratio (downside deviation)
            let downside_returns: Vec<f64> = self.returns
                .iter()
                .filter(|&&r| r < 0.0)
                .copied()
                .collect();

            if !downside_returns.is_empty() {
                let downside_variance = downside_returns
                    .iter()
                    .map(|r| r.powi(2))
                    .sum::<f64>()
                    / downside_returns.len() as f64;
                let downside_std = downside_variance.sqrt();

                if downside_std > 0.0 {
                    self.sortino_ratio = mean_return / downside_std * (252.0_f64).sqrt();
                }
            }

            // Annual return
            let n_periods = self.returns.len() as f64;
            self.annual_return = (1.0 + self.total_return).powf(252.0 / n_periods) - 1.0;
        }

        // Maximum drawdown
        if !self.equity_curve.is_empty() {
            let mut peak = self.equity_curve[0];
            for &value in &self.equity_curve {
                if value > peak {
                    peak = value;
                }
                let drawdown = (peak - value) / peak;
                if drawdown > self.max_drawdown {
                    self.max_drawdown = drawdown;
                }
            }
        }

        // Trade statistics
        if !self.trades.is_empty() {
            let wins: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
            let losses: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl <= 0.0).collect();

            self.win_rate = wins.len() as f64 / self.trades.len() as f64;

            let gross_profit: f64 = wins.iter().map(|t| t.pnl).sum();
            let gross_loss: f64 = losses.iter().map(|t| t.pnl.abs()).sum();

            if gross_loss > 0.0 {
                self.profit_factor = gross_profit / gross_loss;
            }

            self.avg_trade_return =
                self.trades.iter().map(|t| t.pnl_pct).sum::<f64>() / self.trades.len() as f64;

            // Uncertainty analysis
            let median_conf = {
                let mut confs: Vec<f64> = self.trades.iter().map(|t| t.entry_confidence).collect();
                confs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                confs[confs.len() / 2]
            };

            let high_conf_trades: Vec<&Trade> = self.trades
                .iter()
                .filter(|t| t.entry_confidence >= median_conf)
                .collect();

            let low_conf_trades: Vec<&Trade> = self.trades
                .iter()
                .filter(|t| t.entry_confidence < median_conf)
                .collect();

            if !high_conf_trades.is_empty() {
                self.high_conf_win_rate = high_conf_trades
                    .iter()
                    .filter(|t| t.pnl > 0.0)
                    .count() as f64
                    / high_conf_trades.len() as f64;
            }

            if !low_conf_trades.is_empty() {
                self.low_conf_win_rate = low_conf_trades
                    .iter()
                    .filter(|t| t.pnl > 0.0)
                    .count() as f64
                    / low_conf_trades.len() as f64;
            }
        }
    }

    /// Generate summary string.
    pub fn summary(&self) -> String {
        format!(
            r#"
Backtest Results
================
Total Return: {:.2}%
Annual Return: {:.2}%
Sharpe Ratio: {:.3}
Sortino Ratio: {:.3}
Max Drawdown: {:.2}%
Win Rate: {:.2}%
Profit Factor: {:.3}
Number of Trades: {}
Avg Trade Return: {:.2}%

Uncertainty Analysis:
- High Confidence Win Rate: {:.2}%
- Low Confidence Win Rate: {:.2}%
"#,
            self.total_return * 100.0,
            self.annual_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.n_trades,
            self.avg_trade_return * 100.0,
            self.high_conf_win_rate * 100.0,
            self.low_conf_win_rate * 100.0,
        )
    }
}
