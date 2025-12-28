//! Backtest result and reporting.

use crate::models::Order;
use crate::utils::PerformanceMetrics;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Complete backtest result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Strategy name
    pub strategy_name: String,
    /// Symbol tested
    pub symbol: String,
    /// Backtest start time
    pub start_time: DateTime<Utc>,
    /// Backtest end time
    pub end_time: DateTime<Utc>,
    /// Initial capital
    pub initial_capital: f64,
    /// Final equity
    pub final_equity: f64,
    /// Total return percentage
    pub total_return_pct: f64,
    /// Annualized return percentage
    pub annualized_return_pct: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Win rate percentage
    pub win_rate_pct: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Total fees paid
    pub total_fees: f64,
    /// Equity curve
    pub equity_curve: Vec<(DateTime<Utc>, f64)>,
    /// Trade returns
    pub trade_returns: Vec<f64>,
}

impl BacktestResult {
    /// Create a new backtest result from metrics.
    pub fn new(
        strategy_name: String,
        symbol: String,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        initial_capital: f64,
        final_equity: f64,
        metrics: &PerformanceMetrics,
        total_fees: f64,
        equity_curve: Vec<(DateTime<Utc>, f64)>,
        trade_returns: Vec<f64>,
    ) -> Self {
        Self {
            strategy_name,
            symbol,
            start_time,
            end_time,
            initial_capital,
            final_equity,
            total_return_pct: metrics.total_return * 100.0,
            annualized_return_pct: metrics.annualized_return * 100.0,
            sharpe_ratio: metrics.sharpe_ratio,
            sortino_ratio: metrics.sortino_ratio,
            max_drawdown_pct: metrics.max_drawdown * 100.0,
            calmar_ratio: metrics.calmar_ratio,
            win_rate_pct: metrics.win_rate * 100.0,
            profit_factor: metrics.profit_factor,
            total_trades: metrics.total_trades,
            total_fees,
            equity_curve,
            trade_returns,
        }
    }

    /// Print a summary report.
    pub fn print_report(&self) {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                    BACKTEST REPORT                           ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Strategy: {:<51} ║", self.strategy_name);
        println!("║ Symbol:   {:<51} ║", self.symbol);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Period:   {} to {}  ║",
            self.start_time.format("%Y-%m-%d"),
            self.end_time.format("%Y-%m-%d")
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║                      PERFORMANCE                             ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Initial Capital:     ${:>15.2}                       ║", self.initial_capital);
        println!("║ Final Equity:        ${:>15.2}                       ║", self.final_equity);
        println!("║ Total Return:        {:>15.2}%                       ║", self.total_return_pct);
        println!("║ Annualized Return:   {:>15.2}%                       ║", self.annualized_return_pct);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║                    RISK METRICS                              ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Sharpe Ratio:        {:>15.2}                        ║", self.sharpe_ratio);
        println!("║ Sortino Ratio:       {:>15.2}                        ║", self.sortino_ratio);
        println!("║ Max Drawdown:        {:>15.2}%                       ║", self.max_drawdown_pct);
        println!("║ Calmar Ratio:        {:>15.2}                        ║", self.calmar_ratio);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║                   TRADE STATISTICS                           ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Total Trades:        {:>15}                        ║", self.total_trades);
        println!("║ Win Rate:            {:>15.2}%                       ║", self.win_rate_pct);
        println!("║ Profit Factor:       {:>15.2}                        ║", self.profit_factor);
        println!("║ Total Fees Paid:     ${:>15.2}                       ║", self.total_fees);
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
    }

    /// Export to JSON file.
    pub fn save_json(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Check if the strategy is profitable.
    pub fn is_profitable(&self) -> bool {
        self.total_return_pct > 0.0
    }

    /// Check if the strategy has acceptable risk metrics.
    pub fn has_good_risk_metrics(&self) -> bool {
        self.sharpe_ratio > 1.0 && self.max_drawdown_pct < 20.0
    }
}

/// Trade record for detailed analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub symbol: String,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub return_pct: f64,
    pub fees: f64,
}

impl TradeRecord {
    /// Create from entry and exit orders.
    pub fn from_orders(entry: &Order, exit: &Order, fee_rate: f64) -> Self {
        let entry_value = entry.avg_fill_price * entry.filled_quantity;
        let exit_value = exit.avg_fill_price * exit.filled_quantity;
        let fees = (entry_value + exit_value) * fee_rate;

        let pnl = if entry.side == crate::models::OrderSide::Buy {
            exit_value - entry_value - fees
        } else {
            entry_value - exit_value - fees
        };

        let return_pct = pnl / entry_value * 100.0;

        Self {
            entry_time: entry.created_at,
            exit_time: exit.created_at,
            symbol: entry.symbol.clone(),
            side: format!("{}", entry.side),
            entry_price: entry.avg_fill_price,
            exit_price: exit.avg_fill_price,
            quantity: entry.filled_quantity,
            pnl,
            return_pct,
            fees,
        }
    }
}
