//! # Backtest Report
//!
//! Report generation for backtest results.

use crate::backtest::engine::{BacktestEngine, EquityPoint, TradeLog};
use crate::metrics::trading::TradingMetrics;
use serde::{Deserialize, Serialize};

/// Complete backtest report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    /// Summary statistics
    pub summary: ReportSummary,
    /// Equity curve
    pub equity_curve: Vec<EquityPoint>,
    /// Trade log
    pub trades: Vec<TradeLog>,
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Initial capital
    pub initial_capital: f64,
    /// Final capital
    pub final_capital: f64,
    /// Total return percentage
    pub total_return_pct: f64,
    /// Number of trades
    pub total_trades: usize,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown_pct: f64,
    /// Average trade P&L
    pub avg_trade_pnl: f64,
    /// Best trade
    pub best_trade: f64,
    /// Worst trade
    pub worst_trade: f64,
}

impl BacktestReport {
    /// Create from engine
    pub fn from_engine(engine: &BacktestEngine, initial_capital: f64) -> Self {
        let metrics = engine.metrics();
        let equity_curve = engine.equity_curve().to_vec();
        let trades = engine.trade_log().to_vec();

        let max_drawdown = equity_curve
            .iter()
            .map(|e| e.drawdown)
            .fold(0.0_f64, f64::max);

        let (best_trade, worst_trade) = if trades.is_empty() {
            (0.0, 0.0)
        } else {
            let best = trades.iter().map(|t| t.pnl).fold(f64::NEG_INFINITY, f64::max);
            let worst = trades.iter().map(|t| t.pnl).fold(f64::INFINITY, f64::min);
            (best, worst)
        };

        let summary = ReportSummary {
            initial_capital,
            final_capital: engine.final_equity(),
            total_return_pct: engine.total_return(),
            total_trades: metrics.total_trades,
            win_rate: metrics.win_rate() * 100.0,
            profit_factor: metrics.profit_factor(),
            sharpe_ratio: metrics.sharpe_ratio(),
            max_drawdown_pct: max_drawdown * 100.0,
            avg_trade_pnl: metrics.expectancy(),
            best_trade,
            worst_trade,
        };

        Self {
            summary,
            equity_curve,
            trades,
        }
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV (trades only)
    pub fn trades_to_csv(&self) -> String {
        let mut csv = String::from("timestamp,side,price,size,pnl,exit_reason\n");

        for trade in &self.trades {
            csv.push_str(&format!(
                "{},{},{},{},{},{}\n",
                trade.timestamp.to_rfc3339(),
                trade.side,
                trade.price,
                trade.size,
                trade.pnl,
                trade.exit_reason
            ));
        }

        csv
    }

    /// Print formatted report
    pub fn print(&self) {
        println!("{}", self.format());
    }

    /// Format report as string
    pub fn format(&self) -> String {
        format!(
            r#"
╔═══════════════════════════════════════════════════════════════╗
║                     BACKTEST REPORT                           ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  CAPITAL SUMMARY                                              ║
║  ─────────────────────────────────────────────────────────   ║
║  Initial Capital:        ${:>12.2}                          ║
║  Final Capital:          ${:>12.2}                          ║
║  Total Return:           {:>12.2}%                          ║
║                                                               ║
║  TRADING STATISTICS                                           ║
║  ─────────────────────────────────────────────────────────   ║
║  Total Trades:           {:>12}                             ║
║  Win Rate:               {:>12.1}%                          ║
║  Profit Factor:          {:>12.2}                           ║
║  Average Trade:          ${:>12.2}                          ║
║  Best Trade:             ${:>12.2}                          ║
║  Worst Trade:            ${:>12.2}                          ║
║                                                               ║
║  RISK METRICS                                                 ║
║  ─────────────────────────────────────────────────────────   ║
║  Sharpe Ratio:           {:>12.2}                           ║
║  Max Drawdown:           {:>12.2}%                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#,
            self.summary.initial_capital,
            self.summary.final_capital,
            self.summary.total_return_pct,
            self.summary.total_trades,
            self.summary.win_rate,
            self.summary.profit_factor,
            self.summary.avg_trade_pnl,
            self.summary.best_trade,
            self.summary.worst_trade,
            self.summary.sharpe_ratio,
            self.summary.max_drawdown_pct
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_formatting() {
        let summary = ReportSummary {
            initial_capital: 10000.0,
            final_capital: 11000.0,
            total_return_pct: 10.0,
            total_trades: 50,
            win_rate: 55.0,
            profit_factor: 1.5,
            sharpe_ratio: 2.1,
            max_drawdown_pct: 5.0,
            avg_trade_pnl: 20.0,
            best_trade: 150.0,
            worst_trade: -80.0,
        };

        let report = BacktestReport {
            summary,
            equity_curve: vec![],
            trades: vec![],
        };

        let formatted = report.format();
        assert!(formatted.contains("Total Trades"));
        assert!(formatted.contains("Sharpe Ratio"));
    }
}
