//! Backtest report and metrics

use serde::{Deserialize, Serialize};
use std::fmt;

/// Backtest performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    /// Total return over the period
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Annualized volatility
    pub annual_volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate (fraction of profitable trades)
    pub win_rate: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average P&L per trade
    pub avg_trade_pnl: f64,
    /// Final equity
    pub final_equity: f64,
    /// Full equity curve
    pub equity_curve: Vec<f64>,
}

impl BacktestReport {
    /// Create an empty report
    pub fn empty() -> Self {
        Self {
            total_return: 0.0,
            annual_return: 0.0,
            annual_volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            num_trades: 0,
            avg_trade_pnl: 0.0,
            final_equity: 0.0,
            equity_curve: Vec::new(),
        }
    }

    /// Check if strategy is profitable
    pub fn is_profitable(&self) -> bool {
        self.total_return > 0.0
    }

    /// Check if Sharpe is acceptable (> 1.0)
    pub fn has_acceptable_sharpe(&self) -> bool {
        self.sharpe_ratio > 1.0
    }

    /// Check if drawdown is manageable (< 20%)
    pub fn has_manageable_drawdown(&self) -> bool {
        self.max_drawdown < 0.20
    }

    /// Overall quality score (0-100)
    pub fn quality_score(&self) -> f64 {
        let mut score = 0.0;

        // Sharpe contribution (max 30 points)
        score += (self.sharpe_ratio * 15.0).min(30.0).max(0.0);

        // Sortino contribution (max 20 points)
        score += (self.sortino_ratio * 10.0).min(20.0).max(0.0);

        // Drawdown contribution (max 20 points)
        score += ((0.3 - self.max_drawdown) * 66.67).min(20.0).max(0.0);

        // Win rate contribution (max 15 points)
        score += (self.win_rate * 25.0).min(15.0);

        // Return contribution (max 15 points)
        score += (self.annual_return * 50.0).min(15.0).max(0.0);

        score.min(100.0)
    }

    /// Generate summary string
    pub fn summary(&self) -> String {
        format!(
            "Total Return: {:.2}% | Sharpe: {:.2} | Max DD: {:.2}% | Trades: {} | Win Rate: {:.1}%",
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.num_trades,
            self.win_rate * 100.0
        )
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV-like format (single row)
    pub fn to_csv_row(&self) -> String {
        format!(
            "{:.6},{:.6},{:.6},{:.4},{:.4},{:.6},{:.4},{},{:.2},{:.2}",
            self.total_return,
            self.annual_return,
            self.annual_volatility,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown,
            self.win_rate,
            self.num_trades,
            self.avg_trade_pnl,
            self.final_equity
        )
    }

    /// CSV header
    pub fn csv_header() -> &'static str {
        "total_return,annual_return,annual_volatility,sharpe_ratio,sortino_ratio,max_drawdown,win_rate,num_trades,avg_trade_pnl,final_equity"
    }
}

impl fmt::Display for BacktestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "                  BACKTEST REPORT")?;
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "Total Return:      {:>8.2}%", self.total_return * 100.0)?;
        writeln!(f, "Annual Return:     {:>8.2}%", self.annual_return * 100.0)?;
        writeln!(f, "Annual Volatility: {:>8.2}%", self.annual_volatility * 100.0)?;
        writeln!(f, "Sharpe Ratio:      {:>8.2}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio:     {:>8.2}", self.sortino_ratio)?;
        writeln!(f, "Max Drawdown:      {:>8.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "───────────────────────────────────────────────────────")?;
        writeln!(f, "Win Rate:          {:>8.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Number of Trades:  {:>8}", self.num_trades)?;
        writeln!(f, "Avg Trade PnL:     ${:>7.2}", self.avg_trade_pnl)?;
        writeln!(f, "Final Equity:      ${:>10,.2}", self.final_equity)?;
        writeln!(f, "───────────────────────────────────────────────────────")?;
        writeln!(f, "Quality Score:     {:>8.1}/100", self.quality_score())?;
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_display() {
        let report = BacktestReport {
            total_return: 0.15,
            annual_return: 0.30,
            annual_volatility: 0.20,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            max_drawdown: 0.10,
            win_rate: 0.55,
            num_trades: 100,
            avg_trade_pnl: 150.0,
            final_equity: 115000.0,
            equity_curve: vec![100000.0, 115000.0],
        };

        let display = format!("{}", report);
        assert!(display.contains("15.00%")); // Total return
        assert!(display.contains("1.50"));   // Sharpe
    }

    #[test]
    fn test_quality_score() {
        let good_report = BacktestReport {
            total_return: 0.20,
            annual_return: 0.40,
            annual_volatility: 0.15,
            sharpe_ratio: 2.5,
            sortino_ratio: 3.0,
            max_drawdown: 0.08,
            win_rate: 0.60,
            num_trades: 50,
            avg_trade_pnl: 200.0,
            final_equity: 120000.0,
            equity_curve: vec![],
        };

        let score = good_report.quality_score();
        assert!(score > 50.0); // Good strategy should score well
    }
}
