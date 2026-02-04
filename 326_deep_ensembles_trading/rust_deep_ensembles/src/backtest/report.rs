//! Backtest report generation

use super::metrics::TradingMetrics;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub entry_time: i64,
    pub exit_time: i64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub pnl: f64,
    pub return_pct: f64,
}

/// Backtest report
#[derive(Debug, Clone)]
pub struct BacktestReport {
    /// Initial capital
    pub initial_capital: f64,
    /// Final capital
    pub final_capital: f64,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Trade records
    pub trades: Vec<TradeRecord>,
    /// Trading metrics
    pub metrics: TradingMetrics,
}

impl BacktestReport {
    /// Create a new backtest report
    pub fn new(
        initial_capital: f64,
        final_capital: f64,
        equity_curve: Vec<f64>,
        trades: Vec<TradeRecord>,
        metrics: TradingMetrics,
    ) -> Self {
        Self {
            initial_capital,
            final_capital,
            equity_curve,
            trades,
            metrics,
        }
    }

    /// Get number of winning trades
    pub fn winning_trades(&self) -> usize {
        self.trades.iter().filter(|t| t.pnl > 0.0).count()
    }

    /// Get number of losing trades
    pub fn losing_trades(&self) -> usize {
        self.trades.iter().filter(|t| t.pnl <= 0.0).count()
    }

    /// Get average trade PnL
    pub fn avg_trade_pnl(&self) -> f64 {
        if self.trades.is_empty() {
            0.0
        } else {
            self.trades.iter().map(|t| t.pnl).sum::<f64>() / self.trades.len() as f64
        }
    }

    /// Get best trade
    pub fn best_trade(&self) -> Option<&TradeRecord> {
        self.trades.iter().max_by(|a, b| {
            a.pnl.partial_cmp(&b.pnl).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get worst trade
    pub fn worst_trade(&self) -> Option<&TradeRecord> {
        self.trades.iter().min_by(|a, b| {
            a.pnl.partial_cmp(&b.pnl).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Generate summary string
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!("\n{}\n", "=".repeat(60)));
        s.push_str("BACKTEST REPORT\n");
        s.push_str(&format!("{}\n\n", "=".repeat(60)));

        s.push_str("Capital:\n");
        s.push_str(&format!("  Initial: ${:.2}\n", self.initial_capital));
        s.push_str(&format!("  Final:   ${:.2}\n", self.final_capital));
        s.push_str(&format!(
            "  Return:  {:.2}%\n\n",
            self.metrics.total_return * 100.0
        ));

        s.push_str("Performance Metrics:\n");
        s.push_str(&format!("  Sharpe Ratio:   {:.2}\n", self.metrics.sharpe_ratio));
        s.push_str(&format!("  Sortino Ratio:  {:.2}\n", self.metrics.sortino_ratio));
        s.push_str(&format!("  Max Drawdown:   {:.2}%\n", self.metrics.max_drawdown * 100.0));
        s.push_str(&format!("  Win Rate:       {:.2}%\n", self.metrics.win_rate * 100.0));
        s.push_str(&format!("  Profit Factor:  {:.2}\n", self.metrics.profit_factor));
        s.push_str(&format!("  Volatility:     {:.2}%\n\n", self.metrics.volatility * 100.0));

        s.push_str("Trading Activity:\n");
        s.push_str(&format!("  Total Trades:   {}\n", self.trades.len()));
        s.push_str(&format!("  Winning Trades: {}\n", self.winning_trades()));
        s.push_str(&format!("  Losing Trades:  {}\n", self.losing_trades()));
        s.push_str(&format!("  Avg Trade PnL:  ${:.2}\n", self.avg_trade_pnl()));

        if let Some(best) = self.best_trade() {
            s.push_str(&format!("  Best Trade:     ${:.2}\n", best.pnl));
        }
        if let Some(worst) = self.worst_trade() {
            s.push_str(&format!("  Worst Trade:    ${:.2}\n", worst.pnl));
        }

        s.push_str(&format!("\n{}\n", "=".repeat(60)));

        s
    }
}

impl fmt::Display for BacktestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_summary() {
        let metrics = TradingMetrics {
            total_return: 0.15,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            max_drawdown: 0.1,
            win_rate: 0.6,
            profit_factor: 1.5,
            volatility: 0.2,
            num_trades: 10,
        };

        let trades = vec![
            TradeRecord {
                symbol: "BTCUSDT".to_string(),
                entry_time: 0,
                exit_time: 100,
                entry_price: 100.0,
                exit_price: 105.0,
                size: 1.0,
                pnl: 5.0,
                return_pct: 0.05,
            },
        ];

        let report = BacktestReport::new(
            100000.0,
            115000.0,
            vec![100000.0, 105000.0, 110000.0, 115000.0],
            trades,
            metrics,
        );

        let summary = report.summary();
        assert!(summary.contains("BACKTEST REPORT"));
        assert!(summary.contains("$100000.00"));
    }
}
