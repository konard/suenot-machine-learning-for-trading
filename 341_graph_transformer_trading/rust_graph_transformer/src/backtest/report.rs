//! Backtesting report generation

use super::metrics::PerformanceMetrics;
use crate::strategy::signal::Signal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub direction: i32,
    pub pnl: f64,
    pub return_pct: f64,
}

/// Backtest report
#[derive(Debug, Clone)]
pub struct BacktestReport {
    /// Strategy name
    pub strategy_name: String,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: DateTime<Utc>,
    /// Initial capital
    pub initial_capital: f64,
    /// Final capital
    pub final_capital: f64,
    /// Trade records
    pub trades: Vec<TradeRecord>,
    /// Daily returns
    pub daily_returns: Vec<f64>,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

impl BacktestReport {
    /// Create a new backtest report
    pub fn new(
        strategy_name: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        initial_capital: f64,
        final_capital: f64,
        trades: Vec<TradeRecord>,
        daily_returns: Vec<f64>,
    ) -> Self {
        let metrics = PerformanceMetrics::from_returns(&daily_returns, 0.02);

        // Build equity curve
        let mut equity_curve = Vec::with_capacity(daily_returns.len() + 1);
        equity_curve.push(initial_capital);
        let mut current = initial_capital;
        for ret in &daily_returns {
            current *= 1.0 + ret;
            equity_curve.push(current);
        }

        Self {
            strategy_name: strategy_name.to_string(),
            start_date,
            end_date,
            initial_capital,
            final_capital,
            trades,
            daily_returns,
            equity_curve,
            metrics,
        }
    }

    /// Get total number of trades
    pub fn num_trades(&self) -> usize {
        self.trades.len()
    }

    /// Get winning trades
    pub fn winning_trades(&self) -> Vec<&TradeRecord> {
        self.trades.iter().filter(|t| t.pnl > 0.0).collect()
    }

    /// Get losing trades
    pub fn losing_trades(&self) -> Vec<&TradeRecord> {
        self.trades.iter().filter(|t| t.pnl < 0.0).collect()
    }

    /// Get average winning trade
    pub fn avg_win(&self) -> f64 {
        let wins = self.winning_trades();
        if wins.is_empty() {
            0.0
        } else {
            wins.iter().map(|t| t.pnl).sum::<f64>() / wins.len() as f64
        }
    }

    /// Get average losing trade
    pub fn avg_loss(&self) -> f64 {
        let losses = self.losing_trades();
        if losses.is_empty() {
            0.0
        } else {
            losses.iter().map(|t| t.pnl).sum::<f64>() / losses.len() as f64
        }
    }

    /// Generate text summary
    pub fn summary(&self) -> String {
        format!(
            "=== Backtest Report: {} ===\n\
             Period: {} to {}\n\
             Initial Capital: ${:.2}\n\
             Final Capital: ${:.2}\n\
             Total Return: {:.2}%\n\
             \n\
             --- Performance Metrics ---\n\
             {}\n\
             \n\
             --- Trade Statistics ---\n\
             Total Trades: {}\n\
             Winning Trades: {}\n\
             Losing Trades: {}\n\
             Avg Win: ${:.2}\n\
             Avg Loss: ${:.2}",
            self.strategy_name,
            self.start_date.format("%Y-%m-%d"),
            self.end_date.format("%Y-%m-%d"),
            self.initial_capital,
            self.final_capital,
            (self.final_capital / self.initial_capital - 1.0) * 100.0,
            self.metrics.summary(),
            self.num_trades(),
            self.winning_trades().len(),
            self.losing_trades().len(),
            self.avg_win(),
            self.avg_loss()
        )
    }

    /// Export to CSV
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("symbol,entry_time,exit_time,entry_price,exit_price,size,direction,pnl,return_pct\n");

        for trade in &self.trades {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{}\n",
                trade.symbol,
                trade.entry_time.to_rfc3339(),
                trade.exit_time.to_rfc3339(),
                trade.entry_price,
                trade.exit_price,
                trade.size,
                trade.direction,
                trade.pnl,
                trade.return_pct
            ));
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_backtest_report() {
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 31, 0, 0, 0).unwrap();

        let trades = vec![
            TradeRecord {
                symbol: "BTCUSDT".to_string(),
                entry_time: start,
                exit_time: end,
                entry_price: 45000.0,
                exit_price: 47000.0,
                size: 1.0,
                direction: 1,
                pnl: 2000.0,
                return_pct: 0.0444,
            },
        ];

        let daily_returns = vec![0.01; 30];

        let report = BacktestReport::new(
            "Test Strategy",
            start,
            end,
            100000.0,
            134000.0,
            trades,
            daily_returns,
        );

        assert_eq!(report.num_trades(), 1);
        assert!(report.metrics.total_return > 0.0);
    }
}
