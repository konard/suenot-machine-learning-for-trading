//! Backtest results and analytics.

use serde::{Deserialize, Serialize};

/// A completed trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trading symbol
    pub symbol: String,
    /// Long (true) or short (false)
    pub is_long: bool,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Entry timestamp (Unix millis)
    pub entry_time: i64,
    /// Exit timestamp (Unix millis)
    pub exit_time: i64,
    /// Position size
    pub size: f64,
    /// Realized PnL
    pub pnl: f64,
    /// Reason for exit
    pub exit_reason: String,
}

impl Trade {
    /// Calculate return percentage.
    pub fn return_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            (self.exit_price - self.entry_price) / self.entry_price * 100.0
                * if self.is_long { 1.0 } else { -1.0 }
        } else {
            0.0
        }
    }

    /// Calculate holding period in minutes.
    pub fn holding_period_minutes(&self) -> i64 {
        (self.exit_time - self.entry_time) / 60000
    }
}

/// Complete backtest results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    /// Starting capital
    pub initial_capital: f64,
    /// Final portfolio value
    pub final_value: f64,
    /// Total return (as fraction, e.g., 0.15 for 15%)
    pub total_return: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate (as fraction)
    pub win_rate: f64,
    /// Average winning trade PnL
    pub avg_win: f64,
    /// Average losing trade PnL (absolute)
    pub avg_loss: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Maximum drawdown (as fraction)
    pub max_drawdown: f64,
    /// Annualized Sharpe ratio
    pub sharpe_ratio: f64,
    /// Buy and hold return for comparison
    pub buy_hold_return: f64,
    /// List of all trades
    pub trades: Vec<Trade>,
    /// Equity curve (portfolio value at each step)
    pub equity_curve: Vec<f64>,
}

impl BacktestResults {
    /// Check if strategy outperformed buy & hold.
    pub fn outperformed_benchmark(&self) -> bool {
        self.total_return > self.buy_hold_return
    }

    /// Calculate risk-adjusted return (return / max drawdown).
    pub fn calmar_ratio(&self) -> f64 {
        if self.max_drawdown > 0.0 {
            self.total_return / self.max_drawdown
        } else {
            0.0
        }
    }

    /// Get trades by exit reason.
    pub fn trades_by_reason(&self, reason: &str) -> Vec<&Trade> {
        self.trades
            .iter()
            .filter(|t| t.exit_reason == reason)
            .collect()
    }

    /// Calculate average holding period in minutes.
    pub fn avg_holding_period(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        self.trades.iter().map(|t| t.holding_period_minutes() as f64).sum::<f64>()
            / self.trades.len() as f64
    }

    /// Get monthly returns (approximate).
    pub fn monthly_returns(&self) -> Vec<f64> {
        if self.equity_curve.len() < 2 {
            return Vec::new();
        }

        // Assuming 15-minute klines, ~2880 per month
        let periods_per_month = 2880;
        let mut monthly = Vec::new();

        for chunk in self.equity_curve.chunks(periods_per_month) {
            if chunk.len() >= 2 {
                let start = chunk[0];
                let end = chunk[chunk.len() - 1];
                monthly.push((end - start) / start);
            }
        }

        monthly
    }

    /// Create a summary report.
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("                    BACKTEST RESULTS SUMMARY                    \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str("📊 PERFORMANCE METRICS\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str(&format!("  Initial Capital:     ${:>12.2}\n", self.initial_capital));
        report.push_str(&format!("  Final Value:         ${:>12.2}\n", self.final_value));
        report.push_str(&format!("  Total Return:        {:>12.2}%\n", self.total_return * 100.0));
        report.push_str(&format!("  Buy & Hold Return:   {:>12.2}%\n", self.buy_hold_return * 100.0));
        report.push_str(&format!(
            "  Outperformed B&H:    {:>12}\n",
            if self.outperformed_benchmark() { "YES ✓" } else { "NO ✗" }
        ));
        report.push_str("\n");

        report.push_str("📈 RISK METRICS\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str(&format!("  Max Drawdown:        {:>12.2}%\n", self.max_drawdown * 100.0));
        report.push_str(&format!("  Sharpe Ratio:        {:>12.2}\n", self.sharpe_ratio));
        report.push_str(&format!("  Calmar Ratio:        {:>12.2}\n", self.calmar_ratio()));
        report.push_str("\n");

        report.push_str("📋 TRADE STATISTICS\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str(&format!("  Total Trades:        {:>12}\n", self.total_trades));
        report.push_str(&format!("  Winning Trades:      {:>12}\n", self.winning_trades));
        report.push_str(&format!("  Losing Trades:       {:>12}\n", self.losing_trades));
        report.push_str(&format!("  Win Rate:            {:>12.2}%\n", self.win_rate * 100.0));
        report.push_str(&format!("  Average Win:         ${:>12.2}\n", self.avg_win));
        report.push_str(&format!("  Average Loss:        ${:>12.2}\n", self.avg_loss));
        report.push_str(&format!("  Profit Factor:       {:>12.2}\n", self.profit_factor));
        report.push_str(&format!("  Avg Holding Period:  {:>10.0} min\n", self.avg_holding_period()));
        report.push_str("\n");

        // Exit reason breakdown
        let stop_losses = self.trades_by_reason("stop_loss").len();
        let take_profits = self.trades_by_reason("take_profit").len();
        let reversals = self.trades_by_reason("signal_reversal").len();
        let end_exits = self.trades_by_reason("end_of_backtest").len();

        report.push_str("🚪 EXIT REASONS\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str(&format!("  Stop Loss:           {:>12}\n", stop_losses));
        report.push_str(&format!("  Take Profit:         {:>12}\n", take_profits));
        report.push_str(&format!("  Signal Reversal:     {:>12}\n", reversals));
        report.push_str(&format!("  End of Backtest:     {:>12}\n", end_exits));
        report.push_str("\n");

        report.push_str("═══════════════════════════════════════════════════════════════\n");

        report
    }
}

impl std::fmt::Display for BacktestResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary_report())
    }
}

/// Comparison between multiple backtest runs.
#[derive(Debug, Clone)]
pub struct BacktestComparison {
    /// Results with labels
    pub results: Vec<(String, BacktestResults)>,
}

impl BacktestComparison {
    /// Create a new comparison.
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    /// Add a result to comparison.
    pub fn add(&mut self, label: String, result: BacktestResults) {
        self.results.push((label, result));
    }

    /// Get best performing strategy by total return.
    pub fn best_by_return(&self) -> Option<&(String, BacktestResults)> {
        self.results
            .iter()
            .max_by(|a, b| a.1.total_return.partial_cmp(&b.1.total_return).unwrap())
    }

    /// Get best performing strategy by Sharpe ratio.
    pub fn best_by_sharpe(&self) -> Option<&(String, BacktestResults)> {
        self.results
            .iter()
            .max_by(|a, b| a.1.sharpe_ratio.partial_cmp(&b.1.sharpe_ratio).unwrap())
    }

    /// Generate comparison table.
    pub fn comparison_table(&self) -> String {
        let mut table = String::new();

        table.push_str("┌──────────────────┬──────────┬──────────┬──────────┬──────────┐\n");
        table.push_str("│     Strategy     │  Return  │  Sharpe  │ Win Rate │ Max DD   │\n");
        table.push_str("├──────────────────┼──────────┼──────────┼──────────┼──────────┤\n");

        for (label, result) in &self.results {
            table.push_str(&format!(
                "│ {:16} │ {:>7.2}% │ {:>8.2} │ {:>7.2}% │ {:>7.2}% │\n",
                &label[..label.len().min(16)],
                result.total_return * 100.0,
                result.sharpe_ratio,
                result.win_rate * 100.0,
                result.max_drawdown * 100.0
            ));
        }

        table.push_str("└──────────────────┴──────────┴──────────┴──────────┴──────────┘\n");

        table
    }
}

impl Default for BacktestComparison {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_calculations() {
        let trade = Trade {
            symbol: "BTCUSDT".to_string(),
            is_long: true,
            entry_price: 50000.0,
            exit_price: 51000.0,
            entry_time: 1000000,
            exit_time: 1060000, // 1 minute later
            size: 1000.0,
            pnl: 20.0,
            exit_reason: "take_profit".to_string(),
        };

        assert!((trade.return_pct() - 2.0).abs() < 0.01);
        assert_eq!(trade.holding_period_minutes(), 1);
    }

    #[test]
    fn test_results_summary() {
        let results = BacktestResults {
            initial_capital: 100000.0,
            final_value: 115000.0,
            total_return: 0.15,
            total_trades: 10,
            winning_trades: 6,
            losing_trades: 4,
            win_rate: 0.6,
            avg_win: 500.0,
            avg_loss: 200.0,
            profit_factor: 1.5,
            max_drawdown: 0.05,
            sharpe_ratio: 1.2,
            buy_hold_return: 0.10,
            trades: Vec::new(),
            equity_curve: vec![100000.0, 115000.0],
        };

        assert!(results.outperformed_benchmark());
        assert!((results.calmar_ratio() - 3.0).abs() < 0.01);

        let report = results.summary_report();
        assert!(report.contains("BACKTEST RESULTS SUMMARY"));
        assert!(report.contains("15.00%"));
    }
}
