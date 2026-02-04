//! Backtest report and metrics.

use super::engine::Trade;
use tabled::{Table, Tabled};

/// Backtest report with performance metrics
#[derive(Debug, Clone)]
pub struct BacktestReport {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub avg_trade_pnl: f64,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<Trade>,
}

impl BacktestReport {
    /// Print formatted report
    pub fn print(&self) {
        println!("\n{}", "=".repeat(60));
        println!("BACKTEST REPORT");
        println!("{}", "=".repeat(60));

        println!("\nPerformance Metrics:");
        println!("  Total Return:    {:>10.2}%", self.total_return * 100.0);
        println!("  Sharpe Ratio:    {:>10.2}", self.sharpe_ratio);
        println!("  Sortino Ratio:   {:>10.2}", self.sortino_ratio);
        println!("  Max Drawdown:    {:>10.2}%", self.max_drawdown * 100.0);

        println!("\nTrade Statistics:");
        println!("  Total Trades:    {:>10}", self.total_trades);
        println!("  Win Rate:        {:>10.2}%", self.win_rate * 100.0);
        println!("  Profit Factor:   {:>10.2}", self.profit_factor);
        println!("  Avg Trade PnL:   {:>10.2}", self.avg_trade_pnl);
    }

    /// Get metrics by confidence level
    pub fn metrics_by_confidence(&self, n_bins: usize) -> Vec<ConfidenceBinMetrics> {
        if self.trades.is_empty() {
            return Vec::new();
        }

        let mut confidences: Vec<f64> = self.trades.iter().map(|t| t.confidence).collect();
        confidences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let bin_size = (self.trades.len() + n_bins - 1) / n_bins;
        let mut bins: Vec<Vec<&Trade>> = vec![Vec::new(); n_bins];

        let sorted_trades: Vec<&Trade> = {
            let mut sorted: Vec<&Trade> = self.trades.iter().collect();
            sorted.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal));
            sorted
        };

        for (i, trade) in sorted_trades.iter().enumerate() {
            let bin_idx = (i / bin_size).min(n_bins - 1);
            bins[bin_idx].push(trade);
        }

        bins.iter()
            .enumerate()
            .filter(|(_, trades)| !trades.is_empty())
            .map(|(i, trades)| {
                let min_conf = trades.iter().map(|t| t.confidence).fold(f64::MAX, f64::min);
                let max_conf = trades.iter().map(|t| t.confidence).fold(f64::MIN, f64::max);
                let num_trades = trades.len();

                let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
                let win_rate = wins as f64 / num_trades as f64;

                let avg_return: f64 = trades.iter().map(|t| t.pnl_pct).sum::<f64>() / num_trades as f64;

                let mean = avg_return;
                let variance: f64 = trades.iter().map(|t| (t.pnl_pct - mean).powi(2)).sum::<f64>()
                    / num_trades as f64;
                let std = variance.sqrt();
                let sharpe = if std > 0.0 { mean / std } else { 0.0 };

                ConfidenceBinMetrics {
                    bin: i + 1,
                    min_confidence: min_conf,
                    max_confidence: max_conf,
                    num_trades,
                    win_rate,
                    avg_return,
                    sharpe,
                }
            })
            .collect()
    }

    /// Print metrics by confidence
    pub fn print_confidence_analysis(&self) {
        let metrics = self.metrics_by_confidence(5);

        if metrics.is_empty() {
            println!("\nNo trades to analyze by confidence level.");
            return;
        }

        println!("\nPerformance by Confidence Level:");
        println!("{:>5} {:>12} {:>12} {:>8} {:>10} {:>12} {:>8}",
            "Bin", "Min Conf", "Max Conf", "Trades", "Win Rate", "Avg Return", "Sharpe");
        println!("{}", "-".repeat(70));

        for m in &metrics {
            println!("{:>5} {:>12.2}% {:>12.2}% {:>8} {:>10.2}% {:>12.4}% {:>8.2}",
                m.bin,
                m.min_confidence * 100.0,
                m.max_confidence * 100.0,
                m.num_trades,
                m.win_rate * 100.0,
                m.avg_return * 100.0,
                m.sharpe);
        }
    }
}

/// Metrics for a confidence bin
#[derive(Debug, Clone)]
pub struct ConfidenceBinMetrics {
    pub bin: usize,
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub avg_return: f64,
    pub sharpe: f64,
}
