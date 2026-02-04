//! Backtest report and metrics

use serde::{Deserialize, Serialize};

/// Results from a backtest run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate (winning / total)
    pub win_rate: f64,
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Mean uncertainty of trades
    pub mean_uncertainty: f64,
    /// 95% confidence interval coverage
    pub uncertainty_coverage_95: f64,
    /// Correlation between uncertainty and error
    pub uncertainty_correlation: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
}

impl Default for BacktestReport {
    fn default() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            mean_uncertainty: 0.0,
            uncertainty_coverage_95: 0.0,
            uncertainty_correlation: 0.0,
            profit_factor: 0.0,
        }
    }
}

impl BacktestReport {
    /// Print summary of the backtest
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("BACKTEST RESULTS");
        println!("{}", "=".repeat(60));

        println!("\n--- Trade Statistics ---");
        println!("Total Trades:    {}", self.total_trades);
        println!("Winning Trades:  {}", self.winning_trades);
        println!("Losing Trades:   {}", self.losing_trades);
        println!("Win Rate:        {:.2}%", self.win_rate * 100.0);

        println!("\n--- Performance Metrics ---");
        println!("Total Return:      {:.2}%", self.total_return * 100.0);
        println!("Annualized Return: {:.2}%", self.annualized_return * 100.0);
        println!("Sharpe Ratio:      {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio:     {:.2}", self.sortino_ratio);
        println!("Max Drawdown:      {:.2}%", self.max_drawdown * 100.0);
        println!("Profit Factor:     {:.2}", self.profit_factor);

        println!("\n--- Uncertainty Metrics ---");
        println!("Mean Uncertainty:       {:.4}", self.mean_uncertainty);
        println!("95% Coverage:           {:.2}%", self.uncertainty_coverage_95 * 100.0);
        println!("Uncertainty-Error Corr: {:.4}", self.uncertainty_correlation);

        println!("{}", "=".repeat(60));
    }

    /// Check if the strategy is profitable
    pub fn is_profitable(&self) -> bool {
        self.total_return > 0.0
    }

    /// Check if uncertainty estimates are well-calibrated
    pub fn is_well_calibrated(&self, tolerance: f64) -> bool {
        (self.uncertainty_coverage_95 - 0.95).abs() < tolerance
    }
}
