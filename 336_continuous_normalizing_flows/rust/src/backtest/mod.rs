//! # Backtesting Module
//!
//! Backtesting framework for CNF trading strategies.

use chrono::{DateTime, Utc};
use ndarray::Array1;

use crate::trading::{CNFTrader, TradingSignal, SignalType};
use crate::utils::Candle;

/// Single backtest result entry
#[derive(Debug, Clone)]
pub struct BacktestEntry {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Close price
    pub close: f64,
    /// Trading signal
    pub signal: SignalType,
    /// Signal confidence
    pub confidence: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Expected return
    pub expected_return: f64,
    /// Current position
    pub position: f64,
    /// Period P&L
    pub pnl: f64,
    /// Cumulative P&L
    pub cumulative_pnl: f64,
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResults {
    /// Individual entries
    pub entries: Vec<BacktestEntry>,
    /// Performance metrics
    pub metrics: BacktestMetrics,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    /// Total return
    pub total_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Number of trades
    pub n_trades: usize,
    /// Average likelihood
    pub avg_likelihood: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Percentage of time in distribution
    pub in_distribution_ratio: f64,
}

/// Backtester for CNF trading
pub struct Backtester {
    /// Lookback period for features
    lookback: usize,
    /// Transaction costs (as fraction)
    transaction_cost: f64,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            transaction_cost: 0.0,
        }
    }

    /// Set transaction costs
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Run backtest on candle data
    pub fn run(&self, trader: &mut CNFTrader, candles: &[Candle]) -> BacktestResults {
        let warmup = self.lookback + 50; // Extra warmup for indicators

        assert!(candles.len() > warmup, "Not enough data for backtest");

        let mut entries = Vec::new();
        let mut position = 0.0;
        let mut cumulative_pnl = 0.0;
        let mut prev_position = 0.0;

        for i in warmup..candles.len() {
            let window = &candles[i - self.lookback..i];
            let signal_info = trader.generate_signal(window);

            // Calculate P&L
            let pnl = if i > warmup {
                let ret = candles[i].close / candles[i - 1].close - 1.0;
                let gross_pnl = prev_position * ret;

                // Apply transaction costs
                let position_change = (signal_info.position_size() - prev_position).abs();
                let costs = position_change * self.transaction_cost;

                gross_pnl - costs
            } else {
                0.0
            };

            cumulative_pnl += pnl;

            // Update position
            prev_position = position;
            position = signal_info.position_size();

            // Reduce position on regime change
            if signal_info.regime_change {
                position *= 0.5;
            }

            entries.push(BacktestEntry {
                timestamp: candles[i].timestamp,
                close: candles[i].close,
                signal: signal_info.signal,
                confidence: signal_info.confidence,
                log_likelihood: signal_info.log_likelihood,
                expected_return: signal_info.expected_return,
                position,
                pnl,
                cumulative_pnl,
            });
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&entries);

        BacktestResults { entries, metrics }
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self, entries: &[BacktestEntry]) -> BacktestMetrics {
        if entries.is_empty() {
            return BacktestMetrics {
                total_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                n_trades: 0,
                avg_likelihood: 0.0,
                avg_confidence: 0.0,
                in_distribution_ratio: 0.0,
            };
        }

        // Extract P&L series
        let pnls: Vec<f64> = entries.iter().map(|e| e.pnl).collect();

        // Total return
        let total_return = entries.last().map(|e| e.cumulative_pnl).unwrap_or(0.0);

        // Sharpe ratio
        let mean_pnl = pnls.iter().sum::<f64>() / pnls.len() as f64;
        let std_pnl = std_dev(&pnls);
        let sharpe_ratio = if std_pnl > 1e-8 {
            mean_pnl / std_pnl * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside: Vec<f64> = pnls.iter().filter(|&&p| p < 0.0).cloned().collect();
        let downside_std = std_dev(&downside);
        let sortino_ratio = if downside_std > 1e-8 {
            mean_pnl / downside_std * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut max_pnl = f64::NEG_INFINITY;
        let mut max_drawdown = 0.0_f64;
        for entry in entries {
            max_pnl = max_pnl.max(entry.cumulative_pnl);
            let drawdown = entry.cumulative_pnl - max_pnl;
            max_drawdown = max_drawdown.min(drawdown);
        }

        // Win rate
        let trading_pnls: Vec<f64> = pnls.iter()
            .filter(|&&p| p.abs() > 1e-10)
            .cloned()
            .collect();
        let wins = trading_pnls.iter().filter(|&&p| p > 0.0).count();
        let win_rate = if !trading_pnls.is_empty() {
            wins as f64 / trading_pnls.len() as f64
        } else {
            0.0
        };

        // Number of trades (position changes)
        let n_trades = entries.windows(2)
            .filter(|w| (w[1].position - w[0].position).abs() > 1e-8)
            .count();

        // Average likelihood
        let avg_likelihood = entries.iter()
            .map(|e| e.log_likelihood)
            .sum::<f64>() / entries.len() as f64;

        // Average confidence
        let avg_confidence = entries.iter()
            .map(|e| e.confidence)
            .sum::<f64>() / entries.len() as f64;

        // In-distribution ratio (likelihood above threshold)
        let in_dist_count = entries.iter()
            .filter(|e| e.log_likelihood > -10.0)
            .count();
        let in_distribution_ratio = in_dist_count as f64 / entries.len() as f64;

        BacktestMetrics {
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            n_trades,
            avg_likelihood,
            avg_confidence,
            in_distribution_ratio,
        }
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(20)
    }
}

/// Calculate standard deviation
fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

impl BacktestMetrics {
    /// Print summary
    pub fn print_summary(&self) {
        println!("=== Backtest Results ===");
        println!("Total Return:      {:.4}", self.total_return);
        println!("Sharpe Ratio:      {:.4}", self.sharpe_ratio);
        println!("Sortino Ratio:     {:.4}", self.sortino_ratio);
        println!("Max Drawdown:      {:.4}", self.max_drawdown);
        println!("Win Rate:          {:.2}%", self.win_rate * 100.0);
        println!("Number of Trades:  {}", self.n_trades);
        println!("Avg Likelihood:    {:.4}", self.avg_likelihood);
        println!("Avg Confidence:    {:.4}", self.avg_confidence);
        println!("In-Dist Ratio:     {:.2}%", self.in_distribution_ratio * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnf::ContinuousNormalizingFlow;
    use crate::utils::generate_synthetic_candles;

    #[test]
    fn test_backtester() {
        let cnf = ContinuousNormalizingFlow::new(9, 32, 2);
        let mut trader = CNFTrader::new(cnf);

        let candles = generate_synthetic_candles(200, 100.0);
        let backtester = Backtester::new(20);

        let results = backtester.run(&mut trader, &candles);

        assert!(!results.entries.is_empty());
        assert!(results.metrics.total_return.is_finite());
        assert!(results.metrics.sharpe_ratio.is_finite());
    }

    #[test]
    fn test_metrics_calculation() {
        let cnf = ContinuousNormalizingFlow::new(9, 32, 2);
        let mut trader = CNFTrader::new(cnf);

        let candles = generate_synthetic_candles(200, 100.0);
        let backtester = Backtester::new(20);

        let results = backtester.run(&mut trader, &candles);
        let metrics = &results.metrics;

        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0);
        assert!(metrics.in_distribution_ratio >= 0.0 && metrics.in_distribution_ratio <= 1.0);
        assert!(metrics.max_drawdown <= 0.0); // Drawdown is negative or zero
    }
}
