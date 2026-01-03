//! # Backtesting Framework
//!
//! Framework for backtesting RealNVP trading strategies.

use ndarray::Array1;
use crate::trading::{RealNVPTrader, TradingSignal};
use crate::utils::{Candle, compute_market_state};

/// Single backtest result entry
#[derive(Debug, Clone)]
pub struct BacktestEntry {
    /// Timestamp (index)
    pub index: usize,
    /// Close price
    pub price: f64,
    /// Trading signal
    pub signal: f64,
    /// Confidence
    pub confidence: f64,
    /// Log probability
    pub log_prob: f64,
    /// Whether in distribution
    pub in_distribution: bool,
    /// Current position
    pub position: f64,
    /// Period PnL
    pub pnl: f64,
    /// Cumulative PnL
    pub cumulative_pnl: f64,
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResults {
    /// Individual entries
    pub entries: Vec<BacktestEntry>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized Sharpe ratio
    pub sharpe_ratio: f64,
    /// Annualized Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Average in-distribution ratio
    pub in_distribution_ratio: f64,
    /// Average log probability
    pub avg_log_prob: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Profit factor
    pub profit_factor: f64,
}

/// Backtester for RealNVP trading strategy
pub struct Backtester {
    /// Lookback period for features
    lookback: usize,
    /// Warmup period
    warmup: usize,
    /// Annualization factor (252 for daily, 365*24 for hourly)
    annualization_factor: f64,
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new()
    }
}

impl Backtester {
    /// Create a new backtester with default settings
    pub fn new() -> Self {
        Self {
            lookback: 20,
            warmup: 100,
            annualization_factor: 252.0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(lookback: usize, warmup: usize, annualization_factor: f64) -> Self {
        Self {
            lookback,
            warmup,
            annualization_factor,
        }
    }

    /// Run backtest on price data
    pub fn run(&self, trader: &RealNVPTrader, candles: &[Candle]) -> BacktestResults {
        let mut entries = Vec::new();
        let mut position = 0.0;
        let mut cumulative_pnl = 0.0;

        let start_idx = self.warmup.max(self.lookback);

        for i in start_idx..candles.len() {
            let window_start = if i >= self.lookback { i - self.lookback } else { 0 };
            let window = &candles[window_start..i];

            // Compute market state
            let state = compute_market_state(window);

            // Get signal
            let signal_info = trader.generate_signal_from_state(&state);

            // Calculate PnL
            let pnl = if i > start_idx && position != 0.0 {
                let daily_return = candles[i].close / candles[i - 1].close - 1.0;
                position * daily_return
            } else {
                0.0
            };
            cumulative_pnl += pnl;

            // Record entry
            entries.push(BacktestEntry {
                index: i,
                price: candles[i].close,
                signal: signal_info.position_size(),
                confidence: signal_info.confidence,
                log_prob: signal_info.log_prob,
                in_distribution: signal_info.in_distribution,
                position,
                pnl,
                cumulative_pnl,
            });

            // Update position for next period
            position = signal_info.position_size();
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&entries);

        BacktestResults { entries, metrics }
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self, entries: &[BacktestEntry]) -> PerformanceMetrics {
        if entries.is_empty() {
            return PerformanceMetrics {
                total_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                in_distribution_ratio: 0.0,
                avg_log_prob: 0.0,
                avg_confidence: 0.0,
                num_trades: 0,
                profit_factor: 0.0,
            };
        }

        let returns: Vec<f64> = entries.iter().map(|e| e.pnl).collect();
        let n = returns.len() as f64;

        // Total return
        let total_return = entries.last().map(|e| e.cumulative_pnl).unwrap_or(0.0);

        // Mean and std of returns
        let mean_return = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / n;
        let std_return = variance.sqrt();

        // Sharpe ratio
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * self.annualization_factor.sqrt()
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if !downside.is_empty() {
            let variance: f64 = downside.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * self.annualization_factor.sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = 0.0_f64;
        let mut max_drawdown = 0.0_f64;
        for entry in entries {
            peak = peak.max(entry.cumulative_pnl);
            let drawdown = peak - entry.cumulative_pnl;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Win rate
        let trading_returns: Vec<f64> = returns.iter().filter(|&&r| r != 0.0).cloned().collect();
        let win_rate = if !trading_returns.is_empty() {
            trading_returns.iter().filter(|&&r| r > 0.0).count() as f64
                / trading_returns.len() as f64
        } else {
            0.0
        };

        // In-distribution ratio
        let in_distribution_ratio = entries.iter()
            .filter(|e| e.in_distribution)
            .count() as f64 / n;

        // Average log probability
        let avg_log_prob = entries.iter()
            .map(|e| e.log_prob)
            .sum::<f64>() / n;

        // Average confidence
        let avg_confidence = entries.iter()
            .map(|e| e.confidence)
            .sum::<f64>() / n;

        // Number of trades (position changes)
        let num_trades = entries.windows(2)
            .filter(|w| (w[0].signal - w[1].signal).abs() > 0.01)
            .count();

        // Profit factor
        let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        PerformanceMetrics {
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            in_distribution_ratio,
            avg_log_prob,
            avg_confidence,
            num_trades,
            profit_factor,
        }
    }
}

impl BacktestResults {
    /// Print summary to console
    pub fn print_summary(&self) {
        println!("=== Backtest Results ===");
        println!("Total Return: {:.4}%", self.metrics.total_return * 100.0);
        println!("Sharpe Ratio: {:.4}", self.metrics.sharpe_ratio);
        println!("Sortino Ratio: {:.4}", self.metrics.sortino_ratio);
        println!("Max Drawdown: {:.4}%", self.metrics.max_drawdown * 100.0);
        println!("Win Rate: {:.2}%", self.metrics.win_rate * 100.0);
        println!("Number of Trades: {}", self.metrics.num_trades);
        println!("Profit Factor: {:.4}", self.metrics.profit_factor);
        println!("In-Distribution Ratio: {:.2}%", self.metrics.in_distribution_ratio * 100.0);
        println!("Avg Log Probability: {:.4}", self.metrics.avg_log_prob);
        println!("Avg Confidence: {:.4}", self.metrics.avg_confidence);
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> Vec<f64> {
        self.entries.iter().map(|e| e.cumulative_pnl).collect()
    }

    /// Get returns
    pub fn returns(&self) -> Vec<f64> {
        self.entries.iter().map(|e| e.pnl).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::flow::RealNVP;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        let base_price = 100.0;
        (0..n).map(|i| {
            let price = base_price + (i as f64 * 0.1).sin() * 5.0;
            Candle {
                timestamp: Utc::now(),
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 1000.0 + (i as f64 * 10.0),
            }
        }).collect()
    }

    #[test]
    fn test_backtester_run() {
        let flow = RealNVP::new(10, 32, 4);
        let trader = RealNVPTrader::new(flow);
        let backtester = Backtester::new();

        let candles = create_test_candles(200);
        let results = backtester.run(&trader, &candles);

        assert!(!results.entries.is_empty());
        assert!(results.metrics.sharpe_ratio.is_finite());
    }

    #[test]
    fn test_metrics_calculation() {
        let entries = vec![
            BacktestEntry {
                index: 0,
                price: 100.0,
                signal: 0.5,
                confidence: 0.8,
                log_prob: -5.0,
                in_distribution: true,
                position: 0.5,
                pnl: 0.01,
                cumulative_pnl: 0.01,
            },
            BacktestEntry {
                index: 1,
                price: 101.0,
                signal: 0.3,
                confidence: 0.7,
                log_prob: -6.0,
                in_distribution: true,
                position: 0.3,
                pnl: -0.005,
                cumulative_pnl: 0.005,
            },
        ];

        let backtester = Backtester::new();
        let metrics = backtester.calculate_metrics(&entries);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0);
    }
}
