//! Metrics collection and reporting

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Trading performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    /// Total number of trades
    pub total_trades: usize,
    /// Winning trades
    pub winning_trades: usize,
    /// Losing trades
    pub losing_trades: usize,
    /// Total profit/loss
    pub total_pnl: f64,
    /// Largest win
    pub largest_win: f64,
    /// Largest loss
    pub largest_loss: f64,
    /// Average win
    pub average_win: f64,
    /// Average loss
    pub average_loss: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Sharpe ratio (if calculated)
    pub sharpe_ratio: Option<f64>,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// News items processed
    pub news_processed: usize,
    /// Signals generated
    pub signals_generated: usize,
    /// API calls made
    pub api_calls: usize,
    /// Average signal confidence
    pub avg_signal_confidence: f64,
    /// Start time
    pub start_time: Option<DateTime<Utc>>,
    /// Last update time
    pub last_update: Option<DateTime<Utc>>,
}

impl Metrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            start_time: Some(Utc::now()),
            ..Default::default()
        }
    }

    /// Calculate derived metrics
    pub fn calculate_derived(&mut self) {
        // Win rate
        if self.total_trades > 0 {
            self.win_rate = self.winning_trades as f64 / self.total_trades as f64;
        }

        // Profit factor
        let total_wins = self.average_win * self.winning_trades as f64;
        let total_losses = self.average_loss.abs() * self.losing_trades as f64;
        if total_losses > 0.0 {
            self.profit_factor = total_wins / total_losses;
        }

        self.last_update = Some(Utc::now());
    }

    /// Record a trade result
    pub fn record_trade(&mut self, pnl: f64) {
        self.total_trades += 1;
        self.total_pnl += pnl;

        if pnl > 0.0 {
            self.winning_trades += 1;
            if pnl > self.largest_win {
                self.largest_win = pnl;
            }
            // Update running average
            self.average_win = (self.average_win * (self.winning_trades - 1) as f64 + pnl)
                / self.winning_trades as f64;
        } else if pnl < 0.0 {
            self.losing_trades += 1;
            if pnl < self.largest_loss {
                self.largest_loss = pnl;
            }
            // Update running average
            self.average_loss = (self.average_loss * (self.losing_trades - 1) as f64 + pnl)
                / self.losing_trades as f64;
        }

        self.calculate_derived();
    }

    /// Get expectancy (average PnL per trade)
    pub fn expectancy(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        self.total_pnl / self.total_trades as f64
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Trades: {} | Win Rate: {:.1}% | PnL: ${:.2} | Profit Factor: {:.2} | Max DD: {:.1}%",
            self.total_trades,
            self.win_rate * 100.0,
            self.total_pnl,
            self.profit_factor,
            self.max_drawdown * 100.0
        )
    }
}

/// Thread-safe metrics recorder
#[derive(Debug, Clone)]
pub struct MetricsRecorder {
    metrics: Arc<Mutex<Metrics>>,
    labels: HashMap<String, String>,
}

impl Default for MetricsRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRecorder {
    /// Create a new metrics recorder
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(Metrics::new())),
            labels: HashMap::new(),
        }
    }

    /// Add a label
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Record a trade
    pub fn record_trade(&self, pnl: f64) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.record_trade(pnl);
        }
    }

    /// Increment news processed counter
    pub fn record_news_processed(&self, count: usize) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.news_processed += count;
            metrics.last_update = Some(Utc::now());
        }
    }

    /// Record signal generated
    pub fn record_signal(&self, confidence: f64) {
        if let Ok(mut metrics) = self.metrics.lock() {
            let prev_count = metrics.signals_generated;
            metrics.signals_generated += 1;

            // Update running average confidence
            metrics.avg_signal_confidence =
                (metrics.avg_signal_confidence * prev_count as f64 + confidence)
                    / metrics.signals_generated as f64;

            metrics.last_update = Some(Utc::now());
        }
    }

    /// Record API call
    pub fn record_api_call(&self) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.api_calls += 1;
        }
    }

    /// Update max drawdown
    pub fn update_drawdown(&self, drawdown: f64) {
        if let Ok(mut metrics) = self.metrics.lock() {
            if drawdown > metrics.max_drawdown {
                metrics.max_drawdown = drawdown;
            }
        }
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> Metrics {
        self.metrics.lock()
            .map(|m| m.clone())
            .unwrap_or_default()
    }

    /// Reset metrics
    pub fn reset(&self) {
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = Metrics::new();
        }
    }

    /// Get labels
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_record_trade() {
        let mut metrics = Metrics::new();

        metrics.record_trade(100.0);
        metrics.record_trade(-50.0);
        metrics.record_trade(75.0);

        assert_eq!(metrics.total_trades, 3);
        assert_eq!(metrics.winning_trades, 2);
        assert_eq!(metrics.losing_trades, 1);
        assert!((metrics.total_pnl - 125.0).abs() < 0.01);
    }

    #[test]
    fn test_win_rate_calculation() {
        let mut metrics = Metrics::new();

        metrics.record_trade(100.0);
        metrics.record_trade(50.0);
        metrics.record_trade(-30.0);
        metrics.record_trade(-20.0);

        // 2 wins, 2 losses = 50% win rate
        assert!((metrics.win_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_metrics_recorder_thread_safety() {
        let recorder = MetricsRecorder::new();

        // Simulate concurrent access
        recorder.record_trade(100.0);
        recorder.record_news_processed(10);
        recorder.record_signal(0.8);
        recorder.record_api_call();

        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.total_trades, 1);
        assert_eq!(snapshot.news_processed, 10);
        assert_eq!(snapshot.signals_generated, 1);
        assert_eq!(snapshot.api_calls, 1);
    }

    #[test]
    fn test_expectancy() {
        let mut metrics = Metrics::new();
        metrics.record_trade(100.0);
        metrics.record_trade(-50.0);

        // Total PnL = 50, 2 trades = 25 expectancy
        assert!((metrics.expectancy() - 25.0).abs() < 0.01);
    }
}
