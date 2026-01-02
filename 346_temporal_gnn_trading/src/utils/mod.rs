//! Utilities module for metrics and helper functions
//!
//! This module provides performance tracking, statistical utilities,
//! and other helper functions.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Performance metrics tracker
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Returns history
    returns: VecDeque<f64>,
    /// Equity curve
    equity_curve: Vec<f64>,
    /// Initial capital
    initial_capital: f64,
    /// Current equity
    current_equity: f64,
    /// Peak equity (for drawdown calculation)
    peak_equity: f64,
    /// Window size for rolling calculations
    window_size: usize,
}

impl PerformanceTracker {
    /// Create new performance tracker
    pub fn new(initial_capital: f64, window_size: usize) -> Self {
        Self {
            returns: VecDeque::with_capacity(window_size),
            equity_curve: vec![initial_capital],
            initial_capital,
            current_equity: initial_capital,
            peak_equity: initial_capital,
            window_size,
        }
    }

    /// Update with new return
    pub fn update(&mut self, pnl: f64) {
        // Update equity
        self.current_equity += pnl;
        self.equity_curve.push(self.current_equity);

        // Update peak
        if self.current_equity > self.peak_equity {
            self.peak_equity = self.current_equity;
        }

        // Calculate and store return
        let return_pct = if self.equity_curve.len() > 1 {
            let prev = self.equity_curve[self.equity_curve.len() - 2];
            if prev > 0.0 { pnl / prev } else { 0.0 }
        } else {
            0.0
        };

        if self.returns.len() >= self.window_size {
            self.returns.pop_front();
        }
        self.returns.push_back(return_pct);
    }

    /// Calculate Sharpe ratio (annualized)
    pub fn sharpe_ratio(&self, risk_free_rate: f64, periods_per_year: f64) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let mean_return: f64 = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let excess_return = mean_return - risk_free_rate / periods_per_year;

        let variance: f64 = self.returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / self.returns.len() as f64;

        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            excess_return / std_dev * periods_per_year.sqrt()
        } else {
            0.0
        }
    }

    /// Calculate Sortino ratio (annualized)
    pub fn sortino_ratio(&self, risk_free_rate: f64, periods_per_year: f64) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let mean_return: f64 = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let excess_return = mean_return - risk_free_rate / periods_per_year;

        // Downside deviation (only negative returns)
        let downside_returns: Vec<f64> = self.returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance: f64 = downside_returns
            .iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / downside_returns.len() as f64;

        let downside_dev = downside_variance.sqrt();

        if downside_dev > 0.0 {
            excess_return / downside_dev * periods_per_year.sqrt()
        } else {
            0.0
        }
    }

    /// Calculate current drawdown
    pub fn current_drawdown(&self) -> f64 {
        if self.peak_equity > 0.0 {
            (self.peak_equity - self.current_equity) / self.peak_equity
        } else {
            0.0
        }
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown(&self) -> f64 {
        if self.equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_dd = 0.0;
        let mut peak = self.equity_curve[0];

        for &equity in &self.equity_curve {
            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    /// Calculate Calmar ratio
    pub fn calmar_ratio(&self, periods_per_year: f64) -> f64 {
        let total_return = (self.current_equity - self.initial_capital) / self.initial_capital;
        let max_dd = self.max_drawdown();

        if max_dd > 0.0 {
            total_return / max_dd * periods_per_year / self.equity_curve.len() as f64
        } else {
            0.0
        }
    }

    /// Get total return
    pub fn total_return(&self) -> f64 {
        (self.current_equity - self.initial_capital) / self.initial_capital
    }

    /// Get current equity
    pub fn current_equity(&self) -> f64 {
        self.current_equity
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    /// Get all metrics as a struct
    pub fn get_metrics(&self, risk_free_rate: f64, periods_per_year: f64) -> Metrics {
        Metrics {
            total_return: self.total_return(),
            sharpe_ratio: self.sharpe_ratio(risk_free_rate, periods_per_year),
            sortino_ratio: self.sortino_ratio(risk_free_rate, periods_per_year),
            max_drawdown: self.max_drawdown(),
            current_drawdown: self.current_drawdown(),
            calmar_ratio: self.calmar_ratio(periods_per_year),
            num_periods: self.equity_curve.len(),
        }
    }
}

/// Performance metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Total return (percentage)
    pub total_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Current drawdown
    pub current_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Number of periods
    pub num_periods: usize,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            current_drawdown: 0.0,
            calmar_ratio: 0.0,
            num_periods: 0,
        }
    }
}

/// Rolling statistics calculator
#[derive(Debug)]
pub struct RollingStats {
    /// Values buffer
    values: VecDeque<f64>,
    /// Window size
    window_size: usize,
    /// Running sum
    sum: f64,
    /// Running sum of squares
    sum_sq: f64,
}

impl RollingStats {
    /// Create new rolling stats
    pub fn new(window_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(window_size),
            window_size,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Update with new value
    pub fn update(&mut self, value: f64) {
        if self.values.len() >= self.window_size {
            let old = self.values.pop_front().unwrap();
            self.sum -= old;
            self.sum_sq -= old * old;
        }

        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    /// Get mean
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Get variance
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }

        let n = self.values.len() as f64;
        let mean = self.mean();
        self.sum_sq / n - mean * mean
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get z-score for a value
    pub fn z_score(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std > 0.0 {
            (value - self.mean()) / std
        } else {
            0.0
        }
    }

    /// Get minimum value
    pub fn min(&self) -> Option<f64> {
        self.values.iter().cloned().fold(None, |acc, v| {
            match acc {
                None => Some(v),
                Some(min) => Some(min.min(v)),
            }
        })
    }

    /// Get maximum value
    pub fn max(&self) -> Option<f64> {
        self.values.iter().cloned().fold(None, |acc, v| {
            match acc {
                None => Some(v),
                Some(max) => Some(max.max(v)),
            }
        })
    }

    /// Check if full
    pub fn is_full(&self) -> bool {
        self.values.len() >= self.window_size
    }

    /// Get number of values
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Exponential moving average
#[derive(Debug)]
pub struct EMA {
    /// Current EMA value
    value: f64,
    /// Alpha (smoothing factor)
    alpha: f64,
    /// Whether initialized
    initialized: bool,
}

impl EMA {
    /// Create new EMA with period
    pub fn new(period: usize) -> Self {
        Self {
            value: 0.0,
            alpha: 2.0 / (period as f64 + 1.0),
            initialized: false,
        }
    }

    /// Create EMA with custom alpha
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.0, 1.0),
            initialized: false,
        }
    }

    /// Update with new value
    pub fn update(&mut self, value: f64) {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value;
        }
    }

    /// Get current EMA value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset EMA
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }
}

/// Calculate correlation between two series
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

/// Calculate rolling correlation
pub fn rolling_correlation(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    if x.len() != y.len() || x.len() < window {
        return vec![];
    }

    let mut correlations = Vec::with_capacity(x.len() - window + 1);

    for i in 0..=(x.len() - window) {
        let x_window = &x[i..i + window];
        let y_window = &y[i..i + window];
        correlations.push(correlation(x_window, y_window));
    }

    correlations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(10000.0, 100);

        // Simulate some returns
        tracker.update(100.0);
        tracker.update(-50.0);
        tracker.update(200.0);

        assert!(tracker.current_equity() > 10000.0);
        assert!(tracker.max_drawdown() > 0.0);
    }

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(5);

        for i in 1..=10 {
            stats.update(i as f64);
        }

        assert!(stats.is_full());
        assert!(stats.mean() > 0.0);
        assert!(stats.std_dev() > 0.0);
    }

    #[test]
    fn test_ema() {
        let mut ema = EMA::new(10);

        for i in 1..=20 {
            ema.update(i as f64);
        }

        assert!(ema.is_initialized());
        assert!(ema.value() > 0.0 && ema.value() < 20.0);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rolling_correlation() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = (1..=10).map(|i| i as f64 * 2.0).collect();

        let corrs = rolling_correlation(&x, &y, 5);
        assert!(!corrs.is_empty());
        assert!(corrs.iter().all(|&c| (c - 1.0).abs() < 0.001));
    }
}
