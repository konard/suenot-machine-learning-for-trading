//! Trading Strategy and Backtesting
//!
//! This module provides trading strategy implementations and backtesting
//! utilities for evaluating time series models.

use std::f64;

/// Trading signal types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Long,
    Neutral,
    Short,
}

impl Signal {
    /// Convert to numeric value
    pub fn as_f64(&self) -> f64 {
        match self {
            Signal::Long => 1.0,
            Signal::Neutral => 0.0,
            Signal::Short => -1.0,
        }
    }

    /// Create from prediction value
    pub fn from_prediction(pred: f64, threshold: f64) -> Self {
        if pred > threshold {
            Signal::Long
        } else if pred < -threshold {
            Signal::Short
        } else {
            Signal::Neutral
        }
    }
}

/// Container for backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Daily/period returns
    pub returns: Vec<f64>,
    /// Cumulative returns
    pub cumulative_returns: Vec<f64>,
    /// Portfolio value over time
    pub equity_curve: Vec<f64>,
    /// Annualized Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Total period return
    pub total_return: f64,
    /// Percentage of winning trades
    pub win_rate: f64,
    /// Number of trades
    pub n_trades: usize,
    /// Additional metrics
    pub metrics: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub annual_return: f64,
    pub volatility: f64,
    pub calmar_ratio: f64,
}

/// Trading strategy configuration
#[derive(Debug, Clone)]
pub struct TradingStrategy {
    /// Signal threshold for position entry
    pub threshold: f64,
    /// Maximum position size (1.0 = 100%)
    pub max_position: f64,
    /// Transaction cost per trade (fraction)
    pub transaction_cost: f64,
    /// Stop loss threshold (fraction)
    pub stop_loss: Option<f64>,
    /// Take profit threshold (fraction)
    pub take_profit: Option<f64>,
}

impl Default for TradingStrategy {
    fn default() -> Self {
        Self {
            threshold: 0.001,
            max_position: 1.0,
            transaction_cost: 0.001,
            stop_loss: None,
            take_profit: None,
        }
    }
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Set maximum position size
    pub fn with_max_position(mut self, max_position: f64) -> Self {
        self.max_position = max_position;
        self
    }

    /// Set transaction cost
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Set stop loss
    pub fn with_stop_loss(mut self, stop_loss: f64) -> Self {
        self.stop_loss = Some(stop_loss);
        self
    }

    /// Set take profit
    pub fn with_take_profit(mut self, take_profit: f64) -> Self {
        self.take_profit = Some(take_profit);
        self
    }

    /// Generate trading signals from predictions
    pub fn generate_signals(&self, predictions: &[f64]) -> Vec<Signal> {
        predictions
            .iter()
            .map(|&p| Signal::from_prediction(p, self.threshold))
            .collect()
    }

    /// Generate position sizes from signals
    pub fn generate_positions(&self, signals: &[Signal], confidences: Option<&[f64]>) -> Vec<f64> {
        match confidences {
            Some(conf) => signals
                .iter()
                .zip(conf.iter())
                .map(|(s, c)| (s.as_f64() * c * self.max_position).clamp(-self.max_position, self.max_position))
                .collect(),
            None => signals
                .iter()
                .map(|s| s.as_f64() * self.max_position)
                .collect(),
        }
    }
}

/// Calculate trading performance metrics
pub fn calculate_metrics(returns: &[f64], periods_per_year: usize) -> PerformanceMetrics {
    // Filter NaN values
    let returns: Vec<f64> = returns.iter().copied().filter(|r| r.is_finite()).collect();

    if returns.is_empty() {
        return PerformanceMetrics {
            annual_return: 0.0,
            volatility: 0.0,
            calmar_ratio: 0.0,
        };
    }

    // Total return
    let cumulative: f64 = returns.iter().map(|r| 1.0 + r).product();
    let total_return = cumulative - 1.0;

    // Annualized return
    let n_periods = returns.len() as f64;
    let annual_return =
        (1.0 + total_return).powf(periods_per_year as f64 / n_periods) - 1.0;

    // Volatility
    let mean = returns.iter().sum::<f64>() / n_periods;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n_periods;
    let volatility = variance.sqrt() * (periods_per_year as f64).sqrt();

    // Maximum drawdown
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    let mut equity = 1.0;
    for r in &returns {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    // Calmar ratio
    let calmar_ratio = if max_dd > 0.0 {
        annual_return / max_dd
    } else {
        0.0
    };

    PerformanceMetrics {
        annual_return,
        volatility,
        calmar_ratio,
    }
}

/// Run backtest on model predictions
pub fn run_backtest(
    predictions: &[f64],
    actual_returns: &[f64],
    strategy: &TradingStrategy,
    initial_capital: f64,
    periods_per_year: usize,
) -> BacktestResult {
    let min_len = predictions.len().min(actual_returns.len());
    let predictions = &predictions[..min_len];
    let actual_returns = &actual_returns[..min_len];

    // Generate signals and positions
    let signals = strategy.generate_signals(predictions);
    let positions = strategy.generate_positions(&signals, None);

    // Calculate strategy returns
    let mut strategy_returns = Vec::with_capacity(min_len);
    let mut position_changes = Vec::with_capacity(min_len);

    for i in 0..min_len {
        let prev_position = if i > 0 { positions[i - 1] } else { 0.0 };
        let position_change = (positions[i] - prev_position).abs();
        position_changes.push(position_change);

        let gross_return = positions[i] * actual_returns[i];
        let transaction_cost = position_change * strategy.transaction_cost;
        strategy_returns.push(gross_return - transaction_cost);
    }

    // Calculate equity curve
    let mut equity = initial_capital;
    let mut equity_curve = Vec::with_capacity(min_len);
    for &r in &strategy_returns {
        equity *= 1.0 + r;
        equity_curve.push(equity);
    }

    // Calculate cumulative returns
    let mut cumulative = 1.0;
    let cumulative_returns: Vec<f64> = strategy_returns
        .iter()
        .map(|&r| {
            cumulative *= 1.0 + r;
            cumulative - 1.0
        })
        .collect();

    // Calculate metrics
    let metrics = calculate_metrics(&strategy_returns, periods_per_year);

    // Sharpe ratio
    let mean_return = strategy_returns.iter().sum::<f64>() / strategy_returns.len() as f64;
    let std_return = (strategy_returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / strategy_returns.len() as f64)
        .sqrt();
    let sharpe_ratio = if std_return > 0.0 {
        metrics.annual_return / (std_return * (periods_per_year as f64).sqrt())
    } else {
        0.0
    };

    // Sortino ratio (downside volatility)
    let downside_returns: Vec<f64> = strategy_returns
        .iter()
        .copied()
        .filter(|&r| r < 0.0)
        .collect();
    let sortino_ratio = if !downside_returns.is_empty() {
        let downside_vol = (downside_returns
            .iter()
            .map(|r| r.powi(2))
            .sum::<f64>()
            / downside_returns.len() as f64)
            .sqrt()
            * (periods_per_year as f64).sqrt();
        if downside_vol > 0.0 {
            metrics.annual_return / downside_vol
        } else {
            0.0
        }
    } else {
        sharpe_ratio * 1.5 // Approximate if no downside
    };

    // Maximum drawdown
    let mut peak = initial_capital;
    let mut max_drawdown = 0.0;
    for &equity_val in &equity_curve {
        if equity_val > peak {
            peak = equity_val;
        }
        let dd = (peak - equity_val) / peak;
        if dd > max_drawdown {
            max_drawdown = dd;
        }
    }

    // Win rate
    let winning_trades = strategy_returns.iter().filter(|&&r| r > 0.0).count();
    let total_trades = strategy_returns.iter().filter(|&&r| r != 0.0).count();
    let win_rate = if total_trades > 0 {
        winning_trades as f64 / total_trades as f64
    } else {
        0.0
    };

    // Number of trades
    let n_trades = position_changes.iter().filter(|&&c| c > 0.0).count();

    // Calculate total return before moving cumulative_returns
    let total_return = cumulative_returns.last().copied().unwrap_or(0.0);

    BacktestResult {
        returns: strategy_returns,
        cumulative_returns,
        equity_curve,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        total_return,
        win_rate,
        n_trades,
        metrics,
    }
}

/// Calculate buy and hold benchmark
pub fn calculate_buy_and_hold(
    actual_returns: &[f64],
    initial_capital: f64,
    periods_per_year: usize,
) -> BacktestResult {
    let mut equity = initial_capital;
    let mut equity_curve = Vec::with_capacity(actual_returns.len());
    let mut cumulative = 1.0;
    let mut cumulative_returns = Vec::with_capacity(actual_returns.len());

    for &r in actual_returns {
        equity *= 1.0 + r;
        equity_curve.push(equity);
        cumulative *= 1.0 + r;
        cumulative_returns.push(cumulative - 1.0);
    }

    let metrics = calculate_metrics(actual_returns, periods_per_year);

    // Sharpe ratio
    let mean_return = actual_returns.iter().sum::<f64>() / actual_returns.len() as f64;
    let std_return = (actual_returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / actual_returns.len() as f64)
        .sqrt();
    let sharpe_ratio = if std_return > 0.0 {
        metrics.annual_return / (std_return * (periods_per_year as f64).sqrt())
    } else {
        0.0
    };

    // Maximum drawdown
    let mut peak = initial_capital;
    let mut max_drawdown = 0.0;
    for &equity_val in &equity_curve {
        if equity_val > peak {
            peak = equity_val;
        }
        let dd = (peak - equity_val) / peak;
        if dd > max_drawdown {
            max_drawdown = dd;
        }
    }

    // Win rate
    let winning = actual_returns.iter().filter(|&&r| r > 0.0).count();
    let total = actual_returns.iter().filter(|&&r| r != 0.0).count();
    let win_rate = if total > 0 {
        winning as f64 / total as f64
    } else {
        0.0
    };

    // Calculate total return before moving cumulative_returns
    let total_return = cumulative_returns.last().copied().unwrap_or(0.0);

    BacktestResult {
        returns: actual_returns.to_vec(),
        cumulative_returns,
        equity_curve,
        sharpe_ratio,
        sortino_ratio: sharpe_ratio * 1.2,
        max_drawdown,
        total_return,
        win_rate,
        n_trades: 1,
        metrics,
    }
}

/// Compare different strategy thresholds
pub fn compare_strategies(
    predictions: &[f64],
    actual_returns: &[f64],
    thresholds: &[f64],
    periods_per_year: usize,
) -> Vec<(f64, BacktestResult)> {
    thresholds
        .iter()
        .map(|&threshold| {
            let strategy = TradingStrategy::new(threshold);
            let result = run_backtest(predictions, actual_returns, &strategy, 100000.0, periods_per_year);
            (threshold, result)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_prediction() {
        assert_eq!(Signal::from_prediction(0.005, 0.001), Signal::Long);
        assert_eq!(Signal::from_prediction(-0.005, 0.001), Signal::Short);
        assert_eq!(Signal::from_prediction(0.0005, 0.001), Signal::Neutral);
    }

    #[test]
    fn test_strategy_generate_signals() {
        let strategy = TradingStrategy::new(0.001);
        let predictions = vec![0.005, -0.005, 0.0001, 0.002];
        let signals = strategy.generate_signals(&predictions);

        assert_eq!(signals[0], Signal::Long);
        assert_eq!(signals[1], Signal::Short);
        assert_eq!(signals[2], Signal::Neutral);
        assert_eq!(signals[3], Signal::Long);
    }

    #[test]
    fn test_strategy_generate_positions() {
        let strategy = TradingStrategy::new(0.001).with_max_position(0.5);
        let signals = vec![Signal::Long, Signal::Short, Signal::Neutral];
        let positions = strategy.generate_positions(&signals, None);

        assert!((positions[0] - 0.5).abs() < 1e-10);
        assert!((positions[1] - (-0.5)).abs() < 1e-10);
        assert!((positions[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_metrics() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let metrics = calculate_metrics(&returns, 252);

        assert!(metrics.annual_return.is_finite());
        assert!(metrics.volatility >= 0.0);
    }

    #[test]
    fn test_run_backtest() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 100;

        let actual_returns: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 0.04 - 0.02).collect();
        let predictions: Vec<f64> = actual_returns
            .iter()
            .map(|&r| r + rng.gen::<f64>() * 0.01 - 0.005)
            .collect();

        let strategy = TradingStrategy::new(0.001);
        let result = run_backtest(&predictions, &actual_returns, &strategy, 100000.0, 8760);

        assert_eq!(result.returns.len(), n);
        assert_eq!(result.equity_curve.len(), n);
        assert!(result.sharpe_ratio.is_finite());
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
    }

    #[test]
    fn test_buy_and_hold() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let result = calculate_buy_and_hold(&returns, 100000.0, 252);

        assert_eq!(result.returns.len(), 5);
        assert_eq!(result.n_trades, 1);
    }

    #[test]
    fn test_compare_strategies() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 100;

        let actual_returns: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 0.04 - 0.02).collect();
        let predictions: Vec<f64> = actual_returns
            .iter()
            .map(|&r| r + rng.gen::<f64>() * 0.01 - 0.005)
            .collect();

        let thresholds = vec![0.0005, 0.001, 0.002];
        let results = compare_strategies(&predictions, &actual_returns, &thresholds, 8760);

        assert_eq!(results.len(), 3);
        for (threshold, result) in results {
            assert!(threshold > 0.0);
            assert!(result.sharpe_ratio.is_finite());
        }
    }
}
