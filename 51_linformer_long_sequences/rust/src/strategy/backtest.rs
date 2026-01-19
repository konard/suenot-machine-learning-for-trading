//! Backtesting framework for Linformer trading strategies.

use crate::strategy::metrics::PerformanceMetrics;

/// Configuration for backtesting.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (percentage)
    pub transaction_cost: f64,
    /// Slippage (percentage)
    pub slippage: f64,
    /// Threshold for long signal
    pub long_threshold: f64,
    /// Threshold for short signal
    pub short_threshold: f64,
    /// Position sizing (fraction of capital)
    pub position_size: f64,
    /// Risk-free rate for metrics
    pub risk_free_rate: f64,
    /// Trading periods per year
    pub periods_per_year: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            long_threshold: 0.0,
            short_threshold: 0.0,
            position_size: 1.0,
            risk_free_rate: 0.02,
            periods_per_year: 252.0,
        }
    }
}

/// Result of a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Final portfolio value
    pub final_value: f64,
    /// Equity curve (portfolio values over time)
    pub equity_curve: Vec<f64>,
    /// Returns series
    pub returns: Vec<f64>,
    /// Positions taken
    pub positions: Vec<i32>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Backtester for evaluating trading strategies.
pub struct Backtester {
    /// Configuration
    pub config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester with default config.
    pub fn new() -> Self {
        Self {
            config: BacktestConfig::default(),
        }
    }

    /// Create a new backtester with custom config.
    pub fn with_config(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with model predictions.
    ///
    /// # Arguments
    /// * `predictions` - Model output (positive = long, negative = short)
    /// * `actual_returns` - Actual market returns
    pub fn run(&self, predictions: &[f64], actual_returns: &[f64]) -> BacktestResult {
        let n = predictions.len().min(actual_returns.len());
        if n == 0 {
            return BacktestResult {
                final_value: self.config.initial_capital,
                equity_curve: vec![self.config.initial_capital],
                returns: vec![],
                positions: vec![],
                metrics: PerformanceMetrics::default(),
            };
        }

        let mut equity_curve = Vec::with_capacity(n + 1);
        let mut returns = Vec::with_capacity(n);
        let mut positions = Vec::with_capacity(n);

        let mut portfolio_value = self.config.initial_capital;
        let mut current_position = 0i32;

        equity_curve.push(portfolio_value);

        for i in 0..n {
            let pred = predictions[i];
            let actual_ret = actual_returns[i];

            // Determine target position
            let target_position = if pred > self.config.long_threshold {
                1
            } else if pred < self.config.short_threshold {
                -1
            } else {
                0
            };

            // Calculate transaction costs for position change
            let position_change = (target_position - current_position).abs();
            let transaction_fee = if position_change > 0 {
                portfolio_value
                    * self.config.position_size
                    * (self.config.transaction_cost + self.config.slippage)
                    * position_change as f64
            } else {
                0.0
            };

            // Update position
            current_position = target_position;
            positions.push(current_position);

            // Calculate period return
            let position_return = current_position as f64
                * self.config.position_size
                * actual_ret;

            let period_return = position_return - transaction_fee / portfolio_value;
            returns.push(period_return);

            // Update portfolio value
            portfolio_value *= 1.0 + period_return;
            equity_curve.push(portfolio_value);
        }

        let metrics = PerformanceMetrics::from_returns(
            &returns,
            self.config.risk_free_rate,
            self.config.periods_per_year,
        );

        BacktestResult {
            final_value: portfolio_value,
            equity_curve,
            returns,
            positions,
            metrics,
        }
    }

    /// Run backtest with direction predictions (1 = up, 0 = down).
    pub fn run_direction(&self, predictions: &[i32], actual_returns: &[f64]) -> BacktestResult {
        let preds: Vec<f64> = predictions
            .iter()
            .map(|&p| if p == 1 { 1.0 } else { -1.0 })
            .collect();

        self.run(&preds, actual_returns)
    }

    /// Compare strategy to buy-and-hold.
    pub fn compare_to_benchmark(
        &self,
        strategy_result: &BacktestResult,
        actual_returns: &[f64],
    ) -> String {
        // Calculate buy-and-hold
        let bh_final = actual_returns
            .iter()
            .fold(self.config.initial_capital, |acc, &r| acc * (1.0 + r));

        let bh_return = (bh_final / self.config.initial_capital - 1.0) * 100.0;
        let strategy_return = (strategy_result.final_value / self.config.initial_capital - 1.0) * 100.0;
        let outperformance = strategy_return - bh_return;

        format!(
            r#"Strategy vs Benchmark Comparison
================================
Strategy Return:   {:>8.2}%
Buy & Hold Return: {:>8.2}%
Outperformance:    {:>8.2}%
Strategy Sharpe:   {:>8.2}
Max Drawdown:      {:>8.2}%"#,
            strategy_return,
            bh_return,
            outperformance,
            strategy_result.metrics.sharpe_ratio,
            strategy_result.metrics.max_drawdown * 100.0,
        )
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_basic() {
        let backtester = Backtester::new();
        let predictions = vec![1.0, 1.0, -1.0, 1.0, 0.0];
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];

        let result = backtester.run(&predictions, &returns);

        assert!(result.final_value > 0.0);
        assert_eq!(result.returns.len(), 5);
        assert_eq!(result.positions.len(), 5);
    }

    #[test]
    fn test_backtest_direction() {
        let backtester = Backtester::new();
        let predictions = vec![1, 1, 0, 1, 0];
        let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005];

        let result = backtester.run_direction(&predictions, &returns);

        assert!(result.final_value > 0.0);
        assert_eq!(result.positions.len(), 5);
    }

    #[test]
    fn test_backtest_empty() {
        let backtester = Backtester::new();
        let result = backtester.run(&[], &[]);

        assert_eq!(result.final_value, backtester.config.initial_capital);
        assert!(result.returns.is_empty());
    }

    #[test]
    fn test_backtest_custom_config() {
        let config = BacktestConfig {
            initial_capital: 50000.0,
            transaction_cost: 0.002,
            ..Default::default()
        };
        let backtester = Backtester::with_config(config);

        let predictions = vec![1.0; 10];
        let returns = vec![0.01; 10];

        let result = backtester.run(&predictions, &returns);

        assert!(result.final_value > 50000.0);
    }

    #[test]
    fn test_compare_to_benchmark() {
        let backtester = Backtester::new();
        let predictions = vec![1.0; 10];
        let returns = vec![0.01; 10];

        let result = backtester.run(&predictions, &returns);
        let comparison = backtester.compare_to_benchmark(&result, &returns);

        assert!(comparison.contains("Strategy"));
        assert!(comparison.contains("Buy & Hold"));
    }
}
