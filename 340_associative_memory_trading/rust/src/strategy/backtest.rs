//! Backtesting framework for associative memory trading strategy

use crate::data::OHLCVSeries;
use crate::features::{Pattern, PatternBuilder};
use crate::memory::DenseAssociativeMemory;
use crate::strategy::{Signal, SignalConfig, SignalGenerator};
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};

/// Single backtest trade
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub entry_price: f64,
    pub exit_price: f64,
    pub direction: f64, // 1.0 for long, -1.0 for short
    pub size: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
}

/// Backtest result for a single step
#[derive(Debug, Clone)]
pub struct BacktestStep {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub signal: Signal,
    pub position: f64,
    pub pnl: f64,
    pub cumulative_pnl: f64,
}

/// Complete backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Individual time steps
    pub steps: Vec<BacktestStep>,
    /// Completed trades
    pub trades: Vec<Trade>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_trade_pnl: f64,
    pub n_trades: usize,
    pub avg_confidence: f64,
}

/// Backtester configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Pattern builder config
    pub pattern_lookback: usize,
    /// Forward period for labels
    pub forward_period: usize,
    /// Warmup period (patterns needed before trading)
    pub warmup_period: usize,
    /// Signal configuration
    pub signal_config: SignalConfig,
    /// Transaction cost (as fraction)
    pub transaction_cost: f64,
    /// Slippage (as fraction)
    pub slippage: f64,
    /// Memory size for storing patterns
    pub memory_size: usize,
    /// Train/test split ratio
    pub train_ratio: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            pattern_lookback: 20,
            forward_period: 5,
            warmup_period: 100,
            signal_config: SignalConfig::default(),
            transaction_cost: 0.001, // 0.1%
            slippage: 0.0005,        // 0.05%
            memory_size: 500,
            train_ratio: 0.7,
        }
    }
}

/// Backtester for associative memory trading
pub struct Backtester {
    config: BacktestConfig,
    pattern_builder: PatternBuilder,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        let pattern_builder = PatternBuilder::new(config.pattern_lookback);

        Self {
            config,
            pattern_builder,
        }
    }

    /// Run backtest on OHLCV data
    pub fn run(&self, data: &OHLCVSeries) -> anyhow::Result<BacktestResult> {
        let min_required = self.config.warmup_period + self.config.forward_period;
        if data.len() < min_required {
            return Err(anyhow::anyhow!(
                "Insufficient data: need at least {} candles, got {}",
                min_required,
                data.len()
            ));
        }

        // Build patterns
        let patterns = self.pattern_builder.build_patterns(data);
        if patterns.is_empty() {
            return Err(anyhow::anyhow!("No patterns could be built"));
        }

        log::info!("Built {} patterns", patterns.len());

        // Split into train/test
        let split_idx = (patterns.len() as f64 * self.config.train_ratio) as usize;
        let train_patterns = &patterns[..split_idx];
        let test_patterns = &patterns[split_idx..];

        log::info!(
            "Train: {} patterns, Test: {} patterns",
            train_patterns.len(),
            test_patterns.len()
        );

        // Build training data
        let (train_features, train_labels) = self.patterns_to_arrays(train_patterns);

        // Initialize memory
        let pattern_dim = train_features.ncols();
        let mut memory = DenseAssociativeMemory::new(
            self.config.memory_size.min(train_patterns.len()),
            pattern_dim,
            1.0,
        );

        // Store training patterns
        memory.store(&train_features, &train_labels);

        // Run backtest on test data
        let signal_generator = SignalGenerator::new(self.config.signal_config.clone());

        let mut steps = Vec::new();
        let mut trades = Vec::new();
        let mut current_position = 0.0;
        let mut cumulative_pnl = 0.0;
        let mut entry_price = 0.0;
        let mut entry_time = test_patterns[0].timestamp;
        let mut total_confidence = 0.0;
        let mut confidence_count = 0;

        for (i, pattern) in test_patterns.iter().enumerate() {
            let price = data.data[split_idx + self.config.pattern_lookback + i].close;

            // Generate signal
            let signal = signal_generator.generate_from_dam(&memory, &pattern.features);

            if signal.is_actionable {
                total_confidence += signal.confidence;
                confidence_count += 1;
            }

            // Calculate PnL from position
            let pnl = if i > 0 && current_position != 0.0 {
                let prev_price = data.data[split_idx + self.config.pattern_lookback + i - 1].close;
                let price_return = (price - prev_price) / prev_price;
                current_position * price_return
            } else {
                0.0
            };

            cumulative_pnl += pnl;

            // Position management
            let target_position = if signal.is_actionable {
                signal.direction * signal.position_size
            } else {
                0.0
            };

            // Check if position changed (trade occurred)
            if target_position != current_position {
                // Close previous trade if exists
                if current_position != 0.0 {
                    let trade_pnl = current_position * (price - entry_price) / entry_price;
                    let trade_pnl_after_costs =
                        trade_pnl - self.config.transaction_cost - self.config.slippage;

                    trades.push(Trade {
                        entry_time,
                        exit_time: pattern.timestamp,
                        entry_price,
                        exit_price: price,
                        direction: current_position.signum(),
                        size: current_position.abs(),
                        pnl: trade_pnl_after_costs * entry_price,
                        pnl_pct: trade_pnl_after_costs,
                    });
                }

                // Open new position
                if target_position != 0.0 {
                    entry_price = price * (1.0 + self.config.slippage * target_position.signum());
                    entry_time = pattern.timestamp;
                }

                current_position = target_position;
            }

            steps.push(BacktestStep {
                timestamp: pattern.timestamp,
                price,
                signal,
                position: current_position,
                pnl,
                cumulative_pnl,
            });
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&steps, &trades, total_confidence, confidence_count);

        Ok(BacktestResult {
            steps,
            trades,
            metrics,
        })
    }

    /// Convert patterns to arrays
    fn patterns_to_arrays(&self, patterns: &[Pattern]) -> (Array2<f64>, Array1<f64>) {
        if patterns.is_empty() {
            return (Array2::zeros((0, 0)), Array1::zeros(0));
        }

        let n = patterns.len();
        let dim = patterns[0].dim();

        let mut features = Array2::zeros((n, dim));
        let mut labels = Array1::zeros(n);

        for (i, pattern) in patterns.iter().enumerate() {
            for (j, &val) in pattern.features.iter().enumerate() {
                features[[i, j]] = val;
            }
            labels[i] = pattern.label.unwrap_or(0.0);
        }

        (features, labels)
    }

    /// Calculate performance metrics
    fn calculate_metrics(
        &self,
        steps: &[BacktestStep],
        trades: &[Trade],
        total_confidence: f64,
        confidence_count: usize,
    ) -> PerformanceMetrics {
        if steps.is_empty() {
            return PerformanceMetrics::default();
        }

        // Returns series
        let returns: Vec<f64> = steps.iter().map(|s| s.pnl).collect();
        let n_days = steps.len() as f64;

        // Total return
        let total_return = steps.last().map(|s| s.cumulative_pnl).unwrap_or(0.0);

        // Annualized return (assuming hourly data)
        let periods_per_year = 365.0 * 24.0;
        let annualized_return = (1.0 + total_return).powf(periods_per_year / n_days) - 1.0;

        // Sharpe ratio
        let mean_return = returns.iter().sum::<f64>() / n_days;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n_days;
        let std_return = variance.sqrt();
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (periods_per_year).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if !downside_returns.is_empty() {
            let variance = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (periods_per_year).sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = 0.0;
        let mut max_dd = 0.0;
        for step in steps {
            if step.cumulative_pnl > peak {
                peak = step.cumulative_pnl;
            }
            let dd = peak - step.cumulative_pnl;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        // Trade statistics
        let n_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if n_trades > 0 {
            winning_trades as f64 / n_trades as f64
        } else {
            0.0
        };

        let total_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let total_loss: f64 = trades
            .iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if total_loss > 0.0 {
            total_profit / total_loss
        } else if total_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_pnl = if n_trades > 0 {
            trades.iter().map(|t| t.pnl).sum::<f64>() / n_trades as f64
        } else {
            0.0
        };

        let avg_confidence = if confidence_count > 0 {
            total_confidence / confidence_count as f64
        } else {
            0.0
        };

        PerformanceMetrics {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: max_dd,
            win_rate,
            profit_factor,
            avg_trade_pnl,
            n_trades,
            avg_confidence,
        }
    }
}

impl BacktestResult {
    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Backtest Results ===\n");
        println!("Total Return:      {:.2}%", self.metrics.total_return * 100.0);
        println!(
            "Annualized Return: {:.2}%",
            self.metrics.annualized_return * 100.0
        );
        println!("Sharpe Ratio:      {:.3}", self.metrics.sharpe_ratio);
        println!("Sortino Ratio:     {:.3}", self.metrics.sortino_ratio);
        println!(
            "Max Drawdown:      {:.2}%",
            self.metrics.max_drawdown * 100.0
        );
        println!("Win Rate:          {:.1}%", self.metrics.win_rate * 100.0);
        println!("Profit Factor:     {:.2}", self.metrics.profit_factor);
        println!("Number of Trades:  {}", self.metrics.n_trades);
        println!(
            "Avg Trade PnL:     {:.4}%",
            self.metrics.avg_trade_pnl * 100.0
        );
        println!(
            "Avg Confidence:    {:.2}%",
            self.metrics.avg_confidence * 100.0
        );
        println!();
    }

    /// Export to CSV
    pub fn to_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;

        wtr.write_record(&[
            "timestamp",
            "price",
            "signal_direction",
            "signal_confidence",
            "position",
            "pnl",
            "cumulative_pnl",
        ])?;

        for step in &self.steps {
            wtr.write_record(&[
                step.timestamp.to_rfc3339(),
                step.price.to_string(),
                step.signal.direction.to_string(),
                step.signal.confidence.to_string(),
                step.position.to_string(),
                step.pnl.to_string(),
                step.cumulative_pnl.to_string(),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::OHLCV;

    fn create_test_data(n: usize) -> OHLCVSeries {
        let mut data = Vec::new();
        let mut price = 100.0;

        for i in 0..n {
            // Add some trending behavior
            let trend = (i as f64 * 0.01).sin() * 2.0;
            let noise = (i as f64 * 0.1).cos() * 0.5;
            price = (price + trend + noise).max(10.0);

            data.push(OHLCV::new(
                Utc::now() + chrono::Duration::hours(i as i64),
                price,
                price * 1.01,
                price * 0.99,
                price + noise,
                1000.0 + (i as f64 % 100.0) * 10.0,
            ));
        }

        OHLCVSeries::with_data("TEST".to_string(), "60".to_string(), data)
    }

    #[test]
    fn test_backtest_runs() {
        let data = create_test_data(500);
        let config = BacktestConfig {
            warmup_period: 50,
            ..Default::default()
        };

        let backtester = Backtester::new(config);
        let result = backtester.run(&data);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.steps.is_empty());
    }
}
