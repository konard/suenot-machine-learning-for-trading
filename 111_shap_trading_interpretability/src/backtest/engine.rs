//! Backtesting engine for SHAP-enhanced trading strategies.

use crate::trading::signals::TradingSignal;
use std::collections::HashMap;

/// Backtest configuration.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital.
    pub initial_capital: f64,
    /// Transaction cost as fraction.
    pub transaction_cost: f64,
    /// Slippage as fraction.
    pub slippage: f64,
    /// Maximum position size as fraction of capital.
    pub max_position_size: f64,
    /// Use SHAP-based filtering.
    pub use_shap_filter: bool,
    /// SHAP stability threshold.
    pub shap_stability_threshold: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            max_position_size: 1.0,
            use_shap_filter: true,
            shap_stability_threshold: 0.3,
        }
    }
}

/// Trade record.
#[derive(Debug, Clone)]
pub struct Trade {
    pub index: usize,
    pub price: f64,
    pub old_position: f64,
    pub new_position: f64,
    pub cost: f64,
    pub pnl: f64,
}

/// Backtest result.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Strategy returns.
    pub returns: Vec<f64>,
    /// Position history.
    pub positions: Vec<f64>,
    /// Equity curve.
    pub equity_curve: Vec<f64>,
    /// Performance metrics.
    pub metrics: HashMap<String, f64>,
    /// Trade history.
    pub trades: Vec<Trade>,
}

impl BacktestResult {
    /// Print summary of results.
    pub fn print_summary(&self) {
        println!("{}", "=".repeat(50));
        println!("Backtest Results");
        println!("{}", "=".repeat(50));
        for (k, v) in &self.metrics {
            println!("  {}: {:.4}", k, v);
        }
        println!("{}", "=".repeat(50));
    }
}

/// Backtesting engine.
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on price data with trading signals.
    pub fn run(
        &self,
        prices: &[f64],
        signals: &[TradingSignal],
    ) -> BacktestResult {
        let n = prices.len();
        assert_eq!(n, signals.len(), "prices and signals must have same length");

        let mut positions = vec![0.0; n];
        let mut returns = vec![0.0; n];
        let mut equity = vec![0.0; n];
        equity[0] = self.config.initial_capital;
        let mut trades = Vec::new();

        // Calculate price returns
        let mut price_returns = vec![0.0; n];
        for i in 1..n {
            price_returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
        }

        let mut current_position = 0.0;
        let mut entry_price = 0.0;

        for i in 1..n {
            let signal = &signals[i - 1];

            // Determine target position
            let target_position = self.compute_target_position(signal);

            // Check if position changes
            if (target_position - current_position).abs() > 1e-6 {
                let trade_cost = (target_position - current_position).abs()
                    * (self.config.transaction_cost + self.config.slippage);

                // Calculate PnL if closing a position
                let pnl = if current_position != 0.0 {
                    current_position * (prices[i] - entry_price) / entry_price
                } else {
                    0.0
                };

                trades.push(Trade {
                    index: i,
                    price: prices[i],
                    old_position: current_position,
                    new_position: target_position,
                    cost: trade_cost,
                    pnl,
                });

                if target_position != 0.0 && current_position == 0.0 {
                    entry_price = prices[i];
                }

                current_position = target_position;
            }

            positions[i] = current_position;

            // Calculate strategy return
            let trade_cost = if i > 0 && (positions[i] - positions[i - 1]).abs() > 1e-6 {
                (positions[i] - positions[i - 1]).abs()
                    * (self.config.transaction_cost + self.config.slippage)
            } else {
                0.0
            };

            let strategy_return = current_position * price_returns[i] - trade_cost;
            returns[i] = strategy_return;
            equity[i] = equity[i - 1] * (1.0 + strategy_return);
        }

        let metrics = self.compute_metrics(&returns, &equity, &trades);

        BacktestResult {
            returns,
            positions,
            equity_curve: equity,
            metrics,
            trades,
        }
    }

    /// Compute target position from signal.
    fn compute_target_position(&self, signal: &TradingSignal) -> f64 {
        let min_confidence = 0.1;

        if signal.confidence < min_confidence {
            return 0.0;
        }

        let mut position = signal.direction as f64 * signal.confidence;

        // Apply SHAP stability filter
        if self.config.use_shap_filter {
            if signal.shap_stability < self.config.shap_stability_threshold {
                position *= signal.shap_stability / self.config.shap_stability_threshold;
            }
        }

        position.clamp(-self.config.max_position_size, self.config.max_position_size)
    }

    /// Compute performance metrics.
    fn compute_metrics(
        &self,
        returns: &[f64],
        equity: &[f64],
        trades: &[Trade],
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Filter active returns
        let active_returns: Vec<f64> = returns.iter().filter(|&&r| r != 0.0).cloned().collect();

        // Annualization factor (assume hourly data)
        let annual_factor = (365.0 * 24.0_f64).sqrt();

        // Sharpe Ratio
        let sharpe = if !active_returns.is_empty() {
            let mean: f64 = active_returns.iter().sum::<f64>() / active_returns.len() as f64;
            let variance: f64 = active_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / active_returns.len() as f64;
            let std = variance.sqrt();
            if std > 0.0 {
                mean / std * annual_factor
            } else {
                0.0
            }
        } else {
            0.0
        };
        metrics.insert("sharpe_ratio".to_string(), sharpe);

        // Sortino Ratio
        let downside: Vec<f64> = active_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        let sortino = if !downside.is_empty() {
            let mean: f64 = active_returns.iter().sum::<f64>() / active_returns.len() as f64;
            let downside_variance: f64 =
                downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
            let downside_std = downside_variance.sqrt();
            if downside_std > 0.0 {
                mean / downside_std * annual_factor
            } else {
                sharpe
            }
        } else {
            sharpe
        };
        metrics.insert("sortino_ratio".to_string(), sortino);

        // Maximum Drawdown
        let mut peak = equity[0];
        let mut max_drawdown = 0.0;
        for &e in equity {
            if e > peak {
                peak = e;
            }
            let drawdown = (e - peak) / peak;
            if drawdown < max_drawdown {
                max_drawdown = drawdown;
            }
        }
        metrics.insert("max_drawdown".to_string(), max_drawdown);

        // Total Return
        let total_return = (equity[equity.len() - 1] - equity[0]) / equity[0];
        metrics.insert("total_return".to_string(), total_return);

        // Calmar Ratio
        let calmar = if max_drawdown != 0.0 {
            total_return / max_drawdown.abs()
        } else {
            0.0
        };
        metrics.insert("calmar_ratio".to_string(), calmar);

        // Win Rate
        let winning: usize = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if !trades.is_empty() {
            winning as f64 / trades.len() as f64
        } else {
            0.0
        };
        metrics.insert("win_rate".to_string(), win_rate);

        // Number of trades
        metrics.insert("n_trades".to_string(), trades.len() as f64);

        // Final equity
        metrics.insert("final_equity".to_string(), equity[equity.len() - 1]);

        metrics
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}
