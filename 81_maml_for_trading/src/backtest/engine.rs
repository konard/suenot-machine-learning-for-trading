//! Backtesting engine for MAML trading strategies.

use crate::maml::algorithm::MAMLTrainer;
use crate::data::bybit::Kline;
use crate::data::features::FeatureGenerator;
use serde::{Deserialize, Serialize};

/// Single backtest trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    /// Entry timestamp
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: i64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position direction (1 for long, -1 for short)
    pub direction: i32,
    /// Profit/loss percentage
    pub pnl_pct: f64,
    /// Profit/loss in absolute terms
    pub pnl_abs: f64,
}

/// Backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    /// Initial capital
    pub initial_capital: f64,
    /// Final capital
    pub final_capital: f64,
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Annualized volatility
    pub annualized_volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Trade records
    pub trades: Vec<TradeRecord>,
    /// Equity curve (capital over time)
    pub equity_curve: Vec<f64>,
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (as percentage)
    pub transaction_cost: f64,
    /// Slippage (as percentage)
    pub slippage: f64,
    /// Prediction threshold for entering trades
    pub threshold: f64,
    /// Adaptation window size
    pub adaptation_window: usize,
    /// Number of adaptation steps
    pub adaptation_steps: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            threshold: 0.001,
            adaptation_window: 20,
            adaptation_steps: 5,
        }
    }
}

/// Backtesting engine for MAML strategies
pub struct BacktestEngine {
    config: BacktestConfig,
    feature_generator: FeatureGenerator,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config,
            feature_generator: FeatureGenerator::default_window(),
        }
    }

    /// Create with default configuration
    pub fn default_engine() -> Self {
        Self::new(BacktestConfig::default())
    }

    /// Run backtest on historical data
    pub fn run(
        &self,
        trainer: &MAMLTrainer,
        klines: &[Kline],
    ) -> BacktestResults {
        // Generate features
        let features = self.feature_generator.compute_features(klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        if features.len() < self.config.adaptation_window + 10 {
            return self.empty_results();
        }

        // Compute returns for targets
        let returns: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect();

        // Start index accounting for feature computation lag
        let start_idx = self.feature_generator.window() + 10;
        let _feature_start = 0;

        let mut capital = self.config.initial_capital;
        let mut position = 0i32; // -1, 0, 1
        let mut entry_price = 0.0;
        let mut entry_time = 0i64;
        let mut trades = Vec::new();
        let mut equity_curve = vec![capital];
        let mut daily_returns = Vec::new();

        // Main backtest loop
        for i in self.config.adaptation_window..features.len().saturating_sub(1) {
            let kline_idx = start_idx + i;
            if kline_idx >= klines.len() - 1 {
                break;
            }

            let current_price = klines[kline_idx].close;
            let next_price = klines[kline_idx + 1].close;
            let timestamp = klines[kline_idx].timestamp;

            // Get adaptation data
            let adapt_start = i.saturating_sub(self.config.adaptation_window);
            let adapt_features: Vec<Vec<f64>> = features[adapt_start..i].to_vec();

            // Ensure we have enough return data
            let return_start = adapt_start;
            let return_end = i.min(returns.len());
            if return_end <= return_start {
                continue;
            }
            let adapt_returns: Vec<f64> = returns[return_start..return_end].to_vec();

            if adapt_features.len() != adapt_returns.len() {
                continue;
            }

            // Adapt model and get prediction
            let adapted_model = trainer.adapt(
                &adapt_features,
                &adapt_returns,
                Some(self.config.adaptation_steps),
            );

            let prediction = adapted_model.predict(&features[i]);

            // Trading logic
            let signal = if prediction > self.config.threshold {
                1
            } else if prediction < -self.config.threshold {
                -1
            } else {
                0
            };

            // Close existing position if signal changes
            if position != 0 && signal != position {
                let exit_price = current_price * (1.0 - self.config.slippage * position.signum() as f64);
                let pnl_pct = if position > 0 {
                    (exit_price / entry_price) - 1.0
                } else {
                    1.0 - (exit_price / entry_price)
                };

                // Apply transaction cost
                let net_pnl_pct = pnl_pct - self.config.transaction_cost;
                let pnl_abs = capital * net_pnl_pct;
                capital *= 1.0 + net_pnl_pct;

                trades.push(TradeRecord {
                    entry_time,
                    exit_time: timestamp,
                    entry_price,
                    exit_price,
                    direction: position,
                    pnl_pct: net_pnl_pct,
                    pnl_abs,
                });

                position = 0;
            }

            // Open new position if signal is non-zero and no current position
            if position == 0 && signal != 0 {
                entry_price = current_price * (1.0 + self.config.slippage * signal.signum() as f64);
                entry_time = timestamp;
                position = signal;

                // Apply transaction cost on entry
                capital *= 1.0 - self.config.transaction_cost;
            }

            // Calculate daily return for equity curve
            let position_return = if position != 0 {
                position as f64 * ((next_price / current_price) - 1.0)
            } else {
                0.0
            };
            daily_returns.push(position_return);

            // Update equity curve
            let mark_to_market = if position != 0 {
                let unrealized_pnl = if position > 0 {
                    (next_price / entry_price) - 1.0
                } else {
                    1.0 - (next_price / entry_price)
                };
                capital * (1.0 + unrealized_pnl)
            } else {
                capital
            };
            equity_curve.push(mark_to_market);
        }

        // Close any remaining position
        if position != 0 && !klines.is_empty() {
            let exit_price = klines.last().unwrap().close;
            let exit_time = klines.last().unwrap().timestamp;
            let pnl_pct = if position > 0 {
                (exit_price / entry_price) - 1.0
            } else {
                1.0 - (exit_price / entry_price)
            };
            let net_pnl_pct = pnl_pct - self.config.transaction_cost;
            let pnl_abs = capital * net_pnl_pct;
            capital *= 1.0 + net_pnl_pct;

            trades.push(TradeRecord {
                entry_time,
                exit_time,
                entry_price,
                exit_price,
                direction: position,
                pnl_pct: net_pnl_pct,
                pnl_abs,
            });
        }

        // Calculate metrics
        self.calculate_metrics(capital, &trades, &equity_curve, &daily_returns)
    }

    /// Calculate performance metrics
    fn calculate_metrics(
        &self,
        final_capital: f64,
        trades: &[TradeRecord],
        equity_curve: &[f64],
        daily_returns: &[f64],
    ) -> BacktestResults {
        let total_return = (final_capital / self.config.initial_capital) - 1.0;

        // Annualized metrics (assuming daily data)
        let n_days = daily_returns.len().max(1);
        let annualized_return = (1.0 + total_return).powf(252.0 / n_days as f64) - 1.0;

        let mean_return = if !daily_returns.is_empty() {
            daily_returns.iter().sum::<f64>() / daily_returns.len() as f64
        } else {
            0.0
        };

        let variance = if !daily_returns.is_empty() {
            daily_returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / daily_returns.len() as f64
        } else {
            0.0
        };
        let annualized_volatility = variance.sqrt() * (252.0_f64).sqrt();

        // Sharpe ratio
        let sharpe_ratio = if annualized_volatility > 0.0 {
            annualized_return / annualized_volatility
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = daily_returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let sortino_ratio = if downside_variance > 0.0 {
            annualized_return / (downside_variance.sqrt() * (252.0_f64).sqrt())
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown = self.calculate_max_drawdown(equity_curve);

        // Win rate
        let winning_trades = trades.iter().filter(|t| t.pnl_pct > 0.0).count();
        let win_rate = if !trades.is_empty() {
            winning_trades as f64 / trades.len() as f64
        } else {
            0.0
        };

        // Profit factor
        let gross_profit: f64 = trades.iter()
            .filter(|t| t.pnl_pct > 0.0)
            .map(|t| t.pnl_abs)
            .sum();
        let gross_loss: f64 = trades.iter()
            .filter(|t| t.pnl_pct < 0.0)
            .map(|t| t.pnl_abs.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestResults {
            initial_capital: self.config.initial_capital,
            final_capital,
            total_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades: trades.len(),
            trades: trades.to_vec(),
            equity_curve: equity_curve.to_vec(),
        }
    }

    /// Calculate maximum drawdown from equity curve
    fn calculate_max_drawdown(&self, equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_equity = equity_curve[0];
        let mut max_drawdown = 0.0;

        for &equity in equity_curve {
            if equity > max_equity {
                max_equity = equity;
            }
            let drawdown = (max_equity - equity) / max_equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Return empty results for insufficient data
    fn empty_results(&self) -> BacktestResults {
        BacktestResults {
            initial_capital: self.config.initial_capital,
            final_capital: self.config.initial_capital,
            total_return: 0.0,
            annualized_return: 0.0,
            annualized_volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            num_trades: 0,
            trades: Vec::new(),
            equity_curve: vec![self.config.initial_capital],
        }
    }

    /// Get configuration
    pub fn config(&self) -> &BacktestConfig {
        &self.config
    }

    /// Get feature generator
    pub fn feature_generator(&self) -> &FeatureGenerator {
        &self.feature_generator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::TradingModel;
    use crate::data::bybit::SimulatedDataGenerator;

    #[test]
    fn test_backtest_engine_creation() {
        let engine = BacktestEngine::default_engine();
        assert!((engine.config().initial_capital - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_backtest_with_simulated_data() {
        let klines = SimulatedDataGenerator::generate_klines(200, 50000.0, 0.02);
        let model = TradingModel::new(11, 32, 1);
        let trainer = MAMLTrainer::new(model, 0.01, 0.001, 3, true);
        let engine = BacktestEngine::default_engine();

        let results = engine.run(&trainer, &klines);

        assert!(results.total_return.is_finite());
        assert!(results.sharpe_ratio.is_finite());
        assert!(results.max_drawdown >= 0.0);
        assert!(results.max_drawdown <= 1.0);
    }

    #[test]
    fn test_max_drawdown_calculation() {
        let engine = BacktestEngine::default_engine();

        // Test with declining equity
        let equity = vec![100.0, 90.0, 80.0, 85.0, 75.0];
        let max_dd = engine.calculate_max_drawdown(&equity);
        assert!((max_dd - 0.25).abs() < 0.01); // 25% max drawdown

        // Test with no drawdown
        let equity_up = vec![100.0, 110.0, 120.0, 130.0];
        let max_dd_up = engine.calculate_max_drawdown(&equity_up);
        assert!((max_dd_up - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_insufficient_data() {
        let klines = SimulatedDataGenerator::generate_klines(10, 50000.0, 0.02);
        let model = TradingModel::new(11, 32, 1);
        let trainer = MAMLTrainer::new(model, 0.01, 0.001, 3, true);
        let engine = BacktestEngine::default_engine();

        let results = engine.run(&trainer, &klines);

        assert_eq!(results.num_trades, 0);
        assert!((results.total_return - 0.0).abs() < 1e-10);
    }
}
