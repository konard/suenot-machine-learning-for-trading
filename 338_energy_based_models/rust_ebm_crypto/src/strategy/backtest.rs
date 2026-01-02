//! Backtesting engine for EBM trading strategy

use ndarray::Array1;

use super::position::{ActionType, PositionManager, PositionSide, TradeAction};
use super::signals::{SignalGenerator, TradingSignal};
use crate::data::Candle;
use crate::ebm::{EnergyModel, MarketRegime, OnlineEnergyEstimator};
use crate::features::FeatureEngine;

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission rate (e.g., 0.001 = 0.1%)
    pub commission_rate: f64,
    /// Slippage rate (e.g., 0.0005 = 0.05%)
    pub slippage_rate: f64,
    /// Whether to allow shorting
    pub allow_short: bool,
    /// Warmup period (skip initial candles)
    pub warmup_period: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission_rate: 0.001,
            slippage_rate: 0.0005,
            allow_short: true,
            warmup_period: 100,
        }
    }
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResults {
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
    /// Calmar ratio (annualized return / max drawdown)
    pub calmar_ratio: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Winning trades
    pub winning_trades: usize,
    /// Losing trades
    pub losing_trades: usize,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade
    pub avg_loss: f64,
    /// Average holding period (in candles)
    pub avg_holding_period: f64,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Trade history
    pub trades: Vec<TradeRecord>,
}

impl BacktestResults {
    /// Print summary
    pub fn print_summary(&self) {
        println!("========== BACKTEST RESULTS ==========");
        println!();
        println!("Performance Metrics:");
        println!("  Total Return:      {:>10.2}%", self.total_return * 100.0);
        println!("  Annualized Return: {:>10.2}%", self.annualized_return * 100.0);
        println!("  Sharpe Ratio:      {:>10.2}", self.sharpe_ratio);
        println!("  Sortino Ratio:     {:>10.2}", self.sortino_ratio);
        println!("  Max Drawdown:      {:>10.2}%", self.max_drawdown * 100.0);
        println!("  Calmar Ratio:      {:>10.2}", self.calmar_ratio);
        println!();
        println!("Trade Statistics:");
        println!("  Total Trades:      {:>10}", self.total_trades);
        println!("  Winning Trades:    {:>10}", self.winning_trades);
        println!("  Losing Trades:     {:>10}", self.losing_trades);
        println!("  Win Rate:          {:>10.2}%", self.win_rate * 100.0);
        println!("  Profit Factor:     {:>10.2}", self.profit_factor);
        println!();
        println!("Trade Details:");
        println!("  Avg Trade Return:  {:>10.2}%", self.avg_trade_return * 100.0);
        println!("  Avg Win:           {:>10.2}%", self.avg_win * 100.0);
        println!("  Avg Loss:          {:>10.2}%", self.avg_loss * 100.0);
        println!("  Avg Holding Period:{:>10.1} candles", self.avg_holding_period);
        println!("======================================");
    }
}

/// Trade record
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Entry time
    pub entry_time: i64,
    /// Exit time
    pub exit_time: i64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position side
    pub side: PositionSide,
    /// Position size
    pub size: f64,
    /// Realized PnL
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Entry reason
    pub entry_reason: String,
    /// Exit reason
    pub exit_reason: String,
}

/// Backtest engine
pub struct BacktestEngine {
    /// Configuration
    pub config: BacktestConfig,
    /// Feature engine
    feature_engine: FeatureEngine,
    /// Online energy estimator
    energy_estimator: OnlineEnergyEstimator,
    /// Signal generator
    signal_generator: SignalGenerator,
    /// Position manager
    position_manager: PositionManager,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config,
            feature_engine: FeatureEngine::default(),
            energy_estimator: OnlineEnergyEstimator::new(100, 0.1),
            signal_generator: SignalGenerator::new(Default::default()),
            position_manager: PositionManager::new(Default::default()),
        }
    }

    /// Run backtest on candle data
    pub fn run(&mut self, candles: &[Candle]) -> BacktestResults {
        let n = candles.len();

        if n < self.config.warmup_period + 10 {
            return self.empty_results();
        }

        // Compute features
        let features = self.feature_engine.compute(candles);

        // Initialize tracking variables
        let mut equity = self.config.initial_capital;
        let mut equity_curve = Vec::with_capacity(n);
        let mut trades = Vec::new();
        let mut peak_equity = equity;
        let mut max_drawdown = 0.0;
        let mut daily_returns = Vec::new();
        let mut current_trade_start = 0i64;
        let mut current_trade_entry = 0.0;
        let mut current_trade_reason = String::new();

        // Run simulation
        for i in self.config.warmup_period..n {
            let candle = &candles[i];
            let feature_row = features.row(i).to_owned();

            // Update energy estimator
            let energy_result = self.energy_estimator.update(&feature_row);

            // Calculate return for this candle
            let ret = if i > 0 {
                (candle.close - candles[i - 1].close) / candles[i - 1].close
            } else {
                0.0
            };

            // Generate signal
            let signal = self.signal_generator.generate(
                energy_result.normalized_energy,
                energy_result.energy,
                ret,
                energy_result.regime,
                candle.timestamp,
            );

            // Process signal through position manager
            if let Some(action) = self.position_manager.process_signal(&signal, candle.close) {
                // Apply commission and slippage
                let trade_cost = action.size * equity * (self.config.commission_rate + self.config.slippage_rate);
                equity -= trade_cost;

                match action.action_type {
                    ActionType::Open => {
                        current_trade_start = candle.timestamp;
                        current_trade_entry = candle.close;
                        current_trade_reason = action.reason.clone();
                    }
                    ActionType::Close => {
                        // Record completed trade
                        let pnl_pct = if action.side == PositionSide::Long {
                            (candle.close - current_trade_entry) / current_trade_entry
                        } else {
                            (current_trade_entry - candle.close) / current_trade_entry
                        };

                        let pnl = pnl_pct * action.size * equity;
                        equity += pnl;

                        trades.push(TradeRecord {
                            entry_time: current_trade_start,
                            exit_time: candle.timestamp,
                            entry_price: current_trade_entry,
                            exit_price: candle.close,
                            side: action.side,
                            size: action.size,
                            pnl,
                            return_pct: pnl_pct,
                            entry_reason: current_trade_reason.clone(),
                            exit_reason: action.reason.clone(),
                        });
                    }
                    ActionType::Reduce => {
                        // Partial close - record as separate trade
                        let pnl_pct = if action.side == PositionSide::Long {
                            (candle.close - current_trade_entry) / current_trade_entry
                        } else {
                            (current_trade_entry - candle.close) / current_trade_entry
                        };

                        let pnl = pnl_pct * action.size * equity;
                        equity += pnl;
                    }
                    _ => {}
                }
            }

            // Update position PnL (for open positions)
            if self.position_manager.position.is_open() {
                let pos = &self.position_manager.position;
                let unrealized_pnl = if pos.side == PositionSide::Long {
                    (candle.close - pos.entry_price) / pos.entry_price * pos.size * equity
                } else {
                    (pos.entry_price - candle.close) / pos.entry_price * pos.size * equity
                };
                equity_curve.push(equity + unrealized_pnl);
            } else {
                equity_curve.push(equity);
            }

            // Track drawdown
            if equity > peak_equity {
                peak_equity = equity;
            }
            let drawdown = (peak_equity - equity) / peak_equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }

            // Track daily returns (for Sharpe calculation)
            if i > self.config.warmup_period {
                let prev_equity = equity_curve.get(equity_curve.len().saturating_sub(2)).copied().unwrap_or(equity);
                if prev_equity > 0.0 {
                    daily_returns.push((equity_curve.last().unwrap() - prev_equity) / prev_equity);
                }
            }
        }

        // Calculate final metrics
        self.calculate_results(equity_curve, trades, daily_returns, max_drawdown)
    }

    /// Calculate backtest results from raw data
    fn calculate_results(
        &self,
        equity_curve: Vec<f64>,
        trades: Vec<TradeRecord>,
        daily_returns: Vec<f64>,
        max_drawdown: f64,
    ) -> BacktestResults {
        let total_return = if equity_curve.is_empty() {
            0.0
        } else {
            (equity_curve.last().unwrap() - self.config.initial_capital) / self.config.initial_capital
        };

        // Annualized return (assuming hourly data)
        let n_periods = equity_curve.len() as f64;
        let periods_per_year = 365.25 * 24.0; // Hourly
        let annualized_return = (1.0 + total_return).powf(periods_per_year / n_periods) - 1.0;

        // Sharpe ratio
        let sharpe_ratio = if daily_returns.len() > 1 {
            let mean_return: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let variance: f64 = daily_returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
                / daily_returns.len() as f64;
            let std_return = variance.sqrt();
            if std_return > 1e-10 {
                mean_return / std_return * (periods_per_year).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let sortino_ratio = if daily_returns.len() > 1 {
            let mean_return: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let downside_returns: Vec<f64> = daily_returns.iter().filter(|&&r| r < 0.0).cloned().collect();
            let downside_var: f64 = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len().max(1) as f64;
            let downside_std = downside_var.sqrt();
            if downside_std > 1e-10 {
                mean_return / downside_std * (periods_per_year).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 1e-10 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let total_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = trades.iter().filter(|t| t.pnl < 0.0).count();
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 1e-10 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };

        let avg_trade_return = if total_trades > 0 {
            trades.iter().map(|t| t.return_pct).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        let avg_win = if winning_trades > 0 {
            trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.return_pct).sum::<f64>() / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss = if losing_trades > 0 {
            trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.return_pct.abs()).sum::<f64>() / losing_trades as f64
        } else {
            0.0
        };

        // Average holding period (approximation based on timestamps)
        let avg_holding_period = if total_trades > 0 {
            trades.iter().map(|t| (t.exit_time - t.entry_time) as f64).sum::<f64>()
                / total_trades as f64
                / 60000.0 // Convert ms to minutes
        } else {
            0.0
        };

        BacktestResults {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            profit_factor,
            avg_trade_return,
            avg_win,
            avg_loss,
            avg_holding_period,
            equity_curve,
            trades,
        }
    }

    /// Return empty results
    fn empty_results(&self) -> BacktestResults {
        BacktestResults {
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_trade_return: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            avg_holding_period: 0.0,
            equity_curve: vec![],
            trades: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles() -> Vec<Candle> {
        (0..500)
            .map(|i| {
                let trend = (i as f64 * 0.001).sin() * 10.0;
                let noise = ((i * 7) as f64).sin() * 2.0;
                let base = 100.0 + trend + noise;

                Candle::new(
                    i as i64 * 3600000, // Hourly
                    base,
                    base + 1.0 + (i % 3) as f64 * 0.5,
                    base - 1.0 - (i % 4) as f64 * 0.3,
                    base + 0.5,
                    1000.0 + (i as f64 * 0.1).cos() * 500.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_backtest_engine() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);

        let candles = create_test_candles();
        let results = engine.run(&candles);

        // Basic sanity checks
        assert!(!results.equity_curve.is_empty());
        assert!(results.max_drawdown >= 0.0);
        assert!(results.max_drawdown <= 1.0);
    }
}
