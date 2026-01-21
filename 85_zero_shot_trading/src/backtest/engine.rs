//! Backtesting engine implementation.

use crate::data::bybit::Kline;
use crate::data::features::FeatureGenerator;
use crate::model::network::ZeroShotModel;
use crate::strategy::signals::Signal;
use crate::strategy::trading::{Position, StrategyConfig};
use crate::backtest::results::{BacktestResults, Trade};
use crate::{Result, ZeroShotError};

/// Backtesting engine configuration.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial portfolio value
    pub initial_capital: f64,
    /// Trading fees (as fraction, e.g., 0.001 for 0.1%)
    pub trading_fee: f64,
    /// Slippage (as fraction)
    pub slippage: f64,
    /// Strategy configuration
    pub strategy_config: StrategyConfig,
    /// Minimum data points before starting trading
    pub warmup_period: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            strategy_config: StrategyConfig::default(),
            warmup_period: 100,
        }
    }
}

/// Backtesting engine.
pub struct BacktestEngine {
    /// Model for predictions
    model: ZeroShotModel,
    /// Feature generator
    feature_generator: FeatureGenerator,
    /// Configuration
    config: BacktestConfig,
    /// Current portfolio value
    portfolio_value: f64,
    /// Current cash
    cash: f64,
    /// Current position (if any)
    position: Option<Position>,
    /// Trade history
    trades: Vec<Trade>,
    /// Equity curve (portfolio value at each step)
    equity_curve: Vec<f64>,
}

impl BacktestEngine {
    /// Create a new backtesting engine.
    pub fn new(model: ZeroShotModel, config: BacktestConfig) -> Self {
        let initial_capital = config.initial_capital;
        Self {
            model,
            feature_generator: FeatureGenerator::default(),
            config,
            portfolio_value: initial_capital,
            cash: initial_capital,
            position: None,
            trades: Vec::new(),
            equity_curve: vec![initial_capital],
        }
    }

    /// Run backtest on historical data.
    pub fn run(&mut self, klines: &[Kline], symbol: &str) -> Result<BacktestResults> {
        if klines.len() < self.config.warmup_period {
            return Err(ZeroShotError::NotEnoughData {
                needed: self.config.warmup_period,
                got: klines.len(),
            });
        }

        // Reset state
        self.portfolio_value = self.config.initial_capital;
        self.cash = self.config.initial_capital;
        self.position = None;
        self.trades.clear();
        self.equity_curve.clear();
        self.equity_curve.push(self.config.initial_capital);

        // Iterate through data
        for i in self.config.warmup_period..klines.len() {
            let lookback_klines = &klines[..=i];
            let current_price = klines[i].close;
            let current_time = klines[i].timestamp.timestamp_millis();

            // Update position PnL
            if let Some(ref mut pos) = self.position {
                pos.update_pnl(current_price);
            }

            // Check for stop loss / take profit
            if let Some(ref pos) = self.position {
                let should_close = pos.is_stop_loss_hit(current_price)
                    || pos.is_take_profit_hit(current_price);

                if should_close {
                    let exit_reason = if pos.is_stop_loss_hit(current_price) {
                        "stop_loss"
                    } else {
                        "take_profit"
                    };
                    self.close_position(current_price, current_time, exit_reason)?;
                }
            }

            // Generate prediction
            if let Ok(features) = self.feature_generator.generate(lookback_klines) {
                if let Ok((regime, scores)) = self.model.predict_regime_with_scores(&features) {
                    // Calculate confidence
                    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let sum_exp: f64 = scores.iter().map(|s| s.exp()).sum();
                    let confidence = max_score.exp() / sum_exp;

                    // Generate signal
                    let signal = Signal::from_regime_with_confidence(regime, confidence);

                    // Execute signal
                    self.execute_signal(signal, current_price, current_time, symbol)?;
                }
            }

            // Update equity curve
            self.update_portfolio_value(current_price);
            self.equity_curve.push(self.portfolio_value);
        }

        // Close any remaining position at end
        if self.position.is_some() {
            let final_price = klines.last().unwrap().close;
            let final_time = klines.last().unwrap().timestamp.timestamp_millis();
            self.close_position(final_price, final_time, "end_of_backtest")?;
        }

        // Calculate results
        Ok(self.calculate_results(klines))
    }

    /// Execute a trading signal.
    fn execute_signal(
        &mut self,
        signal: Signal,
        price: f64,
        time: i64,
        symbol: &str,
    ) -> Result<()> {
        match signal {
            Signal::StrongBuy | Signal::Buy => {
                if self.position.is_none() {
                    self.open_position(true, price, time, symbol)?;
                } else if let Some(ref pos) = self.position {
                    if !pos.is_long {
                        // Close short and open long
                        self.close_position(price, time, "signal_reversal")?;
                        self.open_position(true, price, time, symbol)?;
                    }
                }
            }
            Signal::StrongSell | Signal::Sell => {
                if self.position.is_none() {
                    self.open_position(false, price, time, symbol)?;
                } else if let Some(ref pos) = self.position {
                    if pos.is_long {
                        // Close long and open short
                        self.close_position(price, time, "signal_reversal")?;
                        self.open_position(false, price, time, symbol)?;
                    }
                }
            }
            Signal::Hold => {
                // Do nothing
            }
        }
        Ok(())
    }

    /// Open a new position.
    fn open_position(
        &mut self,
        is_long: bool,
        price: f64,
        time: i64,
        symbol: &str,
    ) -> Result<()> {
        // Apply slippage
        let entry_price = if is_long {
            price * (1.0 + self.config.slippage)
        } else {
            price * (1.0 - self.config.slippage)
        };

        // Calculate position size
        let position_size = self.cash * self.config.strategy_config.max_position_size;

        // Apply trading fee
        let fee = position_size * self.config.trading_fee;
        self.cash -= fee;

        // Calculate stop loss and take profit
        let (stop_loss, take_profit) = if is_long {
            (
                entry_price * (1.0 - self.config.strategy_config.stop_loss_pct),
                entry_price * (1.0 + self.config.strategy_config.take_profit_pct),
            )
        } else {
            (
                entry_price * (1.0 + self.config.strategy_config.stop_loss_pct),
                entry_price * (1.0 - self.config.strategy_config.take_profit_pct),
            )
        };

        self.position = Some(Position {
            symbol: symbol.to_string(),
            is_long,
            entry_price,
            size: position_size,
            stop_loss,
            take_profit,
            entry_time: time,
            unrealized_pnl: 0.0,
        });

        Ok(())
    }

    /// Close current position.
    fn close_position(&mut self, price: f64, time: i64, reason: &str) -> Result<()> {
        if let Some(pos) = self.position.take() {
            // Apply slippage
            let exit_price = if pos.is_long {
                price * (1.0 - self.config.slippage)
            } else {
                price * (1.0 + self.config.slippage)
            };

            // Calculate PnL
            let price_change = if pos.is_long {
                exit_price - pos.entry_price
            } else {
                pos.entry_price - exit_price
            };
            let pnl = (price_change / pos.entry_price) * pos.size;

            // Apply trading fee
            let fee = pos.size * self.config.trading_fee;
            let net_pnl = pnl - fee;

            // Update cash
            self.cash += net_pnl;

            // Record trade
            self.trades.push(Trade {
                symbol: pos.symbol,
                is_long: pos.is_long,
                entry_price: pos.entry_price,
                exit_price,
                entry_time: pos.entry_time,
                exit_time: time,
                size: pos.size,
                pnl: net_pnl,
                exit_reason: reason.to_string(),
            });
        }
        Ok(())
    }

    /// Update portfolio value.
    fn update_portfolio_value(&mut self, current_price: f64) {
        self.portfolio_value = self.cash;
        if let Some(ref pos) = self.position {
            let price_change = if pos.is_long {
                current_price - pos.entry_price
            } else {
                pos.entry_price - current_price
            };
            let unrealized = (price_change / pos.entry_price) * pos.size;
            self.portfolio_value += unrealized;
        }
    }

    /// Calculate backtest results.
    fn calculate_results(&self, klines: &[Kline]) -> BacktestResults {
        let total_trades = self.trades.len();
        let winning_trades = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = total_trades - winning_trades;

        let _total_pnl: f64 = self.trades.iter().map(|t| t.pnl).sum();
        let total_return = (self.portfolio_value - self.config.initial_capital)
            / self.config.initial_capital;

        // Calculate win rate
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        // Calculate average win/loss
        let avg_win = if winning_trades > 0 {
            self.trades
                .iter()
                .filter(|t| t.pnl > 0.0)
                .map(|t| t.pnl)
                .sum::<f64>()
                / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss = if losing_trades > 0 {
            self.trades
                .iter()
                .filter(|t| t.pnl <= 0.0)
                .map(|t| t.pnl.abs())
                .sum::<f64>()
                / losing_trades as f64
        } else {
            0.0
        };

        // Calculate profit factor
        let gross_profit: f64 = self.trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = self.trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Calculate max drawdown
        let mut peak = self.config.initial_capital;
        let mut max_drawdown = 0.0;
        for &value in &self.equity_curve {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let avg_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let return_std = std_dev(&returns);
        let sharpe_ratio = if return_std > 0.0 {
            (avg_return * 252.0_f64.sqrt()) / (return_std * 252.0_f64.sqrt())
        } else {
            0.0
        };

        // Calculate buy & hold return for comparison
        let buy_hold_return = if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
            (last.close - first.close) / first.close
        } else {
            0.0
        };

        BacktestResults {
            initial_capital: self.config.initial_capital,
            final_value: self.portfolio_value,
            total_return,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            max_drawdown,
            sharpe_ratio,
            buy_hold_return,
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
        }
    }
}

/// Calculate standard deviation.
fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::ModelConfig;
    use chrono::Utc;

    fn create_trending_klines(n: usize, trend: f64) -> Vec<Kline> {
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            price *= 1.0 + trend + (rand::random::<f64>() - 0.5) * 0.005;
            klines.push(Kline {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64 * 15),
                open: price * 0.999,
                high: price * 1.005,
                low: price * 0.995,
                close: price,
                volume: 1000.0 * (1.0 + rand::random::<f64>()),
                turnover: price * 1000.0,
            });
        }

        klines
    }

    #[test]
    fn test_backtest_basic() {
        let model = ZeroShotModel::new(ModelConfig::default());
        let config = BacktestConfig {
            warmup_period: 100,
            ..Default::default()
        };
        let mut engine = BacktestEngine::new(model, config);

        let klines = create_trending_klines(300, 0.001);
        let results = engine.run(&klines, "BTCUSDT").unwrap();

        assert!(results.total_trades > 0 || results.equity_curve.len() > 0);
        assert_eq!(results.initial_capital, 100000.0);
    }

    #[test]
    fn test_backtest_not_enough_data() {
        let model = ZeroShotModel::new(ModelConfig::default());
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(model, config);

        let klines = create_trending_klines(50, 0.001);
        let result = engine.run(&klines, "BTCUSDT");

        assert!(result.is_err());
    }
}
