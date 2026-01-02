//! Backtesting engine
//!
//! This module provides the main backtesting engine for evaluating
//! trading strategies on historical data.

use anyhow::Result;
use tracing::info;

use super::analytics::{BacktestResult, PerformanceMetrics, Trade};
use crate::data::OHLCVDataset;
use crate::strategy::{PositionManager, RiskManager, Signal, SignalWithConfidence};

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission rate (fraction)
    pub commission_rate: f64,
    /// Slippage rate (fraction)
    pub slippage_rate: f64,
    /// Maximum position size (fraction of capital)
    pub max_position_size: f64,
    /// Maximum drawdown before stopping
    pub max_drawdown: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            commission_rate: 0.001,
            slippage_rate: 0.0005,
            max_position_size: 0.2,
            max_drawdown: 0.15,
        }
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
    position_manager: PositionManager,
    risk_manager: RiskManager,
    trades: Vec<Trade>,
    equity_curve: Vec<f64>,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        let position_manager =
            PositionManager::new(config.initial_capital, config.max_position_size);
        let risk_manager = RiskManager::new(
            crate::strategy::risk::RiskConfig {
                max_drawdown: config.max_drawdown,
                ..Default::default()
            },
            config.initial_capital,
        );

        Self {
            config,
            position_manager,
            risk_manager,
            trades: Vec::new(),
            equity_curve: vec![config.initial_capital],
        }
    }

    /// Run backtest with pre-computed signals
    pub fn run_with_signals(
        &mut self,
        data: &OHLCVDataset,
        signals: &[SignalWithConfidence],
    ) -> Result<BacktestResult> {
        info!("Running backtest on {} candles with {} signals", data.len(), signals.len());

        let mut current_trade: Option<Trade> = None;

        for (i, signal) in signals.iter().enumerate() {
            if i >= data.len() {
                break;
            }

            let candle = &data.data[i];
            let price = candle.close;

            // Apply slippage
            let execution_price = match signal.signal {
                Signal::Buy => price * (1.0 + self.config.slippage_rate),
                Signal::Sell => price * (1.0 - self.config.slippage_rate),
                Signal::Hold => price,
            };

            // Update position with current price
            self.position_manager.update(price);

            // Check risk limits
            let risk_action = self.risk_manager.update_equity(self.position_manager.equity());
            if self.risk_manager.is_halted() {
                info!("Trading halted due to risk limit at index {}", i);
                break;
            }

            // Check position risk
            let position = self.position_manager.position();
            let position_risk = self.risk_manager.check_position(position, price, None);

            // Process risk action
            if position_risk != crate::strategy::risk::RiskAction::Continue {
                if let Some(mut trade) = current_trade.take() {
                    trade.exit_price = execution_price;
                    trade.exit_time = candle.timestamp;
                    trade.pnl = self.position_manager.close_position(execution_price);
                    trade.commission = (trade.entry_price + trade.exit_price) * trade.size * self.config.commission_rate;
                    trade.pnl -= trade.commission;
                    self.trades.push(trade);
                    self.risk_manager.reset_trailing_stop();
                }
            }

            // Process signal
            match signal.signal {
                Signal::Buy | Signal::Sell => {
                    if current_trade.is_some() {
                        // Close existing position if different direction
                        let should_close = match (&self.position_manager.position().state, signal.signal) {
                            (crate::strategy::position::PositionState::Long, Signal::Sell) => true,
                            (crate::strategy::position::PositionState::Short, Signal::Buy) => true,
                            _ => false,
                        };

                        if should_close {
                            if let Some(mut trade) = current_trade.take() {
                                trade.exit_price = execution_price;
                                trade.exit_time = candle.timestamp;
                                trade.pnl = self.position_manager.close_position(execution_price);
                                trade.commission = (trade.entry_price + trade.exit_price) * trade.size * self.config.commission_rate;
                                trade.pnl -= trade.commission;
                                self.trades.push(trade);
                            }
                        }
                    }

                    // Open new position
                    if current_trade.is_none() {
                        let size = self.position_manager.calculate_size(execution_price, signal.confidence);
                        if self.position_manager.open_position(signal.signal, execution_price, size, candle.timestamp) {
                            current_trade = Some(Trade {
                                entry_time: candle.timestamp,
                                entry_price: execution_price,
                                exit_time: 0,
                                exit_price: 0.0,
                                size,
                                side: signal.signal,
                                pnl: 0.0,
                                commission: 0.0,
                            });
                        }
                    }
                }
                Signal::Hold => {}
            }

            // Record equity
            self.equity_curve.push(self.position_manager.equity());
        }

        // Close any remaining position
        if let Some(mut trade) = current_trade.take() {
            let last_price = data.data.last().map(|c| c.close).unwrap_or(trade.entry_price);
            trade.exit_price = last_price;
            trade.exit_time = data.data.last().map(|c| c.timestamp).unwrap_or(0);
            trade.pnl = self.position_manager.close_position(last_price);
            trade.commission = (trade.entry_price + trade.exit_price) * trade.size * self.config.commission_rate;
            trade.pnl -= trade.commission;
            self.trades.push(trade);
        }

        // Calculate metrics
        let metrics = PerformanceMetrics::from_trades(&self.trades, &self.equity_curve, self.config.initial_capital);

        Ok(BacktestResult {
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            metrics,
        })
    }

    /// Get trades
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_capital, 100000.0);
        assert_eq!(config.commission_rate, 0.001);
    }
}
