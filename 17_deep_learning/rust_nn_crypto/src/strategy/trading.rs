//! Trading Strategy Implementation
//!
//! Combines signals with position management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::position::{Position, PositionSide, PositionSizing};
use super::signals::{Signal, SignalConfig, SignalGenerator};

/// Trading strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub signal_config: SignalConfig,
    pub position_sizing: PositionSizing,
    pub max_position_size: f64,
    pub stop_loss_pct: Option<f64>,
    pub take_profit_pct: Option<f64>,
    pub allow_short: bool,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            signal_config: SignalConfig::default(),
            position_sizing: PositionSizing::PercentOfCapital(0.1),
            max_position_size: 1.0,
            stop_loss_pct: Some(0.02),   // 2% stop loss
            take_profit_pct: Some(0.04), // 4% take profit
            allow_short: true,
        }
    }
}

/// Trade action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeAction {
    Open {
        side: PositionSide,
        size: f64,
        price: f64,
        timestamp: DateTime<Utc>,
    },
    Close {
        reason: CloseReason,
        price: f64,
        pnl: f64,
        timestamp: DateTime<Utc>,
    },
    Hold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloseReason {
    Signal,
    StopLoss,
    TakeProfit,
    SignalReverse,
}

/// Trading strategy
pub struct TradingStrategy {
    pub config: StrategyConfig,
    pub signal_generator: SignalGenerator,
    pub current_position: Option<Position>,
    pub capital: f64,
    pub trades: Vec<TradeRecord>,
}

/// Record of a completed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub side: PositionSide,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub pnl: f64,
    pub pnl_percent: f64,
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub close_reason: CloseReason,
}

impl TradingStrategy {
    /// Create new trading strategy
    pub fn new(config: StrategyConfig, initial_capital: f64) -> Self {
        Self {
            signal_generator: SignalGenerator::new(config.signal_config.clone()),
            config,
            current_position: None,
            capital: initial_capital,
            trades: Vec::new(),
        }
    }

    /// Process a signal and return trade action
    pub fn process_signal(
        &mut self,
        signal: Signal,
        price: f64,
        timestamp: DateTime<Utc>,
        symbol: &str,
        volatility: Option<f64>,
    ) -> TradeAction {
        // Update current position price
        if let Some(ref mut pos) = self.current_position {
            pos.update_price(price);

            // Check stop loss and take profit
            if pos.is_stop_loss_hit() {
                return self.close_position(price, timestamp, CloseReason::StopLoss);
            }
            if pos.is_take_profit_hit() {
                return self.close_position(price, timestamp, CloseReason::TakeProfit);
            }
        }

        match signal {
            Signal::StrongBuy | Signal::Buy => {
                if let Some(ref pos) = self.current_position {
                    if pos.side == PositionSide::Short {
                        // Close short and open long
                        let close_action = self.close_position(price, timestamp, CloseReason::SignalReverse);
                        // Note: In a real implementation, we'd handle both actions
                        return close_action;
                    } else if pos.side == PositionSide::Long {
                        return TradeAction::Hold;
                    }
                }
                // Open long position
                self.open_position(symbol, PositionSide::Long, price, timestamp, volatility)
            }
            Signal::StrongSell | Signal::Sell => {
                if !self.config.allow_short {
                    if let Some(ref pos) = self.current_position {
                        if pos.side == PositionSide::Long {
                            return self.close_position(price, timestamp, CloseReason::Signal);
                        }
                    }
                    return TradeAction::Hold;
                }

                if let Some(ref pos) = self.current_position {
                    if pos.side == PositionSide::Long {
                        return self.close_position(price, timestamp, CloseReason::SignalReverse);
                    } else if pos.side == PositionSide::Short {
                        return TradeAction::Hold;
                    }
                }
                // Open short position
                self.open_position(symbol, PositionSide::Short, price, timestamp, volatility)
            }
            Signal::Hold => {
                TradeAction::Hold
            }
        }
    }

    /// Open a new position
    fn open_position(
        &mut self,
        symbol: &str,
        side: PositionSide,
        price: f64,
        timestamp: DateTime<Utc>,
        volatility: Option<f64>,
    ) -> TradeAction {
        let size = self.config.position_sizing.calculate(self.capital, price, volatility);
        let size = size.min(self.config.max_position_size);

        let mut position = Position::new(
            symbol.to_string(),
            side,
            size,
            price,
            timestamp,
        );

        // Set stop loss
        if let Some(sl_pct) = self.config.stop_loss_pct {
            let sl_price = match side {
                PositionSide::Long => price * (1.0 - sl_pct),
                PositionSide::Short => price * (1.0 + sl_pct),
                PositionSide::Flat => price,
            };
            position = position.with_stop_loss(sl_price);
        }

        // Set take profit
        if let Some(tp_pct) = self.config.take_profit_pct {
            let tp_price = match side {
                PositionSide::Long => price * (1.0 + tp_pct),
                PositionSide::Short => price * (1.0 - tp_pct),
                PositionSide::Flat => price,
            };
            position = position.with_take_profit(tp_price);
        }

        self.current_position = Some(position);

        TradeAction::Open {
            side,
            size,
            price,
            timestamp,
        }
    }

    /// Close current position
    fn close_position(
        &mut self,
        price: f64,
        timestamp: DateTime<Utc>,
        reason: CloseReason,
    ) -> TradeAction {
        if let Some(mut pos) = self.current_position.take() {
            pos.update_price(price);
            let pnl = pos.pnl;

            // Record trade
            self.trades.push(TradeRecord {
                symbol: pos.symbol.clone(),
                side: pos.side,
                entry_price: pos.entry_price,
                exit_price: price,
                size: pos.size,
                pnl,
                pnl_percent: pos.pnl_percent,
                entry_time: pos.entry_time,
                exit_time: timestamp,
                close_reason: reason.clone(),
            });

            // Update capital
            self.capital += pnl;

            TradeAction::Close {
                reason,
                price,
                pnl,
                timestamp,
            }
        } else {
            TradeAction::Hold
        }
    }

    /// Get current position
    pub fn get_position(&self) -> Option<&Position> {
        self.current_position.as_ref()
    }

    /// Get trade history
    pub fn get_trades(&self) -> &[TradeRecord] {
        &self.trades
    }

    /// Get total number of trades
    pub fn trade_count(&self) -> usize {
        self.trades.len()
    }

    /// Get winning trades
    pub fn winning_trades(&self) -> Vec<&TradeRecord> {
        self.trades.iter().filter(|t| t.pnl > 0.0).collect()
    }

    /// Get losing trades
    pub fn losing_trades(&self) -> Vec<&TradeRecord> {
        self.trades.iter().filter(|t| t.pnl < 0.0).collect()
    }

    /// Calculate win rate
    pub fn win_rate(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        self.winning_trades().len() as f64 / self.trades.len() as f64
    }

    /// Calculate total PnL
    pub fn total_pnl(&self) -> f64 {
        self.trades.iter().map(|t| t.pnl).sum()
    }

    /// Calculate average trade PnL
    pub fn average_pnl(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        self.total_pnl() / self.trades.len() as f64
    }

    /// Get current capital
    pub fn get_capital(&self) -> f64 {
        self.capital
    }

    /// Reset strategy state
    pub fn reset(&mut self, initial_capital: f64) {
        self.current_position = None;
        self.capital = initial_capital;
        self.trades.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let strategy = TradingStrategy::new(StrategyConfig::default(), 10000.0);
        assert_eq!(strategy.capital, 10000.0);
        assert!(strategy.current_position.is_none());
    }

    #[test]
    fn test_open_position() {
        let mut strategy = TradingStrategy::new(StrategyConfig::default(), 10000.0);

        let action = strategy.process_signal(
            Signal::Buy,
            50000.0,
            Utc::now(),
            "BTCUSDT",
            None,
        );

        match action {
            TradeAction::Open { side, .. } => assert_eq!(side, PositionSide::Long),
            _ => panic!("Expected Open action"),
        }

        assert!(strategy.current_position.is_some());
    }

    #[test]
    fn test_close_position() {
        let mut strategy = TradingStrategy::new(StrategyConfig::default(), 10000.0);

        // Open position
        strategy.process_signal(Signal::Buy, 50000.0, Utc::now(), "BTCUSDT", None);

        // Close position
        let action = strategy.process_signal(
            Signal::Sell,
            51000.0,
            Utc::now(),
            "BTCUSDT",
            None,
        );

        match action {
            TradeAction::Close { pnl, .. } => assert!(pnl > 0.0),
            _ => panic!("Expected Close action"),
        }

        assert_eq!(strategy.trade_count(), 1);
    }
}
