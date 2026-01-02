//! Strategy module for trading signal generation and execution
//!
//! This module provides trading signal generation based on TGN predictions
//! and position management utilities.

use serde::{Deserialize, Serialize};

/// Signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Go long (buy)
    Long,
    /// Go short (sell)
    Short,
    /// Close position
    Close,
    /// No action
    Hold,
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Symbol
    pub symbol: String,
    /// Signal type
    pub signal_type: SignalType,
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp when signal was generated
    pub timestamp: u64,
}

impl Signal {
    /// Create a new signal
    pub fn new(symbol: &str, signal_type: SignalType, strength: f64, confidence: f64, timestamp: u64) -> Self {
        Self {
            symbol: symbol.to_string(),
            signal_type,
            strength: strength.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            timestamp,
        }
    }

    /// Create a long signal
    pub fn long(symbol: &str, strength: f64, confidence: f64, timestamp: u64) -> Self {
        Self::new(symbol, SignalType::Long, strength, confidence, timestamp)
    }

    /// Create a short signal
    pub fn short(symbol: &str, strength: f64, confidence: f64, timestamp: u64) -> Self {
        Self::new(symbol, SignalType::Short, strength, confidence, timestamp)
    }

    /// Check if signal is actionable (high enough confidence)
    pub fn is_actionable(&self, min_confidence: f64) -> bool {
        self.confidence >= min_confidence && self.signal_type != SignalType::Hold
    }

    /// Calculate position size using Kelly criterion
    pub fn kelly_size(&self, win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
        if avg_loss == 0.0 {
            return 0.0;
        }

        let b = avg_win / avg_loss;
        let p = win_rate;
        let q = 1.0 - p;

        let kelly = (b * p - q) / b;

        // Apply confidence scaling and cap at 25%
        (kelly * self.confidence).max(0.0).min(0.25)
    }
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Minimum confidence to act on signal
    pub min_confidence: f64,
    /// Maximum position size (as fraction of portfolio)
    pub max_position_size: f64,
    /// Maximum total exposure (as fraction of portfolio)
    pub max_exposure: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Maximum leverage
    pub max_leverage: f64,
    /// Cooldown period between trades (milliseconds)
    pub cooldown_ms: u64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_position_size: 0.05,
            max_exposure: 0.15,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.04,
            max_leverage: 3.0,
            cooldown_ms: 60000, // 1 minute
        }
    }
}

/// Position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Position side (1 for long, -1 for short)
    pub side: i32,
    /// Position size
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Current unrealized PnL
    pub unrealized_pnl: f64,
    /// Entry timestamp
    pub entry_time: u64,
    /// Stop loss price
    pub stop_loss: Option<f64>,
    /// Take profit price
    pub take_profit: Option<f64>,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: &str, side: i32, size: f64, entry_price: f64, entry_time: u64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side,
            size,
            entry_price,
            unrealized_pnl: 0.0,
            entry_time,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Set stop loss and take profit
    pub fn with_stops(mut self, stop_loss_pct: f64, take_profit_pct: f64) -> Self {
        if self.side > 0 {
            // Long position
            self.stop_loss = Some(self.entry_price * (1.0 - stop_loss_pct));
            self.take_profit = Some(self.entry_price * (1.0 + take_profit_pct));
        } else {
            // Short position
            self.stop_loss = Some(self.entry_price * (1.0 + stop_loss_pct));
            self.take_profit = Some(self.entry_price * (1.0 - take_profit_pct));
        }
        self
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        let price_change = current_price - self.entry_price;
        self.unrealized_pnl = price_change * self.size * self.side as f64;
    }

    /// Check if stop loss is hit
    pub fn is_stop_loss_hit(&self, current_price: f64) -> bool {
        if let Some(stop) = self.stop_loss {
            if self.side > 0 {
                current_price <= stop
            } else {
                current_price >= stop
            }
        } else {
            false
        }
    }

    /// Check if take profit is hit
    pub fn is_take_profit_hit(&self, current_price: f64) -> bool {
        if let Some(tp) = self.take_profit {
            if self.side > 0 {
                current_price >= tp
            } else {
                current_price <= tp
            }
        } else {
            false
        }
    }

    /// Calculate return percentage
    pub fn return_pct(&self, current_price: f64) -> f64 {
        let price_change_pct = (current_price - self.entry_price) / self.entry_price;
        price_change_pct * self.side as f64
    }
}

/// Trading strategy manager
#[derive(Debug)]
pub struct TradingStrategy {
    /// Configuration
    pub config: StrategyConfig,
    /// Current positions
    positions: std::collections::HashMap<String, Position>,
    /// Last trade times per symbol
    last_trade_times: std::collections::HashMap<String, u64>,
    /// Trade history
    trade_history: Vec<TradeRecord>,
}

/// Record of a completed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    /// Symbol
    pub symbol: String,
    /// Side (1 for long, -1 for short)
    pub side: i32,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Size
    pub size: f64,
    /// Realized PnL
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Entry time
    pub entry_time: u64,
    /// Exit time
    pub exit_time: u64,
    /// Exit reason
    pub exit_reason: String,
}

impl TradingStrategy {
    /// Create new trading strategy
    pub fn new(config: StrategyConfig) -> Self {
        Self {
            config,
            positions: std::collections::HashMap::new(),
            last_trade_times: std::collections::HashMap::new(),
            trade_history: Vec::new(),
        }
    }

    /// Process a signal and return action
    pub fn process_signal(&mut self, signal: &Signal, current_price: f64) -> Option<TradeAction> {
        // Check if in cooldown
        if let Some(&last_time) = self.last_trade_times.get(&signal.symbol) {
            if signal.timestamp - last_time < self.config.cooldown_ms {
                return None;
            }
        }

        // Check confidence threshold
        if signal.confidence < self.config.min_confidence {
            return None;
        }

        // Check existing position
        if let Some(position) = self.positions.get(&signal.symbol) {
            // Check for exit signals
            if position.is_stop_loss_hit(current_price) {
                return Some(TradeAction::Close {
                    symbol: signal.symbol.clone(),
                    reason: "stop_loss".to_string(),
                });
            }

            if position.is_take_profit_hit(current_price) {
                return Some(TradeAction::Close {
                    symbol: signal.symbol.clone(),
                    reason: "take_profit".to_string(),
                });
            }

            // Check for reversal
            let position_side = if position.side > 0 { SignalType::Long } else { SignalType::Short };
            if signal.signal_type != position_side && signal.signal_type != SignalType::Hold {
                return Some(TradeAction::Reverse {
                    symbol: signal.symbol.clone(),
                    new_side: signal.signal_type,
                    size: self.calculate_position_size(signal),
                });
            }

            return None;
        }

        // No existing position - check for entry
        match signal.signal_type {
            SignalType::Long | SignalType::Short => {
                let size = self.calculate_position_size(signal);
                if size > 0.0 {
                    Some(TradeAction::Open {
                        symbol: signal.symbol.clone(),
                        side: signal.signal_type,
                        size,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Calculate position size based on signal
    fn calculate_position_size(&self, signal: &Signal) -> f64 {
        let base_size = self.config.max_position_size;
        let confidence_adjusted = base_size * signal.confidence * signal.strength;
        confidence_adjusted.min(self.config.max_position_size)
    }

    /// Open a new position
    pub fn open_position(&mut self, symbol: &str, side: SignalType, size: f64, price: f64, timestamp: u64) {
        let side_int = if side == SignalType::Long { 1 } else { -1 };
        let position = Position::new(symbol, side_int, size, price, timestamp)
            .with_stops(self.config.stop_loss_pct, self.config.take_profit_pct);

        self.positions.insert(symbol.to_string(), position);
        self.last_trade_times.insert(symbol.to_string(), timestamp);
    }

    /// Close a position
    pub fn close_position(&mut self, symbol: &str, exit_price: f64, exit_time: u64, reason: &str) -> Option<TradeRecord> {
        if let Some(position) = self.positions.remove(symbol) {
            let return_pct = position.return_pct(exit_price);
            let pnl = (exit_price - position.entry_price) * position.size * position.side as f64;

            let record = TradeRecord {
                symbol: symbol.to_string(),
                side: position.side,
                entry_price: position.entry_price,
                exit_price,
                size: position.size,
                pnl,
                return_pct,
                entry_time: position.entry_time,
                exit_time,
                exit_reason: reason.to_string(),
            };

            self.trade_history.push(record.clone());
            self.last_trade_times.insert(symbol.to_string(), exit_time);

            Some(record)
        } else {
            None
        }
    }

    /// Update all positions with current prices
    pub fn update_positions(&mut self, prices: &std::collections::HashMap<String, f64>) {
        for (symbol, position) in &mut self.positions {
            if let Some(&price) = prices.get(symbol) {
                position.update_pnl(price);
            }
        }
    }

    /// Get current position for a symbol
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> &std::collections::HashMap<String, Position> {
        &self.positions
    }

    /// Get trade history
    pub fn get_trade_history(&self) -> &[TradeRecord] {
        &self.trade_history
    }

    /// Calculate performance metrics
    pub fn calculate_metrics(&self) -> PerformanceMetrics {
        if self.trade_history.is_empty() {
            return PerformanceMetrics::default();
        }

        let total_trades = self.trade_history.len();
        let winning_trades = self.trade_history.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = total_trades - winning_trades;

        let win_rate = winning_trades as f64 / total_trades as f64;

        let total_pnl: f64 = self.trade_history.iter().map(|t| t.pnl).sum();

        let avg_win = if winning_trades > 0 {
            self.trade_history.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum::<f64>() / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss = if losing_trades > 0 {
            self.trade_history.iter().filter(|t| t.pnl <= 0.0).map(|t| t.pnl.abs()).sum::<f64>() / losing_trades as f64
        } else {
            0.0
        };

        let profit_factor = if avg_loss > 0.0 {
            avg_win * winning_trades as f64 / (avg_loss * losing_trades as f64)
        } else {
            f64::INFINITY
        };

        PerformanceMetrics {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            total_pnl,
            avg_win,
            avg_loss,
            profit_factor,
        }
    }
}

/// Trade action to execute
#[derive(Debug, Clone)]
pub enum TradeAction {
    /// Open a new position
    Open {
        symbol: String,
        side: SignalType,
        size: f64,
    },
    /// Close an existing position
    Close {
        symbol: String,
        reason: String,
    },
    /// Reverse position (close and open opposite)
    Reverse {
        symbol: String,
        new_side: SignalType,
        size: f64,
    },
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate (0.0 to 1.0)
    pub win_rate: f64,
    /// Total PnL
    pub total_pnl: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade
    pub avg_loss: f64,
    /// Profit factor
    pub profit_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::long("BTCUSDT", 0.8, 0.75, 1000);
        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.is_actionable(0.6));
        assert!(!signal.is_actionable(0.8));
    }

    #[test]
    fn test_position_pnl() {
        let mut position = Position::new("BTCUSDT", 1, 1.0, 50000.0, 1000);
        position.update_pnl(51000.0);
        assert_eq!(position.unrealized_pnl, 1000.0);
    }

    #[test]
    fn test_position_stops() {
        let position = Position::new("BTCUSDT", 1, 1.0, 50000.0, 1000)
            .with_stops(0.02, 0.04);

        assert!(position.stop_loss.is_some());
        assert!(position.take_profit.is_some());
        assert!(position.is_stop_loss_hit(49000.0));
        assert!(position.is_take_profit_hit(52100.0));
    }

    #[test]
    fn test_trading_strategy() {
        let config = StrategyConfig::default();
        let mut strategy = TradingStrategy::new(config);

        let signal = Signal::long("BTCUSDT", 0.8, 0.75, 1000);
        let action = strategy.process_signal(&signal, 50000.0);

        assert!(action.is_some());
    }
}
