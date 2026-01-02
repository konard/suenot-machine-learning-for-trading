//! Portfolio management
//!
//! Track positions and compute portfolio metrics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Portfolio position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

impl Position {
    /// Create new position
    pub fn new(symbol: &str, quantity: f64, entry_price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            quantity,
            entry_price,
            current_price: entry_price,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        }
    }

    /// Update current price
    pub fn update_price(&mut self, price: f64) {
        self.current_price = price;
        self.unrealized_pnl = self.quantity * (price - self.entry_price);
    }

    /// Get position value
    pub fn value(&self) -> f64 {
        self.quantity * self.current_price
    }

    /// Get return percentage
    pub fn return_pct(&self) -> f64 {
        (self.current_price - self.entry_price) / self.entry_price
    }

    /// Is long position
    pub fn is_long(&self) -> bool {
        self.quantity > 0.0
    }
}

/// Portfolio state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// Cash balance
    pub cash: f64,
    /// Positions by symbol
    pub positions: HashMap<String, Position>,
    /// Initial capital
    pub initial_capital: f64,
    /// Total realized PnL
    pub realized_pnl: f64,
    /// Trade history
    pub trades: Vec<Trade>,
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: i64,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub value: f64,
    pub pnl: f64,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl Portfolio {
    /// Create new portfolio
    pub fn new(initial_capital: f64) -> Self {
        Self {
            cash: initial_capital,
            positions: HashMap::new(),
            initial_capital,
            realized_pnl: 0.0,
            trades: Vec::new(),
        }
    }

    /// Get total portfolio value
    pub fn total_value(&self) -> f64 {
        self.cash + self.positions.values().map(|p| p.value()).sum::<f64>()
    }

    /// Get total unrealized PnL
    pub fn unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }

    /// Get total return
    pub fn total_return(&self) -> f64 {
        (self.total_value() - self.initial_capital) / self.initial_capital
    }

    /// Execute buy order
    pub fn buy(&mut self, symbol: &str, quantity: f64, price: f64, timestamp: i64) -> bool {
        let value = quantity * price;

        if value > self.cash {
            return false;
        }

        self.cash -= value;

        let position = self
            .positions
            .entry(symbol.to_string())
            .or_insert_with(|| Position::new(symbol, 0.0, price));

        // Update average entry price
        let total_quantity = position.quantity + quantity;
        if total_quantity != 0.0 {
            position.entry_price = (position.quantity * position.entry_price + quantity * price)
                / total_quantity;
        }
        position.quantity = total_quantity;
        position.current_price = price;

        self.trades.push(Trade {
            timestamp,
            symbol: symbol.to_string(),
            side: TradeSide::Buy,
            quantity,
            price,
            value,
            pnl: 0.0,
        });

        true
    }

    /// Execute sell order
    pub fn sell(&mut self, symbol: &str, quantity: f64, price: f64, timestamp: i64) -> bool {
        let position = match self.positions.get_mut(symbol) {
            Some(p) => p,
            None => return false,
        };

        if quantity > position.quantity {
            return false;
        }

        let value = quantity * price;
        let pnl = quantity * (price - position.entry_price);

        self.cash += value;
        self.realized_pnl += pnl;
        position.quantity -= quantity;
        position.realized_pnl += pnl;
        position.current_price = price;

        self.trades.push(Trade {
            timestamp,
            symbol: symbol.to_string(),
            side: TradeSide::Sell,
            quantity,
            price,
            value,
            pnl,
        });

        // Remove position if empty
        if position.quantity.abs() < 1e-10 {
            self.positions.remove(symbol);
        }

        true
    }

    /// Update all positions with current prices
    pub fn update_prices(&mut self, prices: &HashMap<String, f64>) {
        for (symbol, position) in self.positions.iter_mut() {
            if let Some(&price) = prices.get(symbol) {
                position.update_price(price);
            }
        }
    }

    /// Get position weights
    pub fn weights(&self) -> HashMap<String, f64> {
        let total = self.total_value();
        if total == 0.0 {
            return HashMap::new();
        }

        self.positions
            .iter()
            .map(|(symbol, position)| (symbol.clone(), position.value() / total))
            .collect()
    }

    /// Rebalance to target weights
    pub fn rebalance(
        &mut self,
        target_weights: &HashMap<String, f64>,
        prices: &HashMap<String, f64>,
        timestamp: i64,
    ) {
        let total_value = self.total_value();

        for (symbol, &target_weight) in target_weights {
            let price = match prices.get(symbol) {
                Some(&p) => p,
                None => continue,
            };

            let target_value = total_value * target_weight;
            let current_value = self
                .positions
                .get(symbol)
                .map(|p| p.value())
                .unwrap_or(0.0);

            let diff_value = target_value - current_value;
            let diff_quantity = diff_value / price;

            if diff_quantity.abs() > 1e-10 {
                if diff_quantity > 0.0 {
                    self.buy(symbol, diff_quantity, price, timestamp);
                } else {
                    self.sell(symbol, -diff_quantity, price, timestamp);
                }
            }
        }
    }

    /// Get trade statistics
    pub fn trade_stats(&self) -> TradeStats {
        if self.trades.is_empty() {
            return TradeStats::default();
        }

        let profits: Vec<f64> = self
            .trades
            .iter()
            .filter(|t| t.side == TradeSide::Sell && t.pnl > 0.0)
            .map(|t| t.pnl)
            .collect();

        let losses: Vec<f64> = self
            .trades
            .iter()
            .filter(|t| t.side == TradeSide::Sell && t.pnl < 0.0)
            .map(|t| t.pnl)
            .collect();

        let total_trades = profits.len() + losses.len();

        TradeStats {
            total_trades,
            winning_trades: profits.len(),
            losing_trades: losses.len(),
            win_rate: if total_trades > 0 {
                profits.len() as f64 / total_trades as f64
            } else {
                0.0
            },
            gross_profit: profits.iter().sum(),
            gross_loss: losses.iter().sum::<f64>().abs(),
            profit_factor: if !losses.is_empty() {
                profits.iter().sum::<f64>() / losses.iter().sum::<f64>().abs()
            } else {
                f64::INFINITY
            },
            avg_win: if !profits.is_empty() {
                profits.iter().sum::<f64>() / profits.len() as f64
            } else {
                0.0
            },
            avg_loss: if !losses.is_empty() {
                losses.iter().sum::<f64>() / losses.len() as f64
            } else {
                0.0
            },
        }
    }
}

/// Trade statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradeStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new(10000.0);
        assert_eq!(portfolio.cash, 10000.0);
        assert_eq!(portfolio.total_value(), 10000.0);
    }

    #[test]
    fn test_buy_sell() {
        let mut portfolio = Portfolio::new(10000.0);

        // Buy
        assert!(portfolio.buy("BTCUSDT", 0.1, 50000.0, 1000));
        assert_eq!(portfolio.cash, 5000.0);
        assert!(portfolio.positions.contains_key("BTCUSDT"));

        // Sell
        assert!(portfolio.sell("BTCUSDT", 0.1, 55000.0, 2000));
        assert_eq!(portfolio.cash, 10500.0);
        assert!(!portfolio.positions.contains_key("BTCUSDT"));
    }

    #[test]
    fn test_portfolio_return() {
        let mut portfolio = Portfolio::new(10000.0);
        portfolio.buy("BTCUSDT", 0.1, 50000.0, 1000);
        portfolio.sell("BTCUSDT", 0.1, 60000.0, 2000);

        assert!((portfolio.total_return() - 0.1).abs() < 0.001);
    }
}
