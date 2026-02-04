//! Backtesting engine

use crate::strategy::signal::{Signal, SignalType};
use super::report::BacktestReport;

/// Single trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: i64,
    pub exit_time: Option<i64>,
    pub symbol: String,
    pub side: SignalType,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub size: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub status: TradeStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeStatus {
    Open,
    Closed,
    StopLoss,
    TakeProfit,
}

/// Backtesting engine
pub struct BacktestEngine {
    /// Initial capital
    initial_capital: f64,
    /// Current cash
    cash: f64,
    /// Commission rate
    commission: f64,
    /// Slippage rate
    slippage: f64,
    /// Stop loss percentage
    stop_loss: f64,
    /// Take profit percentage
    take_profit: f64,
    /// Trade history
    trades: Vec<Trade>,
    /// Current trade (if any)
    current_trade: Option<Trade>,
    /// Equity history
    equity_history: Vec<(i64, f64)>,
    /// Stop loss price for current trade
    stop_loss_price: f64,
    /// Take profit price for current trade
    take_profit_price: f64,
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(10000.0)
    }
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            cash: initial_capital,
            commission: 0.001,
            slippage: 0.0005,
            stop_loss: 0.02,
            take_profit: 0.04,
            trades: Vec::new(),
            current_trade: None,
            equity_history: Vec::new(),
            stop_loss_price: 0.0,
            take_profit_price: 0.0,
        }
    }

    /// Configure commission rate
    pub fn with_commission(mut self, commission: f64) -> Self {
        self.commission = commission;
        self
    }

    /// Configure slippage rate
    pub fn with_slippage(mut self, slippage: f64) -> Self {
        self.slippage = slippage;
        self
    }

    /// Configure stop loss
    pub fn with_stop_loss(mut self, stop_loss: f64) -> Self {
        self.stop_loss = stop_loss;
        self
    }

    /// Configure take profit
    pub fn with_take_profit(mut self, take_profit: f64) -> Self {
        self.take_profit = take_profit;
        self
    }

    /// Reset the engine
    pub fn reset(&mut self) {
        self.cash = self.initial_capital;
        self.trades.clear();
        self.current_trade = None;
        self.equity_history.clear();
    }

    /// Run backtest
    pub fn run(
        &mut self,
        signals: &[Signal],
        prices: &[f64],
        timestamps: &[i64],
        symbol: &str,
    ) -> BacktestReport {
        self.reset();

        for (i, (signal, &price)) in signals.iter().zip(prices.iter()).enumerate() {
            let timestamp = timestamps[i];

            // Update equity
            let equity = self.calculate_equity(price);
            self.equity_history.push((timestamp, equity));

            // Check stops for current trade
            if self.current_trade.is_some() {
                let closed = self.check_stops(price, timestamp);
                if !closed && self.should_close_position(signal) {
                    self.close_position(price, timestamp, TradeStatus::Closed);
                }
            }

            // Open new position if no current trade
            if self.current_trade.is_none() && signal.is_actionable() {
                self.open_position(signal, price, timestamp, symbol);
            }
        }

        // Close remaining position at end
        if self.current_trade.is_some() && !prices.is_empty() {
            let final_price = *prices.last().unwrap();
            let final_time = *timestamps.last().unwrap();
            self.close_position(final_price, final_time, TradeStatus::Closed);
        }

        self.generate_report()
    }

    /// Calculate current equity
    fn calculate_equity(&self, current_price: f64) -> f64 {
        let mut equity = self.cash;

        if let Some(ref trade) = self.current_trade {
            let unrealized_pnl = match trade.side {
                SignalType::Long => (current_price - trade.entry_price) * trade.size,
                SignalType::Short => (trade.entry_price - current_price) * trade.size,
                SignalType::Hold => 0.0,
            };
            equity += unrealized_pnl;
        }

        equity
    }

    /// Open a new position
    fn open_position(&mut self, signal: &Signal, price: f64, timestamp: i64, symbol: &str) {
        // Apply slippage
        let adjusted_price = match signal.signal_type {
            SignalType::Long => price * (1.0 + self.slippage),
            SignalType::Short => price * (1.0 - self.slippage),
            SignalType::Hold => return,
        };

        // Calculate position size
        let position_value = self.cash * signal.position_size;
        let commission = position_value * self.commission;
        let size = (position_value - commission) / adjusted_price;

        // Calculate stop loss and take profit prices
        let (sl, tp) = match signal.signal_type {
            SignalType::Long => {
                let sl = adjusted_price * (1.0 - self.stop_loss);
                let tp = adjusted_price * (1.0 + self.take_profit);
                (sl, tp)
            }
            SignalType::Short => {
                let sl = adjusted_price * (1.0 + self.stop_loss);
                let tp = adjusted_price * (1.0 - self.take_profit);
                (sl, tp)
            }
            SignalType::Hold => (0.0, 0.0),
        };

        self.stop_loss_price = sl;
        self.take_profit_price = tp;

        self.current_trade = Some(Trade {
            entry_time: timestamp,
            exit_time: None,
            symbol: symbol.to_string(),
            side: signal.signal_type,
            entry_price: adjusted_price,
            exit_price: None,
            size,
            pnl: 0.0,
            pnl_pct: 0.0,
            status: TradeStatus::Open,
        });

        self.cash -= position_value;
    }

    /// Close current position
    fn close_position(&mut self, price: f64, timestamp: i64, status: TradeStatus) {
        if let Some(mut trade) = self.current_trade.take() {
            // Apply slippage
            let adjusted_price = match trade.side {
                SignalType::Long => price * (1.0 - self.slippage),
                SignalType::Short => price * (1.0 + self.slippage),
                SignalType::Hold => price,
            };

            // Calculate PnL
            let pnl = match trade.side {
                SignalType::Long => (adjusted_price - trade.entry_price) * trade.size,
                SignalType::Short => (trade.entry_price - adjusted_price) * trade.size,
                SignalType::Hold => 0.0,
            };

            // Apply exit commission
            let position_value = adjusted_price * trade.size;
            let commission = position_value * self.commission;
            let net_pnl = pnl - commission;

            trade.exit_time = Some(timestamp);
            trade.exit_price = Some(adjusted_price);
            trade.pnl = net_pnl;
            trade.pnl_pct = net_pnl / (trade.entry_price * trade.size);
            trade.status = status;

            self.cash += position_value + net_pnl;
            self.trades.push(trade);
        }
    }

    /// Check stop loss and take profit
    fn check_stops(&mut self, price: f64, timestamp: i64) -> bool {
        if let Some(ref trade) = self.current_trade {
            match trade.side {
                SignalType::Long => {
                    if price <= self.stop_loss_price {
                        self.close_position(self.stop_loss_price, timestamp, TradeStatus::StopLoss);
                        return true;
                    }
                    if price >= self.take_profit_price {
                        self.close_position(self.take_profit_price, timestamp, TradeStatus::TakeProfit);
                        return true;
                    }
                }
                SignalType::Short => {
                    if price >= self.stop_loss_price {
                        self.close_position(self.stop_loss_price, timestamp, TradeStatus::StopLoss);
                        return true;
                    }
                    if price <= self.take_profit_price {
                        self.close_position(self.take_profit_price, timestamp, TradeStatus::TakeProfit);
                        return true;
                    }
                }
                SignalType::Hold => {}
            }
        }
        false
    }

    /// Check if position should be closed due to signal change
    fn should_close_position(&self, signal: &Signal) -> bool {
        if let Some(ref trade) = self.current_trade {
            match (trade.side, signal.signal_type) {
                (SignalType::Long, SignalType::Short) => true,
                (SignalType::Short, SignalType::Long) => true,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Generate backtest report
    fn generate_report(&self) -> BacktestReport {
        BacktestReport::from_trades(&self.trades, &self.equity_history, self.initial_capital)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_basic() {
        let mut engine = BacktestEngine::new(10000.0);

        // Create simple signals
        let signals: Vec<Signal> = (0..10).map(|i| {
            let st = if i % 2 == 0 { SignalType::Long } else { SignalType::Hold };
            Signal::new(st, 0.7, 0.01, 0.005, 0.8, 0).with_position_size(0.05)
        }).collect();

        let prices: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        let timestamps: Vec<i64> = (0..10).map(|i| i * 60000).collect();

        let report = engine.run(&signals, &prices, &timestamps, "BTCUSDT");

        assert!(report.total_trades > 0 || report.equity_curve.len() > 0);
    }
}
