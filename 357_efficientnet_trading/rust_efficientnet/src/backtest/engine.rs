//! Backtesting engine

use chrono::{DateTime, Utc};

use super::report::BacktestReport;
use crate::api::types::KlineData;
use crate::strategy::signal::{Signal, SignalType};

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: DateTime<Utc>,
    pub entry_price: f64,
    pub direction: TradeDirection,
    pub size: f64,
    pub confidence: f64,
    pub exit_time: Option<DateTime<Utc>>,
    pub exit_price: Option<f64>,
    pub pnl: Option<f64>,
    pub pnl_pct: Option<f64>,
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeDirection {
    Long,
    Short,
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub position_size: f64,
    pub transaction_cost: f64,
    pub slippage: f64,
    pub allow_short: bool,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            position_size: 1.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            allow_short: false,
            stop_loss: None,
            take_profit: None,
        }
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
    capital: f64,
    position: f64,
    position_direction: Option<TradeDirection>,
    entry_price: f64,
    trades: Vec<Trade>,
    equity_curve: Vec<f64>,
    timestamps: Vec<DateTime<Utc>>,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            capital: config.initial_capital,
            config,
            position: 0.0,
            position_direction: None,
            entry_price: 0.0,
            trades: Vec::new(),
            equity_curve: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Reset the engine state
    pub fn reset(&mut self) {
        self.capital = self.config.initial_capital;
        self.position = 0.0;
        self.position_direction = None;
        self.entry_price = 0.0;
        self.trades.clear();
        self.equity_curve.clear();
        self.timestamps.clear();
    }

    /// Run backtest with pre-computed signals
    pub fn run(&mut self, data: &KlineData, signals: &[Signal]) -> BacktestReport {
        self.reset();

        for (i, kline) in data.klines.iter().enumerate() {
            let signal = signals.get(i).cloned().unwrap_or_else(|| {
                Signal::new(SignalType::Hold, 0.0)
            });

            let price = kline.close;
            let timestamp = kline.timestamp;

            // Check stop-loss / take-profit
            if self.position != 0.0 {
                let pnl_pct = self.calculate_unrealized_pnl_pct(price);

                if let Some(sl) = self.config.stop_loss {
                    if pnl_pct <= -sl {
                        self.close_position(timestamp, price, "stop_loss");
                    }
                }

                if let Some(tp) = self.config.take_profit {
                    if pnl_pct >= tp {
                        self.close_position(timestamp, price, "take_profit");
                    }
                }
            }

            // Process signal
            match signal.signal_type {
                SignalType::Buy if self.position <= 0.0 => {
                    if self.position < 0.0 {
                        self.close_position(timestamp, price, "signal");
                    }
                    self.open_long(timestamp, price, signal.confidence);
                }
                SignalType::Sell => {
                    if self.position > 0.0 {
                        self.close_position(timestamp, price, "signal");
                    } else if self.position == 0.0 && self.config.allow_short {
                        self.open_short(timestamp, price, signal.confidence);
                    }
                }
                _ => {}
            }

            // Record equity
            let equity = self.calculate_equity(price);
            self.equity_curve.push(equity);
            self.timestamps.push(timestamp);
        }

        // Close any remaining position
        if self.position != 0.0 {
            if let Some(last_kline) = data.klines.last() {
                self.close_position(last_kline.timestamp, last_kline.close, "end");
            }
        }

        self.generate_report()
    }

    /// Open a long position
    fn open_long(&mut self, timestamp: DateTime<Utc>, price: f64, confidence: f64) {
        let trade_capital = self.capital * self.config.position_size;
        let cost = trade_capital * (self.config.transaction_cost + self.config.slippage);
        let entry_price = price * (1.0 + self.config.slippage);

        let shares = (trade_capital - cost) / entry_price;

        self.position = shares;
        self.position_direction = Some(TradeDirection::Long);
        self.entry_price = entry_price;
        self.capital -= trade_capital;

        self.trades.push(Trade {
            entry_time: timestamp,
            entry_price,
            direction: TradeDirection::Long,
            size: shares,
            confidence,
            exit_time: None,
            exit_price: None,
            pnl: None,
            pnl_pct: None,
        });
    }

    /// Open a short position
    fn open_short(&mut self, timestamp: DateTime<Utc>, price: f64, confidence: f64) {
        let trade_capital = self.capital * self.config.position_size;
        let cost = trade_capital * (self.config.transaction_cost + self.config.slippage);
        let entry_price = price * (1.0 - self.config.slippage);

        let shares = (trade_capital - cost) / entry_price;

        self.position = -shares;
        self.position_direction = Some(TradeDirection::Short);
        self.entry_price = entry_price;

        self.trades.push(Trade {
            entry_time: timestamp,
            entry_price,
            direction: TradeDirection::Short,
            size: shares,
            confidence,
            exit_time: None,
            exit_price: None,
            pnl: None,
            pnl_pct: None,
        });
    }

    /// Close the current position
    fn close_position(&mut self, timestamp: DateTime<Utc>, price: f64, _reason: &str) {
        if self.position == 0.0 {
            return;
        }

        let (exit_price, pnl) = match self.position_direction {
            Some(TradeDirection::Long) => {
                let exit = price * (1.0 - self.config.slippage);
                let proceeds = self.position.abs() * exit;
                let cost = proceeds * self.config.transaction_cost;
                let pnl = proceeds - cost - (self.position.abs() * self.entry_price);
                (exit, pnl)
            }
            Some(TradeDirection::Short) => {
                let exit = price * (1.0 + self.config.slippage);
                let proceeds = self.position.abs() * self.entry_price;
                let cost = (self.position.abs() * exit) * self.config.transaction_cost;
                let pnl = proceeds - (self.position.abs() * exit) - cost;
                (exit, pnl)
            }
            None => return,
        };

        let pnl_pct = pnl / (self.position.abs() * self.entry_price);

        // Update last trade
        if let Some(trade) = self.trades.last_mut() {
            trade.exit_time = Some(timestamp);
            trade.exit_price = Some(exit_price);
            trade.pnl = Some(pnl);
            trade.pnl_pct = Some(pnl_pct);
        }

        // Update capital
        match self.position_direction {
            Some(TradeDirection::Long) => {
                let proceeds = self.position.abs() * exit_price;
                let cost = proceeds * self.config.transaction_cost;
                self.capital += proceeds - cost;
            }
            Some(TradeDirection::Short) => {
                self.capital += pnl;
            }
            None => {}
        }

        self.position = 0.0;
        self.position_direction = None;
        self.entry_price = 0.0;
    }

    /// Calculate unrealized PnL percentage
    fn calculate_unrealized_pnl_pct(&self, current_price: f64) -> f64 {
        if self.position == 0.0 {
            return 0.0;
        }

        match self.position_direction {
            Some(TradeDirection::Long) => {
                (current_price - self.entry_price) / self.entry_price
            }
            Some(TradeDirection::Short) => {
                (self.entry_price - current_price) / self.entry_price
            }
            None => 0.0,
        }
    }

    /// Calculate current equity
    fn calculate_equity(&self, current_price: f64) -> f64 {
        if self.position == 0.0 {
            return self.capital;
        }

        let position_value = match self.position_direction {
            Some(TradeDirection::Long) => self.position.abs() * current_price,
            Some(TradeDirection::Short) => {
                self.position.abs() * self.entry_price
                    + self.position.abs() * (self.entry_price - current_price)
            }
            None => 0.0,
        };

        self.capital + position_value
    }

    /// Generate backtest report
    fn generate_report(&self) -> BacktestReport {
        BacktestReport::from_backtest(
            &self.trades,
            &self.equity_curve,
            &self.timestamps,
            self.config.initial_capital,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::Kline;
    use chrono::TimeZone;

    fn create_test_data() -> KlineData {
        let klines: Vec<Kline> = (0..100)
            .map(|i| {
                let timestamp = Utc.timestamp_opt(1700000000 + i * 3600, 0).unwrap();
                let price = 100.0 + (i as f64 * 0.1);
                Kline::new(timestamp, price, price + 1.0, price - 1.0, price, 1000.0, 100000.0)
            })
            .collect();

        KlineData::new("TEST".to_string(), "60".to_string(), klines)
    }

    #[test]
    fn test_backtest_no_signals() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);

        let data = create_test_data();
        let signals: Vec<Signal> = vec![Signal::new(SignalType::Hold, 0.0); 100];

        let report = engine.run(&data, &signals);

        assert_eq!(report.total_trades, 0);
        assert!((report.total_return_pct - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_backtest_with_signals() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);

        let data = create_test_data();
        let mut signals: Vec<Signal> = vec![Signal::new(SignalType::Hold, 0.0); 100];

        // Buy at index 10, sell at index 50
        signals[10] = Signal::new(SignalType::Buy, 0.8);
        signals[50] = Signal::new(SignalType::Sell, 0.8);

        let report = engine.run(&data, &signals);

        assert!(report.total_trades > 0);
    }
}
