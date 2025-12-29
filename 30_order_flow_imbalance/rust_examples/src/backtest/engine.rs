//! # Backtest Engine
//!
//! Intraday backtesting framework for order flow strategies.

use crate::data::orderbook::OrderBook;
use crate::data::snapshot::FeatureVector;
use crate::data::trade::Trade;
use crate::features::engine::FeatureEngine;
use crate::metrics::trading::TradingMetrics;
use crate::strategy::position::{ExitReason, Position, PositionManager, PositionSide};
use crate::strategy::signal::{Signal, SignalGenerator, TradingSignal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size (in base currency)
    pub position_size: f64,
    /// Commission rate (as fraction)
    pub commission_rate: f64,
    /// Slippage (bps)
    pub slippage_bps: f64,
    /// Maximum holding time (seconds)
    pub max_holding_time: i64,
    /// Maximum daily loss
    pub max_daily_loss: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            position_size: 0.1, // 0.1 BTC
            commission_rate: 0.0004, // 4 bps maker
            slippage_bps: 1.0,
            max_holding_time: 300, // 5 minutes
            max_daily_loss: 500.0,
        }
    }
}

/// Backtest engine
pub struct BacktestEngine {
    /// Configuration
    config: BacktestConfig,
    /// Feature engine
    feature_engine: FeatureEngine,
    /// Signal generator
    signal_generator: SignalGenerator,
    /// Position manager
    position_manager: PositionManager,
    /// Trading metrics
    metrics: TradingMetrics,
    /// Equity curve
    equity_curve: Vec<EquityPoint>,
    /// Current equity
    current_equity: f64,
    /// Trade log
    trade_log: Vec<TradeLog>,
}

/// Equity point for curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub timestamp: DateTime<Utc>,
    pub equity: f64,
    pub drawdown: f64,
}

/// Trade log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeLog {
    pub timestamp: DateTime<Utc>,
    pub side: String,
    pub price: f64,
    pub size: f64,
    pub pnl: f64,
    pub exit_reason: String,
}

impl BacktestEngine {
    /// Create new engine
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            current_equity: config.initial_capital,
            position_manager: PositionManager::new(
                config.position_size * 2.0,
                config.max_holding_time,
                config.max_daily_loss,
            ),
            config,
            feature_engine: FeatureEngine::new(),
            signal_generator: SignalGenerator::default(),
            metrics: TradingMetrics::new(),
            equity_curve: Vec::new(),
            trade_log: Vec::new(),
        }
    }

    /// Set signal generator
    pub fn set_signal_generator(&mut self, generator: SignalGenerator) {
        self.signal_generator = generator;
    }

    /// Process an order book update
    pub fn process_orderbook(&mut self, book: &OrderBook) {
        // Update feature engine
        self.feature_engine.update_orderbook(book);

        // Get current price
        let mid_price = match book.mid_price() {
            Some(p) => p,
            None => return,
        };

        let spread_bps = book.spread_bps().unwrap_or(100.0);

        // Check for exit first
        if let Some(exit_reason) = self.position_manager.update(mid_price) {
            self.execute_exit(mid_price, exit_reason);
        }

        // Generate features
        let features = self.feature_engine.extract_features(book);

        // Generate signal
        let signal = self.signal_generator.generate(&features, mid_price, spread_bps);

        // Execute signal if flat
        if self.position_manager.is_flat() && signal.signal != Signal::Hold {
            self.execute_entry(&signal, mid_price);
        }

        // Update equity curve
        self.update_equity(book.timestamp, mid_price);
    }

    /// Process a trade
    pub fn process_trade(&mut self, trade: &Trade) {
        self.feature_engine.update_trade(trade);
    }

    /// Execute entry
    fn execute_entry(&mut self, signal: &TradingSignal, price: f64) {
        let slippage = price * self.config.slippage_bps / 10000.0;

        let (side, entry_price) = match signal.signal {
            Signal::Long => (PositionSide::Long, price + slippage),
            Signal::Short => (PositionSide::Short, price - slippage),
            _ => return,
        };

        let position = Position::new(
            "BTCUSDT".to_string(),
            side,
            self.config.position_size,
            entry_price,
            signal.stop_loss,
            signal.take_profit,
        );

        self.position_manager.open_position(position);
    }

    /// Execute exit
    fn execute_exit(&mut self, price: f64, reason: ExitReason) {
        let slippage = price * self.config.slippage_bps / 10000.0;

        let exit_price = if let Some(pos) = self.position_manager.position() {
            match pos.side {
                PositionSide::Long => price - slippage,
                PositionSide::Short => price + slippage,
                PositionSide::Flat => price,
            }
        } else {
            price
        };

        if let Some(trade) = self.position_manager.close_position(exit_price, reason) {
            // Calculate costs
            let notional = trade.size * (trade.entry_price + trade.exit_price) / 2.0;
            let commission = notional * self.config.commission_rate * 2.0; // Entry + exit

            let net_pnl = trade.pnl - commission;

            // Record in metrics
            self.metrics.record_trade(trade.pnl, commission);
            self.current_equity += net_pnl;

            // Log trade
            self.trade_log.push(TradeLog {
                timestamp: trade.exit_time,
                side: format!("{:?}", trade.side),
                price: trade.exit_price,
                size: trade.size,
                pnl: net_pnl,
                exit_reason: format!("{:?}", trade.exit_reason),
            });
        }
    }

    /// Update equity curve
    fn update_equity(&mut self, timestamp: DateTime<Utc>, current_price: f64) {
        // Include unrealized P&L
        let mut equity = self.current_equity;

        if let Some(pos) = self.position_manager.position() {
            let mut pos_clone = pos.clone();
            pos_clone.update_pnl(current_price);
            equity += pos_clone.unrealized_pnl;
        }

        // Calculate drawdown
        let peak = self
            .equity_curve
            .iter()
            .map(|e| e.equity)
            .fold(self.config.initial_capital, f64::max);
        let drawdown = (peak - equity) / peak;

        self.equity_curve.push(EquityPoint {
            timestamp,
            equity,
            drawdown,
        });
    }

    /// Get trading metrics
    pub fn metrics(&self) -> &TradingMetrics {
        &self.metrics
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> &[EquityPoint] {
        &self.equity_curve
    }

    /// Get trade log
    pub fn trade_log(&self) -> &[TradeLog] {
        &self.trade_log
    }

    /// Get final equity
    pub fn final_equity(&self) -> f64 {
        self.current_equity
    }

    /// Get total return
    pub fn total_return(&self) -> f64 {
        (self.current_equity - self.config.initial_capital) / self.config.initial_capital * 100.0
    }

    /// Generate report
    pub fn report(&self) -> String {
        format!(
            r#"
═══════════════════════════════════════════════════════════════
                     BACKTEST REPORT
═══════════════════════════════════════════════════════════════

CAPITAL
───────────────────────────────────────
Initial Capital:     ${:.2}
Final Capital:       ${:.2}
Total Return:        {:.2}%

{}

EQUITY CURVE
───────────────────────────────────────
Data Points:         {}
Peak Equity:         ${:.2}
Max Drawdown:        {:.2}%

═══════════════════════════════════════════════════════════════
"#,
            self.config.initial_capital,
            self.current_equity,
            self.total_return(),
            self.metrics.summary(),
            self.equity_curve.len(),
            self.equity_curve
                .iter()
                .map(|e| e.equity)
                .fold(0.0_f64, f64::max),
            self.equity_curve
                .iter()
                .map(|e| e.drawdown)
                .fold(0.0_f64, f64::max)
                * 100.0
        )
    }

    /// Reset engine
    pub fn reset(&mut self) {
        self.feature_engine.reset();
        self.position_manager = PositionManager::new(
            self.config.position_size * 2.0,
            self.config.max_holding_time,
            self.config.max_daily_loss,
        );
        self.metrics = TradingMetrics::new();
        self.equity_curve.clear();
        self.trade_log.clear();
        self.current_equity = self.config.initial_capital;
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::orderbook::OrderBookLevel;

    fn create_book(bid: f64, ask: f64) -> OrderBook {
        let bids = vec![OrderBookLevel::new(bid, 10.0, 1)];
        let asks = vec![OrderBookLevel::new(ask, 10.0, 1)];
        OrderBook::new("BTCUSDT".to_string(), Utc::now(), bids, asks)
    }

    #[test]
    fn test_backtest_engine() {
        let mut engine = BacktestEngine::new(BacktestConfig::default());

        // Process some order books
        for i in 0..100 {
            let price = 50000.0 + (i as f64) * 10.0;
            let book = create_book(price - 0.5, price + 0.5);
            engine.process_orderbook(&book);
        }

        assert!(!engine.equity_curve.is_empty());
    }
}
