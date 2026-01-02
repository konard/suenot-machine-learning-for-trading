//! Backtesting engine

use crate::backtest::metrics::{PerformanceMetrics, TradeStats};
use crate::data::Candle;
use crate::strategy::{Position, PositionManager, PositionSide, Signal, SignalType, TradeAction};

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub max_position_size: f64,
    pub commission_rate: f64,
    pub slippage_rate: f64,
    pub risk_per_trade: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            max_position_size: 1.0,
            commission_rate: 0.001, // 0.1%
            slippage_rate: 0.0005,  // 0.05%
            risk_per_trade: 0.02,   // 2%
            stop_loss_pct: 0.02,    // 2%
            take_profit_pct: 0.04,  // 4%
        }
    }
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub metrics: PerformanceMetrics,
    pub trade_stats: TradeStats,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<TradeRecord>,
}

/// Individual trade record
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub entry_time: u64,
    pub exit_time: u64,
    pub side: PositionSide,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub pnl: f64,
    pub pnl_percent: f64,
    pub exit_reason: ExitReason,
}

/// Reason for trade exit
#[derive(Debug, Clone, Copy)]
pub enum ExitReason {
    Signal,
    StopLoss,
    TakeProfit,
    EndOfData,
}

/// Backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
    capital: f64,
    position_manager: PositionManager,
    equity_curve: Vec<f64>,
    trades: Vec<TradeRecord>,
    current_entry: Option<(u64, f64, PositionSide)>,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        let mut pm = PositionManager::new(config.max_position_size);
        pm.risk_per_trade = config.risk_per_trade;
        pm.stop_loss_pct = config.stop_loss_pct;
        pm.take_profit_pct = config.take_profit_pct;

        Self {
            capital: config.initial_capital,
            config,
            position_manager: pm,
            equity_curve: Vec::new(),
            trades: Vec::new(),
            current_entry: None,
        }
    }

    /// Run backtest with signals
    pub fn run(&mut self, candles: &[Candle], signals: &[Signal]) -> BacktestResult {
        if candles.is_empty() {
            return self.empty_result();
        }

        self.equity_curve.push(self.capital);

        // Create signal lookup by timestamp
        let signal_map: std::collections::HashMap<u64, &Signal> =
            signals.iter().map(|s| (s.timestamp, s)).collect();

        for candle in candles {
            let current_price = candle.close;

            // Check stop loss / take profit first
            if let Some(action) = self.position_manager.check_exit(current_price) {
                self.process_exit(&action, candle.timestamp, current_price);
            }

            // Process signal if exists for this timestamp
            if let Some(signal) = signal_map.get(&candle.timestamp) {
                if let Some(action) = self.position_manager.process_signal(signal, self.capital) {
                    self.process_action(&action, candle.timestamp);
                }
            }

            // Update equity
            self.position_manager.update(current_price);
            let equity = self.capital + self.position_manager.position.unrealized_pnl;
            self.equity_curve.push(equity);
        }

        // Close any open position at end
        if self.position_manager.position.is_open() {
            let last_price = candles.last().unwrap().close;
            let last_time = candles.last().unwrap().timestamp;
            self.close_position(last_price, last_time, ExitReason::EndOfData);
        }

        self.calculate_result()
    }

    fn process_action(&mut self, action: &TradeAction, timestamp: u64) {
        match action {
            TradeAction::Open { side, size, price } => {
                let adjusted_price = self.apply_slippage(*price, *side);
                let commission = adjusted_price * size * self.config.commission_rate;
                self.capital -= commission;
                self.current_entry = Some((timestamp, adjusted_price, *side));
            }
            TradeAction::Close { size, price, pnl: _ } => {
                self.close_position(*price, timestamp, ExitReason::Signal);
            }
            _ => {}
        }
    }

    fn process_exit(&mut self, action: &TradeAction, timestamp: u64, price: f64) {
        match action {
            TradeAction::StopLoss { .. } => {
                self.close_position(price, timestamp, ExitReason::StopLoss);
            }
            TradeAction::TakeProfit { .. } => {
                self.close_position(price, timestamp, ExitReason::TakeProfit);
            }
            _ => {}
        }
    }

    fn close_position(&mut self, exit_price: f64, exit_time: u64, reason: ExitReason) {
        if let Some((entry_time, entry_price, side)) = self.current_entry.take() {
            let size = self.position_manager.position.size;
            let adjusted_exit = self.apply_slippage(exit_price, side.opposite());

            let pnl = match side {
                PositionSide::Long => (adjusted_exit - entry_price) * size,
                PositionSide::Short => (entry_price - adjusted_exit) * size,
                PositionSide::Flat => 0.0,
            };

            let pnl_percent = match side {
                PositionSide::Long => (adjusted_exit / entry_price - 1.0) * 100.0,
                PositionSide::Short => (1.0 - adjusted_exit / entry_price) * 100.0,
                PositionSide::Flat => 0.0,
            };

            // Apply commission
            let commission = adjusted_exit * size * self.config.commission_rate;
            let net_pnl = pnl - commission;

            self.capital += net_pnl;

            self.trades.push(TradeRecord {
                entry_time,
                exit_time,
                side,
                entry_price,
                exit_price: adjusted_exit,
                size,
                pnl: net_pnl,
                pnl_percent,
                exit_reason: reason,
            });

            self.position_manager.position = Position::flat();
        }
    }

    fn apply_slippage(&self, price: f64, side: PositionSide) -> f64 {
        match side {
            PositionSide::Long => price * (1.0 + self.config.slippage_rate),
            PositionSide::Short => price * (1.0 - self.config.slippage_rate),
            PositionSide::Flat => price,
        }
    }

    fn calculate_result(&self) -> BacktestResult {
        let metrics = PerformanceMetrics::from_equity_curve(
            &self.equity_curve,
            self.config.initial_capital,
        );
        let trade_stats = TradeStats::from_trades(&self.trades);

        BacktestResult {
            metrics,
            trade_stats,
            equity_curve: self.equity_curve.clone(),
            trades: self.trades.clone(),
        }
    }

    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            metrics: PerformanceMetrics::default(),
            trade_stats: TradeStats::default(),
            equity_curve: vec![self.config.initial_capital],
            trades: Vec::new(),
        }
    }
}

impl PositionSide {
    fn opposite(&self) -> PositionSide {
        match self {
            PositionSide::Long => PositionSide::Short,
            PositionSide::Short => PositionSide::Long,
            PositionSide::Flat => PositionSide::Flat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candles() -> Vec<Candle> {
        vec![
            Candle::new(0, 100.0, 102.0, 99.0, 101.0, 1000.0),
            Candle::new(1, 101.0, 104.0, 100.0, 103.0, 1200.0),
            Candle::new(2, 103.0, 105.0, 102.0, 104.0, 1100.0),
            Candle::new(3, 104.0, 106.0, 103.0, 105.0, 1300.0),
            Candle::new(4, 105.0, 107.0, 104.0, 106.0, 1400.0),
        ]
    }

    #[test]
    fn test_backtest_no_trades() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        let candles = sample_candles();
        let signals: Vec<Signal> = vec![];

        let result = engine.run(&candles, &signals);

        assert_eq!(result.trades.len(), 0);
        assert_eq!(result.metrics.total_return, 0.0);
    }

    #[test]
    fn test_backtest_with_trade() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        let candles = sample_candles();
        let signals = vec![
            Signal::new(SignalType::Buy, 0.8, 0, 101.0),
            Signal::new(SignalType::Sell, 0.8, 4, 106.0),
        ];

        let result = engine.run(&candles, &signals);

        assert!(!result.trades.is_empty());
    }
}
