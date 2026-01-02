//! Backtesting Engine
//!
//! Backtesting framework for DenseNet trading strategies.

use crate::data::candle::CandleSeries;
use super::position::{Position, PositionSide, ClosedPosition, ExitReason};
use super::signal::{TradingSignal, SignalDirection};
use super::metrics::PerformanceMetrics;

/// Backtester configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,

    /// Fee rate per trade (e.g., 0.0004 = 0.04%)
    pub fee_rate: f64,

    /// Slippage rate (e.g., 0.0001 = 0.01%)
    pub slippage: f64,

    /// Maximum position size as fraction of capital
    pub max_position_size: f64,

    /// Minimum confidence to take a trade
    pub min_confidence: f64,

    /// Use stop losses
    pub use_stop_loss: bool,

    /// Stop loss ATR multiplier
    pub stop_loss_atr: f64,

    /// Use take profit
    pub use_take_profit: bool,

    /// Take profit risk-reward ratio
    pub take_profit_rr: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            fee_rate: 0.0004, // Bybit taker fee
            slippage: 0.0001,
            max_position_size: 0.25,
            min_confidence: 0.4,
            use_stop_loss: true,
            stop_loss_atr: 2.0,
            use_take_profit: true,
            take_profit_rr: 2.0,
        }
    }
}

/// Backtester for trading strategies
pub struct Backtester {
    /// Configuration
    pub config: BacktestConfig,

    /// Current capital
    capital: f64,

    /// Current position
    position: Option<Position>,

    /// Closed trades
    trades: Vec<ClosedPosition>,

    /// Equity curve (capital over time)
    equity_curve: Vec<(u64, f64)>,

    /// Peak equity for drawdown calculation
    peak_equity: f64,

    /// Maximum drawdown seen
    max_drawdown: f64,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        let initial = config.initial_capital;
        Self {
            config,
            capital: initial,
            position: None,
            trades: Vec::new(),
            equity_curve: Vec::new(),
            peak_equity: initial,
            max_drawdown: 0.0,
        }
    }

    /// Run backtest with signals
    pub fn run(
        &mut self,
        candles: &CandleSeries,
        signals: &[TradingSignal],
    ) -> BacktestResult {
        assert_eq!(candles.len(), signals.len(), "Candles and signals must have same length");

        for (i, (candle, signal)) in candles.candles.iter().zip(signals.iter()).enumerate() {
            let current_price = candle.close;
            let timestamp = candle.timestamp;

            // Calculate ATR for stops (simplified)
            let atr = self.calculate_atr(candles, i, 14);

            // Update position excursions
            if let Some(ref mut pos) = self.position {
                pos.update_excursions(current_price);

                // Check stop loss
                if self.config.use_stop_loss && pos.is_stop_hit(current_price) {
                    self.close_position(current_price, timestamp, ExitReason::StopLoss);
                }
                // Check take profit
                else if self.config.use_take_profit && pos.is_target_hit(current_price) {
                    self.close_position(current_price, timestamp, ExitReason::TakeProfit);
                }
            }

            // Process signal
            if signal.is_actionable(self.config.min_confidence) {
                self.process_signal(signal, current_price, timestamp, atr);
            }

            // Update equity curve
            let equity = self.current_equity(current_price);
            self.equity_curve.push((timestamp, equity));

            // Update drawdown
            if equity > self.peak_equity {
                self.peak_equity = equity;
            }
            let dd = (self.peak_equity - equity) / self.peak_equity;
            if dd > self.max_drawdown {
                self.max_drawdown = dd;
            }
        }

        // Close any remaining position at last price
        if let Some(last) = candles.candles.last() {
            if self.position.is_some() {
                self.close_position(last.close, last.timestamp, ExitReason::Manual);
            }
        }

        self.generate_result()
    }

    /// Process a trading signal
    fn process_signal(&mut self, signal: &TradingSignal, price: f64, timestamp: u64, atr: f64) {
        let target_side = PositionSide::from(signal.direction);

        // Check if we need to flip or close position
        if let Some(ref pos) = self.position {
            if pos.side != target_side {
                // Close current position first
                self.close_position(price, timestamp, ExitReason::Signal);
            } else {
                // Already in correct direction
                return;
            }
        }

        // Open new position if not flat
        if target_side != PositionSide::Flat {
            self.open_position(target_side, signal, price, timestamp, atr);
        }
    }

    /// Open a new position
    fn open_position(
        &mut self,
        side: PositionSide,
        signal: &TradingSignal,
        price: f64,
        timestamp: u64,
        atr: f64,
    ) {
        // Apply slippage
        let entry_price = match side {
            PositionSide::Long => price * (1.0 + self.config.slippage),
            PositionSide::Short => price * (1.0 - self.config.slippage),
            PositionSide::Flat => return,
        };

        // Calculate position size
        let position_value = self.capital * self.config.max_position_size * signal.position_size;
        let size = position_value / entry_price;

        let mut position = Position::new(side, entry_price, size, timestamp);

        // Set stop loss
        if self.config.use_stop_loss {
            let sl = signal.calculate_stop_loss(atr, self.config.stop_loss_atr);
            position = position.with_stop_loss(sl);
        }

        // Set take profit
        if self.config.use_take_profit {
            if let Some(sl) = position.stop_loss {
                let tp = signal.calculate_take_profit(sl, self.config.take_profit_rr);
                position = position.with_take_profit(tp);
            }
        }

        self.position = Some(position);
    }

    /// Close the current position
    fn close_position(&mut self, price: f64, timestamp: u64, reason: ExitReason) {
        if let Some(pos) = self.position.take() {
            // Apply slippage
            let exit_price = match pos.side {
                PositionSide::Long => price * (1.0 - self.config.slippage),
                PositionSide::Short => price * (1.0 + self.config.slippage),
                PositionSide::Flat => price,
            };

            let closed = ClosedPosition::close(pos, exit_price, timestamp, self.config.fee_rate, reason);

            // Update capital
            self.capital += closed.realized_pnl;

            self.trades.push(closed);
        }
    }

    /// Calculate current equity
    fn current_equity(&self, current_price: f64) -> f64 {
        let unrealized = self.position
            .as_ref()
            .map(|p| p.unrealized_pnl(current_price))
            .unwrap_or(0.0);

        self.capital + unrealized
    }

    /// Calculate ATR
    fn calculate_atr(&self, candles: &CandleSeries, index: usize, period: usize) -> f64 {
        if index < period || candles.len() <= period {
            return candles.candles.get(index)
                .map(|c| c.high - c.low)
                .unwrap_or(0.0);
        }

        let start = index.saturating_sub(period);
        let mut tr_sum = 0.0;

        for i in (start + 1)..=index {
            let curr = &candles.candles[i];
            let prev = &candles.candles[i - 1];

            let h_l = curr.high - curr.low;
            let h_c = (curr.high - prev.close).abs();
            let l_c = (curr.low - prev.close).abs();

            tr_sum += h_l.max(h_c).max(l_c);
        }

        tr_sum / period as f64
    }

    /// Generate backtest result
    fn generate_result(&self) -> BacktestResult {
        let final_equity = self.equity_curve.last().map(|(_, e)| *e).unwrap_or(self.config.initial_capital);

        BacktestResult {
            initial_capital: self.config.initial_capital,
            final_capital: final_equity,
            total_return: (final_equity / self.config.initial_capital) - 1.0,
            max_drawdown: self.max_drawdown,
            num_trades: self.trades.len(),
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            metrics: PerformanceMetrics::from_trades(&self.trades, self.config.initial_capital),
        }
    }

    /// Reset backtester for new run
    pub fn reset(&mut self) {
        self.capital = self.config.initial_capital;
        self.position = None;
        self.trades.clear();
        self.equity_curve.clear();
        self.peak_equity = self.config.initial_capital;
        self.max_drawdown = 0.0;
    }
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub initial_capital: f64,
    pub final_capital: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub num_trades: usize,
    pub trades: Vec<ClosedPosition>,
    pub equity_curve: Vec<(u64, f64)>,
    pub metrics: PerformanceMetrics,
}

impl BacktestResult {
    /// Print summary
    pub fn summary(&self) -> String {
        format!(
            "Backtest Result\n\
             ===============\n\
             Initial Capital: ${:.2}\n\
             Final Capital: ${:.2}\n\
             Total Return: {:.2}%\n\
             Max Drawdown: {:.2}%\n\
             Number of Trades: {}\n\
             \n{}",
            self.initial_capital,
            self.final_capital,
            self.total_return * 100.0,
            self.max_drawdown * 100.0,
            self.num_trades,
            self.metrics,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::candle::Candle;

    fn create_test_candles() -> CandleSeries {
        let mut series = CandleSeries::new();
        let mut price = 100.0;

        for i in 0..100 {
            let change = (i as f64 * 0.1).sin() * 2.0;
            price = (price + change).max(50.0);

            series.push(Candle::new(
                i * 60000,
                price,
                price + 1.0,
                price - 1.0,
                price + change * 0.3,
                1000.0,
                price * 1000.0,
            ));
        }

        series
    }

    fn create_test_signals(n: usize) -> Vec<TradingSignal> {
        (0..n).map(|i| {
            let direction = match i % 10 {
                0..=2 => SignalDirection::Long,
                7..=9 => SignalDirection::Short,
                _ => SignalDirection::Neutral,
            };

            TradingSignal::new(
                direction,
                0.7,
                i as u64 * 60000,
                100.0,
                vec![0.2, 0.3, 0.5],
            )
        }).collect()
    }

    #[test]
    fn test_backtester_creation() {
        let config = BacktestConfig::default();
        let bt = Backtester::new(config);
        assert_eq!(bt.capital, 10000.0);
    }

    #[test]
    fn test_backtest_run() {
        let config = BacktestConfig::default();
        let mut bt = Backtester::new(config);

        let candles = create_test_candles();
        let signals = create_test_signals(candles.len());

        let result = bt.run(&candles, &signals);

        assert!(result.num_trades > 0);
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_backtest_reset() {
        let config = BacktestConfig::default();
        let mut bt = Backtester::new(config);

        let candles = create_test_candles();
        let signals = create_test_signals(candles.len());

        bt.run(&candles, &signals);
        bt.reset();

        assert!(bt.trades.is_empty());
        assert!(bt.equity_curve.is_empty());
        assert_eq!(bt.capital, bt.config.initial_capital);
    }
}
