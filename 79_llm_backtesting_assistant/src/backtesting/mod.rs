//! Backtesting module
//!
//! Provides the core backtesting engine and strategy implementations
//! for testing trading strategies on historical data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::data::Candle;
use crate::metrics::{MetricsCalculator, PerformanceMetrics, Trade, TradeSide};

/// Backtest results containing all relevant data for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    /// Strategy name
    pub strategy_name: String,
    /// Trading symbol
    pub symbol: String,
    /// Start date of the backtest
    pub start_date: DateTime<Utc>,
    /// End date of the backtest
    pub end_date: DateTime<Utc>,
    /// Initial capital
    pub initial_capital: f64,
    /// Final capital
    pub final_capital: f64,
    /// List of all trades executed
    pub trades: Vec<Trade>,
    /// Daily equity curve
    pub equity_curve: Vec<f64>,
    /// Calculated performance metrics
    pub metrics: PerformanceMetrics,
    /// Strategy parameters
    pub parameters: serde_json::Value,
    /// Market type (crypto, stock, etc.)
    pub market_type: MarketType,
}

impl BacktestResults {
    /// Create sample backtest results for testing
    pub fn sample() -> Self {
        let now = Utc::now();
        let start = now - chrono::Duration::days(365);

        // Generate sample trades
        let trades = vec![
            Trade::new(
                start + chrono::Duration::days(10),
                start + chrono::Duration::days(15),
                100.0,
                108.0,
                100.0,
                TradeSide::Long,
                "SAMPLE".to_string(),
            ),
            Trade::new(
                start + chrono::Duration::days(20),
                start + chrono::Duration::days(25),
                110.0,
                105.0,
                100.0,
                TradeSide::Long,
                "SAMPLE".to_string(),
            ),
            Trade::new(
                start + chrono::Duration::days(30),
                start + chrono::Duration::days(40),
                105.0,
                115.0,
                100.0,
                TradeSide::Long,
                "SAMPLE".to_string(),
            ),
            Trade::new(
                start + chrono::Duration::days(50),
                start + chrono::Duration::days(55),
                118.0,
                112.0,
                100.0,
                TradeSide::Long,
                "SAMPLE".to_string(),
            ),
            Trade::new(
                start + chrono::Duration::days(60),
                start + chrono::Duration::days(70),
                110.0,
                125.0,
                100.0,
                TradeSide::Long,
                "SAMPLE".to_string(),
            ),
        ];

        // Generate sample equity curve
        let equity_curve: Vec<f64> = (0..365)
            .map(|i| {
                let base = 10000.0;
                let trend = i as f64 * 5.0;
                let noise = ((i as f64 * 0.1).sin() * 200.0) + ((i as f64 * 0.05).cos() * 100.0);
                base + trend + noise
            })
            .collect();

        let calculator = MetricsCalculator::new();
        let metrics = calculator.calculate(&trades, &equity_curve);

        Self {
            strategy_name: "Sample MA Crossover".to_string(),
            symbol: "SAMPLE".to_string(),
            start_date: start,
            end_date: now,
            initial_capital: 10000.0,
            final_capital: *equity_curve.last().unwrap_or(&10000.0),
            trades,
            equity_curve,
            metrics,
            parameters: serde_json::json!({
                "fast_period": 10,
                "slow_period": 30,
                "position_size": 0.1
            }),
            market_type: MarketType::Stock,
        }
    }

    /// Create crypto-specific sample results
    pub fn sample_crypto() -> Self {
        let mut results = Self::sample();
        results.strategy_name = "Crypto MA Crossover".to_string();
        results.symbol = "BTCUSDT".to_string();
        results.market_type = MarketType::Crypto;

        // Recalculate metrics with crypto settings
        let calculator = MetricsCalculator::for_crypto();
        results.metrics = calculator.calculate(&results.trades, &results.equity_curve);

        results
    }
}

/// Market type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MarketType {
    Stock,
    Crypto,
    Forex,
    Futures,
}

impl std::fmt::Display for MarketType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketType::Stock => write!(f, "Stock"),
            MarketType::Crypto => write!(f, "Cryptocurrency"),
            MarketType::Forex => write!(f, "Forex"),
            MarketType::Futures => write!(f, "Futures"),
        }
    }
}

/// Signal generated by a strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

/// Trait for trading strategies
pub trait Strategy: Send + Sync {
    /// Generate a signal based on current market data
    fn generate_signal(&mut self, candles: &[Candle]) -> Signal;

    /// Get the strategy name
    fn name(&self) -> &str;

    /// Get strategy parameters as JSON
    fn parameters(&self) -> serde_json::Value;
}

/// Simple Moving Average Crossover Strategy
pub struct MovingAverageCrossover {
    fast_period: usize,
    slow_period: usize,
    position: Option<TradeSide>,
}

impl MovingAverageCrossover {
    /// Create a new MA crossover strategy
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            position: None,
        }
    }

    /// Calculate simple moving average
    fn calculate_sma(&self, prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period {
            return None;
        }
        let sum: f64 = prices[prices.len() - period..].iter().sum();
        Some(sum / period as f64)
    }
}

impl Strategy for MovingAverageCrossover {
    fn generate_signal(&mut self, candles: &[Candle]) -> Signal {
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

        let fast_ma = match self.calculate_sma(&closes, self.fast_period) {
            Some(ma) => ma,
            None => return Signal::Hold,
        };

        let slow_ma = match self.calculate_sma(&closes, self.slow_period) {
            Some(ma) => ma,
            None => return Signal::Hold,
        };

        // Check for crossover
        if closes.len() < 2 {
            return Signal::Hold;
        }

        let prev_closes = &closes[..closes.len() - 1];
        let prev_fast = match self.calculate_sma(prev_closes, self.fast_period) {
            Some(ma) => ma,
            None => return Signal::Hold,
        };
        let prev_slow = match self.calculate_sma(prev_closes, self.slow_period) {
            Some(ma) => ma,
            None => return Signal::Hold,
        };

        // Golden cross (bullish)
        if prev_fast <= prev_slow && fast_ma > slow_ma {
            self.position = Some(TradeSide::Long);
            return Signal::Buy;
        }

        // Death cross (bearish)
        if prev_fast >= prev_slow && fast_ma < slow_ma {
            if self.position.is_some() {
                self.position = None;
                return Signal::Sell;
            }
        }

        Signal::Hold
    }

    fn name(&self) -> &str {
        "Moving Average Crossover"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "fast_period": self.fast_period,
            "slow_period": self.slow_period
        })
    }
}

/// Simple backtesting engine
pub struct Backtester {
    initial_capital: f64,
    position_size: f64,
    commission: f64,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            position_size: 0.1, // 10% of capital per trade
            commission: 0.001,  // 0.1% commission
        }
    }

    /// Set position size as fraction of capital
    pub fn with_position_size(mut self, size: f64) -> Self {
        self.position_size = size;
        self
    }

    /// Set commission rate
    pub fn with_commission(mut self, commission: f64) -> Self {
        self.commission = commission;
        self
    }

    /// Run a backtest
    pub fn run<S: Strategy>(
        &self,
        strategy: &mut S,
        candles: &[Candle],
        symbol: &str,
        market_type: MarketType,
    ) -> BacktestResults {
        let mut capital = self.initial_capital;
        let mut equity_curve = vec![capital];
        let mut trades = Vec::new();
        let mut position: Option<(DateTime<Utc>, f64, f64)> = None; // (entry_time, entry_price, quantity)

        for i in self.min_lookback(strategy)..candles.len() {
            let candle_slice = &candles[..=i];
            let current_candle = &candles[i];
            let signal = strategy.generate_signal(candle_slice);

            match signal {
                Signal::Buy if position.is_none() => {
                    // Open long position
                    let trade_value = capital * self.position_size;
                    let quantity = trade_value / current_candle.close;
                    let commission_cost = trade_value * self.commission;
                    capital -= commission_cost;
                    position = Some((current_candle.timestamp, current_candle.close, quantity));
                }
                Signal::Sell if position.is_some() => {
                    // Close position
                    let (entry_time, entry_price, quantity) = position.take().unwrap();
                    let exit_value = quantity * current_candle.close;
                    let commission_cost = exit_value * self.commission;
                    let pnl = (current_candle.close - entry_price) * quantity - commission_cost;
                    capital += pnl;

                    trades.push(Trade::new(
                        entry_time,
                        current_candle.timestamp,
                        entry_price,
                        current_candle.close,
                        quantity,
                        TradeSide::Long,
                        symbol.to_string(),
                    ));
                }
                _ => {}
            }

            // Update equity with unrealized P&L
            let unrealized = position
                .as_ref()
                .map(|(_, entry_price, qty)| (current_candle.close - entry_price) * qty)
                .unwrap_or(0.0);
            equity_curve.push(capital + unrealized);
        }

        // Close any open position at the end
        if let Some((entry_time, entry_price, quantity)) = position {
            let last_candle = candles.last().unwrap();
            let pnl = (last_candle.close - entry_price) * quantity;
            capital += pnl;

            trades.push(Trade::new(
                entry_time,
                last_candle.timestamp,
                entry_price,
                last_candle.close,
                quantity,
                TradeSide::Long,
                symbol.to_string(),
            ));
        }

        let calculator = match market_type {
            MarketType::Crypto => MetricsCalculator::for_crypto(),
            _ => MetricsCalculator::new(),
        };
        let metrics = calculator.calculate(&trades, &equity_curve);

        BacktestResults {
            strategy_name: strategy.name().to_string(),
            symbol: symbol.to_string(),
            start_date: candles.first().map(|c| c.timestamp).unwrap_or_else(Utc::now),
            end_date: candles.last().map(|c| c.timestamp).unwrap_or_else(Utc::now),
            initial_capital: self.initial_capital,
            final_capital: capital,
            trades,
            equity_curve,
            metrics,
            parameters: strategy.parameters(),
            market_type,
        }
    }

    /// Get minimum lookback period based on strategy
    fn min_lookback<S: Strategy>(&self, _strategy: &S) -> usize {
        // Default lookback, strategies can override
        50
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_sample_candles;

    #[test]
    fn test_sample_results() {
        let results = BacktestResults::sample();
        assert!(!results.trades.is_empty());
        assert!(!results.equity_curve.is_empty());
    }

    #[test]
    fn test_ma_crossover_strategy() {
        let candles = generate_sample_candles(100, 100.0);
        let mut strategy = MovingAverageCrossover::new(10, 30);
        let backtester = Backtester::new(10000.0);
        let results = backtester.run(&mut strategy, &candles, "TEST", MarketType::Stock);

        assert_eq!(results.initial_capital, 10000.0);
        assert!(!results.equity_curve.is_empty());
    }
}
