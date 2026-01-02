//! Backtesting engine for evaluating trading strategies.

mod metrics;

pub use metrics::*;

use crate::data::Candle;
use crate::strategy::Signal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Position in a specific asset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Position {
    /// Quantity held
    pub quantity: f64,
    /// Average entry price
    pub avg_entry_price: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Realized PnL
    pub realized_pnl: f64,
}

/// A single trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Timestamp
    pub timestamp: u64,
    /// Symbol
    pub symbol: String,
    /// Side (positive = buy, negative = sell)
    pub quantity: f64,
    /// Execution price
    pub price: f64,
    /// Commission paid
    pub commission: f64,
    /// PnL from this trade (for closing trades)
    pub pnl: f64,
}

/// Portfolio state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// Cash balance
    pub cash: f64,
    /// Positions by symbol
    pub positions: HashMap<String, Position>,
    /// Total portfolio value
    pub total_value: f64,
    /// Trade history
    pub trades: Vec<Trade>,
}

impl Portfolio {
    /// Create a new portfolio with initial cash.
    pub fn new(initial_cash: f64) -> Self {
        Self {
            cash: initial_cash,
            positions: HashMap::new(),
            total_value: initial_cash,
            trades: Vec::new(),
        }
    }

    /// Update portfolio value with current prices.
    pub fn update_value(&mut self, prices: &HashMap<String, f64>) {
        let mut position_value = 0.0;

        for (symbol, position) in &mut self.positions {
            if let Some(&price) = prices.get(symbol) {
                let value = position.quantity * price;
                position.unrealized_pnl = value - (position.quantity * position.avg_entry_price);
                position_value += value;
            }
        }

        self.total_value = self.cash + position_value;
    }

    /// Execute a trade.
    pub fn execute_trade(
        &mut self,
        symbol: &str,
        quantity: f64,
        price: f64,
        commission_rate: f64,
        timestamp: u64,
    ) -> Option<Trade> {
        let trade_value = quantity.abs() * price;
        let commission = trade_value * commission_rate;

        // Check if we have enough cash for a buy
        if quantity > 0.0 && self.cash < trade_value + commission {
            return None;
        }

        let position = self.positions.entry(symbol.to_string()).or_default();

        let pnl = if quantity > 0.0 {
            // Buy
            let new_quantity = position.quantity + quantity;
            let new_avg = if new_quantity > 0.0 {
                (position.quantity * position.avg_entry_price + quantity * price) / new_quantity
            } else {
                price
            };
            position.quantity = new_quantity;
            position.avg_entry_price = new_avg;
            self.cash -= trade_value + commission;
            0.0
        } else {
            // Sell
            let sell_quantity = quantity.abs().min(position.quantity);
            let pnl = sell_quantity * (price - position.avg_entry_price);
            position.quantity -= sell_quantity;
            position.realized_pnl += pnl;
            self.cash += sell_quantity * price - commission;
            pnl
        };

        let trade = Trade {
            timestamp,
            symbol: symbol.to_string(),
            quantity,
            price,
            commission,
            pnl,
        };

        self.trades.push(trade.clone());
        Some(trade)
    }

    /// Get current position for a symbol.
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get total realized PnL.
    pub fn total_realized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.realized_pnl).sum()
    }

    /// Get total unrealized PnL.
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }
}

/// Backtesting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission rate (as fraction, e.g., 0.001 = 0.1%)
    pub commission_rate: f64,
    /// Slippage model (as fraction)
    pub slippage: f64,
    /// Maximum position size as fraction of portfolio
    pub max_position_size: f64,
    /// Rebalance frequency (in number of bars)
    pub rebalance_frequency: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission_rate: 0.001,
            slippage: 0.0005,
            max_position_size: 0.2,
            rebalance_frequency: 1,
        }
    }
}

/// Backtest result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Final portfolio value
    pub final_value: f64,
    /// Total return
    pub total_return: f64,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Equity curve (timestamps and values)
    pub equity_curve: Vec<(u64, f64)>,
    /// Trade history
    pub trades: Vec<Trade>,
    /// Drawdown series
    pub drawdowns: Vec<f64>,
}

/// Backtester engine.
pub struct Backtester {
    /// Configuration
    config: BacktestConfig,
    /// Portfolio
    portfolio: Portfolio,
    /// Equity curve
    equity_curve: Vec<(u64, f64)>,
    /// Peak portfolio value (for drawdown calculation)
    peak_value: f64,
    /// Returns for metrics calculation
    returns: Vec<f64>,
    /// Previous portfolio value
    prev_value: f64,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(config: BacktestConfig) -> Self {
        let portfolio = Portfolio::new(config.initial_capital);
        Self {
            prev_value: config.initial_capital,
            peak_value: config.initial_capital,
            config,
            portfolio,
            equity_curve: Vec::new(),
            returns: Vec::new(),
        }
    }

    /// Run backtest on historical data.
    pub fn run(
        &mut self,
        candles: &HashMap<String, Vec<Candle>>,
        signals: &[Vec<Signal>],
    ) -> BacktestResult {
        // Get the minimum length across all symbols
        let min_len = candles
            .values()
            .map(|c| c.len())
            .min()
            .unwrap_or(0);

        let signals_len = signals.len();
        let bars_to_process = min_len.min(signals_len);

        for i in 0..bars_to_process {
            // Get current prices
            let prices: HashMap<String, f64> = candles
                .iter()
                .filter_map(|(symbol, c)| {
                    c.get(i).map(|candle| (symbol.clone(), candle.close))
                })
                .collect();

            // Get timestamp
            let timestamp = candles
                .values()
                .next()
                .and_then(|c| c.get(i))
                .map(|c| c.timestamp)
                .unwrap_or(0);

            // Update portfolio value
            self.portfolio.update_value(&prices);

            // Record equity
            self.equity_curve.push((timestamp, self.portfolio.total_value));

            // Calculate return
            let ret = (self.portfolio.total_value - self.prev_value) / self.prev_value;
            self.returns.push(ret);
            self.prev_value = self.portfolio.total_value;

            // Update peak
            if self.portfolio.total_value > self.peak_value {
                self.peak_value = self.portfolio.total_value;
            }

            // Rebalance if needed
            if i % self.config.rebalance_frequency == 0 && i < signals_len {
                self.rebalance(&signals[i], &prices, timestamp);
            }
        }

        // Calculate final metrics
        let metrics = self.calculate_metrics();
        let drawdowns = self.calculate_drawdowns();

        BacktestResult {
            final_value: self.portfolio.total_value,
            total_return: (self.portfolio.total_value - self.config.initial_capital)
                / self.config.initial_capital,
            metrics,
            equity_curve: self.equity_curve.clone(),
            trades: self.portfolio.trades.clone(),
            drawdowns,
        }
    }

    /// Rebalance portfolio based on signals.
    fn rebalance(
        &mut self,
        signals: &[Signal],
        prices: &HashMap<String, f64>,
        timestamp: u64,
    ) {
        // Calculate target weights
        let mut target_weights: HashMap<String, f64> = HashMap::new();

        // Sum of absolute scores for normalization
        let total_abs_score: f64 = signals
            .iter()
            .filter(|s| s.score.abs() > 0.1)
            .map(|s| s.score.abs())
            .sum();

        if total_abs_score > 0.0 {
            for signal in signals {
                if signal.score.abs() > 0.1 {
                    let weight = (signal.score / total_abs_score) * self.config.max_position_size;
                    target_weights.insert(signal.symbol.clone(), weight.max(0.0)); // Long only
                }
            }
        }

        // Calculate current weights
        let total_value = self.portfolio.total_value;
        let current_weights: HashMap<String, f64> = self
            .portfolio
            .positions
            .iter()
            .filter_map(|(symbol, pos)| {
                prices.get(symbol).map(|&price| {
                    (symbol.clone(), (pos.quantity * price) / total_value)
                })
            })
            .collect();

        // Execute trades to reach target weights
        for (symbol, &target_weight) in &target_weights {
            let current_weight = current_weights.get(symbol).copied().unwrap_or(0.0);
            let weight_diff = target_weight - current_weight;

            if weight_diff.abs() > 0.02 {
                // 2% threshold
                if let Some(&price) = prices.get(symbol) {
                    let trade_value = weight_diff * total_value;
                    let quantity = trade_value / price;

                    // Apply slippage
                    let execution_price = if quantity > 0.0 {
                        price * (1.0 + self.config.slippage)
                    } else {
                        price * (1.0 - self.config.slippage)
                    };

                    self.portfolio.execute_trade(
                        symbol,
                        quantity,
                        execution_price,
                        self.config.commission_rate,
                        timestamp,
                    );
                }
            }
        }

        // Close positions not in target
        let symbols_to_close: Vec<String> = self
            .portfolio
            .positions
            .keys()
            .filter(|s| !target_weights.contains_key(*s))
            .cloned()
            .collect();

        for symbol in symbols_to_close {
            if let Some(pos) = self.portfolio.positions.get(&symbol) {
                if pos.quantity > 0.0 {
                    if let Some(&price) = prices.get(&symbol) {
                        let execution_price = price * (1.0 - self.config.slippage);
                        self.portfolio.execute_trade(
                            &symbol,
                            -pos.quantity,
                            execution_price,
                            self.config.commission_rate,
                            timestamp,
                        );
                    }
                }
            }
        }
    }

    /// Calculate performance metrics.
    fn calculate_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics::from_returns(&self.returns, 252) // Assuming daily data
    }

    /// Calculate drawdown series.
    fn calculate_drawdowns(&self) -> Vec<f64> {
        let mut peak = self.config.initial_capital;
        self.equity_curve
            .iter()
            .map(|(_, value)| {
                if *value > peak {
                    peak = *value;
                }
                (peak - value) / peak
            })
            .collect()
    }

    /// Get portfolio reference.
    pub fn portfolio(&self) -> &Portfolio {
        &self.portfolio
    }
}

/// Simple backtest runner for quick testing.
pub fn quick_backtest(
    candles: &HashMap<String, Vec<Candle>>,
    signals_fn: impl Fn(&HashMap<String, f64>) -> HashMap<String, f64>,
    initial_capital: f64,
) -> BacktestResult {
    let config = BacktestConfig {
        initial_capital,
        ..Default::default()
    };

    let mut backtester = Backtester::new(config);

    // Generate signals for each bar
    let min_len = candles.values().map(|c| c.len()).min().unwrap_or(0);

    let signals: Vec<Vec<Signal>> = (0..min_len)
        .map(|i| {
            let prices: HashMap<String, f64> = candles
                .iter()
                .filter_map(|(symbol, c)| c.get(i).map(|candle| (symbol.clone(), candle.close)))
                .collect();

            let timestamp = candles
                .values()
                .next()
                .and_then(|c| c.get(i))
                .map(|c| c.timestamp)
                .unwrap_or(0);

            let raw_signals = signals_fn(&prices);

            raw_signals
                .into_iter()
                .map(|(symbol, score)| Signal::new(symbol, score, 0.8, timestamp))
                .collect()
        })
        .collect();

    backtester.run(candles, &signals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles(prices: &[f64], symbol: &str) -> Vec<Candle> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| Candle {
                timestamp: 1000 + i as u64 * 3600,
                open: p,
                high: p * 1.01,
                low: p * 0.99,
                close: p,
                volume: 1000.0,
                symbol: symbol.to_string(),
            })
            .collect()
    }

    #[test]
    fn test_portfolio_trade() {
        let mut portfolio = Portfolio::new(10000.0);

        // Buy
        let trade = portfolio.execute_trade("BTCUSDT", 1.0, 100.0, 0.001, 1000);
        assert!(trade.is_some());
        assert!((portfolio.cash - (10000.0 - 100.0 - 0.1)).abs() < 0.01);

        // Sell
        let trade = portfolio.execute_trade("BTCUSDT", -0.5, 110.0, 0.001, 2000);
        assert!(trade.is_some());

        let pos = portfolio.get_position("BTCUSDT").unwrap();
        assert!((pos.quantity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_backtest() {
        let mut candles = HashMap::new();
        let btc_prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
        candles.insert("BTCUSDT".to_string(), make_candles(&btc_prices, "BTCUSDT"));

        let signals: Vec<Vec<Signal>> = (0..100)
            .map(|i| {
                vec![Signal::new("BTCUSDT", 0.5, 0.8, 1000 + i * 3600)]
            })
            .collect();

        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config);
        let result = backtester.run(&candles, &signals);

        // With rising prices and long position, should have positive return
        assert!(result.total_return > 0.0);
    }
}
