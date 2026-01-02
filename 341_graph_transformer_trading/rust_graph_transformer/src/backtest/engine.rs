//! Backtesting engine

use super::metrics::PerformanceMetrics;
use super::report::{BacktestReport, TradeRecord};
use crate::api::types::Kline;
use crate::strategy::portfolio::Portfolio;
use crate::strategy::signal::{Signal, SignalType};
use chrono::{DateTime, TimeZone, Utc};
use std::collections::HashMap;

/// Backtest engine configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Trading fee (as decimal)
    pub trading_fee: f64,
    /// Slippage (as decimal)
    pub slippage: f64,
    /// Rebalance frequency in bars
    pub rebalance_frequency: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            trading_fee: 0.001,  // 0.1%
            slippage: 0.0005,   // 0.05%
            rebalance_frequency: 1,
        }
    }
}

/// Backtest engine
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with signals and price data
    pub fn run(
        &self,
        signals_by_time: &[(i64, Vec<Signal>)],
        prices: &HashMap<String, Vec<Kline>>,
    ) -> BacktestReport {
        let mut capital = self.config.initial_capital;
        let mut positions: HashMap<String, f64> = HashMap::new();  // symbol -> position size
        let mut trades: Vec<TradeRecord> = Vec::new();
        let mut daily_returns: Vec<f64> = Vec::new();
        let mut prev_equity = capital;

        // Get time range
        let start_time = signals_by_time.first().map(|(t, _)| *t).unwrap_or(0);
        let end_time = signals_by_time.last().map(|(t, _)| *t).unwrap_or(0);

        for (timestamp, signals) in signals_by_time {
            // Get current prices
            let current_prices: HashMap<String, f64> = prices
                .iter()
                .filter_map(|(symbol, klines)| {
                    klines
                        .iter()
                        .find(|k| k.start_time <= *timestamp)
                        .map(|k| (symbol.clone(), k.close))
                })
                .collect();

            // Calculate current equity
            let mut equity = capital;
            for (symbol, &size) in &positions {
                if let Some(&price) = current_prices.get(symbol) {
                    equity += size * price;
                }
            }

            // Record daily return
            if prev_equity > 0.0 {
                daily_returns.push(equity / prev_equity - 1.0);
            }
            prev_equity = equity;

            // Close existing positions that have opposite signals
            let mut closed_positions = Vec::new();
            for (symbol, &size) in &positions {
                let signal = signals.iter().find(|s| s.symbol == *symbol);

                let should_close = match signal {
                    Some(s) if s.signal_type == SignalType::Long && size < 0.0 => true,
                    Some(s) if s.signal_type == SignalType::Short && size > 0.0 => true,
                    Some(s) if s.signal_type == SignalType::Hold => true,
                    None => true,  // No signal = close
                    _ => false,
                };

                if should_close {
                    if let Some(&price) = current_prices.get(symbol) {
                        let exit_price = price * (1.0 - self.config.slippage.copysign(size));
                        let pnl = size * (exit_price - price) - exit_price.abs() * self.config.trading_fee;
                        capital += size * exit_price - exit_price.abs() * self.config.trading_fee;

                        trades.push(TradeRecord {
                            symbol: symbol.clone(),
                            entry_time: Utc.timestamp_millis_opt(*timestamp).unwrap(),
                            exit_time: Utc.timestamp_millis_opt(*timestamp).unwrap(),
                            entry_price: price,
                            exit_price,
                            size,
                            direction: if size > 0.0 { 1 } else { -1 },
                            pnl,
                            return_pct: pnl / (size.abs() * price),
                        });

                        closed_positions.push(symbol.clone());
                    }
                }
            }

            for symbol in closed_positions {
                positions.remove(&symbol);
            }

            // Open new positions
            for signal in signals {
                if !signal.is_actionable() {
                    continue;
                }

                if positions.contains_key(&signal.symbol) {
                    continue;
                }

                if let Some(&price) = current_prices.get(&signal.symbol) {
                    // Size based on confidence and available capital
                    let position_value = capital * signal.confidence * 0.1;  // Max 10% per position
                    let size = position_value / price * signal.direction() as f64;

                    let entry_price = price * (1.0 + self.config.slippage * signal.direction() as f64);
                    capital -= size.abs() * entry_price + size.abs() * entry_price * self.config.trading_fee;

                    positions.insert(signal.symbol.clone(), size);
                }
            }
        }

        // Close all remaining positions at end
        let final_capital = capital
            + positions
                .iter()
                .filter_map(|(symbol, &size)| {
                    prices.get(symbol).and_then(|klines| {
                        klines.last().map(|k| size * k.close)
                    })
                })
                .sum::<f64>();

        BacktestReport::new(
            "Graph Transformer Strategy",
            Utc.timestamp_millis_opt(start_time).unwrap(),
            Utc.timestamp_millis_opt(end_time).unwrap(),
            self.config.initial_capital,
            final_capital,
            trades,
            daily_returns,
        )
    }

    /// Run simple backtest with fixed portfolio
    pub fn run_simple(
        &self,
        portfolio: &Portfolio,
        prices: &HashMap<String, Vec<Kline>>,
    ) -> BacktestReport {
        let mut capital = self.config.initial_capital;
        let mut daily_returns: Vec<f64> = Vec::new();

        // Get minimum length across all price series
        let min_len = prices.values().map(|k| k.len()).min().unwrap_or(0);

        if min_len < 2 {
            return BacktestReport::new(
                "Graph Transformer Strategy",
                Utc::now(),
                Utc::now(),
                capital,
                capital,
                vec![],
                vec![],
            );
        }

        // Calculate returns for each day
        for i in 1..min_len {
            let mut day_return = 0.0;

            for position in &portfolio.positions {
                if let Some(klines) = prices.get(&position.symbol) {
                    let prev_price = klines[i - 1].close;
                    let curr_price = klines[i].close;
                    let asset_return = (curr_price - prev_price) / prev_price;
                    day_return += position.weight * asset_return * position.direction as f64;
                }
            }

            // Subtract trading costs (simplified)
            if i == 1 {
                day_return -= self.config.trading_fee * portfolio.invested_weight();
            }

            daily_returns.push(day_return);
            capital *= 1.0 + day_return;
        }

        let start_time = prices
            .values()
            .next()
            .and_then(|k| k.first())
            .map(|k| k.start_time)
            .unwrap_or(0);
        let end_time = prices
            .values()
            .next()
            .and_then(|k| k.last())
            .map(|k| k.start_time)
            .unwrap_or(0);

        BacktestReport::new(
            "Graph Transformer Strategy",
            Utc.timestamp_millis_opt(start_time).unwrap(),
            Utc.timestamp_millis_opt(end_time).unwrap(),
            self.config.initial_capital,
            capital,
            vec![],
            daily_returns,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_engine() {
        let engine = BacktestEngine::default();

        // Create mock data
        let mut prices = HashMap::new();
        let klines: Vec<Kline> = (0..30)
            .map(|i| Kline {
                start_time: i * 86400000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect();
        prices.insert("BTCUSDT".to_string(), klines);

        let portfolio = Portfolio {
            positions: vec![crate::strategy::portfolio::Position {
                symbol: "BTCUSDT".to_string(),
                weight: 1.0,
                direction: 1,
            }],
            cash_weight: 0.0,
        };

        let report = engine.run_simple(&portfolio, &prices);
        assert!(report.metrics.total_return != 0.0);
    }
}
