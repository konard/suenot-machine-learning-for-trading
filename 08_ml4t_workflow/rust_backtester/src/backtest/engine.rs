//! Main backtesting engine.

use crate::backtest::{BrokerConfig, SimulatedBroker, BacktestResult};
use crate::models::Candle;
use crate::strategies::{Signal, Strategy};
use crate::utils::PerformanceMetrics;
use std::collections::HashMap;
use tracing::info;

/// Configuration for the backtest engine.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Broker configuration
    pub broker: BrokerConfig,
    /// Position sizing as fraction of equity (0.0 to 1.0)
    pub position_size: f64,
    /// Warmup period (number of candles to skip for indicator calculation)
    pub warmup_period: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            broker: BrokerConfig::default(),
            position_size: 0.95, // Use 95% of available capital
            warmup_period: 50,
        }
    }
}

/// Event-driven backtesting engine.
pub struct BacktestEngine {
    config: BacktestConfig,
    broker: SimulatedBroker,
}

impl BacktestEngine {
    /// Create a new backtest engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(BacktestConfig::default())
    }

    /// Create a new backtest engine with custom configuration.
    pub fn with_config(config: BacktestConfig) -> Self {
        let broker = SimulatedBroker::new(config.broker.clone());
        Self { config, broker }
    }

    /// Run a backtest with the given strategy and data.
    pub fn run<S: Strategy>(
        &mut self,
        strategy: &mut S,
        candles: &[Candle],
    ) -> BacktestResult {
        info!(
            "Starting backtest with {} candles, warmup: {}",
            candles.len(),
            self.config.warmup_period
        );

        // Reset broker state
        self.broker.reset();

        // Track trade returns
        let mut trade_returns: Vec<f64> = Vec::new();
        let mut entry_equity: Option<f64> = None;

        let symbol = candles.first().map(|c| c.symbol.clone()).unwrap_or_default();

        // Process each candle
        for (i, candle) in candles.iter().enumerate() {
            let prices: HashMap<String, f64> =
                [(candle.symbol.clone(), candle.close)].into_iter().collect();

            // Process pending orders at this bar's price
            self.broker.process_orders(&candle.symbol, candle.close);

            // Skip warmup period
            if i < self.config.warmup_period {
                continue;
            }

            // Get historical data up to current bar
            let historical = &candles[..=i];

            // Generate signal from strategy
            let signal = strategy.on_candle(candle, historical);

            // Get current position
            let has_position = self.broker.get_position(&candle.symbol).is_some();

            // Execute based on signal
            match signal {
                Signal::Buy(strength) if !has_position => {
                    let equity = self.broker.equity(&prices);
                    let position_value = equity * self.config.position_size * strength;
                    let quantity = position_value / candle.close;

                    if quantity > 0.0 {
                        self.broker.buy(&candle.symbol, quantity);
                        entry_equity = Some(equity);
                    }
                }
                Signal::Sell(strength) if has_position => {
                    if let Some(pos) = self.broker.get_position(&candle.symbol) {
                        let sell_qty = pos.size * strength;
                        if sell_qty > 0.0 {
                            self.broker.sell(&candle.symbol, sell_qty);

                            // Record trade return
                            if let Some(entry_eq) = entry_equity.take() {
                                let current_eq = self.broker.equity(&prices);
                                let trade_return = (current_eq - entry_eq) / entry_eq;
                                trade_returns.push(trade_return);
                            }
                        }
                    }
                }
                Signal::Close if has_position => {
                    self.broker.close_all_positions(&prices);

                    // Record trade return
                    if let Some(entry_eq) = entry_equity.take() {
                        let current_eq = self.broker.equity(&prices);
                        let trade_return = (current_eq - entry_eq) / entry_eq;
                        trade_returns.push(trade_return);
                    }
                }
                _ => {}
            }

            // Record equity
            self.broker.record_equity(candle.timestamp, &prices);
        }

        // Close any remaining positions
        if let Some(last_candle) = candles.last() {
            let prices: HashMap<String, f64> =
                [(last_candle.symbol.clone(), last_candle.close)].into_iter().collect();
            self.broker.close_all_positions(&prices);

            if let Some(entry_eq) = entry_equity {
                let current_eq = self.broker.equity(&prices);
                let trade_return = (current_eq - entry_eq) / entry_eq;
                trade_returns.push(trade_return);
            }
        }

        // Calculate performance metrics
        let periods_per_year = 365.0 * 24.0; // Assuming hourly data
        let metrics = PerformanceMetrics::from_returns(&trade_returns, periods_per_year);

        let start_time = candles.first().map(|c| c.timestamp).unwrap_or_else(chrono::Utc::now);
        let end_time = candles.last().map(|c| c.timestamp).unwrap_or_else(chrono::Utc::now);

        let final_prices: HashMap<String, f64> = candles
            .last()
            .map(|c| [(c.symbol.clone(), c.close)].into_iter().collect())
            .unwrap_or_default();

        BacktestResult::new(
            strategy.name().to_string(),
            symbol,
            start_time,
            end_time,
            self.config.broker.initial_cash,
            self.broker.equity(&final_prices),
            &metrics,
            self.broker.total_fees(),
            self.broker.equity_history().to_vec(),
            trade_returns,
        )
    }

    /// Run a simple vectorized backtest (faster but less realistic).
    pub fn run_vectorized(
        signals: &[f64],
        returns: &[f64],
        transaction_cost: f64,
    ) -> Vec<f64> {
        assert_eq!(signals.len(), returns.len());

        let mut strategy_returns = Vec::with_capacity(signals.len());
        let mut prev_signal = 0.0;

        for (signal, ret) in signals.iter().zip(returns.iter()) {
            // Position change incurs transaction cost
            let position_change = (signal - prev_signal).abs();
            let cost = position_change * transaction_cost;

            let strategy_return = signal * ret - cost;
            strategy_returns.push(strategy_return);

            prev_signal = *signal;
        }

        strategy_returns
    }

    /// Get a reference to the broker.
    pub fn broker(&self) -> &SimulatedBroker {
        &self.broker
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Run multiple backtests with different parameters (optimization).
pub fn optimize<S, F>(
    strategy_factory: F,
    candles: &[Candle],
    param_grid: Vec<HashMap<String, f64>>,
) -> Vec<(HashMap<String, f64>, BacktestResult)>
where
    S: Strategy,
    F: Fn(&HashMap<String, f64>) -> S,
{
    let mut results = Vec::new();

    for params in param_grid {
        let mut strategy = strategy_factory(&params);
        let mut engine = BacktestEngine::new();
        let result = engine.run(&mut strategy, candles);

        info!(
            "Params: {:?} -> Sharpe: {:.2}, Return: {:.2}%",
            params, result.sharpe_ratio, result.total_return_pct
        );

        results.push((params, result));
    }

    // Sort by Sharpe ratio descending
    results.sort_by(|a, b| {
        b.1.sharpe_ratio
            .partial_cmp(&a.1.sharpe_ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    struct BuyAndHold;

    impl Strategy for BuyAndHold {
        fn name(&self) -> &str {
            "Buy and Hold"
        }

        fn on_candle(&mut self, _candle: &Candle, historical: &[Candle]) -> Signal {
            if historical.len() == 51 {
                // Buy at warmup end
                Signal::Buy(1.0)
            } else {
                Signal::Hold
            }
        }
    }

    #[test]
    fn test_buy_and_hold() {
        let mut candles = Vec::new();
        let base_price = 100.0;

        for i in 0..100 {
            let price = base_price + i as f64;
            candles.push(Candle::new(
                Utc::now(),
                "BTCUSDT".to_string(),
                price,
                price + 1.0,
                price - 1.0,
                price + 0.5,
                1000.0,
            ));
        }

        let mut strategy = BuyAndHold;
        let mut engine = BacktestEngine::new();
        let result = engine.run(&mut strategy, &candles);

        assert!(result.is_profitable());
    }
}
