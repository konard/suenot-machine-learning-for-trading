//! Backtesting framework for strategy validation.

use crate::data::Candle;
use crate::graph::{FinancialGraph, GraphBuilder};
use crate::mining::{GSpanMiner, Pattern};
use crate::trading::{Signal, SignalGenerator, Strategy, StrategyConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Start timestamp
    pub start_time: i64,
    /// End timestamp
    pub end_time: i64,
    /// Rebalance frequency in seconds
    pub rebalance_interval: i64,
    /// Correlation threshold for graph building
    pub correlation_threshold: f64,
    /// Window size for correlation calculation
    pub window_size: usize,
    /// Minimum pattern support
    pub min_support: usize,
    /// Strategy configuration
    pub strategy_config: StrategyConfig,
    /// Trading fee percentage
    pub trading_fee_pct: f64,
    /// Slippage percentage
    pub slippage_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            start_time: 0,
            end_time: i64::MAX,
            rebalance_interval: 3600, // 1 hour
            correlation_threshold: 0.7,
            window_size: 24,
            min_support: 2,
            strategy_config: StrategyConfig::default(),
            trading_fee_pct: 0.001, // 0.1%
            slippage_pct: 0.0005,   // 0.05%
        }
    }
}

/// A trade in the backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: i64,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub price: f64,
    pub fee: f64,
    pub reason: String,
}

/// Backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total trades
    pub total_trades: usize,
    /// Winning trades
    pub winning_trades: usize,
    /// Losing trades
    pub losing_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Final capital
    pub final_capital: f64,
    /// Equity curve (timestamp, value)
    pub equity_curve: Vec<(i64, f64)>,
    /// All trades
    pub trades: Vec<Trade>,
    /// Daily returns
    pub daily_returns: Vec<f64>,
}

impl BacktestResult {
    /// Check if strategy is profitable
    pub fn is_profitable(&self) -> bool {
        self.total_return > 0.0
    }

    /// Get risk-adjusted metrics summary
    pub fn summary(&self) -> String {
        format!(
            "Return: {:.2}%, Sharpe: {:.2}, MaxDD: {:.2}%, WinRate: {:.2}%, Trades: {}",
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.total_trades
        )
    }
}

/// Backtester for strategy validation
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on historical data
    pub fn run(
        &self,
        data: &HashMap<String, Vec<Candle>>,
    ) -> Result<BacktestResult, BacktestError> {
        // Validate data
        if data.is_empty() {
            return Err(BacktestError::NoData);
        }

        // Find common time range
        let (start_time, end_time) = self.find_time_range(data)?;

        // Initialize components
        let graph_builder = GraphBuilder::new()
            .correlation_threshold(self.config.correlation_threshold)
            .window_size(self.config.window_size);

        let miner = GSpanMiner::new().min_support(self.config.min_support);

        let mut strategy = Strategy::new(self.config.strategy_config.clone());

        // Tracking variables
        let mut equity_curve = Vec::new();
        let mut trades = Vec::new();
        let mut daily_returns = Vec::new();
        let mut prev_capital = self.config.strategy_config.initial_capital;

        // Time loop
        let mut current_time = start_time;
        let mut signal_generator: Option<SignalGenerator> = None;

        while current_time <= end_time {
            // Get data up to current time
            let window_data = self.get_window_data(data, current_time);

            if window_data.values().all(|v| v.len() >= self.config.window_size) {
                // Build graph
                if let Ok(graph) = graph_builder.build_correlation_network(&window_data) {
                    // Mine patterns (periodically)
                    let patterns = if signal_generator.is_none() {
                        miner.mine(&graph).unwrap_or_default()
                    } else {
                        Vec::new()
                    };

                    // Initialize or use existing signal generator
                    if signal_generator.is_none() && !patterns.is_empty() {
                        signal_generator = Some(SignalGenerator::new(patterns));
                    }

                    // Generate signals
                    if let Some(ref mut gen) = signal_generator {
                        let signals = gen.generate(&graph);

                        // Get current prices
                        let prices = self.get_current_prices(&window_data);

                        // Process signals
                        let orders = strategy.process_signals(&signals, &prices);

                        // Execute orders
                        for order in orders {
                            if let Some(&price) = prices.get(&order.symbol) {
                                // Apply slippage
                                let fill_price = match order.side {
                                    crate::trading::strategy::OrderSide::Buy => {
                                        price * (1.0 + self.config.slippage_pct)
                                    }
                                    crate::trading::strategy::OrderSide::Sell => {
                                        price * (1.0 - self.config.slippage_pct)
                                    }
                                };

                                // Calculate fee
                                let fee = order.quantity * fill_price * self.config.trading_fee_pct;

                                // Execute
                                strategy.execute_order(&order, fill_price);

                                // Record trade
                                trades.push(Trade {
                                    timestamp: current_time,
                                    symbol: order.symbol,
                                    side: format!("{:?}", order.side),
                                    quantity: order.quantity,
                                    price: fill_price,
                                    fee,
                                    reason: order.reason,
                                });
                            }
                        }

                        // Update positions with current prices
                        let _ = strategy.process_signals(&[], &prices);
                    }
                }
            }

            // Record equity
            let current_capital = strategy.total_capital();
            equity_curve.push((current_time, current_capital));

            // Calculate daily return (approximate)
            if equity_curve.len() > 1 && current_time % 86400 == 0 {
                let daily_return = (current_capital - prev_capital) / prev_capital;
                daily_returns.push(daily_return);
                prev_capital = current_capital;
            }

            current_time += self.config.rebalance_interval;
        }

        // Calculate final metrics
        Ok(self.calculate_metrics(
            &strategy,
            &equity_curve,
            &trades,
            &daily_returns,
        ))
    }

    /// Find common time range across all symbols
    fn find_time_range(
        &self,
        data: &HashMap<String, Vec<Candle>>,
    ) -> Result<(i64, i64), BacktestError> {
        let mut min_start = i64::MAX;
        let mut max_end = i64::MIN;

        for candles in data.values() {
            if candles.is_empty() {
                continue;
            }
            min_start = min_start.min(candles.first().unwrap().timestamp);
            max_end = max_end.max(candles.last().unwrap().timestamp);
        }

        if min_start == i64::MAX {
            return Err(BacktestError::NoData);
        }

        // Apply config time bounds
        let start = min_start.max(self.config.start_time);
        let end = max_end.min(self.config.end_time);

        Ok((start, end))
    }

    /// Get data window up to specified time
    fn get_window_data(
        &self,
        data: &HashMap<String, Vec<Candle>>,
        until: i64,
    ) -> HashMap<String, Vec<Candle>> {
        data.iter()
            .map(|(symbol, candles)| {
                let filtered: Vec<_> = candles
                    .iter()
                    .filter(|c| c.timestamp <= until)
                    .take(self.config.window_size * 2) // Take extra for return calculation
                    .cloned()
                    .collect();
                (symbol.clone(), filtered)
            })
            .collect()
    }

    /// Get current prices from data
    fn get_current_prices(&self, data: &HashMap<String, Vec<Candle>>) -> HashMap<String, f64> {
        data.iter()
            .filter_map(|(symbol, candles)| {
                candles.last().map(|c| (symbol.clone(), c.close))
            })
            .collect()
    }

    /// Calculate performance metrics
    fn calculate_metrics(
        &self,
        strategy: &Strategy,
        equity_curve: &[(i64, f64)],
        trades: &[Trade],
        daily_returns: &[f64],
    ) -> BacktestResult {
        let initial = self.config.strategy_config.initial_capital;
        let final_capital = strategy.total_capital();
        let total_return = (final_capital - initial) / initial;

        // Annualized return (assuming daily returns)
        let n_days = daily_returns.len().max(1) as f64;
        let annualized_return = (1.0 + total_return).powf(365.0 / n_days) - 1.0;

        // Sharpe ratio
        let avg_return = if daily_returns.is_empty() {
            0.0
        } else {
            daily_returns.iter().sum::<f64>() / daily_returns.len() as f64
        };

        let std_return = if daily_returns.len() > 1 {
            let variance: f64 = daily_returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>()
                / (daily_returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            1.0
        };

        let sharpe_ratio = if std_return > 0.0 {
            (avg_return * 252.0_f64.sqrt()) / std_return
        } else {
            0.0
        };

        // Sortino ratio (only downside deviation)
        let downside_returns: Vec<_> = daily_returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_std = if downside_returns.len() > 1 {
            let variance: f64 = downside_returns
                .iter()
                .map(|r| r.powi(2))
                .sum::<f64>()
                / downside_returns.len() as f64;
            variance.sqrt()
        } else {
            1.0
        };

        let sortino_ratio = if downside_std > 0.0 {
            (avg_return * 252.0_f64.sqrt()) / downside_std
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = initial;
        let mut max_drawdown = 0.0;
        for (_, value) in equity_curve {
            peak = peak.max(*value);
            let drawdown = (peak - value) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Trade analysis
        let trade_returns = self.calculate_trade_returns(trades);
        let winning_trades = trade_returns.iter().filter(|&&r| r > 0.0).count();
        let losing_trades = trade_returns.iter().filter(|&&r| r < 0.0).count();
        let total_trades = trades.len() / 2; // Buy + Sell = 1 round trip

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = trade_returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = trade_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = if !trade_returns.is_empty() {
            trade_returns.iter().sum::<f64>() / trade_returns.len() as f64
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
            avg_trade_return,
            final_capital,
            equity_curve: equity_curve.to_vec(),
            trades: trades.to_vec(),
            daily_returns: daily_returns.to_vec(),
        }
    }

    /// Calculate returns for each round-trip trade
    fn calculate_trade_returns(&self, trades: &[Trade]) -> Vec<f64> {
        let mut returns = Vec::new();
        let mut open_trades: HashMap<String, &Trade> = HashMap::new();

        for trade in trades {
            if trade.side == "Buy" {
                open_trades.insert(trade.symbol.clone(), trade);
            } else if trade.side == "Sell" {
                if let Some(buy_trade) = open_trades.remove(&trade.symbol) {
                    let ret = (trade.price - buy_trade.price) / buy_trade.price;
                    returns.push(ret);
                }
            }
        }

        returns
    }
}

/// Backtest errors
#[derive(Debug, thiserror::Error)]
pub enum BacktestError {
    #[error("No data provided")]
    NoData,
    #[error("Invalid time range")]
    InvalidTimeRange,
    #[error("Insufficient data for analysis")]
    InsufficientData,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_config() {
        let config = BacktestConfig::default();
        assert_eq!(config.rebalance_interval, 3600);
        assert_eq!(config.correlation_threshold, 0.7);
    }

    #[test]
    fn test_backtest_result_summary() {
        let result = BacktestResult {
            total_return: 0.25,
            annualized_return: 0.50,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            max_drawdown: 0.10,
            win_rate: 0.60,
            profit_factor: 1.8,
            total_trades: 50,
            winning_trades: 30,
            losing_trades: 20,
            avg_trade_return: 0.005,
            final_capital: 12500.0,
            equity_curve: vec![],
            trades: vec![],
            daily_returns: vec![],
        };

        assert!(result.is_profitable());
        assert!(result.summary().contains("25.00%"));
    }
}
