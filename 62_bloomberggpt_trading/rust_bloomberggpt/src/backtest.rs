//! Backtesting Module
//!
//! Provides backtesting capabilities for LLM-derived trading strategies.
//! Handles irregular signal timing, position sizing, and performance metrics.

use crate::data::OHLCVBar;
use crate::error::{Error, Result};
use crate::signals::{SignalType, TradingSignal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Maximum position size as fraction of portfolio
    pub max_position_size: f64,
    /// Maximum total exposure
    pub max_total_exposure: f64,
    /// Transaction cost in basis points
    pub transaction_cost_bps: f64,
    /// Slippage in basis points
    pub slippage_bps: f64,
    /// Signal decay time in hours
    pub signal_decay_hours: f64,
    /// Rebalance frequency ("daily", "hourly", "on_signal")
    pub rebalance_frequency: String,
    /// Minimum signal confidence to act on
    pub min_signal_confidence: f64,
    /// Allow short selling
    pub short_selling: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            max_position_size: 0.1,
            max_total_exposure: 0.8,
            transaction_cost_bps: 10.0,
            slippage_bps: 5.0,
            signal_decay_hours: 24.0,
            rebalance_frequency: "daily".to_string(),
            min_signal_confidence: 0.5,
            short_selling: false,
        }
    }
}

/// A single trade executed during backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    /// Symbol traded
    pub symbol: String,
    /// Trade type (buy/sell)
    pub trade_type: TradeType,
    /// Quantity traded
    pub quantity: f64,
    /// Execution price
    pub price: f64,
    /// Total value of trade
    pub value: f64,
    /// Transaction costs
    pub costs: f64,
    /// Signal that triggered the trade
    pub signal_type: SignalType,
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradeType {
    Buy,
    Sell,
    Short,
    Cover,
}

impl std::fmt::Display for TradeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeType::Buy => write!(f, "BUY"),
            TradeType::Sell => write!(f, "SELL"),
            TradeType::Short => write!(f, "SHORT"),
            TradeType::Cover => write!(f, "COVER"),
        }
    }
}

/// Position in a single asset
#[derive(Debug, Clone, Default)]
struct Position {
    /// Current quantity (negative for short)
    quantity: f64,
    /// Average entry price
    avg_price: f64,
    /// Realized P&L
    realized_pnl: f64,
}

/// Portfolio state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Cash balance
    pub cash: f64,
    /// Total equity value
    pub equity: f64,
    /// Position values by symbol
    pub positions: HashMap<String, f64>,
    /// Daily return
    pub daily_return: f64,
}

/// Backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Configuration used
    pub config: BacktestConfig,
    /// All trades executed
    pub trades: Vec<Trade>,
    /// Portfolio history
    pub portfolio_history: Vec<PortfolioState>,
    /// Performance metrics
    pub metrics: BacktestMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Annualized volatility
    pub volatility: f64,
    /// Sharpe ratio (assuming 2% risk-free rate)
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total trades
    pub total_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
}

impl Default for BacktestMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            annualized_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            avg_trade_return: 0.0,
        }
    }
}

/// Backtester for LLM trading signals
pub struct Backtester {
    config: BacktestConfig,
    positions: HashMap<String, Position>,
    cash: f64,
    trades: Vec<Trade>,
    portfolio_history: Vec<PortfolioState>,
    active_signals: Vec<TradingSignal>,
}

impl Backtester {
    /// Create new backtester with config
    pub fn new(config: BacktestConfig) -> Self {
        let initial_capital = config.initial_capital;
        Self {
            config,
            positions: HashMap::new(),
            cash: initial_capital,
            trades: Vec::new(),
            portfolio_history: Vec::new(),
            active_signals: Vec::new(),
        }
    }

    /// Run backtest with signals and price data
    pub fn run(
        &mut self,
        signals: &[TradingSignal],
        prices: &HashMap<String, Vec<OHLCVBar>>,
    ) -> Result<BacktestResult> {
        // Reset state
        self.reset();

        // Sort signals by timestamp
        let mut sorted_signals: Vec<_> = signals.iter().collect();
        sorted_signals.sort_by_key(|s| s.timestamp);

        // Get all unique timestamps from prices
        let mut all_timestamps: Vec<DateTime<Utc>> = prices
            .values()
            .flat_map(|bars| bars.iter().map(|b| b.timestamp))
            .collect();
        all_timestamps.sort();
        all_timestamps.dedup();

        if all_timestamps.is_empty() {
            return Err(Error::InsufficientData("No price data provided".to_string()));
        }

        let mut signal_idx = 0;
        let mut prev_equity = self.config.initial_capital;

        // Process each timestamp
        for timestamp in &all_timestamps {
            // Add new signals that arrived before this timestamp
            while signal_idx < sorted_signals.len()
                && sorted_signals[signal_idx].timestamp <= *timestamp
            {
                if sorted_signals[signal_idx].confidence >= self.config.min_signal_confidence {
                    self.active_signals.push(sorted_signals[signal_idx].clone());
                }
                signal_idx += 1;
            }

            // Remove expired signals
            self.active_signals
                .retain(|s| s.expiry > *timestamp && s.is_valid());

            // Get current prices
            let current_prices: HashMap<String, f64> = prices
                .iter()
                .filter_map(|(symbol, bars)| {
                    bars.iter()
                        .find(|b| b.timestamp == *timestamp)
                        .map(|b| (symbol.clone(), b.close))
                })
                .collect();

            // Apply signal decay
            for signal in &mut self.active_signals {
                let hours = (*timestamp - signal.timestamp).num_hours() as f64;
                let decay = (-hours / self.config.signal_decay_hours).exp();
                signal.strength *= decay;
            }

            // Determine target positions
            if self.should_rebalance(timestamp) {
                self.rebalance(&current_prices, timestamp)?;
            }

            // Calculate equity
            let equity = self.calculate_equity(&current_prices);
            let daily_return = if prev_equity > 0.0 {
                (equity - prev_equity) / prev_equity
            } else {
                0.0
            };

            // Record portfolio state
            let positions: HashMap<String, f64> = self
                .positions
                .iter()
                .filter_map(|(symbol, pos)| {
                    current_prices
                        .get(symbol)
                        .map(|price| (symbol.clone(), pos.quantity * price))
                })
                .collect();

            self.portfolio_history.push(PortfolioState {
                timestamp: *timestamp,
                cash: self.cash,
                equity,
                positions,
                daily_return,
            });

            prev_equity = equity;
        }

        // Calculate metrics
        let metrics = self.calculate_metrics()?;

        Ok(BacktestResult {
            config: self.config.clone(),
            trades: self.trades.clone(),
            portfolio_history: self.portfolio_history.clone(),
            metrics,
        })
    }

    /// Reset backtester state
    fn reset(&mut self) {
        self.positions.clear();
        self.cash = self.config.initial_capital;
        self.trades.clear();
        self.portfolio_history.clear();
        self.active_signals.clear();
    }

    /// Check if should rebalance at this timestamp
    fn should_rebalance(&self, _timestamp: &DateTime<Utc>) -> bool {
        // Simplified: always rebalance when we have signals
        !self.active_signals.is_empty()
    }

    /// Rebalance portfolio based on active signals
    fn rebalance(
        &mut self,
        prices: &HashMap<String, f64>,
        timestamp: &DateTime<Utc>,
    ) -> Result<()> {
        // Aggregate signals by symbol
        let mut target_weights: HashMap<String, f64> = HashMap::new();

        for signal in &self.active_signals {
            let weight = signal.strength
                * signal.confidence
                * signal.signal_type.to_position();

            *target_weights.entry(signal.symbol.clone()).or_default() += weight;
        }

        // Normalize weights
        let total_weight: f64 = target_weights.values().map(|w| w.abs()).sum();
        if total_weight > 0.0 {
            let scale = self.config.max_total_exposure / total_weight;
            for weight in target_weights.values_mut() {
                *weight *= scale;
                // Cap individual positions
                *weight = weight.clamp(
                    -self.config.max_position_size,
                    self.config.max_position_size,
                );
            }
        }

        // Execute trades to reach target weights
        let equity = self.calculate_equity(prices);

        for (symbol, target_weight) in target_weights {
            // Skip if no price
            let price = match prices.get(&symbol) {
                Some(p) => *p,
                None => continue,
            };

            if price <= 0.0 {
                continue;
            }

            // Skip short positions if not allowed
            if !self.config.short_selling && target_weight < 0.0 {
                continue;
            }

            let target_value = equity * target_weight;
            let current_pos = self.positions.get(&symbol).cloned().unwrap_or_default();
            let current_value = current_pos.quantity * price;

            let value_diff = target_value - current_value;
            let quantity_diff = value_diff / price;

            // Skip small trades
            if quantity_diff.abs() < 0.001 {
                continue;
            }

            // Execute trade
            let trade_type = if quantity_diff > 0.0 {
                if current_pos.quantity >= 0.0 {
                    TradeType::Buy
                } else {
                    TradeType::Cover
                }
            } else if current_pos.quantity <= 0.0 {
                TradeType::Short
            } else {
                TradeType::Sell
            };

            // Apply slippage
            let slippage = price * self.config.slippage_bps / 10000.0;
            let exec_price = if quantity_diff > 0.0 {
                price + slippage
            } else {
                price - slippage
            };

            let trade_value = quantity_diff.abs() * exec_price;
            let costs = trade_value * self.config.transaction_cost_bps / 10000.0;

            // Update position
            let pos = self.positions.entry(symbol.clone()).or_default();

            if quantity_diff > 0.0 {
                // Buying
                let total_cost = quantity_diff * exec_price;
                let new_quantity = pos.quantity + quantity_diff;
                if new_quantity != 0.0 {
                    pos.avg_price = (pos.quantity * pos.avg_price + total_cost) / new_quantity;
                }
                pos.quantity = new_quantity;
                self.cash -= total_cost + costs;
            } else {
                // Selling
                let sale_proceeds = quantity_diff.abs() * exec_price;
                let cost_basis = quantity_diff.abs() * pos.avg_price;
                pos.realized_pnl += sale_proceeds - cost_basis;
                pos.quantity += quantity_diff;
                self.cash += sale_proceeds - costs;
            }

            // Find the signal that triggered this trade
            let signal_type = self.active_signals
                .iter()
                .find(|s| s.symbol == symbol)
                .map(|s| s.signal_type)
                .unwrap_or(SignalType::Hold);

            self.trades.push(Trade {
                timestamp: *timestamp,
                symbol,
                trade_type,
                quantity: quantity_diff.abs(),
                price: exec_price,
                value: trade_value,
                costs,
                signal_type,
            });
        }

        Ok(())
    }

    /// Calculate total equity
    fn calculate_equity(&self, prices: &HashMap<String, f64>) -> f64 {
        let position_value: f64 = self
            .positions
            .iter()
            .filter_map(|(symbol, pos)| {
                prices.get(symbol).map(|price| pos.quantity * price)
            })
            .sum();

        self.cash + position_value
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self) -> Result<BacktestMetrics> {
        if self.portfolio_history.len() < 2 {
            return Ok(BacktestMetrics::default());
        }

        let returns: Vec<f64> = self
            .portfolio_history
            .iter()
            .map(|s| s.daily_return)
            .collect();

        let initial_equity = self.config.initial_capital;
        let final_equity = self
            .portfolio_history
            .last()
            .map(|s| s.equity)
            .unwrap_or(initial_equity);

        let total_return = (final_equity - initial_equity) / initial_equity;

        // Annualized return (assuming 252 trading days)
        let n_days = self.portfolio_history.len() as f64;
        let annualized_return = (1.0 + total_return).powf(252.0 / n_days) - 1.0;

        // Volatility
        let mean_return: f64 = returns.iter().sum::<f64>() / n_days;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (n_days - 1.0);
        let volatility = variance.sqrt() * (252.0_f64).sqrt();

        // Sharpe ratio (2% risk-free rate)
        let risk_free_rate = 0.02;
        let sharpe_ratio = if volatility > 0.0 {
            (annualized_return - risk_free_rate) / volatility
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns
            .iter()
            .filter(|r| **r < 0.0)
            .copied()
            .collect();
        let downside_variance: f64 = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_deviation = downside_variance.sqrt() * (252.0_f64).sqrt();
        let sortino_ratio = if downside_deviation > 0.0 {
            (annualized_return - risk_free_rate) / downside_deviation
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = initial_equity;
        let mut max_drawdown = 0.0;
        for state in &self.portfolio_history {
            if state.equity > peak {
                peak = state.equity;
            }
            let drawdown = (peak - state.equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let winning_trades: Vec<&Trade> = self
            .trades
            .iter()
            .filter(|t| match t.trade_type {
                TradeType::Sell | TradeType::Cover => t.value > 0.0,
                _ => false,
            })
            .collect();

        let total_trades = self.trades.len();
        let win_rate = if total_trades > 0 {
            winning_trades.len() as f64 / total_trades as f64
        } else {
            0.0
        };

        // Profit factor
        let gross_profit: f64 = self
            .trades
            .iter()
            .filter(|t| t.value > 0.0)
            .map(|t| t.value)
            .sum();
        let gross_loss: f64 = self
            .trades
            .iter()
            .filter(|t| t.value < 0.0)
            .map(|t| t.value.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        };

        let avg_trade_return = if total_trades > 0 {
            total_return / total_trades as f64
        } else {
            0.0
        };

        Ok(BacktestMetrics {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            total_trades,
            avg_trade_return,
        })
    }
}

/// Print formatted backtest report
pub fn print_backtest_report(result: &BacktestResult) {
    println!("\n{}", "=".repeat(60));
    println!("BACKTEST RESULTS");
    println!("{}", "=".repeat(60));

    let m = &result.metrics;

    println!("\nPERFORMANCE METRICS");
    println!("{}", "-".repeat(40));
    println!("Total Return:      {:>10.2}%", m.total_return * 100.0);
    println!("Annualized Return: {:>10.2}%", m.annualized_return * 100.0);
    println!("Volatility:        {:>10.2}%", m.volatility * 100.0);
    println!("Sharpe Ratio:      {:>10.2}", m.sharpe_ratio);
    println!("Sortino Ratio:     {:>10.2}", m.sortino_ratio);
    println!("Max Drawdown:      {:>10.2}%", m.max_drawdown * 100.0);
    println!("Calmar Ratio:      {:>10.2}", m.calmar_ratio);

    println!("\nTRADE STATISTICS");
    println!("{}", "-".repeat(40));
    println!("Total Trades:      {:>10}", m.total_trades);
    println!("Win Rate:          {:>10.2}%", m.win_rate * 100.0);
    println!("Profit Factor:     {:>10.2}", m.profit_factor);
    println!("Avg Trade Return:  {:>10.4}%", m.avg_trade_return * 100.0);

    if let Some(first) = result.portfolio_history.first() {
        if let Some(last) = result.portfolio_history.last() {
            println!("\nPORTFOLIO");
            println!("{}", "-".repeat(40));
            println!("Initial Capital:   ${:>12.2}", first.equity);
            println!("Final Equity:      ${:>12.2}", last.equity);
            println!(
                "Net P&L:           ${:>12.2}",
                last.equity - first.equity
            );
        }
    }

    println!("\n{}", "=".repeat(60));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_prices() -> HashMap<String, Vec<OHLCVBar>> {
        let mut prices = HashMap::new();
        let base_time = Utc::now();

        let bars: Vec<OHLCVBar> = (0..10)
            .map(|i| OHLCVBar {
                timestamp: base_time + chrono::Duration::days(i),
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 98.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 1000000.0,
            })
            .collect();

        prices.insert("AAPL".to_string(), bars);
        prices
    }

    fn create_test_signals() -> Vec<TradingSignal> {
        let base_time = Utc::now();

        vec![
            TradingSignal {
                symbol: "AAPL".to_string(),
                signal_type: SignalType::Buy,
                strength: 0.8,
                confidence: 0.9,
                source: crate::signals::SignalSource::News,
                timestamp: base_time + chrono::Duration::days(1),
                expiry: base_time + chrono::Duration::days(5),
                reason: "Positive earnings".to_string(),
                sentiment_result: None,
            },
        ]
    }

    #[test]
    fn test_backtest_runs() {
        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config);

        let prices = create_test_prices();
        let signals = create_test_signals();

        let result = backtester.run(&signals, &prices).unwrap();

        assert!(!result.portfolio_history.is_empty());
        assert!(result.metrics.total_trades > 0 || result.metrics.total_return != 0.0);
    }

    #[test]
    fn test_no_short_selling() {
        let config = BacktestConfig {
            short_selling: false,
            ..Default::default()
        };
        let mut backtester = Backtester::new(config);

        let prices = create_test_prices();
        let mut signals = create_test_signals();
        signals[0].signal_type = SignalType::Sell;
        signals[0].strength = -0.8;

        let result = backtester.run(&signals, &prices).unwrap();

        // Should not have any short trades
        for trade in &result.trades {
            assert_ne!(trade.trade_type, TradeType::Short);
        }
    }
}
