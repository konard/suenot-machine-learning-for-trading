//! Backtesting engine for multi-agent trading strategies.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::agents::{Agent, Analysis, Signal, TraderAgent};
use crate::communication::{Debate, DebateResult};
use crate::data::MarketData;
use crate::error::{Result, TradingError};

/// A single trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub action: String,
    pub price: f64,
    pub quantity: f64,
    pub signal: Signal,
    pub confidence: f64,
    pub pnl: f64,
    pub fees: f64,
}

/// Current position in an asset.
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_time: DateTime<Utc>,
    pub current_price: f64,
}

impl Position {
    /// Calculate unrealized P&L.
    pub fn unrealized_pnl(&self) -> f64 {
        (self.current_price - self.entry_price) * self.quantity
    }

    /// Calculate unrealized P&L percentage.
    pub fn unrealized_pnl_pct(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        (self.current_price / self.entry_price - 1.0) * 100.0
    }
}

/// Backtest performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub initial_capital: f64,
    pub final_capital: f64,
    pub total_return: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<(DateTime<Utc>, f64)>,
}

impl BacktestResult {
    /// Print a summary of the backtest results.
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(50));
        println!("BACKTEST RESULTS");
        println!("{}", "=".repeat(50));
        println!(
            "Period: {} to {}",
            self.start_date.format("%Y-%m-%d"),
            self.end_date.format("%Y-%m-%d")
        );
        println!("Initial Capital: ${:.2}", self.initial_capital);
        println!("Final Capital: ${:.2}", self.final_capital);
        println!("Total Return: {:.2}%", self.total_return);
        println!("Number of Trades: {}", self.num_trades);
        println!("Win Rate: {:.2}%", self.win_rate);
        println!("Sharpe Ratio: {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio: {:.2}", self.sortino_ratio);
        println!("Max Drawdown: {:.2}%", self.max_drawdown);
        println!("Calmar Ratio: {:.2}", self.calmar_ratio);
        println!("{}", "=".repeat(50));
    }
}

/// Multi-agent backtester.
pub struct MultiAgentBacktester<'a> {
    agents: Vec<&'a dyn Agent>,
    trader: TraderAgent,
    initial_capital: f64,
    position_size_pct: f64,
    transaction_cost_pct: f64,
    capital: f64,
    positions: HashMap<String, Position>,
    trades: Vec<Trade>,
    equity_history: Vec<(DateTime<Utc>, f64)>,
}

impl<'a> MultiAgentBacktester<'a> {
    /// Create a new backtester.
    pub fn new(
        agents: Vec<&'a dyn Agent>,
        initial_capital: f64,
        position_size_pct: f64,
        transaction_cost_pct: f64,
    ) -> Self {
        Self {
            agents,
            trader: TraderAgent::new("Backtest-Trader"),
            initial_capital,
            position_size_pct,
            transaction_cost_pct,
            capital: initial_capital,
            positions: HashMap::new(),
            trades: Vec::new(),
            equity_history: Vec::new(),
        }
    }

    /// Reset the backtester state.
    pub fn reset(&mut self) {
        self.capital = self.initial_capital;
        self.positions.clear();
        self.trades.clear();
        self.equity_history.clear();
    }

    /// Calculate total equity.
    fn get_equity(&self) -> f64 {
        let position_value: f64 = self
            .positions
            .values()
            .map(|p| p.quantity * p.current_price)
            .sum();
        self.capital + position_value
    }

    /// Update position prices.
    fn update_position_prices(&mut self, symbol: &str, price: f64) {
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.current_price = price;
        }
    }

    /// Execute a trading signal.
    fn execute_signal(
        &mut self,
        timestamp: DateTime<Utc>,
        symbol: &str,
        price: f64,
        signal: Signal,
        confidence: f64,
    ) -> Option<Trade> {
        let has_position = self.positions.contains_key(symbol);

        match (signal, has_position) {
            (Signal::StrongBuy | Signal::Buy, false) => {
                // Open long position
                let position_value = self.capital * self.position_size_pct
                    * if matches!(signal, Signal::Buy) {
                        confidence
                    } else {
                        1.0
                    };

                let quantity = position_value / price;
                let cost = position_value * (1.0 + self.transaction_cost_pct);

                if cost <= self.capital {
                    self.capital -= cost;
                    self.positions.insert(
                        symbol.to_string(),
                        Position {
                            symbol: symbol.to_string(),
                            quantity,
                            entry_price: price,
                            entry_time: timestamp,
                            current_price: price,
                        },
                    );

                    Some(Trade {
                        timestamp,
                        symbol: symbol.to_string(),
                        action: "BUY".to_string(),
                        price,
                        quantity,
                        signal,
                        confidence,
                        pnl: 0.0,
                        fees: position_value * self.transaction_cost_pct,
                    })
                } else {
                    None
                }
            }
            (Signal::StrongSell | Signal::Sell, true) => {
                // Close long position
                let position = self.positions.remove(symbol)?;
                let sale_value = position.quantity * price * (1.0 - self.transaction_cost_pct);
                let pnl = sale_value - (position.quantity * position.entry_price);

                self.capital += sale_value;

                Some(Trade {
                    timestamp,
                    symbol: symbol.to_string(),
                    action: "SELL".to_string(),
                    price,
                    quantity: position.quantity,
                    signal,
                    confidence,
                    pnl,
                    fees: position.quantity * price * self.transaction_cost_pct,
                })
            }
            _ => None,
        }
    }

    /// Run the backtest.
    pub async fn run(
        &mut self,
        symbol: &str,
        data: &MarketData,
        lookback: usize,
        step: usize,
    ) -> Result<BacktestResult> {
        self.reset();

        if data.len() < lookback + 10 {
            return Err(TradingError::InsufficientData {
                required: lookback + 10,
                actual: data.len(),
            });
        }

        let candles = &data.candles;
        let mut daily_returns = Vec::new();
        let mut prev_equity = self.initial_capital;

        for i in lookback..candles.len() {
            if (i - lookback) % step != 0 {
                // Just update prices
                let price = candles[i].close;
                self.update_position_prices(symbol, price);
                continue;
            }

            let timestamp = candles[i].timestamp;
            let current_price = candles[i].close;
            self.update_position_prices(symbol, current_price);

            // Create window data
            let window_candles = candles[i.saturating_sub(lookback)..=i].to_vec();
            let window_data = MarketData::new(symbol, window_candles, &data.source);

            // Run all agents
            let mut analyses = Vec::new();
            for agent in &self.agents {
                if let Ok(analysis) = agent.analyze(symbol, &window_data, None).await {
                    analyses.push(analysis);
                }
            }

            // Aggregate with trader
            let final_analysis = self.trader.aggregate(symbol, &analyses);

            // Execute signal
            if let Some(trade) = self.execute_signal(
                timestamp,
                symbol,
                current_price,
                final_analysis.signal,
                final_analysis.confidence,
            ) {
                self.trades.push(trade);
            }

            // Record equity
            let current_equity = self.get_equity();
            self.equity_history.push((timestamp, current_equity));

            // Calculate daily return
            let daily_return = if prev_equity > 0.0 {
                current_equity / prev_equity - 1.0
            } else {
                0.0
            };
            daily_returns.push(daily_return);
            prev_equity = current_equity;
        }

        // Close remaining positions
        if let Some(last_candle) = candles.last() {
            let final_price = last_candle.close;
            if self.positions.contains_key(symbol) {
                if let Some(trade) = self.execute_signal(
                    last_candle.timestamp,
                    symbol,
                    final_price,
                    Signal::Sell,
                    1.0,
                ) {
                    self.trades.push(trade);
                }
            }
        }

        // Calculate metrics
        let final_capital = self.get_equity();
        let total_return = (final_capital / self.initial_capital - 1.0) * 100.0;
        let num_trades = self.trades.len();

        let win_rate = if num_trades > 0 {
            let wins = self.trades.iter().filter(|t| t.pnl > 0.0).count();
            wins as f64 / num_trades as f64 * 100.0
        } else {
            0.0
        };

        // Sharpe ratio
        let sharpe_ratio = if !daily_returns.is_empty() {
            let mean: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let variance: f64 = daily_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / daily_returns.len() as f64;
            let std = variance.sqrt();
            if std > 0.0 {
                mean / std * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let sortino_ratio = if !daily_returns.is_empty() {
            let mean: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let downside: Vec<f64> = daily_returns.iter().filter(|&&r| r < 0.0).copied().collect();
            if !downside.is_empty() {
                let downside_variance: f64 =
                    downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
                let downside_std = downside_variance.sqrt();
                if downside_std > 0.0 {
                    mean / downside_std * (252.0_f64).sqrt()
                } else {
                    f64::INFINITY
                }
            } else {
                f64::INFINITY
            }
        } else {
            0.0
        };

        // Max drawdown
        let max_drawdown = if !self.equity_history.is_empty() {
            let mut peak = self.equity_history[0].1;
            let mut max_dd = 0.0;
            for (_, equity) in &self.equity_history {
                if *equity > peak {
                    peak = *equity;
                }
                let dd = (peak - equity) / peak * 100.0;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
            max_dd
        } else {
            0.0
        };

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            let annual_return =
                total_return * 252.0 / daily_returns.len().max(1) as f64;
            annual_return / max_drawdown
        } else {
            0.0
        };

        Ok(BacktestResult {
            start_date: candles[lookback].timestamp,
            end_date: candles.last().unwrap().timestamp,
            initial_capital: self.initial_capital,
            final_capital,
            total_return,
            num_trades,
            win_rate,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            trades: self.trades.clone(),
            equity_curve: self.equity_history.clone(),
        })
    }
}

/// Calculate buy-and-hold benchmark.
pub fn buy_and_hold_benchmark(data: &MarketData, initial_capital: f64) -> BacktestResult {
    let candles = &data.candles;

    if candles.is_empty() {
        return BacktestResult {
            start_date: Utc::now(),
            end_date: Utc::now(),
            initial_capital,
            final_capital: initial_capital,
            total_return: 0.0,
            num_trades: 0,
            win_rate: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            trades: vec![],
            equity_curve: vec![],
        };
    }

    let start_price = candles[0].close;
    let shares = initial_capital / start_price;

    let equity_curve: Vec<(DateTime<Utc>, f64)> = candles
        .iter()
        .map(|c| (c.timestamp, shares * c.close))
        .collect();

    let final_capital = shares * candles.last().unwrap().close;
    let total_return = (final_capital / initial_capital - 1.0) * 100.0;

    // Calculate daily returns
    let daily_returns: Vec<f64> = candles
        .windows(2)
        .map(|w| w[1].close / w[0].close - 1.0)
        .collect();

    // Sharpe
    let mean: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len().max(1) as f64;
    let variance: f64 = daily_returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / daily_returns.len().max(1) as f64;
    let std = variance.sqrt();
    let sharpe_ratio = if std > 0.0 {
        mean / std * (252.0_f64).sqrt()
    } else {
        0.0
    };

    // Max drawdown
    let mut peak = equity_curve[0].1;
    let mut max_drawdown = 0.0;
    for (_, equity) in &equity_curve {
        if *equity > peak {
            peak = *equity;
        }
        let dd = (peak - equity) / peak * 100.0;
        if dd > max_drawdown {
            max_drawdown = dd;
        }
    }

    BacktestResult {
        start_date: candles[0].timestamp,
        end_date: candles.last().unwrap().timestamp,
        initial_capital,
        final_capital,
        total_return,
        num_trades: 1,
        win_rate: if total_return > 0.0 { 100.0 } else { 0.0 },
        sharpe_ratio,
        sortino_ratio: 0.0, // Simplified
        max_drawdown,
        calmar_ratio: if max_drawdown > 0.0 {
            total_return / max_drawdown
        } else {
            0.0
        },
        trades: vec![],
        equity_curve,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::{BearAgent, BullAgent, TechnicalAgent};
    use crate::data::create_mock_data;

    #[tokio::test]
    async fn test_backtest() {
        let tech = TechnicalAgent::new("Tech");
        let bull = BullAgent::new("Bull");
        let bear = BearAgent::new("Bear");

        let agents: Vec<&dyn Agent> = vec![&tech, &bull, &bear];

        let mut backtester = MultiAgentBacktester::new(agents, 100000.0, 0.2, 0.001);

        let data = create_mock_data("TEST", 200, 100.0);
        let result = backtester.run("TEST", &data, 50, 5).await.unwrap();

        assert!(result.final_capital > 0.0);
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_buy_and_hold() {
        let data = create_mock_data("TEST", 252, 100.0);
        let result = buy_and_hold_benchmark(&data, 100000.0);

        assert!(result.final_capital > 0.0);
        assert_eq!(result.num_trades, 1);
    }
}
