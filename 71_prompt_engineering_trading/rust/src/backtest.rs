//! Backtesting engine for LLM-generated signals.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::signals::{SignalDirection, TradingSignal};

/// Configuration for backtesting.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub position_size_pct: f64,
    pub max_positions: usize,
    pub commission_pct: f64,
    pub slippage_pct: f64,
    pub use_stop_loss: bool,
    pub use_take_profit: bool,
    pub allow_short: bool,
    pub max_drawdown_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            position_size_pct: 0.1,
            max_positions: 5,
            commission_pct: 0.001,
            slippage_pct: 0.0005,
            use_stop_loss: true,
            use_take_profit: true,
            allow_short: true,
            max_drawdown_pct: 0.2,
        }
    }
}

/// A single trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub direction: SignalDirection,
    pub entry_price: f64,
    pub entry_time: DateTime<Utc>,
    pub size: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub exit_price: Option<f64>,
    pub exit_time: Option<DateTime<Utc>>,
    pub exit_reason: Option<String>,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub llm_confidence: f32,
}

/// Results from backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_return: f64,
    pub total_return_pct: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub avg_holding_period_hours: f64,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
    pub confidence_correlation: f64,
}

impl BacktestResult {
    /// Convert to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// OHLCV bar data.
#[derive(Debug, Clone)]
pub struct Bar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Backtester for LLM-generated signals.
pub struct Backtester {
    config: BacktestConfig,
    capital: f64,
    positions: HashMap<String, Trade>,
    closed_trades: Vec<Trade>,
    equity_history: Vec<f64>,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(config: BacktestConfig) -> Self {
        let initial_capital = config.initial_capital;
        Self {
            config,
            capital: initial_capital,
            positions: HashMap::new(),
            closed_trades: Vec::new(),
            equity_history: vec![initial_capital],
        }
    }

    /// Run backtest on signals.
    pub fn run(
        &mut self,
        signals: &[TradingSignal],
        price_data: &HashMap<String, Vec<Bar>>,
    ) -> BacktestResult {
        // Reset state
        self.capital = self.config.initial_capital;
        self.positions.clear();
        self.closed_trades.clear();
        self.equity_history = vec![self.capital];

        // Sort signals by timestamp
        let mut sorted_signals: Vec<_> = signals.iter().collect();
        sorted_signals.sort_by_key(|s| s.timestamp);

        for signal in sorted_signals {
            if let Some(timestamp) = signal.timestamp {
                // Get price at signal time
                if let Some(prices) = price_data.get(&signal.symbol) {
                    if let Some(current_price) = self.get_price_at_time(prices, timestamp) {
                        // Update existing positions
                        self.update_positions(price_data, timestamp);

                        // Process new signal
                        if signal.direction != SignalDirection::Hold {
                            self.process_signal(signal, current_price, timestamp);
                        }

                        // Record equity
                        let equity = self.calculate_equity(price_data, timestamp);
                        self.equity_history.push(equity);

                        // Check max drawdown
                        if self.check_max_drawdown() {
                            break;
                        }
                    }
                }
            }
        }

        // Close all remaining positions
        self.close_all_positions(price_data);

        self.calculate_results()
    }

    /// Process a trading signal.
    fn process_signal(&mut self, signal: &TradingSignal, price: f64, time: DateTime<Utc>) {
        // Check if we already have position
        if let Some(existing) = self.positions.get(&signal.symbol) {
            if existing.direction != signal.direction {
                self.close_position(&signal.symbol.clone(), price, time, "signal_reversal");
            } else {
                return;
            }
        }

        // Check max positions
        if self.positions.len() >= self.config.max_positions {
            return;
        }

        // Check if short allowed
        if signal.direction == SignalDirection::Sell && !self.config.allow_short {
            return;
        }

        // Calculate position size
        let position_value = self.capital * self.config.position_size_pct;

        // Apply slippage
        let entry_price = match signal.direction {
            SignalDirection::Buy => price * (1.0 + self.config.slippage_pct),
            SignalDirection::Sell => price * (1.0 - self.config.slippage_pct),
            SignalDirection::Hold => return,
        };

        let size = position_value / entry_price;
        let commission = position_value * self.config.commission_pct;

        // Create trade
        let trade = Trade {
            symbol: signal.symbol.clone(),
            direction: signal.direction,
            entry_price,
            entry_time: time,
            size,
            stop_loss: if self.config.use_stop_loss {
                Some(signal.stop_loss)
            } else {
                None
            },
            take_profit: if self.config.use_take_profit {
                Some(signal.take_profit)
            } else {
                None
            },
            exit_price: None,
            exit_time: None,
            exit_reason: None,
            pnl: 0.0,
            pnl_pct: 0.0,
            llm_confidence: signal.confidence,
        };

        self.positions.insert(signal.symbol.clone(), trade);
        self.capital -= commission;
    }

    /// Update positions and check stop/take profit.
    fn update_positions(&mut self, price_data: &HashMap<String, Vec<Bar>>, current_time: DateTime<Utc>) {
        let mut to_close = Vec::new();

        for (symbol, trade) in &self.positions {
            if let Some(prices) = price_data.get(symbol) {
                if let Some(current_price) = self.get_price_at_time(prices, current_time) {
                    // Check stop loss
                    if let Some(stop_loss) = trade.stop_loss {
                        let triggered = match trade.direction {
                            SignalDirection::Buy => current_price <= stop_loss,
                            SignalDirection::Sell => current_price >= stop_loss,
                            SignalDirection::Hold => false,
                        };
                        if triggered {
                            to_close.push((symbol.clone(), current_price, "stop_loss"));
                            continue;
                        }
                    }

                    // Check take profit
                    if let Some(take_profit) = trade.take_profit {
                        let triggered = match trade.direction {
                            SignalDirection::Buy => current_price >= take_profit,
                            SignalDirection::Sell => current_price <= take_profit,
                            SignalDirection::Hold => false,
                        };
                        if triggered {
                            to_close.push((symbol.clone(), current_price, "take_profit"));
                        }
                    }
                }
            }
        }

        for (symbol, price, reason) in to_close {
            self.close_position(&symbol, price, current_time, reason);
        }
    }

    /// Close a position.
    fn close_position(&mut self, symbol: &str, price: f64, time: DateTime<Utc>, reason: &str) {
        if let Some(mut trade) = self.positions.remove(symbol) {
            // Apply slippage
            let exit_price = match trade.direction {
                SignalDirection::Buy => price * (1.0 - self.config.slippage_pct),
                SignalDirection::Sell => price * (1.0 + self.config.slippage_pct),
                SignalDirection::Hold => price,
            };

            // Calculate P&L
            let pnl = match trade.direction {
                SignalDirection::Buy => (exit_price - trade.entry_price) * trade.size,
                SignalDirection::Sell => (trade.entry_price - exit_price) * trade.size,
                SignalDirection::Hold => 0.0,
            };

            // Subtract commission
            let commission = (trade.size * exit_price).abs() * self.config.commission_pct;
            let net_pnl = pnl - commission;

            // Update trade
            trade.exit_price = Some(exit_price);
            trade.exit_time = Some(time);
            trade.exit_reason = Some(reason.to_string());
            trade.pnl = net_pnl;
            trade.pnl_pct = (net_pnl / (trade.entry_price * trade.size)) * 100.0;

            // Update capital
            self.capital += trade.entry_price * trade.size + net_pnl;

            self.closed_trades.push(trade);
        }
    }

    /// Close all remaining positions.
    fn close_all_positions(&mut self, price_data: &HashMap<String, Vec<Bar>>) {
        let symbols: Vec<_> = self.positions.keys().cloned().collect();

        for symbol in symbols {
            if let Some(prices) = price_data.get(&symbol) {
                if let Some(last_bar) = prices.last() {
                    self.close_position(&symbol, last_bar.close, last_bar.timestamp, "end_of_backtest");
                }
            }
        }
    }

    /// Get price at specific time.
    fn get_price_at_time(&self, prices: &[Bar], target_time: DateTime<Utc>) -> Option<f64> {
        for bar in prices {
            if bar.timestamp >= target_time {
                return Some(bar.close);
            }
        }
        prices.last().map(|b| b.close)
    }

    /// Calculate current equity.
    fn calculate_equity(&self, price_data: &HashMap<String, Vec<Bar>>, current_time: DateTime<Utc>) -> f64 {
        let mut equity = self.capital;

        for (symbol, trade) in &self.positions {
            if let Some(prices) = price_data.get(symbol) {
                if let Some(current_price) = self.get_price_at_time(prices, current_time) {
                    let unrealized = match trade.direction {
                        SignalDirection::Buy => (current_price - trade.entry_price) * trade.size,
                        SignalDirection::Sell => (trade.entry_price - current_price) * trade.size,
                        SignalDirection::Hold => 0.0,
                    };
                    equity += trade.entry_price * trade.size + unrealized;
                }
            }
        }

        equity
    }

    /// Check if max drawdown exceeded.
    fn check_max_drawdown(&self) -> bool {
        if self.equity_history.is_empty() {
            return false;
        }

        let peak = self.equity_history.iter().cloned().fold(f64::MIN, f64::max);
        let current = *self.equity_history.last().unwrap();
        let drawdown_pct = (peak - current) / peak;

        drawdown_pct >= self.config.max_drawdown_pct
    }

    /// Calculate backtest results.
    fn calculate_results(&self) -> BacktestResult {
        if self.closed_trades.is_empty() {
            return BacktestResult {
                total_return: 0.0,
                total_return_pct: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                max_drawdown_pct: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                avg_win: 0.0,
                avg_loss: 0.0,
                avg_holding_period_hours: 0.0,
                trades: vec![],
                equity_curve: self.equity_history.clone(),
                confidence_correlation: 0.0,
            };
        }

        // Basic metrics
        let total_pnl: f64 = self.closed_trades.iter().map(|t| t.pnl).sum();
        let total_return_pct = (total_pnl / self.config.initial_capital) * 100.0;

        // Win/loss metrics
        let winners: Vec<_> = self.closed_trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losers: Vec<_> = self.closed_trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if !self.closed_trades.is_empty() {
            (winners.len() as f64 / self.closed_trades.len() as f64) * 100.0
        } else {
            0.0
        };

        let total_wins: f64 = winners.iter().map(|t| t.pnl).sum();
        let total_losses: f64 = losers.iter().map(|t| t.pnl.abs()).sum();

        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else {
            f64::INFINITY
        };

        let avg_win = if !winners.is_empty() {
            total_wins / winners.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losers.is_empty() {
            total_losses / losers.len() as f64
        } else {
            0.0
        };

        // Drawdown
        let mut peak = self.equity_history[0];
        let mut max_dd = 0.0;
        let mut max_dd_pct = 0.0;

        for &equity in &self.equity_history {
            if equity > peak {
                peak = equity;
            }
            let dd = peak - equity;
            let dd_pct = if peak > 0.0 { dd / peak } else { 0.0 };
            if dd > max_dd {
                max_dd = dd;
                max_dd_pct = dd_pct;
            }
        }

        // Sharpe ratio (simplified)
        let sharpe = if self.equity_history.len() > 1 {
            let returns: Vec<f64> = self.equity_history
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();

            if returns.len() > 1 {
                let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns.iter()
                    .map(|r| (r - avg_return).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                let std_dev = variance.sqrt();

                if std_dev > 0.0 {
                    (avg_return * 252.0_f64.sqrt()) / std_dev
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Average holding period
        let holding_periods: Vec<f64> = self.closed_trades
            .iter()
            .filter_map(|t| {
                t.exit_time.map(|exit| {
                    (exit - t.entry_time).num_hours() as f64
                })
            })
            .collect();

        let avg_holding = if !holding_periods.is_empty() {
            holding_periods.iter().sum::<f64>() / holding_periods.len() as f64
        } else {
            0.0
        };

        // Confidence correlation
        let conf_corr = self.calculate_confidence_correlation();

        BacktestResult {
            total_return: total_pnl,
            total_return_pct,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            max_drawdown_pct: max_dd_pct * 100.0,
            win_rate,
            profit_factor,
            total_trades: self.closed_trades.len(),
            winning_trades: winners.len(),
            losing_trades: losers.len(),
            avg_win,
            avg_loss,
            avg_holding_period_hours: avg_holding,
            trades: self.closed_trades.clone(),
            equity_curve: self.equity_history.clone(),
            confidence_correlation: conf_corr,
        }
    }

    /// Calculate correlation between LLM confidence and returns.
    fn calculate_confidence_correlation(&self) -> f64 {
        if self.closed_trades.len() < 3 {
            return 0.0;
        }

        let confidences: Vec<f64> = self.closed_trades.iter().map(|t| t.llm_confidence as f64).collect();
        let returns: Vec<f64> = self.closed_trades.iter().map(|t| t.pnl_pct).collect();

        let n = confidences.len() as f64;
        let mean_conf: f64 = confidences.iter().sum::<f64>() / n;
        let mean_ret: f64 = returns.iter().sum::<f64>() / n;

        let num: f64 = confidences.iter()
            .zip(returns.iter())
            .map(|(c, r)| (c - mean_conf) * (r - mean_ret))
            .sum();

        let var_conf: f64 = confidences.iter().map(|c| (c - mean_conf).powi(2)).sum();
        let var_ret: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum();

        let denom = (var_conf * var_ret).sqrt();

        if denom > 0.0 {
            num / denom
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn create_test_signal(symbol: &str, direction: SignalDirection, price: f64) -> TradingSignal {
        TradingSignal {
            symbol: symbol.to_string(),
            direction,
            confidence: 75.0,
            entry_price: price,
            stop_loss: price * 0.95,
            take_profit: price * 1.10,
            position_size_pct: 5.0,
            timeframe: "1d".to_string(),
            reasoning: "Test signal".to_string(),
            key_factors: vec![],
            timestamp: Some(Utc::now()),
        }
    }

    fn create_test_bars(base_price: f64, days: usize) -> Vec<Bar> {
        let mut bars = Vec::new();
        let mut price = base_price;

        for i in 0..days {
            price *= 1.0 + (i as f64 * 0.001); // Small upward drift
            bars.push(Bar {
                timestamp: Utc::now() + Duration::days(i as i64),
                open: price,
                high: price * 1.01,
                low: price * 0.99,
                close: price,
                volume: 1000000.0,
            });
        }

        bars
    }

    #[test]
    fn test_backtester_basic() {
        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config);

        let signals = vec![create_test_signal("AAPL", SignalDirection::Buy, 185.0)];

        let mut price_data = HashMap::new();
        price_data.insert("AAPL".to_string(), create_test_bars(185.0, 30));

        let result = backtester.run(&signals, &price_data);

        assert!(result.total_trades > 0 || result.equity_curve.len() > 1);
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_capital, 100000.0);
        assert_eq!(config.position_size_pct, 0.1);
        assert_eq!(config.max_positions, 5);
    }
}
