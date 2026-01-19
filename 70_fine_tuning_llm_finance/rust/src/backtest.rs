//! Backtesting framework for LLM trading signals.
//!
//! This module provides:
//! - `LlmBacktester`: Full backtesting simulation
//! - `BacktestResult`: Performance metrics and statistics
//! - `BacktestConfig`: Configuration for backtesting

use crate::signals::{SignalType, TradingSignal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size as fraction of capital (0.0 to 1.0)
    pub position_size: f64,
    /// Stop loss percentage (e.g., 0.05 for 5%)
    pub stop_loss: f64,
    /// Take profit percentage (e.g., 0.15 for 15%)
    pub take_profit: f64,
    /// Transaction cost percentage
    pub transaction_cost: f64,
    /// Whether to allow short selling
    pub allow_short: bool,
    /// Maximum number of concurrent positions
    pub max_positions: usize,
    /// Slippage percentage
    pub slippage: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.2,
            stop_loss: 0.05,
            take_profit: 0.15,
            transaction_cost: 0.001,
            allow_short: false,
            max_positions: 5,
            slippage: 0.0005,
        }
    }
}

impl BacktestConfig {
    /// Set initial capital.
    pub fn with_capital(mut self, capital: f64) -> Self {
        self.initial_capital = capital;
        self
    }

    /// Set position size.
    pub fn with_position_size(mut self, size: f64) -> Self {
        self.position_size = size.clamp(0.01, 1.0);
        self
    }

    /// Set stop loss.
    pub fn with_stop_loss(mut self, stop: f64) -> Self {
        self.stop_loss = stop.clamp(0.01, 0.5);
        self
    }

    /// Set take profit.
    pub fn with_take_profit(mut self, tp: f64) -> Self {
        self.take_profit = tp.clamp(0.01, 1.0);
        self
    }

    /// Enable short selling.
    pub fn with_short_selling(mut self, allow: bool) -> Self {
        self.allow_short = allow;
        self
    }
}

/// A single trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: Option<DateTime<Utc>>,
    /// Exit timestamp
    pub exit_time: Option<DateTime<Utc>>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: Option<f64>,
    /// Position size (positive = long, negative = short)
    pub size: f64,
    /// Profit/loss
    pub pnl: Option<f64>,
    /// Return percentage
    pub return_pct: Option<f64>,
    /// Whether trade is still open
    pub is_open: bool,
    /// Exit reason
    pub exit_reason: Option<String>,
}

impl Trade {
    /// Create a new trade.
    pub fn new(entry_price: f64, size: f64) -> Self {
        Self {
            entry_time: None,
            exit_time: None,
            entry_price,
            exit_price: None,
            size,
            pnl: None,
            return_pct: None,
            is_open: true,
            exit_reason: None,
        }
    }

    /// Close the trade.
    pub fn close(&mut self, exit_price: f64, reason: &str) {
        self.exit_price = Some(exit_price);
        self.is_open = false;
        self.exit_reason = Some(reason.to_string());

        // Calculate P&L
        let price_change = exit_price - self.entry_price;
        self.pnl = Some(price_change * self.size);
        self.return_pct = Some(price_change / self.entry_price);
    }

    /// Check if trade hit stop loss.
    pub fn check_stop_loss(&self, current_price: f64, stop_pct: f64) -> bool {
        let stop_price = if self.size > 0.0 {
            self.entry_price * (1.0 - stop_pct)
        } else {
            self.entry_price * (1.0 + stop_pct)
        };

        if self.size > 0.0 {
            current_price <= stop_price
        } else {
            current_price >= stop_price
        }
    }

    /// Check if trade hit take profit.
    pub fn check_take_profit(&self, current_price: f64, tp_pct: f64) -> bool {
        let tp_price = if self.size > 0.0 {
            self.entry_price * (1.0 + tp_pct)
        } else {
            self.entry_price * (1.0 - tp_pct)
        };

        if self.size > 0.0 {
            current_price >= tp_price
        } else {
            current_price <= tp_price
        }
    }
}

/// Backtest results and performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Initial capital
    pub initial_capital: f64,
    /// Final capital
    pub final_capital: f64,
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (assuming 0% risk-free rate)
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Annualized volatility
    pub volatility: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate
    pub win_rate: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade
    pub avg_loss: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Drawdown series
    pub drawdown_series: Vec<f64>,
    /// Daily returns
    pub daily_returns: Vec<f64>,
    /// All trades
    pub trades: Vec<Trade>,
}

impl BacktestResult {
    /// Create a new empty result.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            final_capital: initial_capital,
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            volatility: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            equity_curve: Vec::new(),
            drawdown_series: Vec::new(),
            daily_returns: Vec::new(),
            trades: Vec::new(),
        }
    }

    /// Calculate all metrics from trades and equity curve.
    pub fn calculate_metrics(&mut self, trading_days: usize) {
        if self.equity_curve.is_empty() {
            return;
        }

        // Final capital
        self.final_capital = *self.equity_curve.last().unwrap_or(&self.initial_capital);

        // Total return
        self.total_return = (self.final_capital - self.initial_capital) / self.initial_capital;

        // Annualized return (assuming 252 trading days)
        let years = trading_days as f64 / 252.0;
        if years > 0.0 {
            self.annualized_return = (1.0 + self.total_return).powf(1.0 / years) - 1.0;
        }

        // Daily returns
        self.daily_returns = self
            .equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Volatility
        if !self.daily_returns.is_empty() {
            let mean: f64 = self.daily_returns.iter().sum::<f64>() / self.daily_returns.len() as f64;
            let variance: f64 = self
                .daily_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / self.daily_returns.len() as f64;
            self.volatility = variance.sqrt() * (252.0_f64).sqrt();
        }

        // Sharpe ratio
        if self.volatility > 0.0 {
            self.sharpe_ratio = self.annualized_return / self.volatility;
        }

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = self
            .daily_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        if !downside_returns.is_empty() {
            let downside_variance: f64 = downside_returns
                .iter()
                .map(|r| r.powi(2))
                .sum::<f64>()
                / downside_returns.len() as f64;
            let downside_dev = downside_variance.sqrt() * (252.0_f64).sqrt();
            if downside_dev > 0.0 {
                self.sortino_ratio = self.annualized_return / downside_dev;
            }
        }

        // Drawdown
        let mut peak = self.initial_capital;
        self.drawdown_series = self
            .equity_curve
            .iter()
            .map(|&equity| {
                if equity > peak {
                    peak = equity;
                }
                (peak - equity) / peak
            })
            .collect();
        self.max_drawdown = self
            .drawdown_series
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);

        // Trade statistics
        self.total_trades = self.trades.len();
        let closed_trades: Vec<&Trade> = self.trades.iter().filter(|t| !t.is_open).collect();

        let winning: Vec<f64> = closed_trades
            .iter()
            .filter_map(|t| t.return_pct)
            .filter(|&r| r > 0.0)
            .collect();
        let losing: Vec<f64> = closed_trades
            .iter()
            .filter_map(|t| t.return_pct)
            .filter(|&r| r <= 0.0)
            .collect();

        self.winning_trades = winning.len();
        self.losing_trades = losing.len();

        if !closed_trades.is_empty() {
            self.win_rate = self.winning_trades as f64 / closed_trades.len() as f64;
        }

        if !winning.is_empty() {
            self.avg_win = winning.iter().sum::<f64>() / winning.len() as f64;
        }
        if !losing.is_empty() {
            self.avg_loss = losing.iter().map(|r| r.abs()).sum::<f64>() / losing.len() as f64;
        }

        // Profit factor
        let gross_profit: f64 = self
            .trades
            .iter()
            .filter_map(|t| t.pnl)
            .filter(|&p| p > 0.0)
            .sum();
        let gross_loss: f64 = self
            .trades
            .iter()
            .filter_map(|t| t.pnl)
            .filter(|&p| p < 0.0)
            .map(|p| p.abs())
            .sum();
        if gross_loss > 0.0 {
            self.profit_factor = gross_profit / gross_loss;
        }
    }

    /// Print a summary report.
    pub fn print_report(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Backtest Report");
        println!("{}", "=".repeat(60));

        println!("\n[Performance Summary]");
        println!("  Initial Capital: ${:.2}", self.initial_capital);
        println!("  Final Capital: ${:.2}", self.final_capital);
        println!("  Total Return: {:.2}%", self.total_return * 100.0);
        println!("  Annualized Return: {:.2}%", self.annualized_return * 100.0);

        println!("\n[Risk Metrics]");
        println!("  Sharpe Ratio: {:.2}", self.sharpe_ratio);
        println!("  Sortino Ratio: {:.2}", self.sortino_ratio);
        println!("  Max Drawdown: {:.2}%", self.max_drawdown * 100.0);
        println!("  Volatility (Ann.): {:.2}%", self.volatility * 100.0);

        println!("\n[Trade Statistics]");
        println!("  Total Trades: {}", self.total_trades);
        println!("  Winning Trades: {}", self.winning_trades);
        println!("  Losing Trades: {}", self.losing_trades);
        println!("  Win Rate: {:.1}%", self.win_rate * 100.0);
        println!("  Avg Win: {:.2}%", self.avg_win * 100.0);
        println!("  Avg Loss: {:.2}%", self.avg_loss * 100.0);
        println!("  Profit Factor: {:.2}", self.profit_factor);

        println!("\n{}", "=".repeat(60));
    }
}

/// LLM-based backtester.
pub struct LlmBacktester {
    /// Configuration
    config: BacktestConfig,
    /// Current capital
    capital: f64,
    /// Current position
    position: Option<Trade>,
    /// All trades
    trades: Vec<Trade>,
    /// Equity curve
    equity_curve: Vec<f64>,
}

impl LlmBacktester {
    /// Create a new backtester.
    pub fn new(config: BacktestConfig) -> Self {
        let capital = config.initial_capital;
        Self {
            config,
            capital,
            position: None,
            trades: Vec::new(),
            equity_curve: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(BacktestConfig::default())
    }

    /// Run the backtest.
    ///
    /// # Arguments
    ///
    /// * `prices` - Price series
    /// * `signals` - Signal series (same length as prices)
    ///
    /// # Returns
    ///
    /// Backtest results with all metrics.
    pub fn run(&mut self, prices: &[f64], signals: &[SignalType]) -> BacktestResult {
        assert_eq!(prices.len(), signals.len(), "Prices and signals must have same length");

        self.capital = self.config.initial_capital;
        self.position = None;
        self.trades.clear();
        self.equity_curve.clear();

        for (i, (&price, &signal)) in prices.iter().zip(signals.iter()).enumerate() {
            // Check existing position
            if let Some(ref mut pos) = self.position {
                // Check stop loss
                if pos.check_stop_loss(price, self.config.stop_loss) {
                    pos.close(price * (1.0 - self.config.slippage), "stop_loss");
                    self.capital += pos.pnl.unwrap_or(0.0);
                    self.trades.push(pos.clone());
                    self.position = None;
                }
                // Check take profit
                else if pos.check_take_profit(price, self.config.take_profit) {
                    pos.close(price * (1.0 - self.config.slippage), "take_profit");
                    self.capital += pos.pnl.unwrap_or(0.0);
                    self.trades.push(pos.clone());
                    self.position = None;
                }
                // Check exit signal
                else if (pos.size > 0.0 && signal == SignalType::Sell)
                    || (pos.size < 0.0 && signal == SignalType::Buy)
                {
                    pos.close(price * (1.0 - self.config.slippage), "signal");
                    self.capital += pos.pnl.unwrap_or(0.0);
                    self.trades.push(pos.clone());
                    self.position = None;
                }
            }

            // Open new position
            if self.position.is_none() {
                match signal {
                    SignalType::Buy => {
                        let position_value = self.capital * self.config.position_size;
                        let entry_price = price * (1.0 + self.config.slippage);
                        let size = position_value / entry_price;

                        // Deduct transaction cost
                        self.capital -= position_value * self.config.transaction_cost;

                        self.position = Some(Trade::new(entry_price, size));
                    }
                    SignalType::Sell if self.config.allow_short => {
                        let position_value = self.capital * self.config.position_size;
                        let entry_price = price * (1.0 - self.config.slippage);
                        let size = -position_value / entry_price;

                        self.capital -= position_value * self.config.transaction_cost;

                        self.position = Some(Trade::new(entry_price, size));
                    }
                    _ => {}
                }
            }

            // Calculate current equity
            let position_value = match &self.position {
                Some(pos) => {
                    let current_value = price * pos.size;
                    let entry_value = pos.entry_price * pos.size;
                    current_value - entry_value
                }
                None => 0.0,
            };

            self.equity_curve.push(self.capital + position_value);
        }

        // Close any remaining position
        if let Some(ref mut pos) = self.position {
            let final_price = *prices.last().unwrap_or(&0.0);
            pos.close(final_price, "end_of_backtest");
            self.capital += pos.pnl.unwrap_or(0.0);
            self.trades.push(pos.clone());
        }

        // Calculate results
        let mut result = BacktestResult::new(self.config.initial_capital);
        result.equity_curve = self.equity_curve.clone();
        result.trades = self.trades.clone();
        result.calculate_metrics(prices.len());

        result
    }

    /// Reset the backtester.
    pub fn reset(&mut self) {
        self.capital = self.config.initial_capital;
        self.position = None;
        self.trades.clear();
        self.equity_curve.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_config() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_capital, 100_000.0);
        assert_eq!(config.position_size, 0.2);
    }

    #[test]
    fn test_trade_creation() {
        let trade = Trade::new(100.0, 10.0);
        assert!(trade.is_open);
        assert_eq!(trade.entry_price, 100.0);
        assert_eq!(trade.size, 10.0);
    }

    #[test]
    fn test_trade_close() {
        let mut trade = Trade::new(100.0, 10.0);
        trade.close(110.0, "signal");

        assert!(!trade.is_open);
        assert_eq!(trade.exit_price, Some(110.0));
        assert_eq!(trade.pnl, Some(100.0)); // 10 * 10 = 100
    }

    #[test]
    fn test_stop_loss() {
        let trade = Trade::new(100.0, 10.0);
        assert!(trade.check_stop_loss(94.0, 0.05)); // 5% stop loss
        assert!(!trade.check_stop_loss(96.0, 0.05));
    }

    #[test]
    fn test_take_profit() {
        let trade = Trade::new(100.0, 10.0);
        assert!(trade.check_take_profit(116.0, 0.15)); // 15% take profit
        assert!(!trade.check_take_profit(114.0, 0.15));
    }

    #[test]
    fn test_backtest_run() {
        let config = BacktestConfig::default()
            .with_capital(100_000.0)
            .with_position_size(0.2);

        let mut backtester = LlmBacktester::new(config);

        // Simple price series
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();

        // Alternating signals
        let signals: Vec<SignalType> = (0..100)
            .map(|i| {
                if i % 20 < 10 {
                    SignalType::Buy
                } else {
                    SignalType::Sell
                }
            })
            .collect();

        let result = backtester.run(&prices, &signals);

        assert!(!result.equity_curve.is_empty());
        assert!(result.total_trades > 0);
    }

    #[test]
    fn test_result_metrics() {
        let mut result = BacktestResult::new(100_000.0);
        result.equity_curve = vec![100_000.0, 101_000.0, 102_000.0, 101_500.0, 103_000.0];

        let mut trade1 = Trade::new(100.0, 10.0);
        trade1.close(110.0, "signal");

        let mut trade2 = Trade::new(110.0, 10.0);
        trade2.close(105.0, "stop_loss");

        result.trades = vec![trade1, trade2];
        result.calculate_metrics(5);

        assert!(result.total_return > 0.0);
        assert_eq!(result.total_trades, 2);
        assert_eq!(result.winning_trades, 1);
        assert_eq!(result.losing_trades, 1);
    }
}
