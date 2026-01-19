//! Backtesting module
//!
//! Framework for backtesting earnings call trading strategies.

use crate::trading::{SignalGenerator, SignalType, TradingSignal};

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: i64,
    pub exit_time: i64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub signal: SignalType,
    pub pnl: f64,
    pub pnl_pct: f64,
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub trades: Vec<Trade>,
    pub total_pnl: f64,
    pub total_pnl_pct: f64,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub num_trades: usize,
}

/// Earnings call event for backtesting
#[derive(Debug, Clone)]
pub struct EarningsEvent {
    pub timestamp: i64,
    pub transcript: String,
    pub price_before: f64,
    pub price_after: f64,
}

/// Backtester configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub position_size: f64,
    pub holding_period_days: i64,
    pub transaction_cost: f64,
    pub risk_free_rate: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            holding_period_days: 5,
            transaction_cost: 0.001,
            risk_free_rate: 0.05,
        }
    }
}

/// Backtester for earnings call strategies
pub struct Backtester {
    signal_generator: SignalGenerator,
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester
    pub fn new() -> Self {
        Self {
            signal_generator: SignalGenerator::new(),
            config: BacktestConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BacktestConfig) -> Self {
        Self {
            signal_generator: SignalGenerator::new(),
            config,
        }
    }

    /// Run backtest on earnings events
    pub fn run(&self, events: &[EarningsEvent]) -> BacktestResults {
        let mut trades = Vec::new();
        let mut returns = Vec::new();

        for event in events {
            let signal = self.signal_generator.generate_signal(&event.transcript);

            // Only trade on non-neutral signals
            if signal.signal_type == SignalType::Neutral {
                continue;
            }

            let trade = self.execute_trade(event, &signal);
            returns.push(trade.pnl_pct);
            trades.push(trade);
        }

        self.calculate_metrics(trades, returns)
    }

    /// Execute a simulated trade
    fn execute_trade(&self, event: &EarningsEvent, signal: &TradingSignal) -> Trade {
        let is_long = matches!(
            signal.signal_type,
            SignalType::StrongBuy | SignalType::Buy
        );

        let entry_price = event.price_before;
        let exit_price = event.price_after;

        let gross_pnl_pct = if is_long {
            (exit_price - entry_price) / entry_price
        } else {
            (entry_price - exit_price) / entry_price
        };

        // Apply transaction costs
        let net_pnl_pct = gross_pnl_pct - 2.0 * self.config.transaction_cost;

        let position_value = self.config.initial_capital * self.config.position_size;
        let pnl = position_value * net_pnl_pct;

        let holding_seconds = self.config.holding_period_days * 24 * 60 * 60;

        Trade {
            entry_time: event.timestamp,
            exit_time: event.timestamp + holding_seconds,
            entry_price,
            exit_price,
            signal: signal.signal_type,
            pnl,
            pnl_pct: net_pnl_pct,
        }
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self, trades: Vec<Trade>, returns: Vec<f64>) -> BacktestResults {
        let num_trades = trades.len();

        if num_trades == 0 {
            return BacktestResults {
                trades,
                total_pnl: 0.0,
                total_pnl_pct: 0.0,
                win_rate: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                num_trades: 0,
            };
        }

        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let total_pnl_pct: f64 = returns.iter().sum();

        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = winning_trades as f64 / num_trades as f64;

        let sharpe_ratio = self.calculate_sharpe(&returns);
        let sortino_ratio = self.calculate_sortino(&returns);
        let max_drawdown = self.calculate_max_drawdown(&returns);

        BacktestResults {
            trades,
            total_pnl,
            total_pnl_pct,
            win_rate,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            num_trades,
        }
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        let std_dev = variance.sqrt();
        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualize assuming quarterly earnings
        let annual_factor = (4.0_f64).sqrt();
        let daily_rf = self.config.risk_free_rate / 252.0;
        let excess_return = mean_return - daily_rf * self.config.holding_period_days as f64;

        (excess_return / std_dev) * annual_factor
    }

    /// Calculate Sortino ratio (downside deviation)
    fn calculate_sortino(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        // Only consider negative returns for downside deviation
        let downside_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|&r| r.powi(2))
            .collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_deviation =
            (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt();

        if downside_deviation == 0.0 {
            return 0.0;
        }

        let annual_factor = (4.0_f64).sqrt();
        let daily_rf = self.config.risk_free_rate / 252.0;
        let excess_return = mean_return - daily_rf * self.config.holding_period_days as f64;

        (excess_return / downside_deviation) * annual_factor
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for &ret in returns {
            cumulative *= 1.0 + ret;
            if cumulative > peak {
                peak = cumulative;
            }
            let drawdown = (peak - cumulative) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for BacktestResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Backtest Results ===")?;
        writeln!(f, "Number of trades: {}", self.num_trades)?;
        writeln!(f, "Total P&L: ${:.2}", self.total_pnl)?;
        writeln!(f, "Total P&L %: {:.2}%", self.total_pnl_pct * 100.0)?;
        writeln!(f, "Win rate: {:.1}%", self.win_rate * 100.0)?;
        writeln!(f, "Sharpe ratio: {:.2}", self.sharpe_ratio)?;
        writeln!(f, "Sortino ratio: {:.2}", self.sortino_ratio)?;
        writeln!(f, "Max drawdown: {:.2}%", self.max_drawdown * 100.0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_bullish_transcript() -> String {
        r#"
John Smith - CEO:
Exceptional quarter with strong growth and record results.
We exceeded expectations and are raising guidance.
We are confident in our trajectory.
        "#
        .to_string()
    }

    fn create_bearish_transcript() -> String {
        r#"
Jane Doe - CFO:
Challenging quarter with declining revenue.
We faced significant headwinds and are lowering guidance.
Outlook remains uncertain.
        "#
        .to_string()
    }

    #[test]
    fn test_backtest_basic() {
        let backtester = Backtester::new();

        let events = vec![
            EarningsEvent {
                timestamp: 1000000,
                transcript: create_bullish_transcript(),
                price_before: 100.0,
                price_after: 110.0,
            },
            EarningsEvent {
                timestamp: 2000000,
                transcript: create_bearish_transcript(),
                price_before: 100.0,
                price_after: 90.0,
            },
        ];

        let results = backtester.run(&events);

        assert!(results.num_trades > 0);
    }

    #[test]
    fn test_empty_backtest() {
        let backtester = Backtester::new();
        let results = backtester.run(&[]);

        assert_eq!(results.num_trades, 0);
        assert_eq!(results.total_pnl, 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let backtester = Backtester::new();
        let returns = vec![0.1, -0.05, -0.1, 0.15, -0.02];
        let max_dd = backtester.calculate_max_drawdown(&returns);

        assert!(max_dd > 0.0);
        assert!(max_dd < 1.0);
    }
}
