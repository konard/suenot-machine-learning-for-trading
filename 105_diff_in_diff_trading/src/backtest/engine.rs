//! Backtesting engine for DiD-based trading strategies.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::data::Candle;
use crate::model::DiDResults;
use crate::trading::{Position, Signal, SignalGenerator, TradingStrategy};

/// Performance metrics from backtesting.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total return over the backtest period
    pub total_return: f64,
    /// Annualized return (assuming 252 trading days)
    pub annualized_return: f64,
    /// Annualized volatility
    pub volatility: f64,
    /// Sharpe ratio (assuming 0% risk-free rate)
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside volatility)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate (fraction of profitable trades)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total number of trades
    pub num_trades: usize,
    /// Average return per trade
    pub avg_trade_return: f64,
    /// Average holding period in days
    pub avg_holding_period: f64,
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PerformanceMetrics(")?;
        writeln!(f, "  total_return       = {:.2}%", self.total_return * 100.0)?;
        writeln!(
            f,
            "  annualized_return  = {:.2}%",
            self.annualized_return * 100.0
        )?;
        writeln!(f, "  volatility         = {:.2}%", self.volatility * 100.0)?;
        writeln!(f, "  sharpe_ratio       = {:.2}", self.sharpe_ratio)?;
        writeln!(f, "  sortino_ratio      = {:.2}", self.sortino_ratio)?;
        writeln!(f, "  max_drawdown       = {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  win_rate           = {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  profit_factor      = {:.2}", self.profit_factor)?;
        writeln!(f, "  num_trades         = {}", self.num_trades)?;
        writeln!(
            f,
            "  avg_trade_return   = {:.2}%",
            self.avg_trade_return * 100.0
        )?;
        writeln!(
            f,
            "  avg_holding_period = {:.1} days",
            self.avg_holding_period
        )?;
        write!(f, ")")
    }
}

/// Event for backtesting.
#[derive(Debug, Clone)]
pub struct Event {
    /// Event date
    pub date: DateTime<Utc>,
    /// Symbol affected by the event
    pub treated_symbol: String,
    /// Control symbol for comparison
    pub control_symbol: String,
    /// DiD estimation results
    pub did_results: DiDResults,
}

/// Backtester for event-driven trading strategies.
pub struct Backtester {
    /// Initial capital
    initial_capital: f64,
    /// Current portfolio value
    portfolio_value: f64,
    /// Cash balance
    cash: f64,
    /// Open positions
    positions: HashMap<String, Position>,
    /// Closed trades
    trades: Vec<Position>,
    /// Equity curve
    equity_curve: Vec<(DateTime<Utc>, f64)>,
    /// Trading strategy
    strategy: TradingStrategy,
    /// Signal generator
    signal_generator: SignalGenerator,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(initial_capital: f64, strategy: TradingStrategy) -> Self {
        Self {
            initial_capital,
            portfolio_value: initial_capital,
            cash: initial_capital,
            positions: HashMap::new(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
            strategy,
            signal_generator: SignalGenerator::new(),
        }
    }

    /// Reset the backtester state.
    pub fn reset(&mut self) {
        self.portfolio_value = self.initial_capital;
        self.cash = self.initial_capital;
        self.positions.clear();
        self.trades.clear();
        self.equity_curve.clear();
    }

    /// Run backtest on a list of events.
    pub fn run(
        &mut self,
        events: &[Event],
        price_data: &HashMap<String, Vec<Candle>>,
    ) -> PerformanceMetrics {
        self.reset();

        // Sort events by date
        let mut sorted_events: Vec<_> = events.iter().collect();
        sorted_events.sort_by_key(|e| e.date);

        // Get all unique dates
        let mut all_dates: Vec<DateTime<Utc>> = price_data
            .values()
            .flat_map(|candles| candles.iter().map(|c| c.timestamp))
            .collect();
        all_dates.sort();
        all_dates.dedup();

        for date in &all_dates {
            // Update portfolio value
            self.update_portfolio_value(*date, price_data);

            // Check exits for existing positions
            self.check_exits(*date, price_data);

            // Process events on this date
            for event in &sorted_events {
                if event.date == *date {
                    let signal = self.signal_generator.generate(&event.did_results);

                    if self.strategy.should_enter(signal, self.positions.len())
                        && !self.positions.contains_key(&event.treated_symbol)
                    {
                        self.enter_position(*date, &event.treated_symbol, signal, price_data);
                    }
                }
            }

            // Record equity
            self.equity_curve.push((*date, self.portfolio_value));
        }

        // Close remaining positions
        if let Some(last_date) = all_dates.last() {
            let symbols: Vec<_> = self.positions.keys().cloned().collect();
            for symbol in symbols {
                self.exit_position(*last_date, &symbol, price_data, "end_of_backtest");
            }
        }

        self.compute_metrics()
    }

    /// Enter a new position.
    fn enter_position(
        &mut self,
        date: DateTime<Utc>,
        symbol: &str,
        signal: Signal,
        price_data: &HashMap<String, Vec<Candle>>,
    ) {
        let candles = match price_data.get(symbol) {
            Some(c) => c,
            None => return,
        };

        let price = match candles.iter().find(|c| c.timestamp == date) {
            Some(c) => c.close,
            None => return,
        };

        let direction = signal.as_i32();
        let entry_price = self.strategy.apply_slippage(price, direction);

        // Calculate position size
        let max_size = self.portfolio_value * self.strategy.max_position_size;
        let units = max_size / entry_price;

        // Deduct transaction cost
        let cost = self.strategy.calc_transaction_cost(max_size);
        self.cash -= cost;

        let position = Position::new(symbol.to_string(), date, entry_price, direction, units);

        self.positions.insert(symbol.to_string(), position);
    }

    /// Exit a position.
    fn exit_position(
        &mut self,
        date: DateTime<Utc>,
        symbol: &str,
        price_data: &HashMap<String, Vec<Candle>>,
        reason: &str,
    ) {
        let position = match self.positions.remove(symbol) {
            Some(p) => p,
            None => return,
        };

        let candles = match price_data.get(symbol) {
            Some(c) => c,
            None => {
                self.positions.insert(symbol.to_string(), position);
                return;
            }
        };

        let price = match candles.iter().find(|c| c.timestamp == date) {
            Some(c) => c.close,
            None => {
                self.positions.insert(symbol.to_string(), position);
                return;
            }
        };

        let exit_price = self.strategy.apply_slippage(price, -position.direction);

        let mut closed_position = position;
        closed_position.close(date, exit_price, reason);

        // Add PnL to cash
        if let Some(pnl) = closed_position.pnl {
            self.cash += closed_position.size * closed_position.entry_price + pnl;
        }

        // Deduct transaction cost
        let cost = self.strategy.calc_transaction_cost(closed_position.size * exit_price);
        self.cash -= cost;

        self.trades.push(closed_position);
    }

    /// Check exit conditions for all open positions.
    fn check_exits(&mut self, date: DateTime<Utc>, price_data: &HashMap<String, Vec<Candle>>) {
        let symbols: Vec<_> = self.positions.keys().cloned().collect();

        for symbol in symbols {
            let candles = match price_data.get(&symbol) {
                Some(c) => c,
                None => continue,
            };

            let current_price = match candles.iter().find(|c| c.timestamp == date) {
                Some(c) => c.close,
                None => continue,
            };

            let position = match self.positions.get(&symbol) {
                Some(p) => p,
                None => continue,
            };

            if let Some(reason) = self.strategy.check_exit(position, current_price, date) {
                self.exit_position(date, &symbol, price_data, &reason);
            }
        }
    }

    /// Update portfolio value based on current prices.
    fn update_portfolio_value(
        &mut self,
        date: DateTime<Utc>,
        price_data: &HashMap<String, Vec<Candle>>,
    ) {
        let mut position_value = 0.0;

        for (symbol, position) in &self.positions {
            let candles = match price_data.get(symbol) {
                Some(c) => c,
                None => continue,
            };

            if let Some(candle) = candles.iter().find(|c| c.timestamp == date) {
                position_value += position.size * candle.close;
            }
        }

        self.portfolio_value = self.cash + position_value;
    }

    /// Compute performance metrics.
    fn compute_metrics(&self) -> PerformanceMetrics {
        if self.equity_curve.is_empty() {
            return self.empty_metrics();
        }

        // Extract equity values
        let equity: Vec<f64> = self.equity_curve.iter().map(|(_, v)| *v).collect();

        // Returns
        let returns: Vec<f64> = equity
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Total return
        let total_return = (equity.last().unwrap() - equity[0]) / equity[0];

        // Annualized return
        let n_days = if self.equity_curve.len() >= 2 {
            (self.equity_curve.last().unwrap().0 - self.equity_curve[0].0).num_days()
        } else {
            1
        };
        let annualized_return = if n_days > 0 {
            (1.0 + total_return).powf(365.0 / n_days as f64) - 1.0
        } else {
            0.0
        };

        // Volatility
        let volatility = if returns.len() > 1 {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            variance.sqrt() * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Sharpe ratio
        let sharpe_ratio = if volatility > 0.0 {
            annualized_return / volatility
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_vol = if downside_returns.len() > 1 {
            let mean = downside_returns.iter().sum::<f64>() / downside_returns.len() as f64;
            let variance = downside_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / (downside_returns.len() - 1) as f64;
            variance.sqrt() * (252.0_f64).sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_vol > 0.0 {
            annualized_return / downside_vol
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = equity[0];
        let mut max_dd = 0.0;
        for &value in &equity {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        // Trade statistics
        let pnls: Vec<f64> = self
            .trades
            .iter()
            .filter_map(|t| t.pnl)
            .collect();

        let win_rate = if !pnls.is_empty() {
            pnls.iter().filter(|&&p| p > 0.0).count() as f64 / pnls.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = pnls.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = if !pnls.is_empty() {
            pnls.iter().sum::<f64>() / pnls.len() as f64 / self.initial_capital
        } else {
            0.0
        };

        let holding_periods: Vec<f64> = self
            .trades
            .iter()
            .filter_map(|t| t.exit_time.map(|exit| (exit - t.entry_time).num_days() as f64))
            .collect();
        let avg_holding_period = if !holding_periods.is_empty() {
            holding_periods.iter().sum::<f64>() / holding_periods.len() as f64
        } else {
            0.0
        };

        PerformanceMetrics {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: max_dd,
            win_rate,
            profit_factor,
            num_trades: self.trades.len(),
            avg_trade_return,
            avg_holding_period,
        }
    }

    /// Return empty metrics.
    fn empty_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            total_return: 0.0,
            annualized_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            num_trades: 0,
            avg_trade_return: 0.0,
            avg_holding_period: 0.0,
        }
    }

    /// Get trade history.
    pub fn get_trades(&self) -> &[Position] {
        &self.trades
    }

    /// Get equity curve.
    pub fn get_equity_curve(&self) -> &[(DateTime<Utc>, f64)] {
        &self.equity_curve
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_synthetic_candles;

    #[test]
    fn test_backtester_basic() {
        let strategy = TradingStrategy::new();
        let mut backtester = Backtester::new(100000.0, strategy);

        // Create price data
        let btc_candles = generate_synthetic_candles(100, 40000.0, "BTCUSDT");
        let eth_candles = generate_synthetic_candles(100, 2500.0, "ETHUSDT");

        let mut price_data = HashMap::new();
        price_data.insert("BTCUSDT".to_string(), btc_candles.clone());
        price_data.insert("ETHUSDT".to_string(), eth_candles);

        // Create a test event
        let events = vec![Event {
            date: btc_candles[50].timestamp,
            treated_symbol: "BTCUSDT".to_string(),
            control_symbol: "ETHUSDT".to_string(),
            did_results: DiDResults {
                did_estimate: 0.05,
                std_error: 0.01,
                t_statistic: 5.0,
                p_value: 0.001,
                ci_lower: 0.03,
                ci_upper: 0.07,
                confidence_level: 0.95,
                n_treated: 100,
                n_control: 100,
            },
        }];

        let metrics = backtester.run(&events, &price_data);

        // Should have executed at least one trade
        assert!(metrics.num_trades >= 1);
    }
}
