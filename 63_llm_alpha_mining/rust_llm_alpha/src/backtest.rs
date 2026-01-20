//! Backtesting module for LLM Alpha Mining.
//!
//! Provides a framework for simulating trading strategies based on alpha factors.

use crate::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single trade in the backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub side: TradeSide,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub exit_timestamp: Option<DateTime<Utc>>,
    pub quantity: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub fees: f64,
    pub exit_reason: String,
}

/// Trade direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Long,
    Short,
}

impl Trade {
    /// Create a new trade.
    pub fn new(timestamp: DateTime<Utc>, side: TradeSide, entry_price: f64, quantity: f64) -> Self {
        Self {
            timestamp,
            side,
            entry_price,
            exit_price: None,
            exit_timestamp: None,
            quantity,
            pnl: 0.0,
            pnl_pct: 0.0,
            fees: 0.0,
            exit_reason: String::new(),
        }
    }

    /// Close the trade.
    pub fn close(&mut self, exit_price: f64, exit_timestamp: DateTime<Utc>, fees: f64, reason: &str) {
        self.exit_price = Some(exit_price);
        self.exit_timestamp = Some(exit_timestamp);
        self.fees = fees;
        self.exit_reason = reason.to_string();

        match self.side {
            TradeSide::Long => {
                self.pnl = (exit_price - self.entry_price) * self.quantity - fees;
                self.pnl_pct = (exit_price / self.entry_price - 1.0) * 100.0;
            }
            TradeSide::Short => {
                self.pnl = (self.entry_price - exit_price) * self.quantity - fees;
                self.pnl_pct = (1.0 - exit_price / self.entry_price) * 100.0;
            }
        }
    }

    /// Check if trade is closed.
    pub fn is_closed(&self) -> bool {
        self.exit_price.is_some()
    }
}

/// Backtest results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
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
    /// Total number of trades
    pub total_trades: usize,
    /// Average trade PnL
    pub avg_trade_pnl: f64,
    /// Average trade duration in periods
    pub avg_trade_duration: f64,
    /// Maximum consecutive losses
    pub max_consecutive_losses: usize,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// List of trades
    pub trades: Vec<Trade>,
    /// Daily returns
    pub daily_returns: Vec<f64>,
}

impl BacktestResult {
    /// Generate a text summary.
    pub fn summary(&self) -> String {
        format!(
            r#"
==================================================
BACKTEST RESULTS
==================================================
Total Return:         {:.2}%
Annualized Return:    {:.2}%
Sharpe Ratio:         {:.2}
Sortino Ratio:        {:.2}
Max Drawdown:         {:.2}%
Calmar Ratio:         {:.2}
--------------------------------------------------
Total Trades:         {}
Win Rate:             {:.2}%
Profit Factor:        {:.2}
Avg Trade PnL:        {:.4}
Avg Trade Duration:   {:.1} periods
Max Consec. Losses:   {}
==================================================
"#,
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.total_trades,
            self.win_rate * 100.0,
            self.profit_factor,
            self.avg_trade_pnl,
            self.avg_trade_duration,
            self.max_consecutive_losses,
        )
    }
}

/// Backtester configuration.
#[derive(Debug, Clone)]
pub struct BacktesterConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size (fraction of capital)
    pub position_size: f64,
    /// Commission rate per trade
    pub commission: f64,
    /// Slippage rate per trade
    pub slippage: f64,
    /// Trading periods per year
    pub periods_per_year: usize,
    /// Long threshold for signals
    pub long_threshold: f64,
    /// Short threshold for signals (None = long only)
    pub short_threshold: Option<f64>,
    /// Maximum holding periods
    pub max_holding_periods: usize,
    /// Stop loss percentage
    pub stop_loss: Option<f64>,
    /// Take profit percentage
    pub take_profit: Option<f64>,
}

impl Default for BacktesterConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.5,
            commission: 0.001,
            slippage: 0.0005,
            periods_per_year: 252,
            long_threshold: 0.0,
            short_threshold: None,
            max_holding_periods: 10,
            stop_loss: None,
            take_profit: None,
        }
    }
}

/// Backtesting engine.
pub struct Backtester {
    config: BacktesterConfig,
}

impl Backtester {
    /// Create a new backtester with initial capital.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            config: BacktesterConfig {
                initial_capital,
                ..Default::default()
            },
        }
    }

    /// Create with full configuration.
    pub fn with_config(config: BacktesterConfig) -> Self {
        Self { config }
    }

    /// Set position size.
    pub fn position_size(mut self, size: f64) -> Self {
        self.config.position_size = size;
        self
    }

    /// Set commission rate.
    pub fn commission(mut self, rate: f64) -> Self {
        self.config.commission = rate;
        self
    }

    /// Set slippage rate.
    pub fn slippage(mut self, rate: f64) -> Self {
        self.config.slippage = rate;
        self
    }

    /// Set long threshold.
    pub fn long_threshold(mut self, threshold: f64) -> Self {
        self.config.long_threshold = threshold;
        self
    }

    /// Set short threshold.
    pub fn short_threshold(mut self, threshold: Option<f64>) -> Self {
        self.config.short_threshold = threshold;
        self
    }

    /// Set max holding periods.
    pub fn max_holding_periods(mut self, periods: usize) -> Self {
        self.config.max_holding_periods = periods;
        self
    }

    /// Set stop loss.
    pub fn stop_loss(mut self, pct: Option<f64>) -> Self {
        self.config.stop_loss = pct;
        self
    }

    /// Set take profit.
    pub fn take_profit(mut self, pct: Option<f64>) -> Self {
        self.config.take_profit = pct;
        self
    }

    /// Run backtest on signals and prices.
    pub fn run(
        &self,
        signals: &[f64],
        prices: &[f64],
        timestamps: &[DateTime<Utc>],
    ) -> Result<BacktestResult> {
        let n = signals.len().min(prices.len()).min(timestamps.len());

        if n < 10 {
            return Err(crate::error::Error::InsufficientData(
                "Need at least 10 data points".to_string(),
            ));
        }

        let mut trades: Vec<Trade> = Vec::new();
        let mut equity = vec![self.config.initial_capital];
        let mut current_position: Option<Trade> = None;
        let mut position_start_idx = 0;

        for i in 0..n {
            let signal = signals[i];
            let price = prices[i];
            let timestamp = timestamps[i];

            // Skip if signal is NaN
            if signal.is_nan() {
                equity.push(*equity.last().unwrap());
                continue;
            }

            // Check exit conditions for existing position
            if let Some(ref mut position) = current_position {
                let holding_periods = i - position_start_idx;
                let mut should_exit = false;
                let mut exit_reason = String::new();

                // Max holding period
                if holding_periods >= self.config.max_holding_periods {
                    should_exit = true;
                    exit_reason = "max_holding".to_string();
                }

                // Stop loss
                if let Some(stop_loss) = self.config.stop_loss {
                    let loss_pct = match position.side {
                        TradeSide::Long => price / position.entry_price - 1.0,
                        TradeSide::Short => 1.0 - price / position.entry_price,
                    };
                    if loss_pct <= -stop_loss {
                        should_exit = true;
                        exit_reason = "stop_loss".to_string();
                    }
                }

                // Take profit
                if let Some(take_profit) = self.config.take_profit {
                    let profit_pct = match position.side {
                        TradeSide::Long => price / position.entry_price - 1.0,
                        TradeSide::Short => 1.0 - price / position.entry_price,
                    };
                    if profit_pct >= take_profit {
                        should_exit = true;
                        exit_reason = "take_profit".to_string();
                    }
                }

                // Signal reversal
                if position.side == TradeSide::Long {
                    if let Some(short_thresh) = self.config.short_threshold {
                        if signal <= short_thresh {
                            should_exit = true;
                            exit_reason = "signal_reversal".to_string();
                        }
                    }
                } else if signal >= self.config.long_threshold {
                    should_exit = true;
                    exit_reason = "signal_reversal".to_string();
                }

                if should_exit {
                    let exit_price = match position.side {
                        TradeSide::Long => price * (1.0 - self.config.slippage),
                        TradeSide::Short => price * (1.0 + self.config.slippage),
                    };
                    let fees = exit_price * position.quantity * self.config.commission;

                    position.close(exit_price, timestamp, fees, &exit_reason);
                    trades.push(position.clone());
                    current_position = None;
                }
            }

            // Check entry conditions
            if current_position.is_none() {
                if signal >= self.config.long_threshold {
                    // Enter long
                    let entry_price = price * (1.0 + self.config.slippage);
                    let quantity = (equity.last().unwrap() * self.config.position_size) / entry_price;
                    let fees = entry_price * quantity * self.config.commission;

                    let mut trade = Trade::new(timestamp, TradeSide::Long, entry_price, quantity);
                    trade.fees = fees;
                    current_position = Some(trade);
                    position_start_idx = i;
                } else if let Some(short_thresh) = self.config.short_threshold {
                    if signal <= short_thresh {
                        // Enter short
                        let entry_price = price * (1.0 - self.config.slippage);
                        let quantity = (equity.last().unwrap() * self.config.position_size) / entry_price;
                        let fees = entry_price * quantity * self.config.commission;

                        let mut trade = Trade::new(timestamp, TradeSide::Short, entry_price, quantity);
                        trade.fees = fees;
                        current_position = Some(trade);
                        position_start_idx = i;
                    }
                }
            }

            // Update equity
            let realized_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
            let unrealized_pnl = if let Some(ref position) = current_position {
                match position.side {
                    TradeSide::Long => (price - position.entry_price) * position.quantity,
                    TradeSide::Short => (position.entry_price - price) * position.quantity,
                }
            } else {
                0.0
            };

            equity.push(self.config.initial_capital + realized_pnl + unrealized_pnl);
        }

        // Close any remaining position
        if let Some(mut position) = current_position {
            let final_price = prices[n - 1];
            let exit_price = match position.side {
                TradeSide::Long => final_price * (1.0 - self.config.slippage),
                TradeSide::Short => final_price * (1.0 + self.config.slippage),
            };
            let fees = exit_price * position.quantity * self.config.commission;

            position.close(exit_price, timestamps[n - 1], fees, "end_of_data");
            trades.push(position);
        }

        // Calculate metrics
        self.calculate_metrics(trades, equity)
    }

    fn calculate_metrics(&self, trades: Vec<Trade>, equity: Vec<f64>) -> Result<BacktestResult> {
        let n = equity.len();

        // Returns
        let total_return = (equity[n - 1] / self.config.initial_capital) - 1.0;
        let annualized_return = (1.0 + total_return).powf(self.config.periods_per_year as f64 / n as f64) - 1.0;

        // Daily returns
        let daily_returns: Vec<f64> = equity
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .filter(|r| !r.is_nan())
            .collect();

        // Sharpe ratio
        let mean_return: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let std_return: f64 = {
            let variance: f64 = daily_returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / daily_returns.len() as f64;
            variance.sqrt()
        };

        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (self.config.periods_per_year as f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = daily_returns.iter().filter(|r| **r < 0.0).cloned().collect();
        let downside_std = if !downside_returns.is_empty() {
            let variance: f64 = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64;
            variance.sqrt()
        } else {
            std_return
        };

        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (self.config.periods_per_year as f64).sqrt()
        } else {
            sharpe_ratio
        };

        // Max drawdown
        let mut peak = self.config.initial_capital;
        let mut max_drawdown = 0.0;
        for &eq in &equity {
            if eq > peak {
                peak = eq;
            }
            let dd = (eq - peak) / peak;
            if dd < max_drawdown {
                max_drawdown = dd;
            }
        }

        // Calmar ratio
        let calmar_ratio = if max_drawdown != 0.0 {
            annualized_return / max_drawdown.abs()
        } else {
            0.0
        };

        // Trade statistics
        let total_trades = trades.len();
        let (win_rate, profit_factor, avg_trade_pnl, avg_trade_duration, max_consecutive_losses) =
            if total_trades > 0 {
                let winning: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
                let losing: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

                let win_rate = winning.len() as f64 / total_trades as f64;

                let gross_profit: f64 = winning.iter().map(|t| t.pnl).sum();
                let gross_loss: f64 = losing.iter().map(|t| t.pnl.abs()).sum();
                let profit_factor = if gross_loss > 0.0 {
                    gross_profit / gross_loss
                } else {
                    f64::INFINITY
                };

                let avg_pnl: f64 = trades.iter().map(|t| t.pnl).sum::<f64>() / total_trades as f64;

                // Average duration (simplified)
                let avg_duration = self.config.max_holding_periods as f64 / 2.0;

                // Max consecutive losses
                let mut max_consec = 0;
                let mut current_consec = 0;
                for trade in &trades {
                    if trade.pnl <= 0.0 {
                        current_consec += 1;
                        if current_consec > max_consec {
                            max_consec = current_consec;
                        }
                    } else {
                        current_consec = 0;
                    }
                }

                (win_rate, profit_factor, avg_pnl, avg_duration, max_consec)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0)
            };

        Ok(BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            total_trades,
            avg_trade_pnl,
            avg_trade_duration,
            max_consecutive_losses,
            equity_curve: equity,
            trades,
            daily_returns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_synthetic_data;
    use crate::alpha::{AlphaEvaluator, predefined_factors};

    #[test]
    fn test_backtester_basic() {
        let data = generate_synthetic_data("TEST", 200, 42);
        let evaluator = AlphaEvaluator::new(&data);

        let factors = predefined_factors();
        let factor_values = evaluator.evaluate(&factors[0]).unwrap();

        // Normalize to z-score
        let mean: f64 = factor_values.iter().filter(|v| !v.is_nan()).sum::<f64>()
            / factor_values.iter().filter(|v| !v.is_nan()).count() as f64;
        let variance: f64 = factor_values
            .iter()
            .filter(|v| !v.is_nan())
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / factor_values.iter().filter(|v| !v.is_nan()).count() as f64;
        let std = variance.sqrt();

        let signals: Vec<f64> = factor_values
            .iter()
            .map(|v| if v.is_nan() { f64::NAN } else { (v - mean) / std })
            .collect();

        let backtester = Backtester::new(100_000.0)
            .position_size(0.5)
            .commission(0.001)
            .long_threshold(0.5)
            .short_threshold(Some(-0.5))
            .max_holding_periods(10);

        let result = backtester.run(&signals, &data.close_prices(), &data.timestamps()).unwrap();

        assert!(result.total_trades > 0);
        println!("{}", result.summary());
    }

    #[test]
    fn test_trade() {
        let now = Utc::now();
        let mut trade = Trade::new(now, TradeSide::Long, 100.0, 10.0);

        assert!(!trade.is_closed());

        trade.close(110.0, now, 1.0, "take_profit");

        assert!(trade.is_closed());
        assert!((trade.pnl - 99.0).abs() < 0.01); // (110-100)*10 - 1 = 99
        assert!((trade.pnl_pct - 10.0).abs() < 0.01); // 10% gain
    }
}
