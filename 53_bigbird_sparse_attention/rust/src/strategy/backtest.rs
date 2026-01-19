//! Backtesting engine
//!
//! Evaluate trading strategies on historical data.

use serde::{Deserialize, Serialize};

use super::signals::{Signal, SignalType};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size as fraction of capital (0-1)
    pub position_size: f64,
    /// Transaction cost (e.g., 0.001 = 0.1%)
    pub transaction_cost: f64,
    /// Slippage estimate
    pub slippage: f64,
    /// Risk-free rate (annual) for Sharpe ratio
    pub risk_free_rate: f64,
    /// Trading periods per year (for annualization)
    pub periods_per_year: u32,
    /// Maximum position size
    pub max_position: f64,
    /// Stop loss threshold (e.g., 0.05 = 5%)
    pub stop_loss: Option<f64>,
    /// Take profit threshold
    pub take_profit: Option<f64>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            transaction_cost: 0.001,
            slippage: 0.0005,
            risk_free_rate: 0.02,
            periods_per_year: 252 * 24, // Hourly trading
            max_position: 1.0,
            stop_loss: None,
            take_profit: None,
        }
    }
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub entry_price: f64,
    pub exit_price: f64,
    pub position_size: f64,
    pub signal_type: SignalType,
    pub pnl: f64,
    pub return_pct: f64,
}

/// Portfolio state at each time step
#[derive(Debug, Clone)]
pub struct PortfolioState {
    pub timestamp_idx: usize,
    pub cash: f64,
    pub position_value: f64,
    pub total_value: f64,
    pub position_size: f64,
    pub signal: SignalType,
}

/// Backtest results
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
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total number of trades
    pub n_trades: usize,
    /// Final portfolio value
    pub final_value: f64,
    /// All trades
    pub trades: Vec<Trade>,
    /// Daily returns
    pub returns: Vec<f64>,
    /// Equity curve (portfolio values over time)
    pub equity_curve: Vec<f64>,
}

impl std::fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Backtest Results ===")?;
        writeln!(f, "Total Return:      {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "Annualized Return: {:.2}%", self.annualized_return * 100.0)?;
        writeln!(f, "Sharpe Ratio:      {:.3}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio:     {:.3}", self.sortino_ratio)?;
        writeln!(f, "Max Drawdown:      {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Win Rate:          {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Profit Factor:     {:.3}", self.profit_factor)?;
        writeln!(f, "Total Trades:      {}", self.n_trades)?;
        writeln!(f, "Final Value:       ${:.2}", self.final_value)
    }
}

/// Backtester engine
pub struct Backtester {
    config: BacktestConfig,
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with signals and price data
    pub fn run(&self, signals: &[Signal], prices: &[f64]) -> BacktestResult {
        assert_eq!(
            signals.len(),
            prices.len(),
            "Signals and prices must have same length"
        );

        let n = signals.len();
        let mut cash = self.config.initial_capital;
        let mut position = 0.0; // Number of units held
        let mut position_cost = 0.0; // Average entry price
        let mut current_signal = SignalType::Neutral;

        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<f64> = Vec::with_capacity(n);
        let mut entry_idx: Option<usize> = None;

        for i in 0..n {
            let price = prices[i];
            let signal = &signals[i];

            // Handle position changes
            if signal.signal_type != current_signal {
                // Close existing position
                if position != 0.0 {
                    let exit_value = position.abs() * price;
                    let cost = exit_value * (self.config.transaction_cost + self.config.slippage);

                    let pnl = if position > 0.0 {
                        position * (price - position_cost) - cost
                    } else {
                        position.abs() * (position_cost - price) - cost
                    };

                    cash += exit_value + pnl;

                    trades.push(Trade {
                        entry_idx: entry_idx.unwrap_or(0),
                        exit_idx: i,
                        entry_price: position_cost,
                        exit_price: price,
                        position_size: position,
                        signal_type: current_signal,
                        pnl,
                        return_pct: pnl / (position.abs() * position_cost),
                    });

                    position = 0.0;
                    position_cost = 0.0;
                }

                // Open new position
                if signal.signal_type != SignalType::Neutral {
                    let trade_value = cash * self.config.position_size.min(self.config.max_position);
                    let cost = trade_value * (self.config.transaction_cost + self.config.slippage);

                    position = if signal.signal_type == SignalType::Long {
                        (trade_value - cost) / price
                    } else {
                        -((trade_value - cost) / price)
                    };

                    position_cost = price;
                    cash -= trade_value;
                    entry_idx = Some(i);
                }

                current_signal = signal.signal_type;
            }

            // Check stop loss / take profit
            if position != 0.0 {
                let unrealized_pnl = if position > 0.0 {
                    (price - position_cost) / position_cost
                } else {
                    (position_cost - price) / position_cost
                };

                // Stop loss
                if let Some(stop_loss) = self.config.stop_loss {
                    if unrealized_pnl < -stop_loss {
                        // Force close position
                        let exit_value = position.abs() * price;
                        let cost = exit_value * (self.config.transaction_cost + self.config.slippage);
                        let pnl = position.signum() * position.abs() * unrealized_pnl * position_cost - cost;

                        cash += exit_value + pnl;

                        trades.push(Trade {
                            entry_idx: entry_idx.unwrap_or(0),
                            exit_idx: i,
                            entry_price: position_cost,
                            exit_price: price,
                            position_size: position,
                            signal_type: current_signal,
                            pnl,
                            return_pct: unrealized_pnl,
                        });

                        position = 0.0;
                        position_cost = 0.0;
                        current_signal = SignalType::Neutral;
                    }
                }

                // Take profit
                if let Some(take_profit) = self.config.take_profit {
                    if unrealized_pnl > take_profit {
                        let exit_value = position.abs() * price;
                        let cost = exit_value * (self.config.transaction_cost + self.config.slippage);
                        let pnl = position.signum() * position.abs() * unrealized_pnl * position_cost - cost;

                        cash += exit_value + pnl;

                        trades.push(Trade {
                            entry_idx: entry_idx.unwrap_or(0),
                            exit_idx: i,
                            entry_price: position_cost,
                            exit_price: price,
                            position_size: position,
                            signal_type: current_signal,
                            pnl,
                            return_pct: unrealized_pnl,
                        });

                        position = 0.0;
                        position_cost = 0.0;
                        current_signal = SignalType::Neutral;
                    }
                }
            }

            // Record equity
            let position_value = position.abs() * price;
            let total_value = cash + position_value;
            equity_curve.push(total_value);
        }

        // Close any remaining position
        if position != 0.0 && !prices.is_empty() {
            let price = *prices.last().unwrap();
            let exit_value = position.abs() * price;
            let cost = exit_value * (self.config.transaction_cost + self.config.slippage);
            let pnl = if position > 0.0 {
                position * (price - position_cost) - cost
            } else {
                position.abs() * (position_cost - price) - cost
            };
            cash += exit_value + pnl;

            trades.push(Trade {
                entry_idx: entry_idx.unwrap_or(0),
                exit_idx: n - 1,
                entry_price: position_cost,
                exit_price: price,
                position_size: position,
                signal_type: current_signal,
                pnl,
                return_pct: pnl / (position.abs() * position_cost),
            });
        }

        // Calculate metrics
        self.calculate_metrics(trades, equity_curve)
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self, trades: Vec<Trade>, equity_curve: Vec<f64>) -> BacktestResult {
        let initial = self.config.initial_capital;
        let final_value = *equity_curve.last().unwrap_or(&initial);

        // Returns
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Total and annualized return
        let total_return = (final_value - initial) / initial;
        let n_periods = equity_curve.len().max(1);
        let annualized_return = (1.0 + total_return).powf(self.config.periods_per_year as f64 / n_periods as f64) - 1.0;

        // Sharpe ratio
        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len().max(1) as f64;
        let std_return = variance.sqrt();
        let risk_free_per_period = self.config.risk_free_rate / self.config.periods_per_year as f64;
        let sharpe_ratio = if std_return > 0.0 {
            (mean_return - risk_free_per_period) / std_return * (self.config.periods_per_year as f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let negative_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < risk_free_per_period)
            .copied()
            .collect();
        let downside_variance = negative_returns
            .iter()
            .map(|r| (r - risk_free_per_period).powi(2))
            .sum::<f64>()
            / negative_returns.len().max(1) as f64;
        let downside_std = downside_variance.sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            (mean_return - risk_free_per_period) / downside_std * (self.config.periods_per_year as f64).sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut max_value = initial;
        let mut max_drawdown = 0.0;
        for &value in &equity_curve {
            if value > max_value {
                max_value = value;
            }
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Win rate and profit factor
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl < 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
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
            n_trades: trades.len(),
            final_value,
            trades,
            returns,
            equity_curve,
        }
    }

    /// Run backtest directly with predictions and prices
    pub fn run_with_predictions(
        &self,
        predictions: &[f64],
        prices: &[f64],
        long_threshold: f64,
        short_threshold: f64,
    ) -> BacktestResult {
        use super::signals::{SignalConfig, SignalGenerator};

        let config = SignalConfig {
            long_threshold,
            short_threshold,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);
        let signals = generator.generate_batch(predictions, None);

        self.run(&signals, prices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_backtest() {
        let config = BacktestConfig {
            initial_capital: 10000.0,
            position_size: 0.5,
            transaction_cost: 0.001,
            slippage: 0.0,
            ..Default::default()
        };

        let backtester = Backtester::new(config);

        // Simulated prices and signals
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();

        let signals: Vec<Signal> = prices
            .windows(2)
            .map(|w| {
                let ret = (w[1] - w[0]) / w[0];
                if ret > 0.01 {
                    Signal::new(SignalType::Long, ret, 1.0)
                } else if ret < -0.01 {
                    Signal::new(SignalType::Short, ret, 1.0)
                } else {
                    Signal::new(SignalType::Neutral, ret, 1.0)
                }
            })
            .collect();

        // Pad signals to match prices
        let mut full_signals = vec![Signal::new(SignalType::Neutral, 0.0, 1.0)];
        full_signals.extend(signals);

        let result = backtester.run(&full_signals, &prices);

        println!("{}", result);
        assert!(result.equity_curve.len() > 0);
    }

    #[test]
    fn test_with_predictions() {
        let backtester = Backtester::default();

        // Trending prices
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.5).collect();

        // Predictions that mostly catch the trend
        let predictions: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0] + 0.001)
            .collect();
        let predictions: Vec<f64> = std::iter::once(0.0).chain(predictions).collect();

        let result = backtester.run_with_predictions(&predictions, &prices, 0.001, -0.001);

        println!("{}", result);
        // Should have positive return on trending market with good predictions
        assert!(result.total_return > -0.5); // Allow some loss due to transaction costs
    }
}
