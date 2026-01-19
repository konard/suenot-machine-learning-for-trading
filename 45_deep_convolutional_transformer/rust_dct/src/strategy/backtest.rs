//! Backtesting engine

use super::signals::{Signal, SignalGenerator, TradingSignal};
use crate::model::DCTModel;
use ndarray::Array3;

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size as fraction of capital
    pub position_size: f64,
    /// Transaction cost per trade
    pub transaction_cost: f64,
    /// Stop loss percentage
    pub stop_loss: Option<f64>,
    /// Take profit percentage
    pub take_profit: Option<f64>,
    /// Confidence threshold for signals
    pub confidence_threshold: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            position_size: 0.1,
            transaction_cost: 0.001,
            stop_loss: Some(0.02),
            take_profit: Some(0.05),
            confidence_threshold: 0.5,
        }
    }
}

/// Individual trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub entry_price: f64,
    pub exit_price: f64,
    pub position: f64,
    pub pnl: f64,
    pub return_pct: f64,
    pub signal: Signal,
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Final portfolio value
    pub final_value: f64,
    /// Total return percentage
    pub total_return: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Win rate
    pub win_rate: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// List of all trades
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<f64>,
}

impl BacktestResult {
    /// Print a summary report
    pub fn print_report(&self) {
        println!("=== Backtest Results ===");
        println!("Final Value: ${:.2}", self.final_value);
        println!("Total Return: {:.2}%", self.total_return * 100.0);
        println!("Number of Trades: {}", self.num_trades);
        println!(
            "Win Rate: {:.2}% ({}/{})",
            self.win_rate * 100.0,
            self.winning_trades,
            self.num_trades
        );
        println!("Max Drawdown: {:.2}%", self.max_drawdown * 100.0);
        println!("Sharpe Ratio: {:.3}", self.sharpe_ratio);
    }
}

/// Backtesting engine
pub struct Backtester {
    config: BacktestConfig,
    signal_generator: SignalGenerator,
}

impl Backtester {
    /// Create new backtester
    pub fn new(config: BacktestConfig) -> Self {
        let signal_generator = SignalGenerator::new(config.confidence_threshold);
        Self {
            config,
            signal_generator,
        }
    }

    /// Run backtest
    pub fn run(
        &self,
        model: &DCTModel,
        features: &Array3<f64>,
        prices: &[f64],
    ) -> BacktestResult {
        let n = features.dim().0;
        assert_eq!(n, prices.len(), "Features and prices must have same length");

        // Generate all predictions
        let predictions = model.predict(features);
        let signals = self.signal_generator.generate(&predictions);

        // Run simulation
        let mut capital = self.config.initial_capital;
        let mut position = 0.0; // Shares held
        let mut entry_price = 0.0;
        let mut entry_idx = 0;
        let mut current_signal = Signal::Hold;

        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<f64> = Vec::with_capacity(n);
        let mut max_equity = capital;
        let mut max_drawdown = 0.0;
        let mut daily_returns: Vec<f64> = Vec::new();
        let mut prev_equity = capital;

        for i in 0..n {
            let price = prices[i];
            let signal = &signals[i];

            // Calculate current equity
            let equity = if position != 0.0 {
                capital + position * price
            } else {
                capital
            };
            equity_curve.push(equity);

            // Track drawdown
            if equity > max_equity {
                max_equity = equity;
            }
            let drawdown = (max_equity - equity) / max_equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }

            // Track daily returns
            if i > 0 {
                daily_returns.push((equity - prev_equity) / prev_equity);
            }
            prev_equity = equity;

            // Check stop loss / take profit if in position
            if position != 0.0 {
                let pnl_pct = if position > 0.0 {
                    (price - entry_price) / entry_price
                } else {
                    (entry_price - price) / entry_price
                };

                let should_exit = self
                    .config
                    .stop_loss
                    .map(|sl| pnl_pct <= -sl)
                    .unwrap_or(false)
                    || self
                        .config
                        .take_profit
                        .map(|tp| pnl_pct >= tp)
                        .unwrap_or(false);

                if should_exit || (signal.signal != current_signal && signal.is_actionable()) {
                    // Exit position
                    let exit_value = position.abs() * price;
                    let cost = exit_value * self.config.transaction_cost;
                    let pnl = if position > 0.0 {
                        position * (price - entry_price) - cost
                    } else {
                        -position * (entry_price - price) - cost
                    };

                    capital += position.abs() * price - cost;

                    trades.push(Trade {
                        entry_idx,
                        exit_idx: i,
                        entry_price,
                        exit_price: price,
                        position,
                        pnl,
                        return_pct: pnl / (position.abs() * entry_price),
                        signal: current_signal,
                    });

                    position = 0.0;
                    current_signal = Signal::Hold;
                }
            }

            // Enter new position if signal changes
            if position == 0.0 && signal.is_actionable() {
                let position_value = capital * self.config.position_size;
                let cost = position_value * self.config.transaction_cost;

                match signal.signal {
                    Signal::Long => {
                        position = (position_value - cost) / price;
                        capital -= position_value;
                    }
                    Signal::Short => {
                        position = -((position_value - cost) / price);
                        // Capital is tied up for short positions same as long
                        capital -= position_value;
                    }
                    Signal::Hold => {}
                }

                if position != 0.0 {
                    entry_price = price;
                    entry_idx = i;
                    current_signal = signal.signal;
                }
            }
        }

        // Close any remaining position
        if position != 0.0 {
            let price = prices[n - 1];
            let exit_value = position.abs() * price;
            let cost = exit_value * self.config.transaction_cost;
            let pnl = if position > 0.0 {
                position * (price - entry_price) - cost
            } else {
                -position * (entry_price - price) - cost
            };

            capital += position.abs() * price - cost;

            trades.push(Trade {
                entry_idx,
                exit_idx: n - 1,
                entry_price,
                exit_price: price,
                position,
                pnl,
                return_pct: pnl / (position.abs() * entry_price),
                signal: current_signal,
            });
        }

        // Calculate statistics
        let final_value = capital;
        let total_return = (final_value - self.config.initial_capital) / self.config.initial_capital;
        let num_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if num_trades > 0 {
            winning_trades as f64 / num_trades as f64
        } else {
            0.0
        };

        // Calculate Sharpe ratio (annualized, assuming daily data)
        let sharpe_ratio = if daily_returns.len() > 1 {
            let mean_return: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let var: f64 = daily_returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (daily_returns.len() - 1) as f64;
            let std = var.sqrt();

            if std > 0.0 {
                (mean_return / std) * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        BacktestResult {
            final_value,
            total_return,
            num_trades,
            winning_trades,
            win_rate,
            max_drawdown,
            sharpe_ratio,
            trades,
            equity_curve,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DCTConfig;

    #[test]
    fn test_backtester() {
        let config = BacktestConfig::default();
        let backtester = Backtester::new(config);

        let model_config = DCTConfig::default();
        let model = DCTModel::new(model_config);

        // Create synthetic data
        let n = 50;
        let features = Array3::from_shape_fn((n, 30, 13), |_| 0.1);
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();

        let result = backtester.run(&model, &features, &prices);

        assert!(result.final_value > 0.0);
        assert!(result.equity_curve.len() == n);
    }
}
