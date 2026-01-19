//! Backtesting engine for FNet trading strategies.

use super::signals::{Signal, SignalGenerator, SignalGeneratorConfig, TradingSignal};

/// Trade record.
#[derive(Debug, Clone)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: i64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size (positive for long, negative for short)
    pub position_size: f64,
    /// Profit/Loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Exit reason
    pub exit_reason: ExitReason,
}

/// Reason for exiting a trade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExitReason {
    /// Take profit hit
    TakeProfit,
    /// Stop loss hit
    StopLoss,
    /// Signal reversed
    SignalReverse,
    /// Max holding period reached
    MaxHoldingPeriod,
    /// End of data
    EndOfData,
}

/// Trade metrics.
#[derive(Debug, Clone)]
pub struct TradeMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return (assuming hourly data)
    pub annual_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total trades
    pub total_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade
    pub avg_loss: f64,
    /// Final equity
    pub final_equity: f64,
}

impl TradeMetrics {
    /// Print metrics summary.
    pub fn summary(&self) -> String {
        format!(
            "Total Return: {:.2}%\n\
             Annual Return: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             Sortino Ratio: {:.2}\n\
             Max Drawdown: {:.2}%\n\
             Win Rate: {:.2}%\n\
             Profit Factor: {:.2}\n\
             Total Trades: {}\n\
             Avg Trade Return: {:.4}%\n\
             Final Equity: ${:.2}",
            self.total_return * 100.0,
            self.annual_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.total_trades,
            self.avg_trade_return * 100.0,
            self.final_equity
        )
    }
}

/// Backtest result.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Trading metrics
    pub metrics: TradeMetrics,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// All trades
    pub trades: Vec<Trade>,
    /// Returns series
    pub returns: Vec<f64>,
    /// Signals generated
    pub signals: Vec<TradingSignal>,
}

/// Backtester configuration.
#[derive(Debug, Clone)]
pub struct BacktesterConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (as fraction)
    pub transaction_cost: f64,
    /// Slippage (as fraction)
    pub slippage: f64,
}

impl Default for BacktesterConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
        }
    }
}

/// Backtesting engine.
pub struct Backtester {
    config: BacktesterConfig,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(config: BacktesterConfig) -> Self {
        Self { config }
    }

    /// Run backtest with predictions.
    pub fn run(
        &self,
        predictions: &[f64],
        prices: &[f64],
        timestamps: &[i64],
        signal_config: SignalGeneratorConfig,
    ) -> BacktestResult {
        let mut signal_generator = SignalGenerator::new(signal_config.clone());

        // Generate all signals
        let mut signals = Vec::with_capacity(predictions.len());
        for i in 0..predictions.len() {
            let signal = signal_generator.generate_signal(predictions[i], timestamps[i], prices[i]);
            signals.push(signal);
        }

        // Run simulation
        self.simulate(&signals, prices, timestamps, &signal_config)
    }

    /// Run simulation with pre-generated signals.
    fn simulate(
        &self,
        signals: &[TradingSignal],
        prices: &[f64],
        timestamps: &[i64],
        config: &SignalGeneratorConfig,
    ) -> BacktestResult {
        let n = signals.len();
        let mut equity = self.config.initial_capital;
        let mut equity_curve = vec![equity];
        let mut trades = Vec::new();
        let mut returns = Vec::new();

        // Position state
        let mut position: Option<(usize, f64, f64)> = None; // (entry_idx, entry_price, size)
        let mut holding_periods = 0_usize;

        for i in 0..n {
            let current_price = prices[i];
            let signal = &signals[i];

            // Check if we need to exit current position
            if let Some((entry_idx, entry_price, position_size)) = position {
                holding_periods += 1;

                let price_change = (current_price - entry_price) / entry_price;
                let position_return = price_change * position_size.signum();

                let mut should_exit = false;
                let mut exit_reason = ExitReason::EndOfData;

                // Check exit conditions
                if position_return >= config.take_profit {
                    should_exit = true;
                    exit_reason = ExitReason::TakeProfit;
                } else if position_return <= -config.stop_loss {
                    should_exit = true;
                    exit_reason = ExitReason::StopLoss;
                } else if holding_periods >= config.max_holding_period {
                    should_exit = true;
                    exit_reason = ExitReason::MaxHoldingPeriod;
                } else if (position_size > 0.0 && signal.signal == Signal::Sell)
                    || (position_size < 0.0 && signal.signal == Signal::Buy)
                {
                    should_exit = true;
                    exit_reason = ExitReason::SignalReverse;
                }

                if should_exit || i == n - 1 {
                    // Calculate trade result
                    // Slippage direction depends on position: long exits by selling (negative slippage),
                    // short exits by buying (positive slippage)
                    let slippage_sign = if position_size > 0.0 { -1.0 } else { 1.0 };
                    let exit_price = current_price * (1.0 + slippage_sign * self.config.slippage);
                    let gross_pnl = (exit_price - entry_price) * position_size;
                    let costs = entry_price.abs() * position_size.abs() * self.config.transaction_cost * 2.0;
                    let net_pnl = gross_pnl - costs;
                    let return_pct = net_pnl / (entry_price * position_size.abs());

                    equity += net_pnl;
                    returns.push(return_pct);

                    trades.push(Trade {
                        entry_time: timestamps[entry_idx],
                        exit_time: timestamps[i],
                        entry_price,
                        exit_price,
                        position_size,
                        pnl: net_pnl,
                        return_pct,
                        exit_reason: if i == n - 1 && !should_exit {
                            ExitReason::EndOfData
                        } else {
                            exit_reason
                        },
                    });

                    position = None;
                    holding_periods = 0;
                }
            }

            // Check if we should enter new position
            if position.is_none() && signal.is_actionable() {
                // Slippage direction depends on signal: buy pays higher (positive slippage),
                // sell pays lower (negative slippage)
                let slippage_sign = if signal.signal == Signal::Buy { 1.0 } else { -1.0 };
                let entry_price = current_price * (1.0 + slippage_sign * self.config.slippage);
                let position_size = if signal.signal == Signal::Buy {
                    config.position_size
                } else {
                    -config.position_size
                };

                position = Some((i, entry_price, position_size));
            }

            equity_curve.push(equity);
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&equity_curve, &trades, &returns);

        BacktestResult {
            metrics,
            equity_curve,
            trades,
            returns,
            signals: signals.to_vec(),
        }
    }

    /// Calculate trading metrics.
    fn calculate_metrics(
        &self,
        equity_curve: &[f64],
        trades: &[Trade],
        returns: &[f64],
    ) -> TradeMetrics {
        let initial = self.config.initial_capital;
        let final_equity = *equity_curve.last().unwrap_or(&initial);
        let total_return = (final_equity - initial) / initial;

        // Annualized return (assuming hourly data)
        let hours = equity_curve.len() as f64;
        let years = hours / (24.0 * 365.0);
        let annual_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Calculate Sharpe ratio
        let sharpe_ratio = if !returns.is_empty() {
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 =
                returns.iter().map(|&r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
            let std_return = variance.sqrt();

            if std_return > 0.0 {
                // Annualize: hourly data -> yearly
                let annual_factor = (24.0 * 365.0_f64).sqrt();
                mean_return / std_return * annual_factor
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calculate Sortino ratio (downside deviation only)
        let sortino_ratio = if !returns.is_empty() {
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

            if !downside_returns.is_empty() {
                let downside_variance: f64 =
                    downside_returns.iter().map(|&r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
                let downside_std = downside_variance.sqrt();

                if downside_std > 0.0 {
                    let annual_factor = (24.0 * 365.0_f64).sqrt();
                    mean_return / downside_std * annual_factor
                } else {
                    f64::INFINITY
                }
            } else {
                f64::INFINITY
            }
        } else {
            0.0
        };

        // Calculate max drawdown
        let max_drawdown = {
            let mut peak = equity_curve[0];
            let mut max_dd = 0.0;

            for &equity in equity_curve {
                if equity > peak {
                    peak = equity;
                }
                let dd = (peak - equity) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
            max_dd
        };

        // Trade statistics
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

        let avg_trade_return = if !trades.is_empty() {
            trades.iter().map(|t| t.return_pct).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };

        let avg_win = if !winning_trades.is_empty() {
            winning_trades.iter().map(|t| t.return_pct).sum::<f64>() / winning_trades.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing_trades.is_empty() {
            losing_trades.iter().map(|t| t.return_pct).sum::<f64>() / losing_trades.len() as f64
        } else {
            0.0
        };

        TradeMetrics {
            total_return,
            annual_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades: trades.len(),
            avg_trade_return,
            avg_win,
            avg_loss,
            final_equity,
        }
    }

    /// Get backtester configuration.
    pub fn config(&self) -> &BacktesterConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_simulation() {
        // Create simple price series with trend
        let n = 100;
        let mut prices = Vec::with_capacity(n);
        let mut price = 50000.0;

        for i in 0..n {
            price *= 1.0 + 0.001 * (i as f64 * 0.1).sin();
            prices.push(price);
        }

        let timestamps: Vec<i64> = (0..n).map(|i| 1700000000 + i as i64 * 3600).collect();

        // Create predictions (perfect foresight for testing)
        let predictions: Vec<f64> = (0..n - 1)
            .map(|i| (prices[i + 1] - prices[i]) / prices[i])
            .chain(std::iter::once(0.0))
            .collect();

        let backtester = Backtester::new(BacktesterConfig::default());
        let signal_config = SignalGeneratorConfig {
            threshold: 0.0001,
            confidence_threshold: 0.0,
            ..Default::default()
        };

        let result = backtester.run(&predictions, &prices, &timestamps, signal_config);

        assert!(!result.trades.is_empty());
        assert_eq!(result.equity_curve.len(), n + 1);
    }

    #[test]
    fn test_metrics_calculation() {
        let backtester = Backtester::new(BacktesterConfig {
            initial_capital: 100_000.0,
            ..Default::default()
        });

        // Simple equity curve
        let equity_curve = vec![100_000.0, 102_000.0, 101_000.0, 105_000.0, 103_000.0];
        let trades = vec![
            Trade {
                entry_time: 0,
                exit_time: 1,
                entry_price: 100.0,
                exit_price: 102.0,
                position_size: 1.0,
                pnl: 2000.0,
                return_pct: 0.02,
                exit_reason: ExitReason::TakeProfit,
            },
            Trade {
                entry_time: 2,
                exit_time: 3,
                entry_price: 101.0,
                exit_price: 100.0,
                position_size: 1.0,
                pnl: -1000.0,
                return_pct: -0.01,
                exit_reason: ExitReason::StopLoss,
            },
        ];
        let returns = vec![0.02, -0.01];

        let metrics = backtester.calculate_metrics(&equity_curve, &trades, &returns);

        assert_eq!(metrics.total_trades, 2);
        assert_eq!(metrics.win_rate, 0.5);
        assert!(metrics.profit_factor > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let backtester = Backtester::new(BacktesterConfig::default());

        // Equity curve with 10% drawdown
        let equity_curve = vec![100_000.0, 110_000.0, 99_000.0, 105_000.0];

        let metrics = backtester.calculate_metrics(&equity_curve, &[], &[]);

        // Max drawdown from 110k to 99k = 10%
        assert!((metrics.max_drawdown - 0.1).abs() < 0.001);
    }
}
