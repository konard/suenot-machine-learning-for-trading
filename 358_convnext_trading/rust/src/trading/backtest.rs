//! Backtesting engine for strategy evaluation

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::data::Candle;

use super::strategy::{Position, Strategy};
use super::Signal;

/// Backtest results metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BacktestMetrics {
    /// Total return (as decimal, e.g., 0.15 = 15%)
    pub total_return: f64,
    /// Annualized Sharpe ratio
    pub sharpe_ratio: f64,
    /// Annualized Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown (as decimal)
    pub max_drawdown: f64,
    /// Win rate (fraction of profitable trades)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average trade duration
    pub avg_trade_duration: Duration,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Average profit per winning trade
    pub avg_win: f64,
    /// Average loss per losing trade
    pub avg_loss: f64,
    /// Maximum consecutive wins
    pub max_consecutive_wins: usize,
    /// Maximum consecutive losses
    pub max_consecutive_losses: usize,
}

impl Default for BacktestMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_trade_duration: Duration::from_secs(0),
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            avg_win: 0.0,
            avg_loss: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
        }
    }
}

/// Trade record
#[derive(Clone, Debug)]
struct Trade {
    entry_time: i64,
    exit_time: i64,
    entry_price: f64,
    exit_price: f64,
    size: f64,
    is_long: bool,
    pnl: f64,
}

impl Trade {
    fn duration(&self) -> Duration {
        Duration::from_millis((self.exit_time - self.entry_time).max(0) as u64)
    }

    fn is_win(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Backtesting engine
pub struct Backtest {
    /// Trading strategy
    strategy: Strategy,
    /// Initial capital
    initial_capital: f64,
    /// Trading fee percentage
    fee_rate: f64,
    /// Slippage percentage
    slippage: f64,
}

impl Backtest {
    /// Create a new backtest
    pub fn new(strategy: Strategy, initial_capital: f64) -> Self {
        Self {
            strategy,
            initial_capital,
            fee_rate: 0.001,  // 0.1% fee
            slippage: 0.0005, // 0.05% slippage
        }
    }

    /// Create with custom fees
    pub fn with_fees(strategy: Strategy, initial_capital: f64, fee_rate: f64, slippage: f64) -> Self {
        Self {
            strategy,
            initial_capital,
            fee_rate,
            slippage,
        }
    }

    /// Run backtest
    pub fn run(&self, candles: &[Candle]) -> Result<BacktestMetrics> {
        let seq_len = self.strategy.seq_length();

        if candles.len() < seq_len + 1 {
            return Ok(BacktestMetrics::default());
        }

        let mut cash = self.initial_capital;
        let mut position = Position::Flat;
        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<f64> = Vec::new();

        // Track position entry details
        let mut entry_time: i64 = 0;
        let mut entry_price: f64 = 0.0;
        let mut position_size: f64 = 0.0;
        let mut is_long: bool = false;

        for i in seq_len..candles.len() {
            let current_candle = &candles[i];
            let current_price = current_candle.close;

            // Calculate current equity
            let equity = cash + position.unrealized_pnl(current_price);
            equity_curve.push(equity);

            // Check for exit conditions
            if position.is_open() && self.strategy.check_exit(&position, current_price) {
                // Close position
                let exit_price = self.apply_slippage(current_price, !is_long);
                let fee = position_size * exit_price * self.fee_rate;

                let pnl = if is_long {
                    (exit_price - entry_price) * position_size - fee
                } else {
                    (entry_price - exit_price) * position_size - fee
                };

                cash += position_size * exit_price - fee;
                if !is_long {
                    cash += 2.0 * (entry_price - exit_price) * position_size; // Adjust for short
                }

                trades.push(Trade {
                    entry_time,
                    exit_time: current_candle.timestamp,
                    entry_price,
                    exit_price,
                    size: position_size,
                    is_long,
                    pnl,
                });

                position = Position::Flat;
            }

            // Generate signal if flat
            if !position.is_open() {
                let history = &candles[..=i];
                let signal = self.strategy.generate_signal(history)?;

                match signal {
                    Signal::Long { confidence: _ } => {
                        let size = self.strategy.calculate_position_size(&signal, cash, current_price);
                        if size > 0.0 {
                            let actual_price = self.apply_slippage(current_price, true);
                            let fee = size * actual_price * self.fee_rate;

                            if size * actual_price + fee <= cash {
                                cash -= size * actual_price + fee;
                                position = Position::Long {
                                    entry_price: actual_price,
                                    size,
                                };
                                entry_time = current_candle.timestamp;
                                entry_price = actual_price;
                                position_size = size;
                                is_long = true;
                            }
                        }
                    }
                    Signal::Short { confidence: _ } => {
                        let size = self.strategy.calculate_position_size(&signal, cash, current_price);
                        if size > 0.0 {
                            let actual_price = self.apply_slippage(current_price, false);
                            let fee = size * actual_price * self.fee_rate;

                            if fee <= cash {
                                cash -= fee;
                                position = Position::Short {
                                    entry_price: actual_price,
                                    size,
                                };
                                entry_time = current_candle.timestamp;
                                entry_price = actual_price;
                                position_size = size;
                                is_long = false;
                            }
                        }
                    }
                    Signal::Hold => {}
                }
            }
        }

        // Close any remaining position at last price
        if position.is_open() {
            let last_price = candles.last().unwrap().close;
            let exit_price = self.apply_slippage(last_price, !is_long);
            let fee = position_size * exit_price * self.fee_rate;

            let pnl = if is_long {
                (exit_price - entry_price) * position_size - fee
            } else {
                (entry_price - exit_price) * position_size - fee
            };

            trades.push(Trade {
                entry_time,
                exit_time: candles.last().unwrap().timestamp,
                entry_price,
                exit_price,
                size: position_size,
                is_long,
                pnl,
            });
        }

        // Calculate metrics
        self.calculate_metrics(&trades, &equity_curve)
    }

    /// Apply slippage to price
    fn apply_slippage(&self, price: f64, is_buy: bool) -> f64 {
        if is_buy {
            price * (1.0 + self.slippage)
        } else {
            price * (1.0 - self.slippage)
        }
    }

    /// Calculate backtest metrics
    fn calculate_metrics(&self, trades: &[Trade], equity_curve: &[f64]) -> Result<BacktestMetrics> {
        if trades.is_empty() || equity_curve.is_empty() {
            return Ok(BacktestMetrics::default());
        }

        let final_equity = *equity_curve.last().unwrap();
        let total_return = (final_equity - self.initial_capital) / self.initial_capital;

        // Calculate returns for Sharpe/Sortino
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_return = (returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64)
            .sqrt();

        // Annualized Sharpe (assuming hourly data, ~8760 periods per year)
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (8760.0_f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio (downside deviation only)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if !downside_returns.is_empty() {
            (downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64)
                .sqrt()
        } else {
            0.0
        };

        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (8760.0_f64).sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = equity_curve[0];
        let mut max_drawdown = 0.0;
        for &equity in equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Trade statistics
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.is_win()).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| !t.is_win()).collect();

        let total_trades = trades.len();
        let n_winning = winning_trades.len();
        let n_losing = losing_trades.len();

        let win_rate = if total_trades > 0 {
            n_winning as f64 / total_trades as f64
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

        let avg_win = if n_winning > 0 {
            gross_profit / n_winning as f64
        } else {
            0.0
        };

        let avg_loss = if n_losing > 0 {
            gross_loss / n_losing as f64
        } else {
            0.0
        };

        // Average trade duration
        let total_duration: Duration = trades.iter().map(|t| t.duration()).sum();
        let avg_trade_duration = if total_trades > 0 {
            total_duration / total_trades as u32
        } else {
            Duration::from_secs(0)
        };

        // Consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) = self.calculate_streaks(trades);

        Ok(BacktestMetrics {
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            avg_trade_duration,
            total_trades,
            winning_trades: n_winning,
            losing_trades: n_losing,
            avg_win,
            avg_loss,
            max_consecutive_wins,
            max_consecutive_losses,
        })
    }

    /// Calculate consecutive win/loss streaks
    fn calculate_streaks(&self, trades: &[Trade]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.is_win() {
                current_wins += 1;
                current_losses = 0;
                if current_wins > max_wins {
                    max_wins = current_wins;
                }
            } else {
                current_losses += 1;
                current_wins = 0;
                if current_losses > max_losses {
                    max_losses = current_losses;
                }
            }
        }

        (max_wins, max_losses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convnext::{ConvNeXt, ConvNeXtConfig};

    fn create_test_candles(n: usize, trend: f64) -> Vec<Candle> {
        let mut price = 100.0;
        (0..n)
            .map(|i| {
                price += trend + (i as f64 * 0.1).sin() * 0.5;
                Candle {
                    timestamp: i as i64 * 3600000,
                    open: price - 0.5,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price,
                    volume: 1000.0,
                    turnover: price * 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_runs() {
        let config = ConvNeXtConfig::tiny();
        let model = ConvNeXt::new(config);
        let strategy = Strategy::new(model, 0.02);
        let backtest = Backtest::new(strategy, 10000.0);

        let candles = create_test_candles(500, 0.1);
        let metrics = backtest.run(&candles).unwrap();

        // Metrics should be calculated
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0);
    }

    #[test]
    fn test_metrics_calculation() {
        let trades = vec![
            Trade {
                entry_time: 0,
                exit_time: 3600000,
                entry_price: 100.0,
                exit_price: 102.0,
                size: 1.0,
                is_long: true,
                pnl: 2.0,
            },
            Trade {
                entry_time: 3600000,
                exit_time: 7200000,
                entry_price: 102.0,
                exit_price: 100.0,
                size: 1.0,
                is_long: true,
                pnl: -2.0,
            },
        ];

        let config = ConvNeXtConfig::tiny();
        let model = ConvNeXt::new(config);
        let strategy = Strategy::new(model, 0.02);
        let backtest = Backtest::new(strategy, 10000.0);

        let equity_curve = vec![10000.0, 10002.0, 10000.0];
        let metrics = backtest.calculate_metrics(&trades, &equity_curve).unwrap();

        assert_eq!(metrics.total_trades, 2);
        assert_eq!(metrics.winning_trades, 1);
        assert_eq!(metrics.losing_trades, 1);
        assert!((metrics.win_rate - 0.5).abs() < 1e-6);
    }
}
