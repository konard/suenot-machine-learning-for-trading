//! # Backtesting
//!
//! Backtesting framework for sentiment-based trading strategies.

use crate::api::Candle;
use crate::strategy::signals::{GeneratedSignal, TradingSignal};
use serde::{Deserialize, Serialize};

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Trading fee percentage
    pub fee_pct: f64,
    /// Slippage percentage
    pub slippage_pct: f64,
    /// Maximum position size (fraction of capital)
    pub max_position_size: f64,
    /// Enable stop loss
    pub use_stop_loss: bool,
    /// Enable take profit
    pub use_take_profit: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            fee_pct: 0.001, // 0.1%
            slippage_pct: 0.0005, // 0.05%
            max_position_size: 0.5,
            use_stop_loss: true,
            use_take_profit: true,
        }
    }
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: i64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size (in quote currency)
    pub position_size: f64,
    /// Direction (1 = long, -1 = short)
    pub direction: i32,
    /// Profit/loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Exit reason
    pub exit_reason: String,
}

/// Backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return percentage
    pub total_return_pct: f64,
    /// Annualized return percentage
    pub annualized_return_pct: f64,
    /// Sharpe ratio (assuming 0% risk-free rate)
    pub sharpe_ratio: f64,
    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade
    pub avg_loss: f64,
    /// Final capital
    pub final_capital: f64,
    /// All trades
    pub trades: Vec<Trade>,
    /// Equity curve (timestamps, values)
    pub equity_curve: Vec<(i64, f64)>,
}

/// Backtester for sentiment strategies
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

    /// Run backtest with candles and signals
    ///
    /// # Arguments
    /// * `candles` - Price data (sorted by timestamp)
    /// * `signals` - Trading signals (one per candle)
    pub fn run(&self, candles: &[Candle], signals: &[GeneratedSignal]) -> BacktestResult {
        assert_eq!(
            candles.len(),
            signals.len(),
            "Candles and signals must have same length"
        );

        if candles.is_empty() {
            return self.empty_result();
        }

        let mut capital = self.config.initial_capital;
        let mut position: Option<Position> = None;
        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<(i64, f64)> = Vec::new();
        let mut peak_equity = capital;
        let mut max_drawdown = 0.0;

        for (_i, (candle, signal)) in candles.iter().zip(signals.iter()).enumerate() {
            // Record equity
            let current_equity = if let Some(ref pos) = position {
                let unrealized_pnl = pos.calculate_pnl(candle.close);
                capital + unrealized_pnl
            } else {
                capital
            };
            equity_curve.push((candle.timestamp, current_equity));

            // Update peak and drawdown
            if current_equity > peak_equity {
                peak_equity = current_equity;
            }
            let drawdown = (peak_equity - current_equity) / peak_equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }

            // Check existing position for exit conditions
            if let Some(ref pos) = position {
                let mut should_exit = false;
                let mut exit_reason = String::new();

                // Check stop loss
                if self.config.use_stop_loss {
                    let stop_price = if pos.direction > 0 {
                        pos.entry_price * (1.0 - signal.stop_loss)
                    } else {
                        pos.entry_price * (1.0 + signal.stop_loss)
                    };

                    if (pos.direction > 0 && candle.low <= stop_price)
                        || (pos.direction < 0 && candle.high >= stop_price)
                    {
                        should_exit = true;
                        exit_reason = "Stop Loss".to_string();
                    }
                }

                // Check take profit
                if !should_exit && self.config.use_take_profit {
                    let tp_price = if pos.direction > 0 {
                        pos.entry_price * (1.0 + signal.take_profit)
                    } else {
                        pos.entry_price * (1.0 - signal.take_profit)
                    };

                    if (pos.direction > 0 && candle.high >= tp_price)
                        || (pos.direction < 0 && candle.low <= tp_price)
                    {
                        should_exit = true;
                        exit_reason = "Take Profit".to_string();
                    }
                }

                // Check for signal reversal
                if !should_exit {
                    let signal_direction = signal.signal.direction();
                    if signal_direction != 0 && signal_direction != pos.direction {
                        should_exit = true;
                        exit_reason = "Signal Reversal".to_string();
                    }
                }

                // Exit position
                if should_exit {
                    let exit_price = self.apply_slippage(candle.close, -pos.direction);
                    let pnl = pos.calculate_pnl(exit_price);
                    let fee = pos.size.abs() * self.config.fee_pct;
                    let net_pnl = pnl - fee;

                    trades.push(Trade {
                        entry_time: pos.entry_time,
                        exit_time: candle.timestamp,
                        entry_price: pos.entry_price,
                        exit_price,
                        position_size: pos.size,
                        direction: pos.direction,
                        pnl: net_pnl,
                        return_pct: net_pnl / pos.size.abs() * 100.0,
                        exit_reason,
                    });

                    capital += net_pnl;
                    position = None;
                }
            }

            // Open new position if no position and valid signal
            if position.is_none() && signal.signal != TradingSignal::Hold {
                let direction = signal.signal.direction();
                let position_value = capital * signal.position_size.min(self.config.max_position_size);
                let entry_price = self.apply_slippage(candle.close, direction);
                let fee = position_value * self.config.fee_pct;

                position = Some(Position {
                    entry_time: candle.timestamp,
                    entry_price,
                    size: position_value,
                    direction,
                });

                capital -= fee;
            }
        }

        // Close any remaining position at last price
        if let Some(pos) = position {
            let last_candle = candles.last().unwrap();
            let exit_price = self.apply_slippage(last_candle.close, -pos.direction);
            let pnl = pos.calculate_pnl(exit_price);
            let fee = pos.size.abs() * self.config.fee_pct;
            let net_pnl = pnl - fee;

            trades.push(Trade {
                entry_time: pos.entry_time,
                exit_time: last_candle.timestamp,
                entry_price: pos.entry_price,
                exit_price,
                position_size: pos.size,
                direction: pos.direction,
                pnl: net_pnl,
                return_pct: net_pnl / pos.size.abs() * 100.0,
                exit_reason: "End of Backtest".to_string(),
            });

            capital += net_pnl;
        }

        self.calculate_metrics(capital, trades, equity_curve, max_drawdown, candles)
    }

    /// Apply slippage to price
    fn apply_slippage(&self, price: f64, direction: i32) -> f64 {
        if direction > 0 {
            price * (1.0 + self.config.slippage_pct)
        } else {
            price * (1.0 - self.config.slippage_pct)
        }
    }

    /// Calculate backtest metrics
    fn calculate_metrics(
        &self,
        final_capital: f64,
        trades: Vec<Trade>,
        equity_curve: Vec<(i64, f64)>,
        max_drawdown: f64,
        candles: &[Candle],
    ) -> BacktestResult {
        let total_return_pct =
            (final_capital - self.config.initial_capital) / self.config.initial_capital * 100.0;

        // Calculate annualized return
        let trading_days = if candles.len() > 1 {
            let first_ts = candles.first().unwrap().timestamp;
            let last_ts = candles.last().unwrap().timestamp;
            (last_ts - first_ts) as f64 / (24.0 * 3600.0 * 1000.0)
        } else {
            1.0
        };
        let years = trading_days / 365.0;
        let annualized_return_pct = if years > 0.0 {
            ((final_capital / self.config.initial_capital).powf(1.0 / years) - 1.0) * 100.0
        } else {
            total_return_pct
        };

        // Calculate Sharpe ratio
        let returns: Vec<f64> = trades.iter().map(|t| t.return_pct / 100.0).collect();
        let sharpe_ratio = self.calculate_sharpe(&returns);

        // Trade statistics
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

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

        BacktestResult {
            total_return_pct,
            annualized_return_pct,
            sharpe_ratio,
            max_drawdown_pct: max_drawdown * 100.0,
            win_rate,
            profit_factor,
            total_trades: trades.len(),
            winning_trades: winning_trades.len(),
            losing_trades: losing_trades.len(),
            avg_trade_return,
            avg_win,
            avg_loss,
            final_capital,
            trades,
            equity_curve,
        }
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            // Annualized Sharpe (assuming daily returns, ~252 trading days)
            mean / std_dev * 252.0_f64.sqrt()
        } else {
            0.0
        }
    }

    /// Return empty result
    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            total_return_pct: 0.0,
            annualized_return_pct: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown_pct: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            avg_trade_return: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            final_capital: self.config.initial_capital,
            trades: vec![],
            equity_curve: vec![],
        }
    }
}

/// Internal position tracking
struct Position {
    entry_time: i64,
    entry_price: f64,
    size: f64,
    direction: i32,
}

impl Position {
    fn calculate_pnl(&self, current_price: f64) -> f64 {
        let price_change = current_price - self.entry_price;
        let pnl_per_unit = price_change * self.direction as f64;
        self.size * pnl_per_unit / self.entry_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::signals::SignalStrength;

    fn make_candle(timestamp: i64, close: f64) -> Candle {
        Candle {
            timestamp,
            open: close,
            high: close * 1.01,
            low: close * 0.99,
            close,
            volume: 1000.0,
            turnover: close * 1000.0,
        }
    }

    fn make_signal(signal: TradingSignal, position_size: f64) -> GeneratedSignal {
        GeneratedSignal {
            signal,
            strength: SignalStrength::Moderate,
            sentiment_score: 0.5,
            confidence: 0.7,
            position_size,
            stop_loss: 0.05,
            take_profit: 0.10,
            reason: "Test".to_string(),
        }
    }

    #[test]
    fn test_empty_backtest() {
        let backtester = Backtester::default();
        let result = backtester.run(&[], &[]);

        assert_eq!(result.total_trades, 0);
        assert_eq!(result.final_capital, 10_000.0);
    }

    #[test]
    fn test_profitable_trade() {
        let backtester = Backtester::default();

        let candles = vec![
            make_candle(1000, 100.0),
            make_candle(2000, 105.0),
            make_candle(3000, 110.0),
        ];

        let signals = vec![
            make_signal(TradingSignal::Buy, 0.5),
            make_signal(TradingSignal::Hold, 0.0),
            make_signal(TradingSignal::Sell, 0.0),
        ];

        let result = backtester.run(&candles, &signals);

        assert!(result.total_return_pct > 0.0);
        assert!(result.final_capital > 10_000.0);
    }

    #[test]
    fn test_losing_trade() {
        let backtester = Backtester::default();

        let candles = vec![
            make_candle(1000, 100.0),
            make_candle(2000, 95.0),
            make_candle(3000, 90.0),
        ];

        let signals = vec![
            make_signal(TradingSignal::Buy, 0.5),
            make_signal(TradingSignal::Hold, 0.0),
            make_signal(TradingSignal::Sell, 0.0),
        ];

        let result = backtester.run(&candles, &signals);

        assert!(result.total_return_pct < 0.0);
        assert!(result.final_capital < 10_000.0);
    }

    #[test]
    fn test_equity_curve() {
        let backtester = Backtester::default();

        let candles: Vec<Candle> = (0..10)
            .map(|i| make_candle(i * 1000, 100.0 + i as f64))
            .collect();

        let signals: Vec<GeneratedSignal> = (0..10)
            .map(|_| make_signal(TradingSignal::Hold, 0.0))
            .collect();

        let result = backtester.run(&candles, &signals);

        assert_eq!(result.equity_curve.len(), 10);
    }
}
