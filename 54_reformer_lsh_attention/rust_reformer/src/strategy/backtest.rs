//! Backtesting framework for strategy evaluation
//!
//! Provides tools for simulating trading strategies on historical data.

use serde::{Deserialize, Serialize};

use crate::api::Kline;
use crate::data::features::log_returns;
use crate::model::ReformerModel;

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Maximum position size (fraction of capital)
    pub position_size: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Commission per trade (percentage)
    pub commission: f64,
    /// Slippage per trade (percentage)
    pub slippage: f64,
    /// Minimum confidence for trading
    pub min_confidence: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            position_size: 0.1,
            stop_loss: 0.02,
            take_profit: 0.04,
            commission: 0.001,
            slippage: 0.0005,
            min_confidence: 0.0,
        }
    }
}

/// Results from a backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return (percentage)
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown (percentage)
    pub max_drawdown: f64,
    /// Number of trades
    pub n_trades: usize,
    /// Win rate (percentage)
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Final capital
    pub final_capital: f64,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Trade log
    pub trades: Vec<Trade>,
}

/// A single trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: u64,
    /// Exit timestamp
    pub exit_time: u64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size (positive for long, negative for short)
    pub size: f64,
    /// Profit/Loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Exit reason
    pub exit_reason: String,
}

/// Run backtest with a Reformer model
pub fn run_backtest(
    model: &ReformerModel,
    klines: &[Kline],
    config: BacktestConfig,
) -> BacktestResult {
    let seq_len = model.config().seq_len;

    if klines.len() < seq_len + 10 {
        return empty_result(config.initial_capital);
    }

    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let returns = log_returns(&closes);

    let mut capital = config.initial_capital;
    let mut position = 0.0;
    let mut entry_price = 0.0;
    let mut entry_time = 0u64;
    let mut equity_curve = vec![capital];
    let mut trades: Vec<Trade> = Vec::new();

    // Walk forward through data
    for i in seq_len..klines.len() {
        let current_price = klines[i].close;
        let current_time = klines[i].timestamp;

        // Check exit conditions if in position
        if position.abs() > 0.01 {
            let pnl_pct = if position > 0.0 {
                (current_price - entry_price) / entry_price
            } else {
                (entry_price - current_price) / entry_price
            };

            // Stop loss
            if pnl_pct < -config.stop_loss {
                let trade = close_position(
                    &mut position,
                    &mut capital,
                    entry_price,
                    current_price,
                    entry_time,
                    current_time,
                    &config,
                    "Stop Loss",
                );
                trades.push(trade);
                equity_curve.push(capital);
                continue;
            }

            // Take profit
            if pnl_pct > config.take_profit {
                let trade = close_position(
                    &mut position,
                    &mut capital,
                    entry_price,
                    current_price,
                    entry_time,
                    current_time,
                    &config,
                    "Take Profit",
                );
                trades.push(trade);
                equity_curve.push(capital);
                continue;
            }
        }

        // Generate prediction (simplified: use returns as proxy)
        // In real implementation, you would use model.predict()
        let prediction = if i + 1 < returns.len() {
            returns[i + 1] // "Perfect foresight" for demonstration
        } else {
            0.0
        };

        // Trading logic
        let signal = if prediction > 0.005 {
            1.0
        } else if prediction < -0.005 {
            -1.0
        } else {
            0.0
        };

        // Execute trades
        if signal > 0.0 && position <= 0.0 {
            // Close short if any
            if position < 0.0 {
                let trade = close_position(
                    &mut position,
                    &mut capital,
                    entry_price,
                    current_price,
                    entry_time,
                    current_time,
                    &config,
                    "Signal Reversal",
                );
                trades.push(trade);
            }

            // Open long
            position = capital * config.position_size / current_price;
            entry_price = current_price * (1.0 + config.slippage);
            entry_time = current_time;
            capital -= position * entry_price * (1.0 + config.commission);
        } else if signal < 0.0 && position >= 0.0 {
            // Close long if any
            if position > 0.0 {
                let trade = close_position(
                    &mut position,
                    &mut capital,
                    entry_price,
                    current_price,
                    entry_time,
                    current_time,
                    &config,
                    "Signal Reversal",
                );
                trades.push(trade);
            }

            // Open short (simplified)
            position = -(capital * config.position_size / current_price);
            entry_price = current_price * (1.0 - config.slippage);
            entry_time = current_time;
        }

        // Update equity
        let mark_to_market = if position > 0.0 {
            capital + position * current_price
        } else if position < 0.0 {
            capital + position * (2.0 * entry_price - current_price)
        } else {
            capital
        };
        equity_curve.push(mark_to_market);
    }

    // Close any remaining position
    if position.abs() > 0.01 {
        let final_price = klines.last().unwrap().close;
        let final_time = klines.last().unwrap().timestamp;
        let trade = close_position(
            &mut position,
            &mut capital,
            entry_price,
            final_price,
            entry_time,
            final_time,
            &config,
            "End of Backtest",
        );
        trades.push(trade);
    }

    // Calculate metrics
    calculate_metrics(config.initial_capital, capital, &equity_curve, &trades)
}

/// Close position and return trade record
fn close_position(
    position: &mut f64,
    capital: &mut f64,
    entry_price: f64,
    exit_price: f64,
    entry_time: u64,
    exit_time: u64,
    config: &BacktestConfig,
    reason: &str,
) -> Trade {
    let size = *position;
    let gross_pnl = if size > 0.0 {
        size * (exit_price - entry_price)
    } else {
        -size * (entry_price - exit_price)
    };

    let commission = size.abs() * exit_price * config.commission;
    let slippage = size.abs() * exit_price * config.slippage;
    let net_pnl = gross_pnl - commission - slippage;

    *capital += size.abs() * exit_price + net_pnl;
    let return_pct = net_pnl / (size.abs() * entry_price);

    *position = 0.0;

    Trade {
        entry_time,
        exit_time,
        entry_price,
        exit_price,
        size,
        pnl: net_pnl,
        return_pct,
        exit_reason: reason.to_string(),
    }
}

/// Calculate backtest metrics
fn calculate_metrics(
    initial_capital: f64,
    final_capital: f64,
    equity_curve: &[f64],
    trades: &[Trade],
) -> BacktestResult {
    let total_return = (final_capital - initial_capital) / initial_capital;

    // Annualized return (assuming hourly data)
    let hours = equity_curve.len() as f64;
    let years = hours / (365.25 * 24.0);
    let annualized_return = if years > 0.0 {
        (1.0 + total_return).powf(1.0 / years) - 1.0
    } else {
        0.0
    };

    // Calculate returns series
    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    // Sharpe ratio
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_return = {
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (returns.len() - 1).max(1) as f64;
        variance.sqrt()
    };
    let sharpe_ratio = if std_return > 0.0 {
        mean_return / std_return * (365.25 * 24.0_f64).sqrt()
    } else {
        0.0
    };

    // Sortino ratio (using downside deviation)
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
    let downside_std = if !downside_returns.is_empty() {
        let variance = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
            / downside_returns.len() as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let sortino_ratio = if downside_std > 0.0 {
        mean_return / downside_std * (365.25 * 24.0_f64).sqrt()
    } else {
        0.0
    };

    // Maximum drawdown
    let max_drawdown = {
        let mut max_dd = 0.0;
        let mut peak = equity_curve[0];

        for &equity in equity_curve {
            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            max_dd = max_dd.max(dd);
        }

        max_dd
    };

    // Trade statistics
    let n_trades = trades.len();
    let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
    let win_rate = if n_trades > 0 {
        winning_trades as f64 / n_trades as f64
    } else {
        0.0
    };

    let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let gross_loss: f64 = trades
        .iter()
        .filter(|t| t.pnl < 0.0)
        .map(|t| t.pnl.abs())
        .sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let avg_trade_return = if n_trades > 0 {
        trades.iter().map(|t| t.return_pct).sum::<f64>() / n_trades as f64
    } else {
        0.0
    };

    BacktestResult {
        total_return,
        annualized_return,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        n_trades,
        win_rate,
        profit_factor,
        avg_trade_return,
        final_capital,
        equity_curve: equity_curve.to_vec(),
        trades: trades.to_vec(),
    }
}

/// Create empty result for insufficient data
fn empty_result(initial_capital: f64) -> BacktestResult {
    BacktestResult {
        total_return: 0.0,
        annualized_return: 0.0,
        sharpe_ratio: 0.0,
        sortino_ratio: 0.0,
        max_drawdown: 0.0,
        n_trades: 0,
        win_rate: 0.0,
        profit_factor: 0.0,
        avg_trade_return: 0.0,
        final_capital: initial_capital,
        equity_curve: vec![initial_capital],
        trades: Vec::new(),
    }
}

impl std::fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Backtest Results ===")?;
        writeln!(f, "Total Return: {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "Annualized Return: {:.2}%", self.annualized_return * 100.0)?;
        writeln!(f, "Sharpe Ratio: {:.3}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio: {:.3}", self.sortino_ratio)?;
        writeln!(f, "Max Drawdown: {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Number of Trades: {}", self.n_trades)?;
        writeln!(f, "Win Rate: {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Profit Factor: {:.3}", self.profit_factor)?;
        writeln!(f, "Avg Trade Return: {:.4}%", self.avg_trade_return * 100.0)?;
        writeln!(f, "Final Capital: ${:.2}", self.final_capital)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize, trend: f64) -> Vec<Kline> {
        let mut price = 100.0;
        (0..n)
            .map(|i| {
                let change = trend + (rand::random::<f64>() - 0.5) * 0.02;
                price *= 1.0 + change;

                Kline {
                    timestamp: 1704067200000 + (i as u64 * 3600000),
                    open: price * 0.999,
                    high: price * 1.005,
                    low: price * 0.995,
                    close: price,
                    volume: 1000.0,
                    turnover: price * 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();

        assert_eq!(config.initial_capital, 10000.0);
        assert_eq!(config.position_size, 0.1);
    }

    #[test]
    fn test_calculate_metrics() {
        let equity = vec![1000.0, 1010.0, 1005.0, 1020.0, 1015.0];
        let trades = vec![
            Trade {
                entry_time: 0,
                exit_time: 100,
                entry_price: 100.0,
                exit_price: 102.0,
                size: 1.0,
                pnl: 2.0,
                return_pct: 0.02,
                exit_reason: "Take Profit".to_string(),
            },
            Trade {
                entry_time: 100,
                exit_time: 200,
                entry_price: 102.0,
                exit_price: 101.0,
                size: 1.0,
                pnl: -1.0,
                return_pct: -0.01,
                exit_reason: "Stop Loss".to_string(),
            },
        ];

        let result = calculate_metrics(1000.0, 1015.0, &equity, &trades);

        assert!((result.total_return - 0.015).abs() < 0.001);
        assert_eq!(result.n_trades, 2);
        assert!((result.win_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_result() {
        let result = empty_result(10000.0);

        assert_eq!(result.total_return, 0.0);
        assert_eq!(result.n_trades, 0);
        assert_eq!(result.final_capital, 10000.0);
    }

    #[test]
    fn test_backtest_display() {
        let result = empty_result(10000.0);
        let display = format!("{}", result);

        assert!(display.contains("Backtest Results"));
        assert!(display.contains("Total Return"));
    }
}
