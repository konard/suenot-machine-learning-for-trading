//! Trading Strategy and Backtesting
//!
//! This module provides backtesting framework for evaluating trading strategies.

use ndarray::Array2;

use crate::model::GQATrader;
use crate::predict::{predict_next, Signal};

/// Represents a single trade
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: usize,
    pub entry_price: f32,
    pub direction: TradeDirection,
    pub exit_time: Option<usize>,
    pub exit_price: Option<f32>,
    pub pnl: Option<f32>,
    pub pnl_percent: Option<f32>,
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeDirection {
    Long,
    Short,
}

impl std::fmt::Display for TradeDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeDirection::Long => write!(f, "LONG"),
            TradeDirection::Short => write!(f, "SHORT"),
        }
    }
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f32,
    pub num_trades: usize,
    pub win_rate: f32,
    pub profit_factor: f32,
    pub max_drawdown: f32,
    pub sharpe_ratio: f32,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f32>,
}

impl BacktestResult {
    /// Print a summary of the backtest results
    pub fn print_summary(&self) {
        println!("\nBacktest Results:");
        println!("  Total Return: {:.2}%", self.total_return * 100.0);
        println!("  Number of Trades: {}", self.num_trades);
        println!("  Win Rate: {:.2}%", self.win_rate * 100.0);
        println!("  Profit Factor: {:.2}", self.profit_factor);
        println!("  Max Drawdown: {:.2}%", self.max_drawdown * 100.0);
        println!("  Sharpe Ratio: {:.2}", self.sharpe_ratio);
    }
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub seq_len: usize,
    pub initial_capital: f32,
    pub position_size: f32,
    pub confidence_threshold: f32,
    pub transaction_cost: f32,
    pub stop_loss: Option<f32>,
    pub take_profit: Option<f32>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            seq_len: 60,
            initial_capital: 10000.0,
            position_size: 1.0,
            confidence_threshold: 0.3,
            transaction_cost: 0.001,
            stop_loss: None,
            take_profit: None,
        }
    }
}

/// Run a backtest on the trading model.
///
/// # Arguments
///
/// * `model` - Trained GQATrader model
/// * `data` - OHLCV data array
/// * `config` - Backtest configuration
///
/// # Returns
///
/// BacktestResult with performance metrics
///
/// # Example
///
/// ```rust,no_run
/// use gqa_trading::{GQATrader, backtest_strategy, BacktestConfig};
/// use gqa_trading::data::generate_synthetic_data;
///
/// let model = GQATrader::new(5, 64, 8, 2, 4);
/// let data = generate_synthetic_data(500, 50000.0, 0.02);
/// let result = backtest_strategy(&model, &data.data, BacktestConfig::default());
/// result.print_summary();
/// ```
pub fn backtest_strategy(
    model: &GQATrader,
    data: &Array2<f32>,
    config: BacktestConfig,
) -> BacktestResult {
    let n_samples = data.shape()[0];

    if n_samples < config.seq_len + 10 {
        log::warn!("Not enough data for backtest");
        return BacktestResult {
            total_return: 0.0,
            num_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            max_drawdown: 0.0,
            sharpe_ratio: 0.0,
            trades: vec![],
            equity_curve: vec![config.initial_capital],
        };
    }

    let mut capital = config.initial_capital;
    let mut position: i8 = 0; // 1 = long, -1 = short, 0 = flat
    let mut entry_price = 0.0f32;
    let mut entry_time = 0usize;

    let mut trades: Vec<Trade> = vec![];
    let mut equity_curve = vec![config.initial_capital];

    log::info!(
        "Backtesting from index {} to {}",
        config.seq_len,
        n_samples - 1
    );

    for i in config.seq_len..n_samples - 1 {
        let current_close = data[[i, 3]];
        let next_close = data[[i + 1, 3]];

        // Get model prediction
        let sequence = data
            .slice(ndarray::s![i - config.seq_len..i, ..])
            .to_owned();
        let result = predict_next(model, &sequence);

        // Check stop loss / take profit
        if position != 0 {
            let pnl_percent = if position == 1 {
                (current_close - entry_price) / entry_price
            } else {
                (entry_price - current_close) / entry_price
            };

            // Stop loss
            if let Some(sl) = config.stop_loss {
                if pnl_percent <= -sl {
                    exit_trade(&mut trades, i, current_close);
                    capital *= 1.0 + pnl_percent - config.transaction_cost;
                    position = 0;
                }
            }

            // Take profit
            if let Some(tp) = config.take_profit {
                if pnl_percent >= tp {
                    exit_trade(&mut trades, i, current_close);
                    capital *= 1.0 + pnl_percent - config.transaction_cost;
                    position = 0;
                }
            }
        }

        // Trading logic
        if result.confidence >= config.confidence_threshold {
            if result.prediction == 2 && position <= 0 {
                // UP signal - go long
                if position == -1 {
                    // Close short first
                    let pnl_percent = (entry_price - current_close) / entry_price;
                    exit_trade(&mut trades, i, current_close);
                    capital *= 1.0 + pnl_percent - config.transaction_cost;
                }

                // Open long
                position = 1;
                entry_price = current_close;
                entry_time = i;
                trades.push(Trade {
                    entry_time: i,
                    entry_price: current_close,
                    direction: TradeDirection::Long,
                    exit_time: None,
                    exit_price: None,
                    pnl: None,
                    pnl_percent: None,
                });
                capital *= 1.0 - config.transaction_cost;
            } else if result.prediction == 0 && position >= 0 {
                // DOWN signal - go short
                if position == 1 {
                    // Close long first
                    let pnl_percent = (current_close - entry_price) / entry_price;
                    exit_trade(&mut trades, i, current_close);
                    capital *= 1.0 + pnl_percent - config.transaction_cost;
                }

                // Open short
                position = -1;
                entry_price = current_close;
                entry_time = i;
                trades.push(Trade {
                    entry_time: i,
                    entry_price: current_close,
                    direction: TradeDirection::Short,
                    exit_time: None,
                    exit_price: None,
                    pnl: None,
                    pnl_percent: None,
                });
                capital *= 1.0 - config.transaction_cost;
            }
        }

        // Update equity
        let equity = if position == 1 {
            capital * (1.0 + (next_close - entry_price) / entry_price)
        } else if position == -1 {
            capital * (1.0 + (entry_price - next_close) / entry_price)
        } else {
            capital
        };
        equity_curve.push(equity);
    }

    // Close any remaining position
    if position != 0 {
        let final_price = data[[n_samples - 1, 3]];
        let pnl_percent = if position == 1 {
            (final_price - entry_price) / entry_price
        } else {
            (entry_price - final_price) / entry_price
        };
        exit_trade(&mut trades, n_samples - 1, final_price);
        capital *= 1.0 + pnl_percent - config.transaction_cost;
    }

    // Calculate metrics
    let metrics = calculate_metrics(&trades, &equity_curve, config.initial_capital);

    BacktestResult {
        total_return: metrics.total_return,
        num_trades: metrics.num_trades,
        win_rate: metrics.win_rate,
        profit_factor: metrics.profit_factor,
        max_drawdown: metrics.max_drawdown,
        sharpe_ratio: metrics.sharpe_ratio,
        trades,
        equity_curve,
    }
}

fn exit_trade(trades: &mut [Trade], time: usize, price: f32) {
    if let Some(trade) = trades.last_mut() {
        if trade.exit_time.is_none() {
            trade.exit_time = Some(time);
            trade.exit_price = Some(price);

            let pnl = match trade.direction {
                TradeDirection::Long => price - trade.entry_price,
                TradeDirection::Short => trade.entry_price - price,
            };

            trade.pnl = Some(pnl);
            trade.pnl_percent = Some(pnl / trade.entry_price);
        }
    }
}

struct Metrics {
    total_return: f32,
    num_trades: usize,
    win_rate: f32,
    profit_factor: f32,
    max_drawdown: f32,
    sharpe_ratio: f32,
}

fn calculate_metrics(trades: &[Trade], equity_curve: &[f32], initial_capital: f32) -> Metrics {
    let completed_trades: Vec<_> = trades.iter().filter(|t| t.exit_time.is_some()).collect();

    if completed_trades.is_empty() {
        return Metrics {
            total_return: 0.0,
            num_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            max_drawdown: 0.0,
            sharpe_ratio: 0.0,
        };
    }

    // Total return
    let total_return = (equity_curve.last().unwrap_or(&initial_capital) - initial_capital)
        / initial_capital;

    // Win rate
    let winning_trades = completed_trades
        .iter()
        .filter(|t| t.pnl.map(|p| p > 0.0).unwrap_or(false))
        .count();
    let win_rate = winning_trades as f32 / completed_trades.len() as f32;

    // Profit factor
    let gross_profit: f32 = completed_trades
        .iter()
        .filter_map(|t| t.pnl)
        .filter(|&p| p > 0.0)
        .sum();
    let gross_loss: f32 = completed_trades
        .iter()
        .filter_map(|t| t.pnl)
        .filter(|&p| p < 0.0)
        .map(|p| p.abs())
        .sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        f32::INFINITY
    };

    // Max drawdown
    let mut peak = equity_curve[0];
    let mut max_drawdown = 0.0f32;
    for &value in equity_curve {
        if value > peak {
            peak = value;
        }
        let drawdown = (peak - value) / peak;
        max_drawdown = max_drawdown.max(drawdown);
    }

    // Sharpe ratio
    let returns: Vec<f32> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let sharpe_ratio = if returns.len() > 1 {
        let mean: f32 = returns.iter().sum::<f32>() / returns.len() as f32;
        let variance: f32 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / returns.len() as f32;
        let std = variance.sqrt();
        if std > 0.0 {
            mean / std * (252.0_f32).sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    Metrics {
        total_return,
        num_trades: completed_trades.len(),
        win_rate,
        profit_factor,
        max_drawdown,
        sharpe_ratio,
    }
}

/// Compare multiple strategy configurations
pub fn compare_strategies(
    model: &GQATrader,
    data: &Array2<f32>,
    configs: &[(&str, BacktestConfig)],
) -> Vec<(String, BacktestResult)> {
    let mut results = vec![];

    println!("\nStrategy Comparison:");
    println!("{:-<60}", "");
    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "Strategy", "Return", "Win Rate", "Sharpe"
    );
    println!("{:-<60}", "");

    for (name, config) in configs {
        let result = backtest_strategy(model, data, config.clone());
        println!(
            "{:<20} {:>10.2}% {:>10.2}% {:>10.2}",
            name,
            result.total_return * 100.0,
            result.win_rate * 100.0,
            result.sharpe_ratio
        );
        results.push((name.to_string(), result));
    }

    println!("{:-<60}", "");

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_synthetic_data;

    #[test]
    fn test_backtest() {
        let model = GQATrader::new(5, 32, 4, 2, 2);
        let data = generate_synthetic_data(200, 100.0, 0.02);

        let config = BacktestConfig {
            seq_len: 30,
            initial_capital: 10000.0,
            confidence_threshold: 0.2,
            ..Default::default()
        };

        let result = backtest_strategy(&model, &data.data, config);

        assert!(result.equity_curve.len() > 1);
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
    }

    #[test]
    fn test_trade_direction_display() {
        assert_eq!(format!("{}", TradeDirection::Long), "LONG");
        assert_eq!(format!("{}", TradeDirection::Short), "SHORT");
    }
}
