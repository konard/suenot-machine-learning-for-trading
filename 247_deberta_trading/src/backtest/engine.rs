//! Backtesting engine for DeBERTa sentiment trading strategies.
//!
//! Provides event-driven backtesting with configurable risk parameters,
//! position sizing, and performance metrics.

use crate::data::bybit::Kline;
use crate::model::deberta::{SentimentModel, TradingSignal};
use tracing::{debug, info};

/// Result of a single trade.
#[derive(Debug, Clone)]
pub struct TradeResult {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub direction: i32,
    pub entry_price: f64,
    pub exit_price: f64,
    pub position_size: f64,
    pub pnl: f64,
    pub return_pct: f64,
    pub sentiment_score: f64,
    pub holding_bars: usize,
}

/// Results from backtesting.
#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub trades: Vec<TradeResult>,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub n_trades: usize,
    pub equity_curve: Vec<f64>,
}

/// Open position tracking.
struct Position {
    entry_idx: usize,
    entry_price: f64,
    direction: i32,
    size: f64,
    sentiment: f64,
}

/// Sentiment backtester for DeBERTa-based strategies.
pub struct SentimentBacktester {
    model: SentimentModel,
    initial_capital: f64,
    position_pct: f64,
    entry_threshold: f64,
    stop_loss: f64,
    take_profit: f64,
    max_holding_bars: usize,
}

impl SentimentBacktester {
    /// Create a new backtester with default parameters.
    pub fn new() -> Self {
        Self {
            model: SentimentModel::new(),
            initial_capital: 100_000.0,
            position_pct: 0.1,
            entry_threshold: 0.6,
            stop_loss: 0.05,
            take_profit: 0.10,
            max_holding_bars: 10,
        }
    }

    /// Set initial capital.
    pub fn with_capital(mut self, capital: f64) -> Self {
        self.initial_capital = capital;
        self
    }

    /// Set position size as fraction of capital.
    pub fn with_position_pct(mut self, pct: f64) -> Self {
        self.position_pct = pct;
        self
    }

    /// Set entry threshold.
    pub fn with_entry_threshold(mut self, threshold: f64) -> Self {
        self.entry_threshold = threshold;
        self
    }

    /// Set stop loss and take profit.
    pub fn with_risk_params(mut self, stop_loss: f64, take_profit: f64) -> Self {
        self.stop_loss = stop_loss;
        self.take_profit = take_profit;
        self
    }

    /// Run backtest with kline data and headlines.
    ///
    /// Each element of `headlines_per_bar` corresponds to a kline bar
    /// and contains the headlines for that period (may be empty).
    pub fn run(
        &self,
        klines: &[Kline],
        headlines_per_bar: &[Vec<&str>],
    ) -> BacktestResults {
        let mut capital = self.initial_capital;
        let mut equity_curve = vec![capital];
        let mut trades: Vec<TradeResult> = Vec::new();
        let mut position: Option<Position> = None;

        let n = klines.len().min(headlines_per_bar.len());

        for i in 1..n {
            let current_price = klines[i].close;

            // Manage existing position
            if let Some(ref pos) = position {
                let bars_held = i - pos.entry_idx;
                let price_change = (current_price - pos.entry_price) / pos.entry_price;
                let pos_return = price_change * pos.direction as f64;

                let should_exit = pos_return <= -self.stop_loss
                    || pos_return >= self.take_profit
                    || bars_held >= self.max_holding_bars;

                if should_exit {
                    let pnl = pos.size * pos_return;
                    capital += pos.size + pnl;

                    trades.push(TradeResult {
                        entry_idx: pos.entry_idx,
                        exit_idx: i,
                        direction: pos.direction,
                        entry_price: pos.entry_price,
                        exit_price: current_price,
                        position_size: pos.size,
                        pnl,
                        return_pct: pos_return,
                        sentiment_score: pos.sentiment,
                        holding_bars: bars_held,
                    });

                    debug!(
                        "Closed position at bar {} with return {:.4}%",
                        i,
                        pos_return * 100.0
                    );
                    position = None;
                }
            }

            // Generate new signal if no position and headlines available
            if position.is_none() && !headlines_per_bar[i].is_empty() {
                let signal: TradingSignal = self
                    .model
                    .get_trading_signal(&headlines_per_bar[i], self.entry_threshold);

                if signal.signal.abs() > 0.0 {
                    let direction = if signal.signal > 0.0 { 1 } else { -1 };
                    let size = capital * self.position_pct * signal.confidence;
                    capital -= size;

                    position = Some(Position {
                        entry_idx: i,
                        entry_price: current_price,
                        direction,
                        size,
                        sentiment: signal.signal,
                    });

                    let dir_str = if direction > 0 { "LONG" } else { "SHORT" };
                    debug!(
                        "Opened {} at bar {} @ {:.2} (sentiment: {:.3})",
                        dir_str, i, current_price, signal.signal
                    );
                }
            }

            let pos_value = position.as_ref().map_or(0.0, |p| p.size);
            equity_curve.push(capital + pos_value);
        }

        // Close remaining position
        if let Some(pos) = position {
            let final_price = klines[n - 1].close;
            let price_change = (final_price - pos.entry_price) / pos.entry_price;
            let pos_return = price_change * pos.direction as f64;
            let pnl = pos.size * pos_return;
            let _final_capital = capital + pos.size + pnl;

            trades.push(TradeResult {
                entry_idx: pos.entry_idx,
                exit_idx: n - 1,
                direction: pos.direction,
                entry_price: pos.entry_price,
                exit_price: final_price,
                position_size: pos.size,
                pnl,
                return_pct: pos_return,
                sentiment_score: pos.sentiment,
                holding_bars: n - 1 - pos.entry_idx,
            });
        }

        let metrics = compute_metrics(&trades, &equity_curve);

        info!(
            "Backtest complete: {} trades, total return {:.2}%, Sharpe {:.4}",
            trades.len(),
            metrics.total_return * 100.0,
            metrics.sharpe_ratio
        );

        BacktestResults {
            n_trades: trades.len(),
            trades,
            total_return: metrics.total_return,
            sharpe_ratio: metrics.sharpe_ratio,
            sortino_ratio: metrics.sortino_ratio,
            max_drawdown: metrics.max_drawdown,
            win_rate: metrics.win_rate,
            profit_factor: metrics.profit_factor,
            equity_curve,
        }
    }
}

impl Default for SentimentBacktester {
    fn default() -> Self {
        Self::new()
    }
}

struct Metrics {
    total_return: f64,
    sharpe_ratio: f64,
    sortino_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    profit_factor: f64,
}

fn compute_metrics(trades: &[TradeResult], equity: &[f64]) -> Metrics {
    if trades.is_empty() || equity.len() < 2 {
        return Metrics {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
        };
    }

    let total_return = (equity.last().unwrap() - equity[0]) / equity[0];

    let returns: Vec<f64> = trades.iter().map(|t| t.return_pct).collect();
    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_ret = {
        let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / returns.len() as f64;
        var.sqrt()
    };

    let sharpe = if std_ret > 0.0 {
        mean_ret / std_ret * (252.0_f64).sqrt()
    } else {
        0.0
    };

    let downside: Vec<f64> = returns.iter().filter(|r| **r < 0.0).copied().collect();
    let sortino = if !downside.is_empty() {
        let ds_var =
            downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
        let ds_std = ds_var.sqrt();
        if ds_std > 0.0 {
            mean_ret / ds_std * (252.0_f64).sqrt()
        } else {
            sharpe
        }
    } else {
        sharpe
    };

    // Max drawdown
    let mut peak = equity[0];
    let mut max_dd = 0.0_f64;
    for &eq in equity {
        if eq > peak {
            peak = eq;
        }
        let dd = (peak - eq) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
    let win_rate = wins as f64 / trades.len() as f64;

    let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let gross_loss: f64 = trades
        .iter()
        .filter(|t| t.pnl < 0.0)
        .map(|t| t.pnl.abs())
        .sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        f64::INFINITY
    };

    Metrics {
        total_return,
        sharpe_ratio: sharpe,
        sortino_ratio: sortino,
        max_drawdown: max_dd,
        win_rate,
        profit_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_klines;

    #[test]
    fn test_backtest_runs() {
        let klines = generate_synthetic_klines(100, "BTCUSDT");
        let mut headlines: Vec<Vec<&str>> = vec![vec![]; 100];

        // Add some headlines at various points
        headlines[10] = vec!["Bitcoin surges on institutional demand"];
        headlines[30] = vec!["Crypto market crashes amid regulatory fears"];
        headlines[50] = vec!["Bitcoin rally continues with record volume"];
        headlines[70] = vec!["Market selloff deepens on recession concerns"];

        let backtester = SentimentBacktester::new()
            .with_capital(100_000.0)
            .with_entry_threshold(0.3);

        let results = backtester.run(&klines, &headlines);

        assert!(results.n_trades > 0);
        assert_eq!(results.trades.len(), results.n_trades);
        assert!(!results.equity_curve.is_empty());
    }
}
