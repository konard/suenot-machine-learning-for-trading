//! Backtesting framework.

use std::fmt;

/// Single trade record.
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: usize,
    pub exit_time: usize,
    pub entry_price: f64,
    pub exit_price: f64,
    pub direction: Direction,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub counterfactual_distance: Option<f64>,
}

/// Trade direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Long,
    Short,
}

/// Backtest results.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: usize,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
}

impl fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Backtest Results")?;
        writeln!(f, "================")?;
        writeln!(f, "Total Return:     {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "Sharpe Ratio:     {:.2}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio:    {:.2}", self.sortino_ratio)?;
        writeln!(f, "Max Drawdown:     {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Win Rate:         {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Number of Trades: {}", self.num_trades)
    }
}

/// Backtester for trading strategies.
pub struct Backtester {
    initial_capital: f64,
    commission: f64,
    slippage: f64,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            commission: 0.001,
            slippage: 0.0005,
        }
    }

    /// Set commission rate.
    pub fn with_commission(mut self, commission: f64) -> Self {
        self.commission = commission;
        self
    }

    /// Set slippage rate.
    pub fn with_slippage(mut self, slippage: f64) -> Self {
        self.slippage = slippage;
        self
    }

    /// Run backtest on price data with signals.
    pub fn run(
        &self,
        prices: &[f64],
        signals: &[usize],
        cf_distances: Option<&[f64]>,
        cf_threshold: Option<f64>,
    ) -> BacktestResult {
        let n = prices.len();
        let mut equity = vec![0.0; n];
        equity[0] = self.initial_capital;

        let mut position = 0i32; // 1=long, -1=short, 0=flat
        let mut entry_price = 0.0;
        let mut entry_time = 0;
        let mut trades = Vec::new();

        // Apply CF threshold filter
        let mut filtered_signals = signals.to_vec();
        if let (Some(cf_dist), Some(threshold)) = (cf_distances, cf_threshold) {
            for i in 0..n {
                if cf_dist[i] < threshold {
                    filtered_signals[i] = 1; // Force HOLD
                }
            }
        }

        for i in 1..n {
            let signal = filtered_signals[i];

            // Entry logic
            if position == 0 {
                if signal == 2 {
                    // BUY
                    position = 1;
                    entry_price = prices[i] * (1.0 + self.slippage);
                    entry_time = i;
                } else if signal == 0 {
                    // SELL
                    position = -1;
                    entry_price = prices[i] * (1.0 - self.slippage);
                    entry_time = i;
                }
            }
            // Exit logic
            else if position == 1 && signal == 0 {
                // Close long
                let exit_price = prices[i] * (1.0 - self.slippage);
                let pnl_pct = (exit_price / entry_price - 1.0) - 2.0 * self.commission;
                let pnl = equity[i - 1] * pnl_pct;

                trades.push(Trade {
                    entry_time,
                    exit_time: i,
                    entry_price,
                    exit_price,
                    direction: Direction::Long,
                    pnl,
                    pnl_pct,
                    counterfactual_distance: cf_distances.map(|d| d[entry_time]),
                });

                equity[i] = equity[i - 1] + pnl;
                position = 0;
            } else if position == -1 && signal == 2 {
                // Close short
                let exit_price = prices[i] * (1.0 + self.slippage);
                let pnl_pct = (entry_price / exit_price - 1.0) - 2.0 * self.commission;
                let pnl = equity[i - 1] * pnl_pct;

                trades.push(Trade {
                    entry_time,
                    exit_time: i,
                    entry_price,
                    exit_price,
                    direction: Direction::Short,
                    pnl,
                    pnl_pct,
                    counterfactual_distance: cf_distances.map(|d| d[entry_time]),
                });

                equity[i] = equity[i - 1] + pnl;
                position = 0;
            }

            // No position change
            if equity[i] == 0.0 {
                equity[i] = equity[i - 1];
            }
        }

        // Calculate metrics
        let total_return = (equity[n - 1] / self.initial_capital) - 1.0;
        let daily_returns = self.calculate_returns(&equity);
        let sharpe = self.calculate_sharpe(&daily_returns);
        let sortino = self.calculate_sortino(&daily_returns);
        let max_dd = self.calculate_max_drawdown(&equity);
        let win_rate = if trades.is_empty() {
            0.0
        } else {
            trades.iter().filter(|t| t.pnl > 0.0).count() as f64 / trades.len() as f64
        };

        BacktestResult {
            total_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            win_rate,
            num_trades: trades.len(),
            trades,
            equity_curve: equity,
        }
    }

    fn calculate_returns(&self, equity: &[f64]) -> Vec<f64> {
        equity
            .windows(2)
            .filter_map(|w| {
                if w[0] > 0.0 {
                    Some((w[1] - w[0]) / w[0])
                } else {
                    None
                }
            })
            .collect()
    }

    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            0.0
        } else {
            (252.0_f64).sqrt() * mean / std
        }
    }

    fn calculate_sortino(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside.is_empty() {
            return 0.0;
        }

        let downside_variance: f64 =
            downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std == 0.0 {
            0.0
        } else {
            (252.0_f64).sqrt() * mean / downside_std
        }
    }

    fn calculate_max_drawdown(&self, equity: &[f64]) -> f64 {
        let mut max_dd = 0.0;
        let mut peak = equity[0];

        for &e in equity.iter() {
            if e > peak {
                peak = e;
            }
            let dd = (peak - e) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(10000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtester_creation() {
        let bt = Backtester::new(10000.0).with_commission(0.001).with_slippage(0.0005);
        assert_eq!(bt.initial_capital, 10000.0);
    }

    #[test]
    fn test_backtest_run() {
        let bt = Backtester::new(10000.0);
        let prices = vec![100.0, 101.0, 102.0, 101.0, 103.0, 102.0, 104.0];
        let signals = vec![1, 2, 1, 0, 2, 1, 0];

        let result = bt.run(&prices, &signals, None, None);
        assert!(result.num_trades > 0);
    }
}
