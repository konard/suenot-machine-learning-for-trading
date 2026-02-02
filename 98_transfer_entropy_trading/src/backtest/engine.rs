//! Backtesting engine for TE-based trading strategies.

use crate::trading::signals::TradingSignal;

/// A single step in the backtest.
#[derive(Debug, Clone)]
pub struct BacktestStep {
    pub timestamp: usize,
    pub price: f64,
    pub signal: TradingSignal,
    pub position: f64,
    pub capital: f64,
    pub returns: f64,
}

/// Performance metrics from a backtest.
#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub n_trades: usize,
}

impl std::fmt::Display for BacktestMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Total Return:       {:.1}%", self.total_return * 100.0)?;
        writeln!(f, "  Annualized Return:  {:.1}%", self.annualized_return * 100.0)?;
        writeln!(f, "  Annualized Vol:     {:.1}%", self.annualized_volatility * 100.0)?;
        writeln!(f, "  Sharpe Ratio:       {:.2}", self.sharpe_ratio)?;
        writeln!(f, "  Sortino Ratio:      {:.2}", self.sortino_ratio)?;
        writeln!(f, "  Max Drawdown:       {:.1}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Win Rate:           {:.1}%", self.win_rate * 100.0)?;
        writeln!(f, "  Profit Factor:      {:.2}", self.profit_factor)?;
        writeln!(f, "  Number of Trades:   {}", self.n_trades)?;
        Ok(())
    }
}

/// Backtesting engine for evaluating TE-based strategies.
pub struct BacktestEngine {
    initial_capital: f64,
    transaction_cost: f64,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    ///
    /// # Arguments
    /// * `initial_capital` - Starting capital.
    /// * `transaction_cost` - Cost per trade as a fraction (e.g., 0.001 = 0.1%).
    pub fn new(initial_capital: f64, transaction_cost: f64) -> Self {
        Self {
            initial_capital,
            transaction_cost,
        }
    }

    /// Run a backtest given signals, follower returns, and prices.
    ///
    /// # Arguments
    /// * `signals` - Vector of trading signals.
    /// * `follower_returns` - Return series of the follower asset.
    /// * `prices` - Price series of the follower asset.
    pub fn run(
        &self,
        signals: &[TradingSignal],
        follower_returns: &[f64],
        prices: &[f64],
    ) -> Vec<BacktestStep> {
        let n = signals.len().min(follower_returns.len()).min(prices.len());
        let mut steps = Vec::with_capacity(n);
        let mut capital = self.initial_capital;
        let mut position = 0.0f64;

        for t in 0..n {
            let new_position = signals[t].position();

            // Apply transaction cost on position change
            if (new_position - position).abs() > 1e-10 {
                capital *= 1.0 - self.transaction_cost;
            }

            position = new_position;

            // Apply return
            let ret = if t > 0 { follower_returns[t] } else { 0.0 };
            capital *= 1.0 + position * ret;

            steps.push(BacktestStep {
                timestamp: t,
                price: prices[t],
                signal: signals[t],
                position,
                capital,
                returns: ret,
            });
        }

        steps
    }

    /// Compute performance metrics from backtest results.
    pub fn compute_metrics(&self, steps: &[BacktestStep]) -> BacktestMetrics {
        if steps.len() < 2 {
            return BacktestMetrics {
                total_return: 0.0,
                annualized_return: 0.0,
                annualized_volatility: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                n_trades: 0,
            };
        }

        let capitals: Vec<f64> = steps.iter().map(|s| s.capital).collect();
        let n = capitals.len();

        // Per-step returns
        let returns: Vec<f64> = capitals
            .windows(2)
            .map(|w| w[1] / w[0] - 1.0)
            .collect();

        let total_return = capitals[n - 1] / capitals[0] - 1.0;
        let ann_factor = 252.0;
        let n_periods = returns.len() as f64;

        // Annualized return
        let ann_return = if total_return > -1.0 {
            (1.0 + total_return).powf(ann_factor / n_periods) - 1.0
        } else {
            -1.0
        };

        // Volatility
        let mean_ret = returns.iter().sum::<f64>() / n_periods;
        let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / n_periods;
        let std_ret = variance.sqrt();
        let ann_vol = std_ret * ann_factor.sqrt();

        // Sharpe ratio
        let sharpe = if std_ret > 0.0 {
            ann_factor.sqrt() * mean_ret / std_ret
        } else {
            0.0
        };

        // Sortino ratio
        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if downside.len() > 1 {
            let ds_mean = downside.iter().sum::<f64>() / downside.len() as f64;
            let ds_var = downside.iter().map(|r| (r - ds_mean).powi(2)).sum::<f64>()
                / downside.len() as f64;
            ds_var.sqrt()
        } else {
            0.0
        };
        let sortino = if downside_std > 0.0 {
            ann_factor.sqrt() * mean_ret / downside_std
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = capitals[0];
        let mut max_dd = 0.0f64;
        for &c in &capitals {
            if c > peak {
                peak = c;
            }
            let dd = (c - peak) / peak;
            if dd < max_dd {
                max_dd = dd;
            }
        }

        // Win rate and profit factor
        let mut trade_returns: Vec<f64> = Vec::new();
        for (i, step) in steps.iter().enumerate() {
            if step.position.abs() > 1e-10 && i > 0 {
                trade_returns.push(returns[i - 1]);
            }
        }

        let wins = trade_returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = if !trade_returns.is_empty() {
            wins as f64 / trade_returns.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = trade_returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = trade_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        };

        // Number of trades
        let n_trades = steps
            .windows(2)
            .filter(|w| (w[1].position - w[0].position).abs() > 1e-10)
            .count();

        BacktestMetrics {
            total_return,
            annualized_return: ann_return,
            annualized_volatility: ann_vol,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            win_rate,
            profit_factor,
            n_trades,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_neutral_no_change() {
        let engine = BacktestEngine::new(100_000.0, 0.001);
        let signals = vec![TradingSignal::Neutral; 10];
        let returns = vec![0.01; 10];
        let prices = vec![100.0; 10];

        let steps = engine.run(&signals, &returns, &prices);
        // With neutral position, capital should stay at initial minus first transaction cost
        let last = steps.last().unwrap();
        // Capital should be close to initial since position is 0
        assert!(
            (last.capital - 100_000.0).abs() < 1.0,
            "Capital should stay near initial with neutral position"
        );
    }

    #[test]
    fn test_backtest_long_positive_return() {
        let engine = BacktestEngine::new(100_000.0, 0.0); // No transaction cost
        let signals = vec![
            TradingSignal::Long,
            TradingSignal::Long,
            TradingSignal::Long,
        ];
        let returns = vec![0.0, 0.01, 0.02];
        let prices = vec![100.0, 101.0, 103.0];

        let steps = engine.run(&signals, &returns, &prices);
        let last = steps.last().unwrap();
        // With long position and positive returns, capital should increase
        assert!(last.capital > 100_000.0);
    }
}
