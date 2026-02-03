//! Backtesting engine with performance metrics.

/// Performance metrics from a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    pub total_return: f64,
    pub annual_return: f64,
    pub annual_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: usize,
}

/// Backtesting engine for evaluating trading signals.
pub struct BacktestEngine {
    /// Transaction cost per trade (fraction)
    pub transaction_cost: f64,
}

impl BacktestEngine {
    /// Create a new backtesting engine.
    pub fn new(transaction_cost: f64) -> Self {
        BacktestEngine { transaction_cost }
    }

    /// Run a backtest given close prices and trading signals.
    ///
    /// # Arguments
    /// * `prices` - Vector of close prices
    /// * `signals` - Vector of signals (-1.0, 0.0, 1.0)
    ///
    /// # Returns
    /// Backtest performance metrics
    pub fn run(&self, prices: &[f64], signals: &[f64]) -> BacktestMetrics {
        let n = prices.len().min(signals.len());
        if n < 2 {
            return BacktestMetrics {
                total_return: 0.0,
                annual_return: 0.0,
                annual_volatility: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                num_trades: 0,
            };
        }

        // Compute returns
        let returns: Vec<f64> = (1..n)
            .map(|i| (prices[i] - prices[i - 1]) / (prices[i - 1] + 1e-10))
            .collect();

        let sig = &signals[..returns.len()];

        // Position changes and costs
        let mut position_changes = Vec::with_capacity(returns.len());
        position_changes.push(sig[0].abs());
        for i in 1..returns.len() {
            position_changes.push((sig[i] - sig[i - 1]).abs());
        }

        // Strategy returns
        let strategy_returns: Vec<f64> = (0..returns.len())
            .map(|i| sig[i] * returns[i] - position_changes[i] * self.transaction_cost)
            .collect();

        // Total return
        let total_return: f64 = strategy_returns
            .iter()
            .fold(1.0, |acc, r| acc * (1.0 + r))
            - 1.0;

        let n_periods = strategy_returns.len() as f64;

        // Annualized return
        let annual_return = if total_return > -1.0 {
            (1.0 + total_return).powf(252.0 / n_periods) - 1.0
        } else {
            -1.0
        };

        // Annualized volatility
        let mean_ret: f64 = strategy_returns.iter().sum::<f64>() / n_periods;
        let variance: f64 = strategy_returns
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / n_periods;
        let annual_volatility = variance.sqrt() * 252.0_f64.sqrt();

        // Sharpe ratio
        let sharpe_ratio = annual_return / (annual_volatility + 1e-10);

        // Max drawdown
        let mut cumulative = Vec::with_capacity(strategy_returns.len());
        let mut cum = 1.0;
        let mut running_max = 1.0_f64;
        let mut max_drawdown = 0.0_f64;
        for r in &strategy_returns {
            cum *= 1.0 + r;
            cumulative.push(cum);
            running_max = running_max.max(cum);
            let dd = (cum - running_max) / (running_max + 1e-10);
            max_drawdown = max_drawdown.min(dd);
        }

        // Sortino ratio
        let downside_returns: Vec<f64> = strategy_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_vol = if !downside_returns.is_empty() {
            let dvar: f64 = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64;
            dvar.sqrt() * 252.0_f64.sqrt()
        } else {
            1e-10
        };
        let sortino_ratio = annual_return / (downside_vol + 1e-10);

        // Win rate
        let active: Vec<&f64> = strategy_returns.iter().filter(|&&r| r != 0.0).collect();
        let wins = active.iter().filter(|&&&r| r > 0.0).count();
        let win_rate = wins as f64 / (active.len() as f64 + 1e-10);

        // Number of trades
        let num_trades = position_changes.iter().filter(|&&c| c > 0.0).count();

        BacktestMetrics {
            total_return,
            annual_return,
            annual_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            num_trades,
        }
    }
}

impl std::fmt::Display for BacktestMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Backtest Results:")?;
        writeln!(f, "  Total Return:      {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "  Annual Return:     {:.2}%", self.annual_return * 100.0)?;
        writeln!(f, "  Annual Volatility: {:.2}%", self.annual_volatility * 100.0)?;
        writeln!(f, "  Sharpe Ratio:      {:.2}", self.sharpe_ratio)?;
        writeln!(f, "  Sortino Ratio:     {:.2}", self.sortino_ratio)?;
        writeln!(f, "  Max Drawdown:      {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Win Rate:          {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  Num Trades:        {}", self.num_trades)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_basic() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0];
        let signals = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let engine = BacktestEngine::new(0.001);
        let metrics = engine.run(&prices, &signals);
        assert!(metrics.total_return > 0.0);
        assert!(metrics.sharpe_ratio.is_finite());
    }

    #[test]
    fn test_backtest_empty() {
        let prices: Vec<f64> = vec![];
        let signals: Vec<f64> = vec![];
        let engine = BacktestEngine::new(0.001);
        let metrics = engine.run(&prices, &signals);
        assert_eq!(metrics.num_trades, 0);
    }
}
