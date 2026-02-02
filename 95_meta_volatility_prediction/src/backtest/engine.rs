//! Backtesting engine for volatility-adaptive strategies.

use crate::trading::strategy::{StrategyConfig, VolatilityStrategy};

/// Configuration for the backtesting engine.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub transaction_cost_bps: f64,
    pub slippage_bps: f64,
    pub strategy_config: StrategyConfig,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost_bps: 10.0,
            slippage_bps: 5.0,
            strategy_config: StrategyConfig::default(),
        }
    }
}

/// Results from a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub equity_curve: Vec<f64>,
    pub returns: Vec<f64>,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub vol_forecast_mse: f64,
    pub vol_forecast_mae: f64,
    pub hit_rate: f64,
    pub num_trades: usize,
}

/// Backtesting engine.
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with volatility-adaptive position sizing.
    ///
    /// # Arguments
    /// * `prices` - Asset prices
    /// * `predicted_vols` - Predicted volatilities
    /// * `realized_vols` - Actual realized volatilities
    pub fn run(
        &self,
        prices: &[f64],
        predicted_vols: &[f64],
        realized_vols: &[f64],
    ) -> BacktestResult {
        let n = prices.len();
        let mut equity = self.config.initial_capital;
        let mut equity_curve = vec![equity];
        let mut daily_returns = Vec::new();

        let cost_rate =
            (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000.0;

        let mut strategy = VolatilityStrategy::new(self.config.strategy_config.clone());

        for i in 1..n {
            let vol = if i - 1 < predicted_vols.len() {
                predicted_vols[i - 1]
            } else {
                0.02
            };

            let (pos_pct, _signal) = strategy.update(vol);
            let position_value = equity * pos_pct;

            let price_return = (prices[i] - prices[i - 1]) / prices[i - 1];
            let gross_pnl = position_value * price_return;
            let cost = position_value * cost_rate * 2.0;
            let net_pnl = gross_pnl - cost;

            let prev_equity = equity;
            equity += net_pnl;
            equity_curve.push(equity);

            if prev_equity.abs() > 1e-10 {
                daily_returns.push(net_pnl / prev_equity);
            }
        }

        // Compute metrics
        let total_return = if self.config.initial_capital > 0.0 {
            (equity / self.config.initial_capital) - 1.0
        } else {
            0.0
        };

        let sharpe = compute_sharpe(&daily_returns);
        let sortino = compute_sortino(&daily_returns);
        let max_dd = compute_max_drawdown(&equity_curve);
        let calmar = if max_dd.abs() > 1e-10 {
            let annual_return = total_return * 252.0 / daily_returns.len().max(1) as f64;
            annual_return / max_dd.abs()
        } else {
            0.0
        };

        let min_len = predicted_vols.len().min(realized_vols.len());
        let (mse, mae, hit_rate) = if min_len > 0 {
            compute_forecast_metrics(
                &predicted_vols[..min_len],
                &realized_vols[..min_len],
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        BacktestResult {
            equity_curve,
            returns: daily_returns,
            total_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            calmar_ratio: calmar,
            vol_forecast_mse: mse,
            vol_forecast_mae: mae,
            hit_rate,
            num_trades: n.saturating_sub(1),
        }
    }

    /// Print backtest summary.
    pub fn print_summary(result: &BacktestResult) {
        println!("==================================================");
        println!("Backtest Results");
        println!("==================================================");
        println!("Total Return:      {:>10.2}%", result.total_return * 100.0);
        println!("Sharpe Ratio:      {:>10.3}", result.sharpe_ratio);
        println!("Sortino Ratio:     {:>10.3}", result.sortino_ratio);
        println!("Max Drawdown:      {:>10.2}%", result.max_drawdown * 100.0);
        println!("Calmar Ratio:      {:>10.3}", result.calmar_ratio);
        println!("Vol Forecast MSE:  {:>10.6}", result.vol_forecast_mse);
        println!("Vol Forecast MAE:  {:>10.6}", result.vol_forecast_mae);
        println!("Hit Rate:          {:>10.2}%", result.hit_rate * 100.0);
        println!("Number of Trades:  {:>10}", result.num_trades);
        println!("==================================================");
    }
}

fn compute_sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();
    if std < 1e-10 {
        return 0.0;
    }
    (mean / std) * 252.0_f64.sqrt()
}

fn compute_sortino(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    if downside.len() < 2 {
        return 0.0;
    }
    let ds_var: f64 =
        downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
    let ds_std = ds_var.sqrt();
    if ds_std < 1e-10 {
        return 0.0;
    }
    (mean / ds_std) * 252.0_f64.sqrt()
}

fn compute_max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = equity[0];
    let mut max_dd = 0.0_f64;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = (e - peak) / peak;
        if dd < max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

fn compute_forecast_metrics(predicted: &[f64], realized: &[f64]) -> (f64, f64, f64) {
    let n = predicted.len() as f64;
    let mse: f64 = predicted
        .iter()
        .zip(realized)
        .map(|(p, r)| (p - r).powi(2))
        .sum::<f64>()
        / n;
    let mae: f64 = predicted
        .iter()
        .zip(realized)
        .map(|(p, r)| (p - r).abs())
        .sum::<f64>()
        / n;

    // Hit rate: direction of vol change matches
    let hit_rate = if predicted.len() > 1 {
        let hits: usize = predicted
            .windows(2)
            .zip(realized.windows(2))
            .filter(|(p, r)| (p[1] - p[0]).signum() == (r[1] - r[0]).signum())
            .count();
        hits as f64 / (predicted.len() - 1) as f64
    } else {
        0.0
    };

    (mse, mae, hit_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_basic() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        let predicted_vols = vec![0.02; 100];
        let realized_vols = vec![0.02; 100];

        let engine = BacktestEngine::new(BacktestConfig::default());
        let result = engine.run(&prices, &predicted_vols, &realized_vols);

        assert!(result.equity_curve.len() == 100);
        assert!(result.max_drawdown <= 0.0);
        assert!(result.num_trades == 99);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 90.0, 95.0, 80.0, 120.0];
        let dd = compute_max_drawdown(&equity);
        // Max drawdown from 110 to 80 = (80-110)/110 ≈ -0.2727
        assert!(dd < -0.27);
        assert!(dd > -0.28);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, -0.005, 0.008, 0.003, -0.002, 0.006];
        let sharpe = compute_sharpe(&returns);
        assert!(sharpe.is_finite());
    }
}
