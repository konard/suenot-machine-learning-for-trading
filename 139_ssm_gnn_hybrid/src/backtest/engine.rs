//! Backtesting engine with performance metrics.

/// Result of a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub calmar_ratio: f64,
    pub n_trades: usize,
    pub equity_curve: Vec<f64>,
}

/// Backtesting engine configuration.
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission: f64,
    pub slippage: f64,
    pub risk_free_rate: f64,
    pub periods_per_year: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission: 0.001,
            slippage: 0.0005,
            risk_free_rate: 0.02,
            periods_per_year: 252.0,
        }
    }
}

/// Run a backtest given signals and prices.
///
/// # Arguments
/// * `config` - Backtest configuration
/// * `signals` - (n_periods, n_assets) signal matrix (values in {-1, 0, 1})
/// * `prices` - (n_periods, n_assets) price matrix
/// * `confidences` - Optional (n_periods, n_assets) confidence matrix
pub fn run_backtest(
    config: &BacktestConfig,
    signals: &[Vec<f64>],
    prices: &[Vec<f64>],
    confidences: Option<&[Vec<f64>]>,
) -> BacktestResult {
    let n_periods = signals.len();
    let n_assets = if n_periods > 0 { signals[0].len() } else { 0 };

    // Compute weights
    let weights: Vec<Vec<f64>> = (0..n_periods)
        .map(|t| {
            let conf: Vec<f64> = if let Some(c) = confidences {
                c[t].clone()
            } else {
                vec![1.0; n_assets]
            };

            let raw: Vec<f64> = (0..n_assets)
                .map(|a| signals[t][a] * conf[a])
                .collect();

            let total_abs: f64 = raw.iter().map(|w| w.abs()).sum();
            if total_abs < 1e-10 {
                vec![0.0; n_assets]
            } else {
                raw.iter().map(|w| w / total_abs).collect()
            }
        })
        .collect();

    // Compute returns
    let mut portfolio_returns = vec![0.0_f64; n_periods];
    let mut n_trades = 0;

    for t in 1..n_periods {
        let mut port_ret = 0.0;
        for a in 0..n_assets {
            if prices[t - 1][a].abs() > 1e-10 {
                let asset_ret = (prices[t][a] - prices[t - 1][a]) / prices[t - 1][a];
                port_ret += weights[t - 1][a] * asset_ret;
            }
        }

        // Transaction costs
        if t >= 2 {
            let turnover: f64 = (0..n_assets)
                .map(|a| (weights[t - 1][a] - weights[t - 2][a]).abs())
                .sum();
            port_ret -= turnover * (config.commission + config.slippage);
            if turnover > 0.01 {
                n_trades += 1;
            }
        }

        portfolio_returns[t] = port_ret;
    }

    // Equity curve
    let mut equity_curve = Vec::with_capacity(n_periods);
    let mut equity = config.initial_capital;
    for &ret in &portfolio_returns {
        equity *= 1.0 + ret;
        equity_curve.push(equity);
    }

    // Total return
    let total_return = equity / config.initial_capital - 1.0;

    // Annualized return
    let n_years = n_periods as f64 / config.periods_per_year;
    let annualized_return = if n_years > 0.0 {
        (1.0 + total_return).powf(1.0 / n_years) - 1.0
    } else {
        0.0
    };

    // Sharpe ratio
    let rf_per_period = config.risk_free_rate / config.periods_per_year;
    let excess: Vec<f64> = portfolio_returns.iter().map(|r| r - rf_per_period).collect();
    let mean_excess = excess.iter().sum::<f64>() / excess.len() as f64;
    let std_returns = std_dev(&portfolio_returns);
    let sharpe_ratio = if std_returns > 1e-10 {
        mean_excess / std_returns * config.periods_per_year.sqrt()
    } else {
        0.0
    };

    // Sortino ratio
    let downside: Vec<f64> = portfolio_returns
        .iter()
        .filter(|&&r| r < 0.0)
        .copied()
        .collect();
    let downside_std = if downside.is_empty() { 1e-10 } else { std_dev(&downside) };
    let sortino_ratio = mean_excess / downside_std * config.periods_per_year.sqrt();

    // Max drawdown
    let mut running_max = 0.0_f64;
    let mut max_drawdown = 0.0_f64;
    for &eq in &equity_curve {
        running_max = running_max.max(eq);
        let dd = (running_max - eq) / (running_max + 1e-10);
        max_drawdown = max_drawdown.max(dd);
    }

    // Win rate
    let winning = portfolio_returns.iter().filter(|&&r| r > 0.0).count();
    let active = portfolio_returns.iter().filter(|&&r| r != 0.0).count();
    let win_rate = if active > 0 { winning as f64 / active as f64 } else { 0.0 };

    // Profit factor
    let gross_profit: f64 = portfolio_returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = portfolio_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    let profit_factor = if gross_loss > 1e-10 { gross_profit / gross_loss } else { 0.0 };

    // Calmar ratio
    let calmar_ratio = if max_drawdown > 1e-10 {
        annualized_return / max_drawdown
    } else {
        0.0
    };

    BacktestResult {
        total_return,
        annualized_return,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        win_rate,
        profit_factor,
        calmar_ratio,
        n_trades,
        equity_curve,
    }
}

fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_basic() {
        let config = BacktestConfig::default();

        let signals = vec![
            vec![1.0, -1.0],
            vec![1.0, -1.0],
            vec![0.0, 0.0],
            vec![-1.0, 1.0],
            vec![-1.0, 1.0],
        ];

        let prices = vec![
            vec![100.0, 50.0],
            vec![102.0, 49.0],
            vec![101.0, 51.0],
            vec![103.0, 48.0],
            vec![100.0, 52.0],
        ];

        let result = run_backtest(&config, &signals, &prices, None);
        assert_eq!(result.equity_curve.len(), 5);
        assert_eq!(result.equity_curve.len(), result.equity_curve.len());
    }
}
