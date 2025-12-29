//! Performance metrics and statistics.

use statrs::statistics::Statistics;

/// Calculate Sharpe Ratio.
///
/// # Arguments
/// * `returns` - Vector of period returns
/// * `risk_free_rate` - Risk-free rate per period (default 0)
/// * `periods_per_year` - Number of periods per year for annualization
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().copied().mean();
    let std_dev = returns.iter().copied().std_dev();

    if std_dev == 0.0 {
        return 0.0;
    }

    let excess_return = mean_return - risk_free_rate / periods_per_year;
    (excess_return / std_dev) * periods_per_year.sqrt()
}

/// Calculate Sortino Ratio (uses downside deviation).
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().copied().mean();
    let target_return = risk_free_rate / periods_per_year;

    // Calculate downside deviation
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < target_return)
        .map(|&r| (r - target_return).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return f64::INFINITY; // No downside risk
    }

    let downside_dev = (downside_returns.iter().sum::<f64>() / returns.len() as f64).sqrt();

    if downside_dev == 0.0 {
        return f64::INFINITY;
    }

    let excess_return = mean_return - target_return;
    (excess_return / downside_dev) * periods_per_year.sqrt()
}

/// Calculate Maximum Drawdown.
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut max_dd = 0.0;
    let mut peak = equity_curve[0];

    for &value in equity_curve {
        if value > peak {
            peak = value;
        }
        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Calculate Calmar Ratio (annualized return / max drawdown).
pub fn calmar_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    // Build equity curve from returns
    let mut equity = vec![1.0];
    for r in returns {
        equity.push(equity.last().unwrap() * (1.0 + r));
    }

    let max_dd = max_drawdown(&equity);
    if max_dd == 0.0 {
        return f64::INFINITY;
    }

    let total_return = equity.last().unwrap() / equity.first().unwrap() - 1.0;
    let periods = returns.len() as f64;
    let annualized_return = (1.0 + total_return).powf(periods_per_year / periods) - 1.0;

    annualized_return / max_dd
}

/// Calculate win rate.
pub fn win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    wins as f64 / returns.len() as f64
}

/// Calculate profit factor (gross profit / gross loss).
pub fn profit_factor(returns: &[f64]) -> f64 {
    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if gross_loss == 0.0 {
        return f64::INFINITY;
    }

    gross_profit / gross_loss
}

/// Calculate average win and average loss.
pub fn avg_win_loss(returns: &[f64]) -> (f64, f64) {
    let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
    let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    let avg_win = if wins.is_empty() {
        0.0
    } else {
        wins.iter().sum::<f64>() / wins.len() as f64
    };

    let avg_loss = if losses.is_empty() {
        0.0
    } else {
        losses.iter().sum::<f64>() / losses.len() as f64
    };

    (avg_win, avg_loss)
}

/// Calculate CAGR (Compound Annual Growth Rate).
pub fn cagr(initial_value: f64, final_value: f64, years: f64) -> f64 {
    if years == 0.0 || initial_value == 0.0 {
        return 0.0;
    }
    (final_value / initial_value).powf(1.0 / years) - 1.0
}

/// Calculate the deflated Sharpe Ratio.
///
/// This adjusts the Sharpe Ratio for multiple testing bias.
/// Based on Bailey & Lopez de Prado (2014).
pub fn deflated_sharpe_ratio(
    observed_sr: f64,
    num_trials: usize,
    sample_length: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    if num_trials == 0 || sample_length == 0 {
        return 0.0;
    }

    // Expected maximum SR under null hypothesis
    let e_max_sr = expected_max_sr(num_trials, sample_length);

    // Variance of SR estimator
    let sr_std = sr_std(sample_length, skewness, kurtosis, observed_sr);

    if sr_std == 0.0 {
        return 0.0;
    }

    // Probability that observed SR is significant
    let z = (observed_sr - e_max_sr) / sr_std;

    // Use standard normal CDF
    0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2))
}

/// Expected maximum Sharpe Ratio under multiple testing.
fn expected_max_sr(num_trials: usize, sample_length: usize) -> f64 {
    let gamma = 0.5772156649; // Euler-Mascheroni constant
    let n = num_trials as f64;

    (1.0 - gamma) * (2.0 * n.ln()).sqrt()
        + gamma * (2.0 * n.ln()).sqrt()
        + (2.0 * n.ln()).sqrt().recip() * (n.ln().ln() + (4.0 * std::f64::consts::PI).ln() - 2.0 * gamma)
        / (2.0 * (sample_length as f64).sqrt())
}

/// Standard deviation of Sharpe Ratio estimator.
fn sr_std(sample_length: usize, skewness: f64, kurtosis: f64, sr: f64) -> f64 {
    let n = sample_length as f64;
    let term1 = 1.0 + 0.25 * sr.powi(2) * (kurtosis - 1.0);
    let term2 = sr * skewness;

    ((1.0 - term2 + term1) / n).sqrt()
}

/// Backtest performance summary.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub total_trades: usize,
}

impl PerformanceMetrics {
    /// Calculate all metrics from returns.
    pub fn from_returns(returns: &[f64], periods_per_year: f64) -> Self {
        let (avg_win, avg_loss) = avg_win_loss(returns);

        let total_return = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;
        let periods = returns.len() as f64;
        let years = periods / periods_per_year;

        Self {
            total_return,
            annualized_return: if years > 0.0 {
                (1.0 + total_return).powf(1.0 / years) - 1.0
            } else {
                0.0
            },
            sharpe_ratio: sharpe_ratio(returns, 0.0, periods_per_year),
            sortino_ratio: sortino_ratio(returns, 0.0, periods_per_year),
            max_drawdown: {
                let mut equity = vec![1.0];
                for r in returns {
                    equity.push(equity.last().unwrap() * (1.0 + r));
                }
                max_drawdown(&equity)
            },
            calmar_ratio: calmar_ratio(returns, periods_per_year),
            win_rate: win_rate(returns),
            profit_factor: profit_factor(returns),
            avg_win,
            avg_loss,
            total_trades: returns.len(),
        }
    }

    /// Pretty print the metrics.
    pub fn print(&self) {
        println!("═══════════════════════════════════════");
        println!("         Performance Summary           ");
        println!("═══════════════════════════════════════");
        println!("Total Return:      {:>10.2}%", self.total_return * 100.0);
        println!("Annualized Return: {:>10.2}%", self.annualized_return * 100.0);
        println!("Sharpe Ratio:      {:>10.2}", self.sharpe_ratio);
        println!("Sortino Ratio:     {:>10.2}", self.sortino_ratio);
        println!("Max Drawdown:      {:>10.2}%", self.max_drawdown * 100.0);
        println!("Calmar Ratio:      {:>10.2}", self.calmar_ratio);
        println!("Win Rate:          {:>10.2}%", self.win_rate * 100.0);
        println!("Profit Factor:     {:>10.2}", self.profit_factor);
        println!("Avg Win:           {:>10.4}%", self.avg_win * 100.0);
        println!("Avg Loss:          {:>10.4}%", self.avg_loss * 100.0);
        println!("Total Trades:      {:>10}", self.total_trades);
        println!("═══════════════════════════════════════");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 120.0, 90.0, 95.0];
        let mdd = max_drawdown(&equity);
        assert!((mdd - 0.25).abs() < 0.01); // 25% drawdown from 120 to 90
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sr = sharpe_ratio(&returns, 0.0, 252.0);
        assert!(sr > 0.0);
    }

    #[test]
    fn test_win_rate() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let wr = win_rate(&returns);
        assert!((wr - 0.8).abs() < 0.01);
    }
}
