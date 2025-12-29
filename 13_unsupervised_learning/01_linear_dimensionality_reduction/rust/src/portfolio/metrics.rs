//! Portfolio performance metrics

use ndarray::Array1;

/// Calculate annualized return
pub fn annualized_return(returns: &Array1<f64>, periods_per_year: f64) -> f64 {
    let mean_return = returns.mean().unwrap_or(0.0);
    mean_return * periods_per_year
}

/// Calculate annualized volatility
pub fn annualized_volatility(returns: &Array1<f64>, periods_per_year: f64) -> f64 {
    returns.std(0.0) * periods_per_year.sqrt()
}

/// Calculate Sharpe ratio
pub fn sharpe_ratio(returns: &Array1<f64>, risk_free_rate: f64, periods_per_year: f64) -> f64 {
    let ann_return = annualized_return(returns, periods_per_year);
    let ann_vol = annualized_volatility(returns, periods_per_year);

    if ann_vol > 1e-10 {
        (ann_return - risk_free_rate) / ann_vol
    } else {
        0.0
    }
}

/// Calculate Sortino ratio (uses downside deviation)
pub fn sortino_ratio(returns: &Array1<f64>, risk_free_rate: f64, periods_per_year: f64) -> f64 {
    let ann_return = annualized_return(returns, periods_per_year);
    let downside_dev = downside_deviation(returns, 0.0, periods_per_year);

    if downside_dev > 1e-10 {
        (ann_return - risk_free_rate) / downside_dev
    } else {
        0.0
    }
}

/// Calculate downside deviation
pub fn downside_deviation(returns: &Array1<f64>, mar: f64, periods_per_year: f64) -> f64 {
    let negative_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < mar)
        .map(|&r| (r - mar).powi(2))
        .collect();

    if negative_returns.is_empty() {
        return 0.0;
    }

    let mean_sq: f64 = negative_returns.iter().sum::<f64>() / negative_returns.len() as f64;
    mean_sq.sqrt() * periods_per_year.sqrt()
}

/// Calculate maximum drawdown
pub fn max_drawdown(returns: &Array1<f64>) -> f64 {
    let n = returns.len();
    if n == 0 {
        return 0.0;
    }

    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;

    for &r in returns.iter() {
        cumulative *= 1.0 + r;
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = (peak - cumulative) / peak;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    max_dd
}

/// Calculate Calmar ratio (annualized return / max drawdown)
pub fn calmar_ratio(returns: &Array1<f64>, periods_per_year: f64) -> f64 {
    let ann_return = annualized_return(returns, periods_per_year);
    let max_dd = max_drawdown(returns);

    if max_dd > 1e-10 {
        ann_return / max_dd
    } else {
        0.0
    }
}

/// Calculate Value at Risk (VaR) at given confidence level
pub fn var_historical(returns: &Array1<f64>, confidence: f64) -> f64 {
    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((1.0 - confidence) * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Calculate Conditional VaR (Expected Shortfall)
pub fn cvar(returns: &Array1<f64>, confidence: f64) -> f64 {
    let var = var_historical(returns, confidence);

    let tail_returns: Vec<f64> = returns.iter().filter(|&&r| r <= var).copied().collect();

    if tail_returns.is_empty() {
        return var;
    }

    tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
}

/// Calculate Information Ratio vs benchmark
pub fn information_ratio(
    returns: &Array1<f64>,
    benchmark_returns: &Array1<f64>,
    periods_per_year: f64,
) -> f64 {
    let excess_returns: Array1<f64> = returns - benchmark_returns;
    let tracking_error = excess_returns.std(0.0) * periods_per_year.sqrt();
    let mean_excess = excess_returns.mean().unwrap_or(0.0) * periods_per_year;

    if tracking_error > 1e-10 {
        mean_excess / tracking_error
    } else {
        0.0
    }
}

/// Calculate Beta vs benchmark
pub fn beta(returns: &Array1<f64>, benchmark_returns: &Array1<f64>) -> f64 {
    let n = returns.len().min(benchmark_returns.len()) as f64;
    if n < 2.0 {
        return 1.0;
    }

    let mean_r = returns.mean().unwrap_or(0.0);
    let mean_b = benchmark_returns.mean().unwrap_or(0.0);

    let mut covariance = 0.0;
    let mut benchmark_variance = 0.0;

    for i in 0..n as usize {
        let r_dev = returns[i] - mean_r;
        let b_dev = benchmark_returns[i] - mean_b;
        covariance += r_dev * b_dev;
        benchmark_variance += b_dev * b_dev;
    }

    if benchmark_variance > 1e-10 {
        covariance / benchmark_variance
    } else {
        1.0
    }
}

/// Calculate Alpha (Jensen's Alpha)
pub fn alpha(
    returns: &Array1<f64>,
    benchmark_returns: &Array1<f64>,
    risk_free_rate: f64,
    periods_per_year: f64,
) -> f64 {
    let portfolio_beta = beta(returns, benchmark_returns);
    let ann_return = annualized_return(returns, periods_per_year);
    let ann_benchmark = annualized_return(benchmark_returns, periods_per_year);

    ann_return - (risk_free_rate + portfolio_beta * (ann_benchmark - risk_free_rate))
}

/// Calculate Treynor ratio
pub fn treynor_ratio(
    returns: &Array1<f64>,
    benchmark_returns: &Array1<f64>,
    risk_free_rate: f64,
    periods_per_year: f64,
) -> f64 {
    let ann_return = annualized_return(returns, periods_per_year);
    let portfolio_beta = beta(returns, benchmark_returns);

    if portfolio_beta.abs() > 1e-10 {
        (ann_return - risk_free_rate) / portfolio_beta
    } else {
        0.0
    }
}

/// Portfolio metrics summary
#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub var_95: f64,
    pub cvar_95: f64,
}

impl PortfolioMetrics {
    /// Calculate all metrics from returns
    pub fn from_returns(returns: &Array1<f64>, periods_per_year: f64) -> Self {
        let total_return = returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;

        Self {
            total_return,
            annualized_return: annualized_return(returns, periods_per_year),
            annualized_volatility: annualized_volatility(returns, periods_per_year),
            sharpe_ratio: sharpe_ratio(returns, 0.0, periods_per_year),
            sortino_ratio: sortino_ratio(returns, 0.0, periods_per_year),
            max_drawdown: max_drawdown(returns),
            calmar_ratio: calmar_ratio(returns, periods_per_year),
            var_95: var_historical(returns, 0.95),
            cvar_95: cvar(returns, 0.95),
        }
    }

    /// Print metrics summary
    pub fn summary(&self) {
        println!("\n=== Portfolio Metrics ===");
        println!("Total Return:          {:>10.2}%", self.total_return * 100.0);
        println!(
            "Annualized Return:     {:>10.2}%",
            self.annualized_return * 100.0
        );
        println!(
            "Annualized Volatility: {:>10.2}%",
            self.annualized_volatility * 100.0
        );
        println!("Sharpe Ratio:          {:>10.2}", self.sharpe_ratio);
        println!("Sortino Ratio:         {:>10.2}", self.sortino_ratio);
        println!("Max Drawdown:          {:>10.2}%", self.max_drawdown * 100.0);
        println!("Calmar Ratio:          {:>10.2}", self.calmar_ratio);
        println!("VaR (95%):             {:>10.2}%", self.var_95 * 100.0);
        println!("CVaR (95%):            {:>10.2}%", self.cvar_95 * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sharpe_ratio() {
        let returns = array![0.01, 0.02, -0.01, 0.015, 0.005];
        let sharpe = sharpe_ratio(&returns, 0.0, 365.0);
        assert!(sharpe > 0.0); // Positive returns should have positive Sharpe
    }

    #[test]
    fn test_max_drawdown() {
        let returns = array![0.1, 0.1, -0.5, 0.1, 0.1];
        let max_dd = max_drawdown(&returns);
        assert!(max_dd > 0.0);
        assert!(max_dd < 1.0);
    }

    #[test]
    fn test_var() {
        let returns = array![-0.05, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05];
        let var = var_historical(&returns, 0.95);
        assert!(var < 0.0); // VaR should be negative (loss)
    }

    #[test]
    fn test_beta() {
        // Portfolio perfectly correlated with benchmark
        let portfolio = array![0.01, 0.02, -0.01, 0.015];
        let benchmark = array![0.01, 0.02, -0.01, 0.015];
        let b = beta(&portfolio, &benchmark);
        assert!((b - 1.0).abs() < 0.1);
    }
}
