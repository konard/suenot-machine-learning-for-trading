//! Performance Metrics

/// Calculate Sharpe Ratio
///
/// # Arguments
/// * `returns` - Series of returns
/// * `risk_free_rate` - Risk-free rate (annualized)
/// * `periods_per_year` - Number of periods per year (e.g., 252 for daily)
pub fn calculate_sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean_return - risk_free_rate / periods_per_year;

    let variance = if returns.len() > 1 {
        returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64
    } else {
        return 0.0;
    };

    let std_dev = variance.sqrt();
    if std_dev == 0.0 {
        return 0.0;
    }

    excess_return / std_dev * periods_per_year.sqrt()
}

/// Calculate Sortino Ratio (uses downside deviation)
///
/// # Arguments
/// * `returns` - Series of returns
/// * `risk_free_rate` - Risk-free rate (annualized)
/// * `periods_per_year` - Number of periods per year
pub fn calculate_sortino_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean_return - risk_free_rate / periods_per_year;

    // Calculate downside deviation
    let target = risk_free_rate / periods_per_year;
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < target)
        .map(|&r| (r - target).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return if excess_return > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
    }

    let downside_variance = downside_returns.iter().sum::<f64>() / downside_returns.len() as f64;
    let downside_dev = downside_variance.sqrt();

    if downside_dev == 0.0 {
        return 0.0;
    }

    excess_return / downside_dev * periods_per_year.sqrt()
}

/// Calculate Maximum Drawdown
///
/// # Arguments
/// * `equity_curve` - Series of equity values
pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut peak = equity_curve[0];
    let mut max_dd = 0.0;

    for &equity in equity_curve {
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Calculate Win Rate
///
/// # Arguments
/// * `pnls` - Series of trade P&Ls
pub fn calculate_win_rate(pnls: &[f64]) -> f64 {
    if pnls.is_empty() {
        return 0.0;
    }

    let wins = pnls.iter().filter(|&&p| p > 0.0).count();
    wins as f64 / pnls.len() as f64
}

/// Calculate Profit Factor
///
/// # Arguments
/// * `pnls` - Series of trade P&Ls
pub fn calculate_profit_factor(pnls: &[f64]) -> f64 {
    let gross_profit: f64 = pnls.iter().filter(|&&p| p > 0.0).sum();
    let gross_loss: f64 = pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();

    if gross_loss == 0.0 {
        if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        gross_profit / gross_loss
    }
}

/// Calculate Calmar Ratio (annualized return / max drawdown)
///
/// # Arguments
/// * `annualized_return` - Annualized return
/// * `max_drawdown` - Maximum drawdown
pub fn calculate_calmar_ratio(annualized_return: f64, max_drawdown: f64) -> f64 {
    if max_drawdown == 0.0 {
        if annualized_return > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        annualized_return / max_drawdown
    }
}

/// Calculate Information Ratio
///
/// # Arguments
/// * `returns` - Portfolio returns
/// * `benchmark_returns` - Benchmark returns
/// * `periods_per_year` - Number of periods per year
pub fn calculate_information_ratio(
    returns: &[f64],
    benchmark_returns: &[f64],
    periods_per_year: f64,
) -> f64 {
    if returns.len() != benchmark_returns.len() || returns.is_empty() {
        return 0.0;
    }

    let excess_returns: Vec<f64> = returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(r, b)| r - b)
        .collect();

    let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;

    let variance = excess_returns
        .iter()
        .map(|r| (r - mean_excess).powi(2))
        .sum::<f64>()
        / (excess_returns.len() - 1) as f64;

    let tracking_error = variance.sqrt();

    if tracking_error == 0.0 {
        return 0.0;
    }

    mean_excess / tracking_error * periods_per_year.sqrt()
}

/// Calculate Value at Risk (VaR)
///
/// # Arguments
/// * `returns` - Series of returns
/// * `confidence` - Confidence level (e.g., 0.95 for 95%)
pub fn calculate_var(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
///
/// # Arguments
/// * `returns` - Series of returns
/// * `confidence` - Confidence level (e.g., 0.95 for 95%)
pub fn calculate_cvar(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((1.0 - confidence) * sorted.len() as f64).ceil() as usize;
    let tail: Vec<f64> = sorted[..idx.min(sorted.len())].to_vec();

    if tail.is_empty() {
        return sorted[0];
    }

    tail.iter().sum::<f64>() / tail.len() as f64
}

/// Calculate average true range
///
/// # Arguments
/// * `highs` - High prices
/// * `lows` - Low prices
/// * `closes` - Close prices
/// * `period` - Period for averaging
pub fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return 0.0;
    }

    let n = highs.len();
    let mut true_ranges = Vec::with_capacity(n);

    true_ranges.push(highs[0] - lows[0]);

    for i in 1..n {
        let hl = highs[i] - lows[i];
        let hc = (highs[i] - closes[i - 1]).abs();
        let lc = (lows[i] - closes[i - 1]).abs();
        true_ranges.push(hl.max(hc).max(lc));
    }

    // Simple average of last 'period' true ranges
    let start = n - period;
    true_ranges[start..].iter().sum::<f64>() / period as f64
}

/// Calculate directional accuracy
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `actuals` - Actual values
pub fn calculate_directional_accuracy(predictions: &[f64], actuals: &[f64]) -> f64 {
    if predictions.len() != actuals.len() || predictions.is_empty() {
        return 0.0;
    }

    let correct = predictions
        .iter()
        .zip(actuals.iter())
        .filter(|(p, a)| (**p > 0.0) == (**a > 0.0))
        .count();

    correct as f64 / predictions.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sharpe = calculate_sharpe_ratio(&returns, 0.02, 252.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_sortino_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sortino = calculate_sortino_ratio(&returns, 0.02, 252.0);
        assert!(sortino > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 115.0, 100.0, 120.0];
        let dd = calculate_max_drawdown(&equity);
        // Max DD is from 115 to 100 = 13.04%
        assert!((dd - 0.1304).abs() < 0.01);
    }

    #[test]
    fn test_win_rate() {
        let pnls = vec![100.0, -50.0, 75.0, -25.0, 60.0];
        let wr = calculate_win_rate(&pnls);
        assert!((wr - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_profit_factor() {
        let pnls = vec![100.0, -50.0, 75.0, -25.0, 60.0];
        let pf = calculate_profit_factor(&pnls);
        // Gross profit = 235, Gross loss = 75
        assert!((pf - 235.0 / 75.0).abs() < 0.001);
    }

    #[test]
    fn test_var() {
        let returns = vec![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, 0.01, -0.01];
        let var = calculate_var(&returns, 0.95);
        // Should be a negative value (worst returns)
        assert!(var < 0.0);
    }

    #[test]
    fn test_directional_accuracy() {
        let predictions = vec![0.01, -0.02, 0.01, -0.01, 0.02];
        let actuals = vec![0.02, -0.01, 0.005, -0.02, 0.01];
        let accuracy = calculate_directional_accuracy(&predictions, &actuals);
        assert!((accuracy - 1.0).abs() < 0.001);
    }
}
