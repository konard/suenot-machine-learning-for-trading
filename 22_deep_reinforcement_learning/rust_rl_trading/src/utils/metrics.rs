//! Financial metrics and performance calculations.

/// Calculate Sharpe Ratio
///
/// # Arguments
/// * `returns` - Vector of periodic returns
/// * `risk_free_rate` - Annual risk-free rate (default 0)
/// * `periods_per_year` - Number of periods per year (252 for daily, 365*24 for hourly)
pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean_return = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    // Annualize
    let annual_return = mean_return * periods_per_year;
    let annual_std = std_dev * periods_per_year.sqrt();

    (annual_return - risk_free_rate) / annual_std
}

/// Calculate Maximum Drawdown
///
/// # Arguments
/// * `nav_series` - Vector of Net Asset Values over time
pub fn calculate_max_drawdown(nav_series: &[f64]) -> f64 {
    if nav_series.is_empty() {
        return 0.0;
    }

    let mut max_drawdown = 0.0;
    let mut peak = nav_series[0];

    for &nav in nav_series {
        peak = peak.max(nav);
        let drawdown = (peak - nav) / peak;
        max_drawdown = max_drawdown.max(drawdown);
    }

    max_drawdown
}

/// Calculate Sortino Ratio (uses downside deviation instead of standard deviation)
///
/// # Arguments
/// * `returns` - Vector of periodic returns
/// * `target_return` - Minimum acceptable return (default 0)
/// * `periods_per_year` - Number of periods per year
pub fn calculate_sortino(returns: &[f64], target_return: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean_return = returns.iter().sum::<f64>() / n;

    // Calculate downside deviation
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < target_return)
        .map(|&r| (r - target_return).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return f64::INFINITY; // No downside risk
    }

    let downside_variance = downside_returns.iter().sum::<f64>() / n;
    let downside_std = downside_variance.sqrt();

    if downside_std == 0.0 {
        return f64::INFINITY;
    }

    // Annualize
    let annual_return = mean_return * periods_per_year;
    let annual_target = target_return * periods_per_year;
    let annual_downside_std = downside_std * periods_per_year.sqrt();

    (annual_return - annual_target) / annual_downside_std
}

/// Calculate Calmar Ratio (return / max drawdown)
///
/// # Arguments
/// * `returns` - Vector of periodic returns
/// * `nav_series` - Vector of Net Asset Values
/// * `periods_per_year` - Number of periods per year
pub fn calculate_calmar(returns: &[f64], nav_series: &[f64], periods_per_year: f64) -> f64 {
    let max_dd = calculate_max_drawdown(nav_series);
    if max_dd == 0.0 {
        return f64::INFINITY;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let annual_return = mean_return * periods_per_year;

    annual_return / max_dd
}

/// Calculate win rate (percentage of positive returns)
pub fn calculate_win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    wins as f64 / returns.len() as f64
}

/// Calculate profit factor (sum of gains / sum of losses)
pub fn calculate_profit_factor(returns: &[f64]) -> f64 {
    let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if losses == 0.0 {
        return f64::INFINITY;
    }

    gains / losses
}

/// Calculate cumulative return
pub fn calculate_cumulative_return(nav_series: &[f64]) -> f64 {
    if nav_series.len() < 2 {
        return 0.0;
    }

    let first = nav_series.first().unwrap();
    let last = nav_series.last().unwrap();

    if *first == 0.0 {
        return 0.0;
    }

    (last - first) / first
}

/// Performance metrics summary
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub num_trades: usize,
}

impl PerformanceMetrics {
    /// Calculate all metrics from returns and NAV series
    pub fn from_data(returns: &[f64], nav_series: &[f64], periods_per_year: f64) -> Self {
        Self {
            total_return: calculate_cumulative_return(nav_series),
            sharpe_ratio: calculate_sharpe(returns, 0.0, periods_per_year),
            sortino_ratio: calculate_sortino(returns, 0.0, periods_per_year),
            max_drawdown: calculate_max_drawdown(nav_series),
            calmar_ratio: calculate_calmar(returns, nav_series, periods_per_year),
            win_rate: calculate_win_rate(returns),
            profit_factor: calculate_profit_factor(returns),
            num_trades: returns.len(),
        }
    }
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Metrics:")?;
        writeln!(f, "  Total Return:   {:>10.2}%", self.total_return * 100.0)?;
        writeln!(f, "  Sharpe Ratio:   {:>10.2}", self.sharpe_ratio)?;
        writeln!(f, "  Sortino Ratio:  {:>10.2}", self.sortino_ratio)?;
        writeln!(f, "  Max Drawdown:   {:>10.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Calmar Ratio:   {:>10.2}", self.calmar_ratio)?;
        writeln!(f, "  Win Rate:       {:>10.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  Profit Factor:  {:>10.2}", self.profit_factor)?;
        writeln!(f, "  Num Trades:     {:>10}", self.num_trades)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01, 0.02];
        let sharpe = calculate_sharpe(&returns, 0.0, 252.0);
        // Should be a reasonable number
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_max_drawdown() {
        let nav = vec![100.0, 110.0, 105.0, 95.0, 100.0, 90.0, 95.0];
        let max_dd = calculate_max_drawdown(&nav);
        // Max drawdown should be (110 - 90) / 110 = 0.1818...
        assert!((max_dd - 0.1818).abs() < 0.01);
    }

    #[test]
    fn test_win_rate() {
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01, 0.02];
        let win_rate = calculate_win_rate(&returns);
        // 4 wins out of 6 = 0.666...
        assert!((win_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cumulative_return() {
        let nav = vec![100.0, 110.0, 105.0, 120.0];
        let total_return = calculate_cumulative_return(&nav);
        // (120 - 100) / 100 = 0.2
        assert!((total_return - 0.2).abs() < 0.001);
    }
}
