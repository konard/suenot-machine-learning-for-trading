//! Trading performance metrics

use serde::{Deserialize, Serialize};

/// Compute returns from a price series
pub fn compute_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Compute cumulative returns
pub fn compute_cumulative_returns(returns: &[f64]) -> Vec<f64> {
    let mut cumulative = Vec::with_capacity(returns.len());
    let mut cum = 1.0;

    for &r in returns {
        cum *= 1.0 + r;
        cumulative.push(cum - 1.0);
    }

    cumulative
}

/// Compute Sharpe ratio
///
/// # Arguments
/// * `returns` - Vector of returns
/// * `risk_free_rate` - Risk-free rate (annualized)
/// * `periods_per_year` - Number of periods per year (252 for daily, 365*24 for hourly)
pub fn compute_sharpe(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    let excess_return = mean_return - risk_free_rate / periods_per_year;
    let annualized_return = excess_return * periods_per_year;
    let annualized_vol = std_dev * periods_per_year.sqrt();

    annualized_return / annualized_vol
}

/// Compute Sortino ratio (uses only downside deviation)
pub fn compute_sortino(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

    // Downside deviation
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < 0.0)
        .copied()
        .collect();

    if downside_returns.is_empty() {
        return f64::INFINITY; // No negative returns
    }

    let downside_variance = downside_returns
        .iter()
        .map(|r| r.powi(2))
        .sum::<f64>()
        / downside_returns.len() as f64;
    let downside_dev = downside_variance.sqrt();

    if downside_dev == 0.0 {
        return 0.0;
    }

    let excess_return = mean_return - risk_free_rate / periods_per_year;
    let annualized_return = excess_return * periods_per_year;
    let annualized_downside = downside_dev * periods_per_year.sqrt();

    annualized_return / annualized_downside
}

/// Compute maximum drawdown
pub fn compute_max_drawdown(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }

    let mut max_price = prices[0];
    let mut max_drawdown = 0.0;

    for &price in prices {
        if price > max_price {
            max_price = price;
        }
        let drawdown = (max_price - price) / max_price;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    max_drawdown
}

/// Compute maximum drawdown from equity curve
pub fn compute_max_drawdown_equity(equity: &[f64]) -> (f64, usize, usize) {
    if equity.is_empty() {
        return (0.0, 0, 0);
    }

    let mut max_equity = equity[0];
    let mut max_equity_idx = 0;
    let mut max_drawdown = 0.0;
    let mut drawdown_start = 0;
    let mut drawdown_end = 0;

    for (i, &eq) in equity.iter().enumerate() {
        if eq > max_equity {
            max_equity = eq;
            max_equity_idx = i;
        }

        let drawdown = (max_equity - eq) / max_equity;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
            drawdown_start = max_equity_idx;
            drawdown_end = i;
        }
    }

    (max_drawdown, drawdown_start, drawdown_end)
}

/// Compute Calmar ratio (annualized return / max drawdown)
pub fn compute_calmar(returns: &[f64], prices: &[f64], periods_per_year: f64) -> f64 {
    let max_dd = compute_max_drawdown(prices);
    if max_dd == 0.0 {
        return 0.0;
    }

    let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
    let n_periods = returns.len() as f64;
    let annualized_return = (1.0 + total_return).powf(periods_per_year / n_periods) - 1.0;

    annualized_return / max_dd
}

/// Compute win rate
pub fn compute_win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    wins as f64 / returns.len() as f64
}

/// Compute profit factor
pub fn compute_profit_factor(returns: &[f64]) -> f64 {
    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if gross_loss == 0.0 {
        return f64::INFINITY;
    }

    gross_profit / gross_loss
}

/// Compute average win / average loss ratio
pub fn compute_win_loss_ratio(returns: &[f64]) -> f64 {
    let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
    let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    if wins.is_empty() || losses.is_empty() {
        return 0.0;
    }

    let avg_win = wins.iter().sum::<f64>() / wins.len() as f64;
    let avg_loss = losses.iter().map(|l| l.abs()).sum::<f64>() / losses.len() as f64;

    if avg_loss == 0.0 {
        return 0.0;
    }

    avg_win / avg_loss
}

/// Comprehensive trading metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradingMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub win_loss_ratio: f64,
    pub n_trades: usize,
    pub volatility: f64,
}

impl TradingMetrics {
    /// Compute all metrics from returns and prices
    pub fn from_returns(returns: &[f64], prices: &[f64], periods_per_year: f64) -> Self {
        let total_return = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
        let n_periods = returns.len().max(1) as f64;
        let annualized_return = (1.0 + total_return).powf(periods_per_year / n_periods) - 1.0;

        let mean_return = returns.iter().sum::<f64>() / n_periods;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / n_periods;
        let volatility = variance.sqrt() * periods_per_year.sqrt();

        Self {
            total_return,
            annualized_return,
            sharpe_ratio: compute_sharpe(returns, 0.0, periods_per_year),
            sortino_ratio: compute_sortino(returns, 0.0, periods_per_year),
            max_drawdown: compute_max_drawdown(prices),
            calmar_ratio: compute_calmar(returns, prices, periods_per_year),
            win_rate: compute_win_rate(returns),
            profit_factor: compute_profit_factor(returns),
            win_loss_ratio: compute_win_loss_ratio(returns),
            n_trades: returns.iter().filter(|&&r| r != 0.0).count(),
            volatility,
        }
    }

    /// Create from PnL series
    pub fn from_pnl(pnl: &[f64], initial_value: f64, periods_per_year: f64) -> Self {
        // Convert PnL to equity curve
        let mut equity = Vec::with_capacity(pnl.len() + 1);
        equity.push(initial_value);

        let mut current = initial_value;
        for &p in pnl {
            current += p;
            equity.push(current);
        }

        // Calculate returns
        let returns = compute_returns(&equity);

        Self::from_returns(&returns, &equity, periods_per_year)
    }
}

impl std::fmt::Display for TradingMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Return: {:.2}% | Sharpe: {:.2} | MaxDD: {:.2}% | WinRate: {:.1}%",
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_returns() {
        let prices = vec![100.0, 105.0, 102.0, 108.0];
        let returns = compute_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.05).abs() < 0.0001);
    }

    #[test]
    fn test_compute_sharpe() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sharpe = compute_sharpe(&returns, 0.0, 252.0);

        // Should be positive (positive average return)
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_compute_max_drawdown() {
        let prices = vec![100.0, 110.0, 105.0, 90.0, 95.0, 100.0];
        let max_dd = compute_max_drawdown(&prices);

        // Peak at 110, trough at 90 -> 18.18% drawdown
        assert!((max_dd - 0.1818).abs() < 0.01);
    }

    #[test]
    fn test_compute_win_rate() {
        let returns = vec![0.01, -0.005, 0.02, 0.015, -0.01];
        let win_rate = compute_win_rate(&returns);

        assert!((win_rate - 0.6).abs() < 0.01); // 3 wins out of 5
    }

    #[test]
    fn test_trading_metrics() {
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 * (1.0 + 0.001 * i as f64))
            .collect();
        let returns = compute_returns(&prices);

        let metrics = TradingMetrics::from_returns(&returns, &prices, 252.0);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.sharpe_ratio > 0.0);
        assert!(metrics.max_drawdown >= 0.0);

        println!("Metrics: {}", metrics);
    }
}
