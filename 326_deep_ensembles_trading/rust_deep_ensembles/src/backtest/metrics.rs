//! Trading metrics calculation

/// Calculate Sharpe ratio
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, annualization: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let daily_rf = risk_free_rate / annualization;

    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std > 0.0 {
        (mean - daily_rf) * annualization.sqrt() / std
    } else {
        0.0
    }
}

/// Calculate Sortino ratio
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, annualization: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let daily_rf = risk_free_rate / annualization;

    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

    if downside_returns.is_empty() {
        return f64::INFINITY;
    }

    let downside_variance: f64 = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
        / downside_returns.len() as f64;
    let downside_std = downside_variance.sqrt();

    if downside_std > 0.0 {
        (mean - daily_rf) * annualization.sqrt() / downside_std
    } else {
        0.0
    }
}

/// Calculate maximum drawdown
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
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

/// Calculate win rate
pub fn win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    wins as f64 / returns.len() as f64
}

/// Calculate profit factor
pub fn profit_factor(returns: &[f64]) -> f64 {
    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

/// Calculate total return
pub fn total_return(initial: f64, final_value: f64) -> f64 {
    if initial > 0.0 {
        (final_value - initial) / initial
    } else {
        0.0
    }
}

/// Calculate CAGR (Compound Annual Growth Rate)
pub fn cagr(initial: f64, final_value: f64, years: f64) -> f64 {
    if initial > 0.0 && years > 0.0 {
        (final_value / initial).powf(1.0 / years) - 1.0
    } else {
        0.0
    }
}

/// Calculate volatility (annualized standard deviation)
pub fn volatility(returns: &[f64], annualization: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;

    variance.sqrt() * annualization.sqrt()
}

/// Trading metrics collection
#[derive(Debug, Clone)]
pub struct TradingMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub volatility: f64,
    pub num_trades: usize,
}

impl TradingMetrics {
    /// Compute all metrics from returns and equity curve
    pub fn compute(
        returns: &[f64],
        equity_curve: &[f64],
        initial_capital: f64,
        final_capital: f64,
        num_trades: usize,
        risk_free_rate: f64,
        annualization: f64,
    ) -> Self {
        Self {
            total_return: total_return(initial_capital, final_capital),
            sharpe_ratio: sharpe_ratio(returns, risk_free_rate, annualization),
            sortino_ratio: sortino_ratio(returns, risk_free_rate, annualization),
            max_drawdown: max_drawdown(equity_curve),
            win_rate: win_rate(returns),
            profit_factor: profit_factor(returns),
            volatility: volatility(returns, annualization),
            num_trades,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sharpe = sharpe_ratio(&returns, 0.02, 252.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 95.0, 100.0];
        let dd = max_drawdown(&equity);
        // Max drawdown should be from 110 to 95 = 13.6%
        assert!((dd - 0.136).abs() < 0.01);
    }

    #[test]
    fn test_win_rate() {
        let returns = vec![0.01, -0.01, 0.02, 0.005, -0.005];
        let wr = win_rate(&returns);
        assert!((wr - 0.6).abs() < 0.001); // 3 wins out of 5
    }

    #[test]
    fn test_profit_factor() {
        let returns = vec![0.10, -0.05, 0.08, -0.03];
        let pf = profit_factor(&returns);
        // Profit = 0.18, Loss = 0.08
        assert!((pf - 2.25).abs() < 0.001);
    }
}
