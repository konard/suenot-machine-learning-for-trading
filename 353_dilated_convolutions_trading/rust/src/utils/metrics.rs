//! Performance Metrics

/// Calculate Sharpe ratio
///
/// # Arguments
/// - `returns` - Slice of returns
/// - `risk_free_rate` - Annual risk-free rate (default 0)
/// - `periods_per_year` - Number of periods per year (252 for daily, 365*24 for hourly)
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std < 1e-10 {
        return 0.0;
    }

    let excess_return = mean - risk_free_rate / periods_per_year;
    (excess_return / std) * periods_per_year.sqrt()
}

/// Calculate Sortino ratio (downside deviation)
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;

    // Downside deviation (only negative returns)
    let downside_variance: f64 = returns
        .iter()
        .filter(|&&r| r < 0.0)
        .map(|r| r.powi(2))
        .sum::<f64>() / n;
    let downside_std = downside_variance.sqrt();

    if downside_std < 1e-10 {
        return 0.0;
    }

    let excess_return = mean - risk_free_rate / periods_per_year;
    (excess_return / downside_std) * periods_per_year.sqrt()
}

/// Calculate maximum drawdown
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut max_equity = equity_curve[0];
    let mut max_dd = 0.0;

    for &equity in equity_curve {
        if equity > max_equity {
            max_equity = equity;
        }
        let dd = (max_equity - equity) / max_equity;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Calculate Calmar ratio (return / max drawdown)
pub fn calmar_ratio(total_return: f64, max_drawdown: f64) -> f64 {
    if max_drawdown < 1e-10 {
        return 0.0;
    }
    total_return / max_drawdown
}

/// Calculate win rate
pub fn win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    wins as f64 / returns.len() as f64
}

/// Calculate profit factor (sum of wins / sum of losses)
pub fn profit_factor(returns: &[f64]) -> f64 {
    let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if losses < 1e-10 {
        return f64::INFINITY;
    }
    gains / losses
}

/// Calculate average win and average loss
pub fn avg_win_loss(returns: &[f64]) -> (f64, f64) {
    let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).cloned().collect();
    let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).collect();

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

/// Calculate cumulative returns from simple returns
pub fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
    let mut cumulative = Vec::with_capacity(returns.len());
    let mut cum = 1.0;

    for &r in returns {
        cum *= 1.0 + r;
        cumulative.push(cum - 1.0);
    }

    cumulative
}

/// Calculate equity curve from returns (starting at 1.0)
pub fn equity_curve(returns: &[f64]) -> Vec<f64> {
    let mut equity = Vec::with_capacity(returns.len() + 1);
    equity.push(1.0);

    let mut current = 1.0;
    for &r in returns {
        current *= 1.0 + r;
        equity.push(current);
    }

    equity
}

/// Calculate Value at Risk (VaR) at given confidence level
pub fn value_at_risk(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((1.0 - confidence) * returns.len() as f64).floor() as usize;
    sorted[index.min(returns.len() - 1)]
}

/// Calculate Expected Shortfall (CVaR) at given confidence level
pub fn expected_shortfall(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let cutoff_index = ((1.0 - confidence) * returns.len() as f64).ceil() as usize;
    let tail: Vec<f64> = sorted.into_iter().take(cutoff_index.max(1)).collect();

    tail.iter().sum::<f64>() / tail.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sr = sharpe_ratio(&returns, 0.0, 252.0);
        assert!(sr > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![1.0, 1.1, 1.2, 1.0, 1.15, 1.25];
        let mdd = max_drawdown(&equity);
        // Max DD from 1.2 to 1.0 = 16.67%
        assert!((mdd - 0.1667).abs() < 0.01);
    }

    #[test]
    fn test_win_rate() {
        let returns = vec![0.01, -0.01, 0.02, 0.01, -0.02];
        let wr = win_rate(&returns);
        assert!((wr - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_profit_factor() {
        let returns = vec![0.02, -0.01, 0.02, -0.01];
        let pf = profit_factor(&returns);
        assert!((pf - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_equity_curve() {
        let returns = vec![0.1, -0.05, 0.1];
        let eq = equity_curve(&returns);
        assert_eq!(eq.len(), 4);
        assert!((eq[0] - 1.0).abs() < 1e-10);
        assert!((eq[3] - 1.1 * 0.95 * 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_var() {
        let returns = vec![-0.05, -0.02, 0.01, 0.02, 0.03, -0.01, 0.015, -0.03, 0.02, 0.01];
        let var95 = value_at_risk(&returns, 0.95);
        assert!(var95 < 0.0); // VaR should be negative (loss)
    }
}
