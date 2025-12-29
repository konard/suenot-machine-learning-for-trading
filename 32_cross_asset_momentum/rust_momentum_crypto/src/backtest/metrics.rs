//! Метрики производительности
//!
//! Этот модуль содержит функции для расчёта различных метрик
//! производительности торговой стратегии.

use super::engine::PortfolioSnapshot;

/// Рассчитать простую доходность
pub fn simple_return(initial: f64, final_val: f64) -> f64 {
    (final_val - initial) / initial
}

/// Рассчитать логарифмическую доходность
pub fn log_return(initial: f64, final_val: f64) -> f64 {
    (final_val / initial).ln()
}

/// Рассчитать CAGR (Compound Annual Growth Rate)
pub fn cagr(initial: f64, final_val: f64, years: f64) -> f64 {
    if years <= 0.0 || initial <= 0.0 {
        return 0.0;
    }
    (final_val / initial).powf(1.0 / years) - 1.0
}

/// Рассчитать волатильность (стандартное отклонение доходностей)
pub fn volatility(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;

    variance.sqrt()
}

/// Рассчитать аннуализированную волатильность
pub fn annualized_volatility(returns: &[f64], periods_per_year: f64) -> f64 {
    volatility(returns) * periods_per_year.sqrt()
}

/// Рассчитать Sharpe Ratio
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean_return = returns.iter().sum::<f64>() / n;
    let annualized_return = mean_return * periods_per_year;

    let vol = annualized_volatility(returns, periods_per_year);

    if vol == 0.0 {
        return 0.0;
    }

    (annualized_return - risk_free_rate) / vol
}

/// Рассчитать Sortino Ratio
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean_return = returns.iter().sum::<f64>() / n;
    let annualized_return = mean_return * periods_per_year;

    // Downside deviation
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    if downside_returns.is_empty() {
        return f64::INFINITY; // Нет отрицательных доходностей
    }

    let sum_sq: f64 = downside_returns.iter().map(|r| r.powi(2)).sum();
    let downside_deviation = (sum_sq / downside_returns.len() as f64).sqrt() * periods_per_year.sqrt();

    if downside_deviation == 0.0 {
        return 0.0;
    }

    (annualized_return - risk_free_rate) / downside_deviation
}

/// Рассчитать максимальную просадку
pub fn max_drawdown(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut max_value = values[0];
    let mut max_dd = 0.0;

    for &value in values {
        max_value = max_value.max(value);
        let dd = (max_value - value) / max_value;
        max_dd = max_dd.max(dd);
    }

    max_dd
}

/// Рассчитать Calmar Ratio
pub fn calmar_ratio(cagr_val: f64, max_dd: f64) -> f64 {
    if max_dd == 0.0 {
        return 0.0;
    }
    cagr_val / max_dd
}

/// Рассчитать Win Rate
pub fn win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let wins = returns.iter().filter(|&&r| r > 0.0).count() as f64;
    wins / returns.len() as f64
}

/// Рассчитать Profit Factor
pub fn profit_factor(returns: &[f64]) -> f64 {
    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if gross_loss == 0.0 {
        return f64::INFINITY;
    }

    gross_profit / gross_loss
}

/// Рассчитать среднюю выигрышную сделку
pub fn avg_win(returns: &[f64]) -> f64 {
    let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
    if wins.is_empty() {
        return 0.0;
    }
    wins.iter().sum::<f64>() / wins.len() as f64
}

/// Рассчитать среднюю проигрышную сделку
pub fn avg_loss(returns: &[f64]) -> f64 {
    let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    if losses.is_empty() {
        return 0.0;
    }
    losses.iter().sum::<f64>() / losses.len() as f64
}

/// Рассчитать Kelly Criterion
pub fn kelly_criterion(win_rate_val: f64, avg_win_val: f64, avg_loss_val: f64) -> f64 {
    if avg_loss_val == 0.0 {
        return 0.0;
    }

    let win_loss_ratio = avg_win_val / avg_loss_val.abs();
    let loss_rate = 1.0 - win_rate_val;

    win_rate_val - (loss_rate / win_loss_ratio)
}

/// Рассчитать VaR (Value at Risk)
pub fn var(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    let index = index.min(sorted.len() - 1);

    -sorted[index]
}

/// Рассчитать CVaR (Conditional Value at Risk / Expected Shortfall)
pub fn cvar(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let var_val = var(returns, confidence);

    let tail_returns: Vec<f64> = returns.iter().filter(|&&r| -r >= var_val).copied().collect();

    if tail_returns.is_empty() {
        return var_val;
    }

    -(tail_returns.iter().sum::<f64>() / tail_returns.len() as f64)
}

/// Рассчитать Information Ratio
pub fn information_ratio(
    strategy_returns: &[f64],
    benchmark_returns: &[f64],
    periods_per_year: f64,
) -> f64 {
    if strategy_returns.len() != benchmark_returns.len() || strategy_returns.is_empty() {
        return 0.0;
    }

    let active_returns: Vec<f64> = strategy_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(s, b)| s - b)
        .collect();

    let mean_active = active_returns.iter().sum::<f64>() / active_returns.len() as f64;
    let annualized_active = mean_active * periods_per_year;

    let tracking_error = annualized_volatility(&active_returns, periods_per_year);

    if tracking_error == 0.0 {
        return 0.0;
    }

    annualized_active / tracking_error
}

/// Рассчитать все метрики для портфеля
pub fn calculate_all_metrics(snapshots: &[PortfolioSnapshot], periods_per_year: f64) -> PerformanceMetrics {
    if snapshots.is_empty() {
        return PerformanceMetrics::default();
    }

    let values: Vec<f64> = snapshots.iter().map(|s| s.value).collect();
    let returns: Vec<f64> = values.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    let initial = values[0];
    let final_val = *values.last().unwrap();
    let total_return = simple_return(initial, final_val);

    let first_date = snapshots.first().unwrap().timestamp;
    let last_date = snapshots.last().unwrap().timestamp;
    let years = (last_date - first_date).num_days() as f64 / 365.0;

    let cagr_val = cagr(initial, final_val, years);
    let vol = annualized_volatility(&returns, periods_per_year);
    let max_dd = max_drawdown(&values);

    PerformanceMetrics {
        total_return,
        cagr: cagr_val,
        volatility: vol,
        sharpe_ratio: sharpe_ratio(&returns, 0.0, periods_per_year),
        sortino_ratio: sortino_ratio(&returns, 0.0, periods_per_year),
        max_drawdown: max_dd,
        calmar_ratio: calmar_ratio(cagr_val, max_dd),
        win_rate: win_rate(&returns),
        profit_factor: profit_factor(&returns),
        avg_win: avg_win(&returns),
        avg_loss: avg_loss(&returns),
        var_95: var(&returns, 0.95),
        cvar_95: cvar(&returns, 0.95),
    }
}

/// Структура с метриками производительности
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub cagr: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub var_95: f64,
    pub cvar_95: f64,
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Metrics")?;
        writeln!(f, "===================")?;
        writeln!(f, "Total Return:    {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "CAGR:            {:.2}%", self.cagr * 100.0)?;
        writeln!(f, "Volatility:      {:.2}%", self.volatility * 100.0)?;
        writeln!(f, "Sharpe Ratio:    {:.2}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio:   {:.2}", self.sortino_ratio)?;
        writeln!(f, "Max Drawdown:    {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Calmar Ratio:    {:.2}", self.calmar_ratio)?;
        writeln!(f, "Win Rate:        {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Profit Factor:   {:.2}", self.profit_factor)?;
        writeln!(f, "VaR (95%):       {:.2}%", self.var_95 * 100.0)?;
        writeln!(f, "CVaR (95%):      {:.2}%", self.cvar_95 * 100.0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_return() {
        assert!((simple_return(100.0, 120.0) - 0.2).abs() < 1e-10);
        assert!((simple_return(100.0, 80.0) - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_volatility() {
        let returns = vec![0.01, -0.01, 0.01, -0.01, 0.01];
        let vol = volatility(&returns);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let values = vec![100.0, 110.0, 90.0, 95.0, 80.0, 100.0];
        let dd = max_drawdown(&values);
        // Максимум был 110, минимум после него 80
        // DD = (110 - 80) / 110 = 0.2727...
        assert!((dd - 0.2727).abs() < 0.01);
    }

    #[test]
    fn test_sharpe_ratio() {
        // Положительные доходности
        let returns = vec![0.01, 0.02, 0.015, 0.012, 0.018];
        let sr = sharpe_ratio(&returns, 0.0, 365.0);
        assert!(sr > 0.0);
    }

    #[test]
    fn test_win_rate() {
        let returns = vec![0.01, -0.01, 0.02, -0.005, 0.015];
        let wr = win_rate(&returns);
        assert!((wr - 0.6).abs() < 1e-10); // 3 из 5 положительные
    }

    #[test]
    fn test_var() {
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.02, -0.02, 0.04, -0.04];
        let var_95 = var(&returns, 0.95);
        assert!(var_95 > 0.0);
    }
}
