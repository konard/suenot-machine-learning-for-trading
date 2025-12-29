//! Метрики производительности бэктеста

use crate::strategies::Trade;

/// Результат бэктеста
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Начальный капитал
    pub initial_equity: f64,
    /// Конечный капитал
    pub final_equity: f64,
    /// Список сделок
    pub trades: Vec<Trade>,
    /// Кривая эквити
    pub equity_curve: Vec<f64>,
    /// Максимальная просадка
    pub max_drawdown: f64,
}

impl BacktestResult {
    /// Создать результат бэктеста
    pub fn new(
        initial_equity: f64,
        final_equity: f64,
        trades: Vec<Trade>,
        equity_curve: Vec<f64>,
        max_drawdown: f64,
    ) -> Self {
        Self {
            initial_equity,
            final_equity,
            trades,
            equity_curve,
            max_drawdown,
        }
    }

    /// Пустой результат
    pub fn empty(initial_equity: f64) -> Self {
        Self {
            initial_equity,
            final_equity: initial_equity,
            trades: vec![],
            equity_curve: vec![initial_equity],
            max_drawdown: 0.0,
        }
    }

    /// Общая доходность
    pub fn total_return(&self) -> f64 {
        (self.final_equity - self.initial_equity) / self.initial_equity
    }

    /// Общая доходность в процентах
    pub fn total_return_percent(&self) -> f64 {
        self.total_return() * 100.0
    }

    /// Количество сделок
    pub fn total_trades(&self) -> usize {
        self.trades.len()
    }

    /// Количество прибыльных сделок
    pub fn winning_trades(&self) -> usize {
        self.trades.iter().filter(|t| t.is_profitable()).count()
    }

    /// Количество убыточных сделок
    pub fn losing_trades(&self) -> usize {
        self.trades.iter().filter(|t| !t.is_profitable()).count()
    }

    /// Процент прибыльных сделок
    pub fn win_rate(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        self.winning_trades() as f64 / self.trades.len() as f64 * 100.0
    }

    /// Средняя прибыль по прибыльным сделкам
    pub fn average_win(&self) -> f64 {
        let wins: Vec<_> = self.trades.iter().filter(|t| t.is_profitable()).collect();
        if wins.is_empty() {
            return 0.0;
        }
        wins.iter().map(|t| t.pnl).sum::<f64>() / wins.len() as f64
    }

    /// Средний убыток по убыточным сделкам
    pub fn average_loss(&self) -> f64 {
        let losses: Vec<_> = self.trades.iter().filter(|t| !t.is_profitable()).collect();
        if losses.is_empty() {
            return 0.0;
        }
        losses.iter().map(|t| t.pnl.abs()).sum::<f64>() / losses.len() as f64
    }

    /// Profit Factor (отношение прибыли к убыткам)
    pub fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self
            .trades
            .iter()
            .filter(|t| t.is_profitable())
            .map(|t| t.pnl)
            .sum();
        let gross_loss: f64 = self
            .trades
            .iter()
            .filter(|t| !t.is_profitable())
            .map(|t| t.pnl.abs())
            .sum();

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

    /// Risk/Reward Ratio
    pub fn risk_reward_ratio(&self) -> f64 {
        let avg_loss = self.average_loss();
        if avg_loss == 0.0 {
            return 0.0;
        }
        self.average_win() / avg_loss
    }

    /// Expectancy (математическое ожидание сделки)
    pub fn expectancy(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        let win_rate = self.win_rate() / 100.0;
        let avg_win = self.average_win();
        let avg_loss = self.average_loss();

        win_rate * avg_win - (1.0 - win_rate) * avg_loss
    }

    /// Коэффициент Шарпа (упрощённый, годовой)
    pub fn sharpe_ratio(&self, risk_free_rate: f64, periods_per_year: f64) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        // Рассчитываем доходности
        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Годовая доходность и волатильность
        let annual_return = mean_return * periods_per_year;
        let annual_std = std_dev * periods_per_year.sqrt();

        (annual_return - risk_free_rate) / annual_std
    }

    /// Коэффициент Сортино (использует только отрицательные отклонения)
    pub fn sortino_ratio(&self, risk_free_rate: f64, periods_per_year: f64) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        // Только отрицательные отклонения
        let downside_variance: f64 = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std == 0.0 {
            return 0.0;
        }

        let annual_return = mean_return * periods_per_year;
        let annual_downside_std = downside_std * periods_per_year.sqrt();

        (annual_return - risk_free_rate) / annual_downside_std
    }

    /// Calmar Ratio (доходность / максимальная просадка)
    pub fn calmar_ratio(&self) -> f64 {
        if self.max_drawdown == 0.0 {
            return 0.0;
        }
        self.total_return() / self.max_drawdown
    }

    /// Получить метрики производительности
    pub fn performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            total_return: self.total_return_percent(),
            total_trades: self.total_trades(),
            winning_trades: self.winning_trades(),
            losing_trades: self.losing_trades(),
            win_rate: self.win_rate(),
            profit_factor: self.profit_factor(),
            max_drawdown: self.max_drawdown * 100.0,
            sharpe_ratio: self.sharpe_ratio(0.0, 365.0),
            sortino_ratio: self.sortino_ratio(0.0, 365.0),
            calmar_ratio: self.calmar_ratio(),
            expectancy: self.expectancy(),
            risk_reward_ratio: self.risk_reward_ratio(),
        }
    }

    /// Вывести отчёт
    pub fn print_report(&self) {
        let metrics = self.performance_metrics();

        println!("════════════════════════════════════════════");
        println!("           BACKTEST REPORT");
        println!("════════════════════════════════════════════");
        println!();
        println!("CAPITAL");
        println!("  Initial:          ${:.2}", self.initial_equity);
        println!("  Final:            ${:.2}", self.final_equity);
        println!("  Total Return:     {:.2}%", metrics.total_return);
        println!();
        println!("TRADES");
        println!("  Total:            {}", metrics.total_trades);
        println!("  Winning:          {}", metrics.winning_trades);
        println!("  Losing:           {}", metrics.losing_trades);
        println!("  Win Rate:         {:.2}%", metrics.win_rate);
        println!();
        println!("RISK METRICS");
        println!("  Max Drawdown:     {:.2}%", metrics.max_drawdown);
        println!("  Profit Factor:    {:.2}", metrics.profit_factor);
        println!("  Risk/Reward:      {:.2}", metrics.risk_reward_ratio);
        println!("  Expectancy:       ${:.2}", metrics.expectancy);
        println!();
        println!("PERFORMANCE RATIOS");
        println!("  Sharpe Ratio:     {:.2}", metrics.sharpe_ratio);
        println!("  Sortino Ratio:    {:.2}", metrics.sortino_ratio);
        println!("  Calmar Ratio:     {:.2}", metrics.calmar_ratio);
        println!("════════════════════════════════════════════");
    }
}

/// Структура с метриками производительности
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub expectancy: f64,
    pub risk_reward_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mock_trades() -> Vec<Trade> {
        vec![
            Trade::new(1000, 100.0, 2000, 110.0, true, 1.0, 0.001), // +10%
            Trade::new(2000, 110.0, 3000, 105.0, true, 1.0, 0.001), // -4.5%
            Trade::new(3000, 105.0, 4000, 115.0, true, 1.0, 0.001), // +9.5%
        ]
    }

    #[test]
    fn test_win_rate() {
        let trades = create_mock_trades();
        let result = BacktestResult::new(10000.0, 11500.0, trades, vec![10000.0, 11500.0], 0.05);

        // 2 из 3 сделок прибыльные
        assert!((result.win_rate() - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_profit_factor() {
        let trades = create_mock_trades();
        let result = BacktestResult::new(10000.0, 11500.0, trades, vec![10000.0, 11500.0], 0.05);

        // Profit factor должен быть > 1 для прибыльной стратегии
        assert!(result.profit_factor() > 1.0);
    }

    #[test]
    fn test_empty_result() {
        let result = BacktestResult::empty(10000.0);

        assert_eq!(result.total_trades(), 0);
        assert_eq!(result.win_rate(), 0.0);
        assert_eq!(result.total_return(), 0.0);
    }
}
