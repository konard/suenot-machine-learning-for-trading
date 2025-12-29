//! Метрики качества исполнения

use serde::{Deserialize, Serialize};

/// Метрики исполнения
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Implementation shortfall (в долях от total value)
    pub implementation_shortfall: f64,
    /// Arrival price slippage (в долях)
    pub arrival_slippage: f64,
    /// VWAP slippage
    pub vwap_slippage: f64,
    /// Общая стоимость исполнения
    pub total_cost: f64,
    /// Средняя цена исполнения
    pub average_execution_price: f64,
    /// Цена при входе
    pub arrival_price: f64,
    /// VWAP рынка за период
    pub market_vwap: f64,
    /// Общий объём
    pub total_quantity: f64,
    /// Количество сделок
    pub num_trades: usize,
}

impl ExecutionMetrics {
    /// Создать метрики из результатов исполнения
    pub fn from_execution(
        quantities: &[f64],
        prices: &[f64],
        arrival_price: f64,
        market_vwap: f64,
    ) -> Self {
        if quantities.is_empty() || prices.is_empty() {
            return Self::default();
        }

        let total_quantity: f64 = quantities.iter().sum();
        let total_value: f64 = quantities.iter()
            .zip(prices.iter())
            .map(|(q, p)| q * p)
            .sum();

        let average_price = if total_quantity > 0.0 {
            total_value / total_quantity
        } else {
            arrival_price
        };

        let arrival_slippage = if arrival_price > 0.0 {
            (average_price - arrival_price) / arrival_price
        } else {
            0.0
        };

        let vwap_slippage = if market_vwap > 0.0 {
            (average_price - market_vwap) / market_vwap
        } else {
            0.0
        };

        let implementation_shortfall = if arrival_price > 0.0 && total_quantity > 0.0 {
            (average_price - arrival_price) * total_quantity / (arrival_price * total_quantity)
        } else {
            0.0
        };

        let total_cost = (average_price - arrival_price) * total_quantity;

        Self {
            implementation_shortfall,
            arrival_slippage,
            vwap_slippage,
            total_cost,
            average_execution_price: average_price,
            arrival_price,
            market_vwap,
            total_quantity,
            num_trades: quantities.len(),
        }
    }

    /// Сравнить с другими метриками (improvement в %)
    pub fn improvement_vs(&self, other: &ExecutionMetrics) -> f64 {
        if other.implementation_shortfall.abs() > 1e-10 {
            (other.implementation_shortfall - self.implementation_shortfall)
                / other.implementation_shortfall.abs()
                * 100.0
        } else {
            0.0
        }
    }
}

/// Статистика производительности агента
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Средний shortfall
    pub mean_shortfall: f64,
    /// Стандартное отклонение shortfall
    pub std_shortfall: f64,
    /// Максимальный shortfall
    pub max_shortfall: f64,
    /// Минимальный shortfall
    pub min_shortfall: f64,
    /// Sharpe ratio (shortfall)
    pub sharpe_ratio: f64,
    /// Средний reward
    pub mean_reward: f64,
    /// Количество эпизодов
    pub num_episodes: usize,
    /// Win rate (% эпизодов лучше baseline)
    pub win_rate: f64,
}

impl PerformanceStats {
    /// Рассчитать статистику из списка shortfalls
    pub fn from_shortfalls(shortfalls: &[f64], rewards: &[f64], baseline_shortfalls: &[f64]) -> Self {
        if shortfalls.is_empty() {
            return Self::default();
        }

        let n = shortfalls.len() as f64;

        let mean_shortfall = shortfalls.iter().sum::<f64>() / n;
        let variance = shortfalls.iter()
            .map(|s| (s - mean_shortfall).powi(2))
            .sum::<f64>() / n;
        let std_shortfall = variance.sqrt();

        let max_shortfall = shortfalls.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_shortfall = shortfalls.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let sharpe_ratio = if std_shortfall > 1e-10 {
            -mean_shortfall / std_shortfall // Negative because lower shortfall is better
        } else {
            0.0
        };

        let mean_reward = if rewards.is_empty() {
            0.0
        } else {
            rewards.iter().sum::<f64>() / rewards.len() as f64
        };

        let win_rate = if !baseline_shortfalls.is_empty() {
            let wins = shortfalls.iter()
                .zip(baseline_shortfalls.iter())
                .filter(|(s, b)| s < b)
                .count();
            wins as f64 / shortfalls.len() as f64
        } else {
            0.0
        };

        Self {
            mean_shortfall,
            std_shortfall,
            max_shortfall,
            min_shortfall,
            sharpe_ratio,
            mean_reward,
            num_episodes: shortfalls.len(),
            win_rate,
        }
    }

    /// Вывести отчёт
    pub fn report(&self) -> String {
        format!(
            "Performance Statistics:\n\
             ├── Episodes: {}\n\
             ├── Mean Shortfall: {:.6} ({:.4} bps)\n\
             ├── Std Shortfall: {:.6}\n\
             ├── Max/Min Shortfall: {:.6} / {:.6}\n\
             ├── Sharpe Ratio: {:.4}\n\
             ├── Mean Reward: {:.4}\n\
             └── Win Rate vs Baseline: {:.2}%",
            self.num_episodes,
            self.mean_shortfall,
            self.mean_shortfall * 10000.0,
            self.std_shortfall,
            self.max_shortfall,
            self.min_shortfall,
            self.sharpe_ratio,
            self.mean_reward,
            self.win_rate * 100.0,
        )
    }
}

/// Сравнение нескольких стратегий
#[derive(Debug, Clone, Default)]
pub struct StrategyComparison {
    pub strategy_names: Vec<String>,
    pub stats: Vec<PerformanceStats>,
}

impl StrategyComparison {
    /// Добавить стратегию
    pub fn add(&mut self, name: impl Into<String>, stats: PerformanceStats) {
        self.strategy_names.push(name.into());
        self.stats.push(stats);
    }

    /// Получить лучшую стратегию
    pub fn best_strategy(&self) -> Option<&str> {
        self.stats.iter()
            .zip(self.strategy_names.iter())
            .min_by(|(a, _), (b, _)| {
                a.mean_shortfall.partial_cmp(&b.mean_shortfall).unwrap()
            })
            .map(|(_, name)| name.as_str())
    }

    /// Вывести сравнительную таблицу
    pub fn report(&self) -> String {
        let mut lines = vec![
            "Strategy Comparison:".to_string(),
            format!("{:<20} {:>12} {:>12} {:>10} {:>10}",
                    "Strategy", "Mean IS (bps)", "Std IS", "Sharpe", "Win Rate"),
            "-".repeat(66),
        ];

        for (name, stats) in self.strategy_names.iter().zip(self.stats.iter()) {
            lines.push(format!(
                "{:<20} {:>12.4} {:>12.6} {:>10.4} {:>9.2}%",
                name,
                stats.mean_shortfall * 10000.0,
                stats.std_shortfall,
                stats.sharpe_ratio,
                stats.win_rate * 100.0,
            ));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_metrics() {
        let quantities = vec![100.0, 100.0, 100.0];
        let prices = vec![100.0, 101.0, 102.0];
        let arrival_price = 100.0;
        let market_vwap = 101.0;

        let metrics = ExecutionMetrics::from_execution(
            &quantities,
            &prices,
            arrival_price,
            market_vwap,
        );

        assert_eq!(metrics.num_trades, 3);
        assert!((metrics.average_execution_price - 101.0).abs() < 0.001);
        assert!((metrics.arrival_slippage - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_performance_stats() {
        let shortfalls = vec![0.001, 0.002, 0.0015, 0.0025];
        let rewards = vec![-1.0, -2.0, -1.5, -2.5];
        let baseline = vec![0.002, 0.003, 0.002, 0.003];

        let stats = PerformanceStats::from_shortfalls(&shortfalls, &rewards, &baseline);

        assert_eq!(stats.num_episodes, 4);
        assert!(stats.win_rate > 0.5);
    }
}
