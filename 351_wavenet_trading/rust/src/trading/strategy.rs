//! Торговые стратегии на основе WaveNet

use crate::types::{Candle, Signal, PerformanceMetrics};
use crate::models::WaveNet;
use crate::analysis::FeatureBuilder;
use super::signals::ThresholdSignalGenerator;

/// Конфигурация стратегии
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Порог для покупки (предсказанная доходность)
    pub buy_threshold: f64,
    /// Порог для продажи
    pub sell_threshold: f64,
    /// Размер позиции (доля от капитала)
    pub position_size: f64,
    /// Максимальная позиция
    pub max_position: f64,
    /// Стоп-лосс (доля)
    pub stop_loss: f64,
    /// Тейк-профит (доля)
    pub take_profit: f64,
    /// Комиссия за сделку
    pub commission: f64,
    /// Размер окна для WaveNet
    pub window_size: usize,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            buy_threshold: 0.001,   // 0.1%
            sell_threshold: -0.001,
            position_size: 0.1,     // 10% капитала
            max_position: 1.0,      // 100% максимум
            stop_loss: 0.02,        // 2% стоп-лосс
            take_profit: 0.04,      // 4% тейк-профит
            commission: 0.001,      // 0.1% комиссия
            window_size: 100,
        }
    }
}

/// Состояние позиции
#[derive(Debug, Clone)]
pub struct Position {
    pub size: f64,      // Положительный = long, отрицательный = short
    pub entry_price: f64,
    pub entry_time: usize,
}

/// Торговая стратегия на основе WaveNet
pub struct WaveNetStrategy {
    pub config: StrategyConfig,
    pub signal_generator: ThresholdSignalGenerator,
}

impl WaveNetStrategy {
    pub fn new(config: StrategyConfig) -> Self {
        let signal_generator = ThresholdSignalGenerator::new(
            config.buy_threshold,
            config.sell_threshold,
        );

        Self {
            config,
            signal_generator,
        }
    }

    /// Получить сигнал на основе предсказания модели
    pub fn get_signal(&self, predicted_return: f64) -> Signal {
        self.signal_generator.generate(predicted_return)
    }

    /// Рассчитать размер позиции
    pub fn calculate_position_size(
        &self,
        capital: f64,
        price: f64,
        volatility: f64,
    ) -> f64 {
        // Размер позиции, скорректированный на волатильность
        let vol_adjusted = if volatility > 0.0 {
            self.config.position_size / volatility.sqrt()
        } else {
            self.config.position_size
        };

        let position_value = capital * vol_adjusted.min(self.config.max_position);
        position_value / price
    }

    /// Проверить стоп-лосс
    pub fn check_stop_loss(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        if is_long {
            (entry_price - current_price) / entry_price > self.config.stop_loss
        } else {
            (current_price - entry_price) / entry_price > self.config.stop_loss
        }
    }

    /// Проверить тейк-профит
    pub fn check_take_profit(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        if is_long {
            (current_price - entry_price) / entry_price > self.config.take_profit
        } else {
            (entry_price - current_price) / entry_price > self.config.take_profit
        }
    }
}

/// Простая стратегия следования за сигналом
pub struct SimpleSignalStrategy {
    pub threshold: f64,
}

impl SimpleSignalStrategy {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Применить стратегию к предсказаниям
    pub fn apply(&self, predictions: &[f64]) -> Vec<f64> {
        predictions
            .iter()
            .map(|&p| {
                if p > self.threshold {
                    1.0
                } else if p < -self.threshold {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Вычислить доходности стратегии
    pub fn calculate_returns(&self, predictions: &[f64], actual_returns: &[f64]) -> Vec<f64> {
        let positions = self.apply(predictions);

        // Позиция на момент t влияет на доходность t+1
        positions
            .iter()
            .zip(actual_returns.iter().skip(1))
            .map(|(&pos, &ret)| pos * ret)
            .collect()
    }
}

/// Расчёт метрик производительности
pub fn calculate_metrics(returns: &[f64]) -> PerformanceMetrics {
    if returns.is_empty() {
        return PerformanceMetrics::default();
    }

    let n = returns.len() as f64;

    // Общая доходность
    let total_return = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;

    // Среднее и стандартное отклонение
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    // Sharpe Ratio (annualized, assuming daily returns)
    let sharpe_ratio = if std > 0.0 {
        (mean / std) * (252_f64).sqrt()
    } else {
        0.0
    };

    // Sortino Ratio (only downside deviation)
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    let downside_std = if !downside_returns.is_empty() {
        let downside_var = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
            / downside_returns.len() as f64;
        downside_var.sqrt()
    } else {
        1e-10
    };
    let sortino_ratio = (mean / downside_std) * (252_f64).sqrt();

    // Maximum Drawdown
    let mut peak = 0.0;
    let mut max_drawdown = 0.0;
    let mut cumulative = 0.0;

    for ret in returns {
        cumulative += ret;
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = peak - cumulative;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    // Win Rate
    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    let win_rate = wins as f64 / n;

    // Profit Factor
    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        f64::INFINITY
    };

    // Total trades (transitions from 0 position)
    let total_trades = returns.iter().filter(|&&r| r != 0.0).count();

    PerformanceMetrics {
        total_return,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        win_rate,
        profit_factor,
        total_trades,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_config() {
        let config = StrategyConfig::default();
        assert!(config.buy_threshold > 0.0);
        assert!(config.sell_threshold < 0.0);
    }

    #[test]
    fn test_simple_strategy() {
        let strategy = SimpleSignalStrategy::new(0.01);
        let predictions = vec![0.02, -0.02, 0.005, 0.03];
        let positions = strategy.apply(&predictions);

        assert_eq!(positions, vec![1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_calculate_metrics() {
        let returns = vec![0.01, -0.005, 0.02, 0.01, -0.01, 0.015];
        let metrics = calculate_metrics(&returns);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.win_rate > 0.0 && metrics.win_rate <= 1.0);
    }

    #[test]
    fn test_stop_loss() {
        let strategy = WaveNetStrategy::new(StrategyConfig::default());

        // Long position
        assert!(strategy.check_stop_loss(100.0, 97.0, true));  // 3% loss > 2% threshold
        assert!(!strategy.check_stop_loss(100.0, 99.0, true)); // 1% loss < 2% threshold

        // Short position
        assert!(strategy.check_stop_loss(100.0, 103.0, false));
        assert!(!strategy.check_stop_loss(100.0, 101.0, false));
    }
}
