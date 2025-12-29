//! Модуль расчёта моментума
//!
//! Содержит функции для расчёта ценового и sentiment-моментума.

mod price;
mod sentiment;

pub use price::{calculate_price_momentum, PriceMomentum};
pub use sentiment::{calculate_sentiment_momentum, SentimentMomentum};

use serde::{Deserialize, Serialize};

/// Результат расчёта моментума
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumResult {
    /// Значение моментума
    pub value: f64,
    /// Период расчёта
    pub period: usize,
    /// Нормализованное значение (z-score)
    pub z_score: f64,
    /// Направление (1 = вверх, -1 = вниз, 0 = нейтрально)
    pub direction: i8,
}

impl MomentumResult {
    /// Создание нового результата
    pub fn new(value: f64, period: usize) -> Self {
        Self {
            value,
            period,
            z_score: 0.0,
            direction: if value > 0.01 {
                1
            } else if value < -0.01 {
                -1
            } else {
                0
            },
        }
    }

    /// Установка z-score
    pub fn with_z_score(mut self, z_score: f64) -> Self {
        self.z_score = z_score;
        self
    }
}

/// Общий калькулятор моментума
pub struct MomentumCalculator {
    /// Периоды для расчёта
    periods: Vec<usize>,
}

impl Default for MomentumCalculator {
    fn default() -> Self {
        Self::new(&[5, 10, 20, 60])
    }
}

impl MomentumCalculator {
    /// Создание калькулятора с заданными периодами
    pub fn new(periods: &[usize]) -> Self {
        Self {
            periods: periods.to_vec(),
        }
    }

    /// Расчёт всех моментумов для ценового ряда
    pub fn calculate_all_price(&self, prices: &[f64]) -> Vec<Vec<MomentumResult>> {
        self.periods
            .iter()
            .map(|&period| calculate_price_momentum(prices, period))
            .collect()
    }

    /// Расчёт всех моментумов для sentiment ряда
    pub fn calculate_all_sentiment(&self, sentiments: &[f64]) -> Vec<Vec<MomentumResult>> {
        self.periods
            .iter()
            .map(|&period| calculate_sentiment_momentum(sentiments, period))
            .collect()
    }

    /// Получение последних значений моментума
    pub fn latest_momentum(&self, results: &[Vec<MomentumResult>]) -> Vec<Option<MomentumResult>> {
        results.iter().map(|r| r.last().cloned()).collect()
    }
}

/// Расчёт z-score для серии значений
pub fn calculate_z_scores(values: &[f64], window: usize) -> Vec<f64> {
    if values.len() < window {
        return vec![0.0; values.len()];
    }

    let mut z_scores = vec![0.0; window - 1];

    for i in (window - 1)..values.len() {
        let window_values = &values[i + 1 - window..=i];
        let mean: f64 = window_values.iter().sum::<f64>() / window as f64;
        let variance: f64 =
            window_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        let std = variance.sqrt();

        let z = if std > 0.0 {
            (values[i] - mean) / std
        } else {
            0.0
        };

        z_scores.push(z);
    }

    z_scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_calculator() {
        let calculator = MomentumCalculator::new(&[3, 5]);
        let prices = vec![100.0, 102.0, 104.0, 103.0, 105.0, 108.0, 107.0];

        let results = calculator.calculate_all_price(&prices);

        assert_eq!(results.len(), 2); // 2 периода
        assert!(!results[0].is_empty());
    }

    #[test]
    fn test_z_scores() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let z_scores = calculate_z_scores(&values, 5);

        assert_eq!(z_scores.len(), values.len());

        // Первые 4 значения должны быть 0 (недостаточно данных)
        assert_eq!(z_scores[0], 0.0);
        assert_eq!(z_scores[3], 0.0);

        // Последние значения должны быть положительными (рост)
        assert!(z_scores[9] > 0.0);
    }
}
