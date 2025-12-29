//! Sentiment-моментум
//!
//! Расчёт моментума на основе sentiment-данных.

use super::{calculate_z_scores, MomentumResult};
use serde::{Deserialize, Serialize};

/// Sentiment-моментум с различными метриками
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentMomentum {
    /// Изменение sentiment за период
    pub change: f64,
    /// Скользящее среднее sentiment
    pub sma: f64,
    /// Отклонение от среднего
    pub deviation: f64,
    /// Ускорение (изменение изменения)
    pub acceleration: f64,
    /// Период расчёта
    pub period: usize,
}

impl SentimentMomentum {
    /// Расчёт sentiment-моментума
    pub fn calculate(sentiments: &[f64], period: usize) -> Option<Self> {
        if sentiments.len() < period + 2 {
            return None;
        }

        let current = *sentiments.last()?;
        let past = sentiments[sentiments.len() - period - 1];

        // Изменение sentiment
        let change = current - past;

        // SMA sentiment
        let recent = &sentiments[sentiments.len() - period..];
        let sma = recent.iter().sum::<f64>() / period as f64;

        // Отклонение от среднего
        let deviation = current - sma;

        // Ускорение (вторая производная)
        let prev_change = sentiments[sentiments.len() - 2] - sentiments[sentiments.len() - period - 2];
        let acceleration = change - prev_change;

        Some(Self {
            change,
            sma,
            deviation,
            acceleration,
            period,
        })
    }

    /// Направление моментума
    pub fn direction(&self) -> i8 {
        if self.change > 0.05 {
            1 // Растущий
        } else if self.change < -0.05 {
            -1 // Падающий
        } else {
            0 // Нейтральный
        }
    }

    /// Является ли sentiment бычьим
    pub fn is_bullish(&self) -> bool {
        self.change > 0.0 && self.deviation > 0.0
    }

    /// Является ли sentiment медвежьим
    pub fn is_bearish(&self) -> bool {
        self.change < 0.0 && self.deviation < 0.0
    }

    /// Ускоряется ли позитивный sentiment
    pub fn is_accelerating_bullish(&self) -> bool {
        self.change > 0.0 && self.acceleration > 0.0
    }

    /// Ускоряется ли негативный sentiment
    pub fn is_accelerating_bearish(&self) -> bool {
        self.change < 0.0 && self.acceleration < 0.0
    }

    /// Сила sentiment-моментума
    pub fn strength(&self) -> f64 {
        // Комбинация изменения и ускорения
        let change_strength = self.change.abs().min(1.0);
        let accel_strength = (self.acceleration.abs() * 2.0).min(1.0);

        (change_strength * 0.7 + accel_strength * 0.3).min(1.0)
    }
}

/// Расчёт sentiment-моментума для временного ряда
pub fn calculate_sentiment_momentum(sentiments: &[f64], period: usize) -> Vec<MomentumResult> {
    if sentiments.len() < period + 1 {
        return vec![];
    }

    let mut results = Vec::with_capacity(sentiments.len() - period);

    // Рассчитываем изменения для всех точек
    let changes: Vec<f64> = (period..sentiments.len())
        .map(|i| sentiments[i] - sentiments[i - period])
        .collect();

    // Рассчитываем z-scores
    let z_scores = calculate_z_scores(&changes, period.min(20));

    for (&change, &z_score) in changes.iter().zip(z_scores.iter()) {
        results.push(MomentumResult::new(change, period).with_z_score(z_score));
    }

    results
}

/// Расчёт объёмно-взвешенного sentiment-моментума
pub fn volume_weighted_sentiment_momentum(
    sentiments: &[f64],
    volumes: &[f64],
    period: usize,
) -> Vec<f64> {
    if sentiments.len() != volumes.len() || sentiments.len() < period {
        return vec![];
    }

    let mut results = Vec::with_capacity(sentiments.len() - period + 1);

    for i in (period - 1)..sentiments.len() {
        let window_start = i + 1 - period;

        let mut weighted_sum = 0.0;
        let mut total_volume = 0.0;

        for j in window_start..=i {
            weighted_sum += sentiments[j] * volumes[j];
            total_volume += volumes[j];
        }

        let vw_sentiment = if total_volume > 0.0 {
            weighted_sum / total_volume
        } else {
            0.0
        };

        results.push(vw_sentiment);
    }

    // Вычисляем моментум (разницу между текущим и прошлым VW sentiment)
    let mut momentum = vec![0.0; period - 1];
    for i in 1..results.len() {
        momentum.push(results[i] - results[i - 1]);
    }

    momentum
}

/// Расчёт sentiment divergence (расхождение)
pub fn sentiment_divergence(
    sentiments: &[f64],
    prices: &[f64],
    period: usize,
) -> Vec<f64> {
    if sentiments.len() != prices.len() || sentiments.len() < period + 1 {
        return vec![];
    }

    let mut divergence = vec![0.0; period];

    for i in period..sentiments.len() {
        let sent_change = sentiments[i] - sentiments[i - period];
        let price_change = if prices[i - period] != 0.0 {
            (prices[i] - prices[i - period]) / prices[i - period]
        } else {
            0.0
        };

        // Нормализуем price_change для сравнения с sentiment
        let normalized_price = (price_change * 10.0).clamp(-1.0, 1.0);

        // Дивергенция = разница между sentiment и нормализованным движением цены
        divergence.push(sent_change - normalized_price);
    }

    divergence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_momentum() {
        let sentiments = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let momentum = SentimentMomentum::calculate(&sentiments, 3);

        assert!(momentum.is_some());
        let m = momentum.unwrap();
        assert!(m.change > 0.0); // Sentiment растёт
        assert!(m.is_bullish());
    }

    #[test]
    fn test_calculate_sentiment_momentum() {
        let sentiments = vec![0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];
        let results = calculate_sentiment_momentum(&sentiments, 3);

        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.value > 0.0)); // Все положительные
    }

    #[test]
    fn test_volume_weighted_momentum() {
        let sentiments = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let volumes = vec![100.0, 200.0, 150.0, 300.0, 250.0];

        let vw_momentum = volume_weighted_sentiment_momentum(&sentiments, &volumes, 3);

        assert_eq!(vw_momentum.len(), sentiments.len());
    }

    #[test]
    fn test_sentiment_divergence() {
        // Цена растёт, sentiment падает = отрицательная дивергенция
        let sentiments = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let prices = vec![100.0, 105.0, 110.0, 115.0, 120.0];

        let div = sentiment_divergence(&sentiments, &prices, 2);

        assert!(!div.is_empty());
        // Последние значения должны показывать дивергенцию
        assert!(div.last().unwrap() < &0.0); // Негативная дивергенция
    }

    #[test]
    fn test_acceleration() {
        // Sentiment ускоряется вверх
        let sentiments = vec![0.0, 0.1, 0.15, 0.22, 0.32, 0.45, 0.62, 0.85];
        let m = SentimentMomentum::calculate(&sentiments, 3).unwrap();

        assert!(m.is_accelerating_bullish());
    }
}
