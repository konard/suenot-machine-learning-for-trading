//! Обнаружение дивергенций
//!
//! Модуль для обнаружения расхождений между ценой и sentiment.

use serde::{Deserialize, Serialize};

/// Тип дивергенции
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceType {
    /// Бычья дивергенция: цена падает, sentiment растёт
    /// Потенциальный сигнал разворота вверх
    Bullish,

    /// Медвежья дивергенция: цена растёт, sentiment падает
    /// Потенциальный сигнал разворота вниз
    Bearish,

    /// Нет дивергенции
    None,
}

impl DivergenceType {
    /// Является ли дивергенция сигналом разворота
    pub fn is_reversal_signal(&self) -> bool {
        matches!(self, DivergenceType::Bullish | DivergenceType::Bearish)
    }

    /// Ожидаемое направление после дивергенции
    pub fn expected_direction(&self) -> i8 {
        match self {
            DivergenceType::Bullish => 1,  // Ожидаем рост
            DivergenceType::Bearish => -1, // Ожидаем падение
            DivergenceType::None => 0,
        }
    }
}

/// Информация о дивергенции
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Divergence {
    /// Тип дивергенции
    pub divergence_type: DivergenceType,
    /// Сила дивергенции (0.0 - 1.0)
    pub strength: f64,
    /// Является ли дивергенция сильной
    pub is_strong: bool,
    /// Значение ценового моментума
    pub price_momentum: f64,
    /// Значение sentiment-моментума
    pub sentiment_momentum: f64,
}

impl Divergence {
    /// Создание новой дивергенции
    pub fn new(
        divergence_type: DivergenceType,
        price_momentum: f64,
        sentiment_momentum: f64,
    ) -> Self {
        let strength = Self::calculate_strength(price_momentum, sentiment_momentum);
        let is_strong = strength > 0.6;

        Self {
            divergence_type,
            strength,
            is_strong,
            price_momentum,
            sentiment_momentum,
        }
    }

    /// Расчёт силы дивергенции
    fn calculate_strength(price_momentum: f64, sentiment_momentum: f64) -> f64 {
        // Сила = насколько сильно расходятся направления
        let price_magnitude = price_momentum.abs();
        let sent_magnitude = sentiment_momentum.abs();

        // Оба должны быть значительными
        let magnitude = (price_magnitude.min(sent_magnitude) * 2.0).min(1.0);

        // Направления должны быть противоположными
        let direction_diff = if price_momentum.signum() != sentiment_momentum.signum() {
            1.0
        } else {
            0.0
        };

        magnitude * direction_diff
    }
}

/// Детектор дивергенций
pub struct DivergenceDetector {
    /// Порог для определения значительного движения
    significance_threshold: f64,
    /// Порог для сильной дивергенции
    strong_threshold: f64,
}

impl Default for DivergenceDetector {
    fn default() -> Self {
        Self::new(0.1, 0.6)
    }
}

impl DivergenceDetector {
    /// Создание нового детектора
    ///
    /// # Аргументы
    ///
    /// * `significance_threshold` - Порог для значительного движения
    /// * `strong_threshold` - Порог силы для сильной дивергенции
    pub fn new(significance_threshold: f64, strong_threshold: f64) -> Self {
        Self {
            significance_threshold,
            strong_threshold,
        }
    }

    /// Обнаружение дивергенции в одной точке
    pub fn detect_at_point(
        &self,
        price_momentum: f64,
        sentiment_momentum: f64,
    ) -> Option<Divergence> {
        // Проверяем, что оба значения значительны
        if price_momentum.abs() < self.significance_threshold
            || sentiment_momentum.abs() < self.significance_threshold
        {
            return None;
        }

        // Определяем направления
        let price_direction = price_momentum.signum() as i8;
        let sent_direction = sentiment_momentum.signum() as i8;

        // Если направления совпадают - нет дивергенции
        if price_direction == sent_direction {
            return None;
        }

        // Определяем тип дивергенции
        let divergence_type = if price_direction < 0 && sent_direction > 0 {
            DivergenceType::Bullish // Цена вниз, sentiment вверх
        } else {
            DivergenceType::Bearish // Цена вверх, sentiment вниз
        };

        Some(Divergence::new(divergence_type, price_momentum, sentiment_momentum))
    }

    /// Обнаружение дивергенций в временном ряде
    pub fn detect_series(
        &self,
        price_momentums: &[f64],
        sentiment_momentums: &[f64],
    ) -> Vec<Option<Divergence>> {
        let len = price_momentums.len().min(sentiment_momentums.len());

        (0..len)
            .map(|i| self.detect_at_point(price_momentums[i], sentiment_momentums[i]))
            .collect()
    }

    /// Подсчёт дивергенций за период
    pub fn count_divergences(
        &self,
        price_momentums: &[f64],
        sentiment_momentums: &[f64],
    ) -> DivergenceStats {
        let divergences = self.detect_series(price_momentums, sentiment_momentums);

        let mut bullish = 0;
        let mut bearish = 0;
        let mut strong = 0;

        for div in divergences.iter().flatten() {
            match div.divergence_type {
                DivergenceType::Bullish => bullish += 1,
                DivergenceType::Bearish => bearish += 1,
                DivergenceType::None => {}
            }
            if div.is_strong {
                strong += 1;
            }
        }

        DivergenceStats {
            bullish_count: bullish,
            bearish_count: bearish,
            strong_count: strong,
            total_points: price_momentums.len().min(sentiment_momentums.len()),
        }
    }

    /// Поиск последней сильной дивергенции
    pub fn find_last_strong_divergence(
        &self,
        price_momentums: &[f64],
        sentiment_momentums: &[f64],
    ) -> Option<(usize, Divergence)> {
        let divergences = self.detect_series(price_momentums, sentiment_momentums);

        divergences
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, div)| {
                div.as_ref()
                    .filter(|d| d.is_strong)
                    .map(|d| (i, d.clone()))
            })
    }
}

/// Статистика дивергенций
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceStats {
    /// Количество бычьих дивергенций
    pub bullish_count: usize,
    /// Количество медвежьих дивергенций
    pub bearish_count: usize,
    /// Количество сильных дивергенций
    pub strong_count: usize,
    /// Общее количество точек
    pub total_points: usize,
}

impl DivergenceStats {
    /// Процент дивергенций
    pub fn divergence_ratio(&self) -> f64 {
        if self.total_points == 0 {
            0.0
        } else {
            (self.bullish_count + self.bearish_count) as f64 / self.total_points as f64
        }
    }

    /// Преобладающий тип дивергенции
    pub fn dominant_type(&self) -> DivergenceType {
        if self.bullish_count > self.bearish_count {
            DivergenceType::Bullish
        } else if self.bearish_count > self.bullish_count {
            DivergenceType::Bearish
        } else {
            DivergenceType::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bullish_divergence() {
        let detector = DivergenceDetector::default();

        // Цена падает, sentiment растёт
        let div = detector.detect_at_point(-0.5, 0.4);

        assert!(div.is_some());
        let d = div.unwrap();
        assert_eq!(d.divergence_type, DivergenceType::Bullish);
    }

    #[test]
    fn test_bearish_divergence() {
        let detector = DivergenceDetector::default();

        // Цена растёт, sentiment падает
        let div = detector.detect_at_point(0.5, -0.4);

        assert!(div.is_some());
        let d = div.unwrap();
        assert_eq!(d.divergence_type, DivergenceType::Bearish);
    }

    #[test]
    fn test_no_divergence() {
        let detector = DivergenceDetector::default();

        // Оба вверх
        assert!(detector.detect_at_point(0.5, 0.4).is_none());

        // Оба вниз
        assert!(detector.detect_at_point(-0.5, -0.4).is_none());

        // Слабые движения
        assert!(detector.detect_at_point(0.05, -0.05).is_none());
    }

    #[test]
    fn test_divergence_series() {
        let detector = DivergenceDetector::default();

        let price_moms = vec![0.3, 0.4, -0.3, 0.2, -0.4];
        let sent_moms = vec![0.2, -0.3, 0.4, 0.1, 0.3];

        let divs = detector.detect_series(&price_moms, &sent_moms);

        assert_eq!(divs.len(), 5);

        // price[1]=0.4, sent[1]=-0.3 → bearish
        assert!(divs[1].is_some());
        assert_eq!(divs[1].as_ref().unwrap().divergence_type, DivergenceType::Bearish);

        // price[2]=-0.3, sent[2]=0.4 → bullish
        assert!(divs[2].is_some());
        assert_eq!(divs[2].as_ref().unwrap().divergence_type, DivergenceType::Bullish);
    }

    #[test]
    fn test_divergence_stats() {
        let detector = DivergenceDetector::default();

        let price_moms = vec![0.3, 0.4, -0.3, 0.5, -0.4];
        let sent_moms = vec![0.2, -0.3, 0.4, -0.5, 0.3];

        let stats = detector.count_divergences(&price_moms, &sent_moms);

        assert!(stats.bullish_count > 0);
        assert!(stats.bearish_count > 0);
        assert_eq!(stats.total_points, 5);
    }

    #[test]
    fn test_divergence_strength() {
        // Сильная дивергенция: большие противоположные движения
        let div = Divergence::new(DivergenceType::Bearish, 0.8, -0.7);
        assert!(div.strength > 0.5);
        assert!(div.is_strong);

        // Слабая дивергенция: небольшие движения
        let div = Divergence::new(DivergenceType::Bullish, -0.2, 0.15);
        assert!(div.strength < 0.5);
        assert!(!div.is_strong);
    }
}
