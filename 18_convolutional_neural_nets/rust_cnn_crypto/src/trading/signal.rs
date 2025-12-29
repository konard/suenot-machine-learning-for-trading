//! Торговые сигналы

use serde::{Deserialize, Serialize};

/// Направление сигнала
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalDirection {
    /// Покупка (Long)
    Long,
    /// Продажа (Short)
    Short,
    /// Нейтрально / без позиции
    Neutral,
}

/// Торговый сигнал
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Временная метка
    pub timestamp: i64,
    /// Направление
    pub direction: SignalDirection,
    /// Сила сигнала (0.0 - 1.0)
    pub strength: f64,
    /// Вероятности классов [down, neutral, up]
    pub probabilities: [f64; 3],
    /// Текущая цена
    pub price: f64,
}

impl Signal {
    /// Создание нового сигнала
    pub fn new(
        timestamp: i64,
        probabilities: [f64; 3],
        price: f64,
        threshold: f64,
    ) -> Self {
        let (direction, strength) = Self::determine_direction(&probabilities, threshold);

        Self {
            timestamp,
            direction,
            strength,
            probabilities,
            price,
        }
    }

    /// Определение направления сигнала
    fn determine_direction(probs: &[f64; 3], threshold: f64) -> (SignalDirection, f64) {
        let max_prob = probs.iter().cloned().fold(0.0, f64::max);
        let max_idx = probs.iter().position(|&p| p == max_prob).unwrap_or(1);

        if max_prob < threshold {
            return (SignalDirection::Neutral, max_prob);
        }

        let direction = match max_idx {
            0 => SignalDirection::Short,
            2 => SignalDirection::Long,
            _ => SignalDirection::Neutral,
        };

        (direction, max_prob)
    }

    /// Преобразование из предсказания класса
    pub fn from_prediction(
        timestamp: i64,
        predicted_class: usize,
        probabilities: [f64; 3],
        price: f64,
        threshold: f64,
    ) -> Self {
        Self::new(timestamp, probabilities, price, threshold)
    }

    /// Сильный сигнал?
    pub fn is_strong(&self, min_strength: f64) -> bool {
        self.strength >= min_strength
    }

    /// Нужно открывать позицию?
    pub fn should_open_position(&self) -> bool {
        self.direction != SignalDirection::Neutral
    }

    /// Обратный сигнал
    pub fn reversed(&self) -> SignalDirection {
        match self.direction {
            SignalDirection::Long => SignalDirection::Short,
            SignalDirection::Short => SignalDirection::Long,
            SignalDirection::Neutral => SignalDirection::Neutral,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new(1234567890, [0.1, 0.2, 0.7], 50000.0, 0.5);
        assert_eq!(signal.direction, SignalDirection::Long);
        assert!(signal.strength >= 0.7);
    }

    #[test]
    fn test_signal_neutral() {
        let signal = Signal::new(1234567890, [0.35, 0.35, 0.3], 50000.0, 0.5);
        assert_eq!(signal.direction, SignalDirection::Neutral);
    }

    #[test]
    fn test_signal_short() {
        let signal = Signal::new(1234567890, [0.8, 0.1, 0.1], 50000.0, 0.5);
        assert_eq!(signal.direction, SignalDirection::Short);
    }
}
