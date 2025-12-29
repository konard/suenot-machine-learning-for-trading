//! Signal Generation
//!
//! Генерация торговых сигналов на основе прогнозов TFT.

use crate::model::QuantilePrediction;
use serde::{Deserialize, Serialize};

/// Тип торгового сигнала
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    /// Покупка (Long)
    Long,
    /// Продажа (Short)
    Short,
    /// Удержание (без позиции)
    Hold,
    /// Закрытие позиции
    Close,
}

/// Торговая позиция
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Направление
    pub signal: Signal,
    /// Размер позиции (0.0 - 1.0)
    pub size: f64,
    /// Уверенность в сигнале (0.0 - 1.0)
    pub confidence: f64,
    /// Временная метка
    pub timestamp: i64,
    /// Цена входа
    pub entry_price: Option<f64>,
}

impl Position {
    /// Создает новую позицию
    pub fn new(signal: Signal, size: f64, confidence: f64, timestamp: i64) -> Self {
        Self {
            signal,
            size: size.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            timestamp,
            entry_price: None,
        }
    }

    /// Пустая позиция (hold)
    pub fn empty(timestamp: i64) -> Self {
        Self {
            signal: Signal::Hold,
            size: 0.0,
            confidence: 0.0,
            timestamp,
            entry_price: None,
        }
    }

    /// Проверяет, открыта ли позиция
    pub fn is_open(&self) -> bool {
        self.signal == Signal::Long || self.signal == Signal::Short
    }
}

/// Генератор торговых сигналов
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Порог для уверенности (минимальная ширина интервала относительно медианы)
    pub confidence_threshold: f64,

    /// Порог для направления (минимальный прогноз возврата)
    pub direction_threshold: f64,

    /// Максимальный размер позиции
    pub max_position_size: f64,

    /// Базовый размер позиции
    pub base_position_size: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.05, // 5% ширина интервала
            direction_threshold: 0.005, // 0.5% минимальный прогноз
            max_position_size: 1.0,
            base_position_size: 0.5,
        }
    }
}

impl SignalGenerator {
    /// Создает генератор с настройками по умолчанию
    pub fn new() -> Self {
        Self::default()
    }

    /// Создает генератор с кастомными порогами
    pub fn with_thresholds(confidence: f64, direction: f64) -> Self {
        Self {
            confidence_threshold: confidence,
            direction_threshold: direction,
            ..Default::default()
        }
    }

    /// Генерирует сигнал на основе прогноза
    pub fn generate(&self, prediction: &QuantilePrediction, timestamp: i64) -> Position {
        // Получаем прогнозные значения
        let lower = prediction.lower();
        let median = prediction.median();
        let upper = prediction.upper();

        // Берем первый горизонт прогноза
        let q10 = lower[0];
        let q50 = median[0];
        let q90 = upper[0];

        // Вычисляем ширину интервала относительно медианы
        let interval_width = (q90 - q10).abs();
        let relative_width = if q50.abs() > 1e-10 {
            interval_width / q50.abs()
        } else {
            interval_width
        };

        // Определяем уверенность (обратно пропорционально ширине интервала)
        let confidence = if relative_width > 0.0 {
            (1.0 / (1.0 + relative_width)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Определяем сигнал
        let signal = if relative_width > self.confidence_threshold {
            // Слишком неуверенный прогноз
            Signal::Hold
        } else if q10 > self.direction_threshold {
            // Весь интервал положительный
            Signal::Long
        } else if q90 < -self.direction_threshold {
            // Весь интервал отрицательный
            Signal::Short
        } else if q50 > self.direction_threshold {
            // Медиана положительная, но интервал пересекает ноль
            Signal::Long
        } else if q50 < -self.direction_threshold {
            // Медиана отрицательная
            Signal::Short
        } else {
            Signal::Hold
        };

        // Размер позиции пропорционален уверенности
        let size = if signal == Signal::Hold {
            0.0
        } else {
            (self.base_position_size * confidence * 2.0).min(self.max_position_size)
        };

        Position::new(signal, size, confidence, timestamp)
    }

    /// Генерирует сигналы для серии прогнозов
    pub fn generate_series(
        &self,
        predictions: &[QuantilePrediction],
        timestamps: &[i64],
    ) -> Vec<Position> {
        predictions
            .iter()
            .zip(timestamps.iter())
            .map(|(pred, &ts)| self.generate(pred, ts))
            .collect()
    }
}

/// Торговая стратегия на основе TFT
#[derive(Debug, Clone)]
pub struct TradingStrategy {
    /// Генератор сигналов
    pub signal_generator: SignalGenerator,

    /// Текущая позиция
    current_position: Option<Position>,

    /// История позиций
    position_history: Vec<Position>,

    /// Максимальное количество одновременных позиций
    pub max_positions: usize,
}

impl TradingStrategy {
    /// Создает новую стратегию
    pub fn new(signal_generator: SignalGenerator) -> Self {
        Self {
            signal_generator,
            current_position: None,
            position_history: Vec::new(),
            max_positions: 1,
        }
    }

    /// Обновляет стратегию на основе нового прогноза
    pub fn update(&mut self, prediction: &QuantilePrediction, timestamp: i64, price: f64) -> Position {
        let new_position = self.signal_generator.generate(prediction, timestamp);

        // Логика перехода между позициями
        let final_position = match (&self.current_position, new_position.signal) {
            // Нет позиции -> открываем новую
            (None, Signal::Long | Signal::Short) => {
                let mut pos = new_position.clone();
                pos.entry_price = Some(price);
                pos
            }

            // Есть позиция, новый сигнал противоположный -> закрываем и открываем
            (Some(current), Signal::Long) if current.signal == Signal::Short => {
                let mut pos = new_position.clone();
                pos.entry_price = Some(price);
                pos
            }
            (Some(current), Signal::Short) if current.signal == Signal::Long => {
                let mut pos = new_position.clone();
                pos.entry_price = Some(price);
                pos
            }

            // Есть позиция, Hold сигнал -> закрываем
            (Some(_), Signal::Hold) => Position::empty(timestamp),

            // Держим текущую позицию
            (Some(current), _) => current.clone(),

            // Нет позиции, Hold -> остаемся без позиции
            (None, Signal::Hold | Signal::Close) => Position::empty(timestamp),
        };

        // Сохраняем историю
        if let Some(old) = &self.current_position {
            if old.signal != final_position.signal {
                self.position_history.push(old.clone());
            }
        }

        self.current_position = Some(final_position.clone());
        final_position
    }

    /// Возвращает текущую позицию
    pub fn current_position(&self) -> Option<&Position> {
        self.current_position.as_ref()
    }

    /// Возвращает историю позиций
    pub fn position_history(&self) -> &[Position] {
        &self.position_history
    }

    /// Сбрасывает стратегию
    pub fn reset(&mut self) {
        self.current_position = None;
        self.position_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_prediction(q10: f64, q50: f64, q90: f64) -> QuantilePrediction {
        let values = Array2::from_shape_vec((1, 3), vec![q10, q50, q90]).unwrap();
        QuantilePrediction::new(vec![0.1, 0.5, 0.9], values)
    }

    #[test]
    fn test_signal_generator_long() {
        let gen = SignalGenerator::new();

        // Все квантили положительные -> Long
        let pred = make_prediction(0.01, 0.02, 0.03);
        let pos = gen.generate(&pred, 0);

        assert_eq!(pos.signal, Signal::Long);
        assert!(pos.size > 0.0);
    }

    #[test]
    fn test_signal_generator_short() {
        let gen = SignalGenerator::new();

        // Все квантили отрицательные -> Short
        let pred = make_prediction(-0.03, -0.02, -0.01);
        let pos = gen.generate(&pred, 0);

        assert_eq!(pos.signal, Signal::Short);
        assert!(pos.size > 0.0);
    }

    #[test]
    fn test_signal_generator_hold() {
        let gen = SignalGenerator::new();

        // Широкий интервал -> Hold
        let pred = make_prediction(-0.1, 0.0, 0.1);
        let pos = gen.generate(&pred, 0);

        assert_eq!(pos.signal, Signal::Hold);
        assert_eq!(pos.size, 0.0);
    }

    #[test]
    fn test_trading_strategy() {
        let gen = SignalGenerator::new();
        let mut strategy = TradingStrategy::new(gen);

        // Открываем long
        let pred = make_prediction(0.01, 0.02, 0.03);
        let pos = strategy.update(&pred, 0, 100.0);
        assert_eq!(pos.signal, Signal::Long);
        assert_eq!(pos.entry_price, Some(100.0));

        // Переворачиваемся в short
        let pred = make_prediction(-0.03, -0.02, -0.01);
        let pos = strategy.update(&pred, 1, 101.0);
        assert_eq!(pos.signal, Signal::Short);
        assert_eq!(pos.entry_price, Some(101.0));
    }
}
