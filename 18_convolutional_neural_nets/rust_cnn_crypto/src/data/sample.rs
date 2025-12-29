//! Структура одного образца данных

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Метка класса для прогнозирования
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Label {
    /// Цена пойдёт вниз (доходность < -threshold)
    Down = 0,
    /// Цена останется примерно на месте
    Neutral = 1,
    /// Цена пойдёт вверх (доходность > threshold)
    Up = 2,
}

impl Label {
    /// Создание метки из доходности
    pub fn from_return(return_pct: f64, threshold: f64) -> Self {
        if return_pct > threshold {
            Self::Up
        } else if return_pct < -threshold {
            Self::Down
        } else {
            Self::Neutral
        }
    }

    /// Преобразование в числовое значение
    pub fn as_usize(&self) -> usize {
        *self as usize
    }

    /// Получение количества классов
    pub fn num_classes() -> usize {
        3
    }
}

impl From<usize> for Label {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::Down,
            1 => Self::Neutral,
            2 => Self::Up,
            _ => Self::Neutral,
        }
    }
}

/// Один образец для обучения/предсказания
#[derive(Debug, Clone)]
pub struct Sample {
    /// Входные данные: матрица [channels, window_size]
    /// channels = количество признаков (OHLCV + индикаторы)
    /// window_size = количество свечей в окне
    pub features: Array2<f32>,

    /// Метка класса (для обучения)
    pub label: Option<Label>,

    /// Временная метка последней свечи
    pub timestamp: i64,

    /// Доходность (для оценки)
    pub actual_return: Option<f64>,
}

impl Sample {
    /// Создание нового образца
    pub fn new(features: Array2<f32>, timestamp: i64) -> Self {
        Self {
            features,
            label: None,
            timestamp,
            actual_return: None,
        }
    }

    /// Установка метки
    pub fn with_label(mut self, label: Label) -> Self {
        self.label = Some(label);
        self
    }

    /// Установка фактической доходности
    pub fn with_return(mut self, return_pct: f64) -> Self {
        self.actual_return = Some(return_pct);
        self
    }

    /// Получение формы входных данных
    pub fn shape(&self) -> (usize, usize) {
        let shape = self.features.shape();
        (shape[0], shape[1])
    }

    /// Количество каналов (признаков)
    pub fn num_channels(&self) -> usize {
        self.features.shape()[0]
    }

    /// Размер окна
    pub fn window_size(&self) -> usize {
        self.features.shape()[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_label_from_return() {
        assert_eq!(Label::from_return(2.0, 1.0), Label::Up);
        assert_eq!(Label::from_return(-2.0, 1.0), Label::Down);
        assert_eq!(Label::from_return(0.5, 1.0), Label::Neutral);
    }

    #[test]
    fn test_sample_creation() {
        let features = Array2::zeros((5, 60));
        let sample = Sample::new(features, 1234567890)
            .with_label(Label::Up)
            .with_return(1.5);

        assert_eq!(sample.num_channels(), 5);
        assert_eq!(sample.window_size(), 60);
        assert_eq!(sample.label, Some(Label::Up));
        assert_eq!(sample.actual_return, Some(1.5));
    }
}
