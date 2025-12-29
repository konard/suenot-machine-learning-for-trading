//! Конфигурация моделей LSTM и GRU

use serde::{Deserialize, Serialize};

/// Конфигурация LSTM/GRU модели
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    /// Количество входных признаков
    pub input_size: usize,
    /// Размер скрытого слоя
    pub hidden_size: usize,
    /// Количество выходов
    pub output_size: usize,
    /// Количество LSTM слоёв
    pub num_layers: usize,
    /// Скорость обучения
    pub learning_rate: f64,
    /// Dropout вероятность
    pub dropout: f64,
    /// Использовать bidirectional
    pub bidirectional: bool,
    /// Размер батча
    pub batch_size: usize,
    /// Градиентный клиппинг
    pub gradient_clip: Option<f64>,
    /// Инициализация весов
    pub weight_init: WeightInit,
}

/// Тип инициализации весов
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WeightInit {
    /// Инициализация Xavier/Glorot
    Xavier,
    /// Инициализация He
    He,
    /// Нормальное распределение
    Normal { mean: f64, std: f64 },
    /// Равномерное распределение
    Uniform { low: f64, high: f64 },
}

impl Default for WeightInit {
    fn default() -> Self {
        WeightInit::Xavier
    }
}

impl LSTMConfig {
    /// Создаёт новую конфигурацию с параметрами по умолчанию
    ///
    /// # Аргументы
    ///
    /// * `input_size` - Количество входных признаков
    /// * `hidden_size` - Размер скрытого слоя
    /// * `output_size` - Количество выходов
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            num_layers: 1,
            learning_rate: 0.001,
            dropout: 0.0,
            bidirectional: false,
            batch_size: 32,
            gradient_clip: Some(1.0),
            weight_init: WeightInit::Xavier,
        }
    }

    /// Устанавливает скорость обучения
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Устанавливает dropout
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Устанавливает количество слоёв
    pub fn with_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Включает bidirectional режим
    pub fn bidirectional(mut self) -> Self {
        self.bidirectional = true;
        self
    }

    /// Устанавливает размер батча
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Устанавливает градиентный клиппинг
    pub fn with_gradient_clip(mut self, clip: f64) -> Self {
        self.gradient_clip = Some(clip);
        self
    }

    /// Устанавливает инициализацию весов
    pub fn with_weight_init(mut self, init: WeightInit) -> Self {
        self.weight_init = init;
        self
    }

    /// Создаёт конфигурацию для маленькой модели
    pub fn small(input_size: usize, output_size: usize) -> Self {
        Self::new(input_size, 32, output_size)
    }

    /// Создаёт конфигурацию для средней модели
    pub fn medium(input_size: usize, output_size: usize) -> Self {
        Self::new(input_size, 64, output_size).with_layers(2)
    }

    /// Создаёт конфигурацию для большой модели
    pub fn large(input_size: usize, output_size: usize) -> Self {
        Self::new(input_size, 128, output_size)
            .with_layers(3)
            .with_dropout(0.2)
    }
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self::new(1, 64, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = LSTMConfig::new(5, 64, 1)
            .with_learning_rate(0.01)
            .with_dropout(0.3)
            .with_layers(2)
            .bidirectional();

        assert_eq!(config.input_size, 5);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.output_size, 1);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.dropout, 0.3);
        assert_eq!(config.num_layers, 2);
        assert!(config.bidirectional);
    }

    #[test]
    fn test_preset_configs() {
        let small = LSTMConfig::small(5, 1);
        assert_eq!(small.hidden_size, 32);

        let medium = LSTMConfig::medium(5, 1);
        assert_eq!(medium.hidden_size, 64);
        assert_eq!(medium.num_layers, 2);

        let large = LSTMConfig::large(5, 1);
        assert_eq!(large.hidden_size, 128);
        assert_eq!(large.num_layers, 3);
    }
}
