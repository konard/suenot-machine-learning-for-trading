//! Конфигурация TFT модели

use serde::{Deserialize, Serialize};

/// Конфигурация Temporal Fusion Transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFTConfig {
    /// Размер скрытого слоя
    pub hidden_size: usize,

    /// Количество голов внимания
    pub num_attention_heads: usize,

    /// Dropout rate
    pub dropout: f64,

    /// Размер скрытого слоя для continuous переменных
    pub hidden_continuous_size: usize,

    /// Количество LSTM слоев
    pub num_lstm_layers: usize,

    /// Длина encoder context
    pub encoder_length: usize,

    /// Длина prediction horizon
    pub prediction_length: usize,

    /// Количество признаков encoder
    pub num_encoder_features: usize,

    /// Количество признаков decoder (known future)
    pub num_decoder_features: usize,

    /// Количество статических признаков
    pub num_static_features: usize,

    /// Квантили для прогноза
    pub quantiles: Vec<f64>,

    /// Скорость обучения
    pub learning_rate: f64,

    /// Размер батча
    pub batch_size: usize,

    /// Количество эпох обучения
    pub max_epochs: usize,

    /// Early stopping patience
    pub patience: usize,

    /// Gradient clipping
    pub gradient_clip_val: Option<f64>,
}

impl Default for TFTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            num_attention_heads: 4,
            dropout: 0.1,
            hidden_continuous_size: 16,
            num_lstm_layers: 2,
            encoder_length: 168,
            prediction_length: 24,
            num_encoder_features: 24,
            num_decoder_features: 4,
            num_static_features: 0,
            quantiles: vec![0.1, 0.5, 0.9],
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            patience: 10,
            gradient_clip_val: Some(1.0),
        }
    }
}

impl TFTConfig {
    /// Конфигурация для часовых данных
    pub fn hourly() -> Self {
        Self::default()
    }

    /// Конфигурация для дневных данных
    pub fn daily() -> Self {
        Self {
            encoder_length: 30,
            prediction_length: 7,
            ..Default::default()
        }
    }

    /// Маленькая модель для быстрого обучения
    pub fn small() -> Self {
        Self {
            hidden_size: 32,
            num_attention_heads: 2,
            hidden_continuous_size: 8,
            num_lstm_layers: 1,
            ..Default::default()
        }
    }

    /// Большая модель для лучшего качества
    pub fn large() -> Self {
        Self {
            hidden_size: 128,
            num_attention_heads: 8,
            hidden_continuous_size: 32,
            num_lstm_layers: 2,
            ..Default::default()
        }
    }

    /// Устанавливает размеры признаков на основе dataset
    pub fn with_feature_sizes(
        mut self,
        encoder_features: usize,
        decoder_features: usize,
        static_features: usize,
    ) -> Self {
        self.num_encoder_features = encoder_features;
        self.num_decoder_features = decoder_features;
        self.num_static_features = static_features;
        self
    }

    /// Устанавливает временные параметры
    pub fn with_lengths(mut self, encoder: usize, prediction: usize) -> Self {
        self.encoder_length = encoder;
        self.prediction_length = prediction;
        self
    }

    /// Устанавливает квантили
    pub fn with_quantiles(mut self, quantiles: Vec<f64>) -> Self {
        self.quantiles = quantiles;
        self
    }

    /// Возвращает количество выходов (квантилей)
    pub fn output_size(&self) -> usize {
        self.quantiles.len()
    }

    /// Проверяет валидность конфигурации
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".to_string());
        }
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".to_string());
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }
        if self.quantiles.is_empty() {
            return Err("quantiles must not be empty".to_string());
        }
        for &q in &self.quantiles {
            if q <= 0.0 || q >= 1.0 {
                return Err(format!("quantile {} must be in (0, 1)", q));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TFTConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_presets() {
        let hourly = TFTConfig::hourly();
        assert_eq!(hourly.encoder_length, 168);

        let daily = TFTConfig::daily();
        assert_eq!(daily.encoder_length, 30);

        let small = TFTConfig::small();
        assert_eq!(small.hidden_size, 32);

        let large = TFTConfig::large();
        assert_eq!(large.hidden_size, 128);
    }

    #[test]
    fn test_invalid_config() {
        let mut config = TFTConfig::default();
        config.hidden_size = 0;
        assert!(config.validate().is_err());

        let mut config = TFTConfig::default();
        config.quantiles = vec![];
        assert!(config.validate().is_err());

        let mut config = TFTConfig::default();
        config.quantiles = vec![1.5];
        assert!(config.validate().is_err());
    }
}
