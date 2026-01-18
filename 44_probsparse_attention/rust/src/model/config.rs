//! Конфигурация модели Informer
//!
//! Определяет все гиперпараметры модели с ProbSparse Attention

use serde::{Deserialize, Serialize};
use crate::defaults;

/// Тип выхода модели
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Регрессия (предсказание значений)
    Regression,
    /// Направление (вверх/вниз/нейтрально)
    Direction,
    /// Квантильная регрессия
    Quantile,
}

impl Default for OutputType {
    fn default() -> Self {
        OutputType::Regression
    }
}

/// Конфигурация модели Informer с ProbSparse Attention
///
/// # Example
///
/// ```
/// use informer_probsparse::InformerConfig;
///
/// let config = InformerConfig {
///     seq_len: 96,
///     pred_len: 24,
///     d_model: 64,
///     n_heads: 4,
///     ..Default::default()
/// };
///
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformerConfig {
    // Input/Output
    /// Длина входной последовательности
    pub seq_len: usize,
    /// Длина начального токена для декодера
    pub label_len: usize,
    /// Горизонт прогнозирования
    pub pred_len: usize,
    /// Количество входных признаков
    pub input_features: usize,

    // Architecture
    /// Размерность модели
    pub d_model: usize,
    /// Количество голов внимания
    pub n_heads: usize,
    /// Размерность feed-forward слоя
    pub d_ff: usize,
    /// Количество слоёв encoder
    pub n_encoder_layers: usize,
    /// Количество слоёв decoder
    pub n_decoder_layers: usize,
    /// Dropout rate
    pub dropout: f64,

    // ProbSparse Attention
    /// Фактор разреженности (u = c·log(L))
    pub sampling_factor: f64,
    /// Использовать distilling
    pub use_distilling: bool,

    // Output
    /// Тип выхода
    pub output_type: OutputType,
    /// Квантили для квантильной регрессии
    pub quantiles: Vec<f64>,

    // Embedding
    /// Размер ядра свёртки
    pub kernel_size: usize,
    /// Использовать позиционное кодирование
    pub use_positional_encoding: bool,
}

impl Default for InformerConfig {
    fn default() -> Self {
        Self {
            seq_len: defaults::SEQ_LEN,
            label_len: defaults::SEQ_LEN / 2,
            pred_len: defaults::PRED_LEN,
            input_features: defaults::INPUT_FEATURES,

            d_model: defaults::D_MODEL,
            n_heads: defaults::N_HEADS,
            d_ff: defaults::D_MODEL * 4,
            n_encoder_layers: defaults::N_ENCODER_LAYERS,
            n_decoder_layers: 1,
            dropout: defaults::DROPOUT,

            sampling_factor: defaults::SAMPLING_FACTOR,
            use_distilling: true,

            output_type: OutputType::Regression,
            quantiles: vec![0.1, 0.5, 0.9],

            kernel_size: 3,
            use_positional_encoding: true,
        }
    }
}

impl InformerConfig {
    /// Создаёт новую конфигурацию с параметрами по умолчанию
    pub fn new() -> Self {
        Self::default()
    }

    /// Валидирует конфигурацию
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }

        if self.kernel_size % 2 == 0 {
            return Err(format!(
                "kernel_size ({}) must be odd",
                self.kernel_size
            ));
        }

        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(format!(
                "dropout ({}) must be in [0, 1]",
                self.dropout
            ));
        }

        if self.sampling_factor <= 0.0 {
            return Err(format!(
                "sampling_factor ({}) must be positive",
                self.sampling_factor
            ));
        }

        if self.seq_len == 0 {
            return Err("seq_len must be > 0".to_string());
        }

        if self.pred_len == 0 {
            return Err("pred_len must be > 0".to_string());
        }

        Ok(())
    }

    /// Возвращает размерность головы внимания
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Вычисляет финальную длину последовательности после distilling
    pub fn final_seq_len(&self) -> usize {
        if !self.use_distilling {
            return self.seq_len;
        }

        let mut seq_len = self.seq_len;
        for _ in 0..(self.n_encoder_layers - 1) {
            seq_len /= 2;
        }
        seq_len
    }

    /// Создаёт конфигурацию для маленькой модели (тестирование)
    pub fn small() -> Self {
        Self {
            seq_len: 48,
            label_len: 24,
            pred_len: 12,
            d_model: 32,
            n_heads: 2,
            d_ff: 64,
            n_encoder_layers: 2,
            ..Default::default()
        }
    }

    /// Создаёт конфигурацию для средней модели
    pub fn medium() -> Self {
        Self::default()
    }

    /// Создаёт конфигурацию для большой модели
    pub fn large() -> Self {
        Self {
            seq_len: 168,
            label_len: 84,
            pred_len: 48,
            d_model: 128,
            n_heads: 8,
            d_ff: 512,
            n_encoder_layers: 4,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = InformerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_d_model() {
        let config = InformerConfig {
            d_model: 65,  // Not divisible by n_heads (4)
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_kernel_size() {
        let config = InformerConfig {
            kernel_size: 4,  // Even number
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_head_dim() {
        let config = InformerConfig {
            d_model: 64,
            n_heads: 4,
            ..Default::default()
        };
        assert_eq!(config.head_dim(), 16);
    }

    #[test]
    fn test_final_seq_len() {
        let config = InformerConfig {
            seq_len: 96,
            n_encoder_layers: 3,
            use_distilling: true,
            ..Default::default()
        };
        // After 2 distilling operations: 96 -> 48 -> 24
        assert_eq!(config.final_seq_len(), 24);
    }

    #[test]
    fn test_small_config() {
        let config = InformerConfig::small();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_large_config() {
        let config = InformerConfig::large();
        assert!(config.validate().is_ok());
    }
}
