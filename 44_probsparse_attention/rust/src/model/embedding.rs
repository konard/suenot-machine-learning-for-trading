//! Слои эмбеддинга для модели Informer
//!
//! - TokenEmbedding: преобразование входных признаков в d_model
//! - PositionalEncoding: синусоидальное позиционное кодирование

use ndarray::{Array1, Array2, Array3};
use std::f64::consts::PI;

use crate::model::config::InformerConfig;

/// Token Embedding с использованием 1D свёртки
///
/// Преобразует [batch, seq_len, features] в [batch, seq_len, d_model]
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    /// Веса свёртки [d_model, input_features, kernel_size]
    conv_weights: Array3<f64>,
    /// Bias [d_model]
    conv_bias: Array1<f64>,
    /// Размер ядра
    kernel_size: usize,
    /// Padding
    padding: usize,
}

impl TokenEmbedding {
    /// Создаёт новый слой эмбеддинга
    pub fn new(config: &InformerConfig) -> Self {
        let kernel_size = config.kernel_size;
        let padding = kernel_size / 2;

        // Xavier инициализация
        let scale = (2.0 / (config.input_features + config.d_model) as f64).sqrt();

        let conv_weights = Array3::from_shape_fn(
            (config.d_model, config.input_features, kernel_size),
            |_| rand_normal() * scale
        );
        let conv_bias = Array1::zeros(config.d_model);

        Self {
            conv_weights,
            conv_bias,
            kernel_size,
            padding,
        }
    }

    /// Прямой проход
    ///
    /// # Arguments
    ///
    /// * `x` - Входной тензор [batch, seq_len, features]
    ///
    /// # Returns
    ///
    /// * `output` - Эмбеддинги [batch, seq_len, d_model]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, in_features) = x.dim();
        let d_model = self.conv_weights.dim().0;

        // Padding
        let padded_len = seq_len + 2 * self.padding;
        let mut padded = Array3::zeros((batch_size, padded_len, in_features));

        for b in 0..batch_size {
            // Zero padding
            for t in 0..seq_len {
                for f in 0..in_features {
                    padded[[b, self.padding + t, f]] = x[[b, t, f]];
                }
            }
        }

        // 1D Convolution
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_out in 0..d_model {
                    let mut sum = self.conv_bias[d_out];
                    for k in 0..self.kernel_size {
                        for d_in in 0..in_features {
                            sum += padded[[b, t + k, d_in]]
                                * self.conv_weights[[d_out, d_in, k]];
                        }
                    }
                    // GELU activation approximation
                    output[[b, t, d_out]] = gelu(sum);
                }
            }
        }

        output
    }
}

/// Синусоидальное позиционное кодирование
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Таблица позиционных кодировок [max_len, d_model]
    pe: Array2<f64>,
    /// Dropout rate
    dropout: f64,
}

impl PositionalEncoding {
    /// Создаёт позиционное кодирование
    pub fn new(d_model: usize, max_len: usize, dropout: f64) -> Self {
        let mut pe = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let div_term = (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
                let angle = pos as f64 / div_term;

                pe[[pos, 2 * i]] = angle.sin();
                if 2 * i + 1 < d_model {
                    pe[[pos, 2 * i + 1]] = angle.cos();
                }
            }
        }

        Self { pe, dropout }
    }

    /// Прямой проход
    ///
    /// # Arguments
    ///
    /// * `x` - Входной тензор [batch, seq_len, d_model]
    ///
    /// # Returns
    ///
    /// * `output` - С добавленным позиционным кодированием
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = x.clone();

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..d_model {
                    output[[b, t, d]] += self.pe[[t, d]];

                    // Dropout (для training)
                    if self.dropout > 0.0 && rand::random::<f64>() < self.dropout {
                        output[[b, t, d]] = 0.0;
                    }
                }
            }
        }

        output
    }

    /// Прямой проход без dropout (для inference)
    pub fn forward_no_dropout(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = x.clone();

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..d_model {
                    output[[b, t, d]] += self.pe[[t, d]];
                }
            }
        }

        output
    }
}

/// GELU activation function
///
/// GELU(x) = x * Φ(x), где Φ - CDF стандартного нормального распределения
/// Аппроксимация: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
fn gelu(x: f64) -> f64 {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
}

/// Генерирует случайное число из стандартного нормального распределения
fn rand_normal() -> f64 {
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embedding_shape() {
        let config = InformerConfig {
            input_features: 6,
            d_model: 32,
            kernel_size: 3,
            ..Default::default()
        };

        let embedding = TokenEmbedding::new(&config);

        // [batch=2, seq_len=16, features=6]
        let x = Array3::from_shape_fn((2, 16, 6), |_| rand_normal());

        let output = embedding.forward(&x);

        assert_eq!(output.dim(), (2, 16, 32));
    }

    #[test]
    fn test_positional_encoding_shape() {
        let pe = PositionalEncoding::new(32, 100, 0.0);

        let x = Array3::from_shape_fn((2, 16, 32), |_| rand_normal());

        let output = pe.forward_no_dropout(&x);

        assert_eq!(output.dim(), (2, 16, 32));
    }

    #[test]
    fn test_positional_encoding_different() {
        let pe = PositionalEncoding::new(32, 100, 0.0);

        // Проверяем, что разные позиции имеют разные кодировки
        assert!((pe.pe[[0, 0]] - pe.pe[[1, 0]]).abs() > 1e-6);
        assert!((pe.pe[[0, 1]] - pe.pe[[1, 1]]).abs() > 1e-6);
    }

    #[test]
    fn test_positional_encoding_sin_cos() {
        let pe = PositionalEncoding::new(4, 10, 0.0);

        // Проверяем, что sin/cos паттерн соблюдается
        // PE[0, 0] должен быть sin(0) = 0
        assert!(pe.pe[[0, 0]].abs() < 1e-10);

        // PE[0, 1] должен быть cos(0) = 1
        assert!((pe.pe[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gelu() {
        // GELU(0) ≈ 0
        assert!(gelu(0.0).abs() < 1e-10);

        // GELU positive for positive input
        assert!(gelu(1.0) > 0.0);

        // GELU close to x for large positive x
        assert!((gelu(3.0) - 3.0).abs() < 0.1);

        // GELU small negative for negative input
        assert!(gelu(-1.0) < 0.0);
        assert!(gelu(-1.0).abs() < 0.2);
    }
}
