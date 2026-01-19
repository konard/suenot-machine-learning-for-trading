//! Полная модель Informer с ProbSparse Attention
//!
//! Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
//! https://arxiv.org/abs/2012.07436
//!
//! Ключевые особенности:
//! 1. ProbSparse Self-Attention: O(L·log(L)) сложность
//! 2. Self-Attention Distilling: прогрессивное уменьшение последовательности
//! 3. Generative Style Decoder: эффективная генерация длинных прогнозов

use ndarray::{Array1, Array2, Array3};

use crate::model::{
    config::{InformerConfig, OutputType},
    attention::{ProbSparseAttention, AttentionDistilling, AttentionWeights},
    embedding::{TokenEmbedding, PositionalEncoding},
};

/// Слой энкодера Informer
#[derive(Debug)]
pub struct EncoderLayer {
    /// ProbSparse Self-Attention
    self_attention: ProbSparseAttention,
    /// LayerNorm 1
    norm1_gamma: Array1<f64>,
    norm1_beta: Array1<f64>,
    /// LayerNorm 2
    norm2_gamma: Array1<f64>,
    norm2_beta: Array1<f64>,
    /// Feed-Forward Network
    ff_w1: Array2<f64>,
    ff_b1: Array1<f64>,
    ff_w2: Array2<f64>,
    ff_b2: Array1<f64>,
    /// Distilling layer (optional)
    distilling: Option<AttentionDistilling>,
    /// Dropout rate
    dropout: f64,
}

impl EncoderLayer {
    /// Создаёт новый слой энкодера
    pub fn new(config: &InformerConfig, use_distilling: bool) -> Self {
        let d_model = config.d_model;
        let d_ff = config.d_ff;

        // Xavier инициализация
        let scale_ff = (2.0 / (d_model + d_ff) as f64).sqrt();

        Self {
            self_attention: ProbSparseAttention::new(config),
            norm1_gamma: Array1::ones(d_model),
            norm1_beta: Array1::zeros(d_model),
            norm2_gamma: Array1::ones(d_model),
            norm2_beta: Array1::zeros(d_model),
            ff_w1: Array2::from_shape_fn((d_model, d_ff), |_| rand_normal() * scale_ff),
            ff_b1: Array1::zeros(d_ff),
            ff_w2: Array2::from_shape_fn((d_ff, d_model), |_| rand_normal() * scale_ff),
            ff_b2: Array1::zeros(d_model),
            distilling: if use_distilling {
                Some(AttentionDistilling::new(d_model))
            } else {
                None
            },
            dropout: config.dropout,
        }
    }

    /// Прямой проход слоя энкодера
    pub fn forward(
        &self,
        x: &Array3<f64>,
        return_attention: bool,
    ) -> (Array3<f64>, Option<AttentionWeights>) {
        let (batch_size, seq_len, d_model) = x.dim();

        // 1. Self-Attention with residual
        let (attn_out, attn_weights) = self.self_attention.forward(x, return_attention);
        let mut out = self.add_norm(&x, &attn_out, &self.norm1_gamma, &self.norm1_beta);

        // 2. Feed-Forward with residual
        let ff_out = self.feed_forward(&out);
        out = self.add_norm(&out, &ff_out, &self.norm2_gamma, &self.norm2_beta);

        // 3. Distilling (if enabled)
        if let Some(ref distill) = self.distilling {
            out = distill.forward(&out);
        }

        (out, if return_attention { Some(attn_weights) } else { None })
    }

    /// Add & LayerNorm
    fn add_norm(
        &self,
        x: &Array3<f64>,
        residual: &Array3<f64>,
        gamma: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::zeros(x.dim());

        for b in 0..batch_size {
            for t in 0..seq_len {
                // Add residual
                let mut combined = Array1::zeros(d_model);
                for d in 0..d_model {
                    combined[d] = x[[b, t, d]] + residual[[b, t, d]];
                }

                // Compute mean and variance
                let mean = combined.mean().unwrap_or(0.0);
                let var = combined.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
                let std = (var + 1e-6).sqrt();

                // Normalize and scale
                for d in 0..d_model {
                    output[[b, t, d]] = gamma[d] * (combined[d] - mean) / std + beta[d];
                }
            }
        }

        output
    }

    /// Feed-Forward Network: Linear -> GELU -> Linear
    fn feed_forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let d_ff = self.ff_w1.dim().1;

        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                // First linear
                let mut hidden = Array1::zeros(d_ff);
                for f in 0..d_ff {
                    let mut sum = self.ff_b1[f];
                    for d in 0..d_model {
                        sum += x[[b, t, d]] * self.ff_w1[[d, f]];
                    }
                    // GELU activation
                    hidden[f] = gelu(sum);
                }

                // Second linear
                for d in 0..d_model {
                    let mut sum = self.ff_b2[d];
                    for f in 0..d_ff {
                        sum += hidden[f] * self.ff_w2[[f, d]];
                    }
                    output[[b, t, d]] = sum;
                }
            }
        }

        output
    }
}

/// Модель Informer
///
/// Полная реализация архитектуры Informer с ProbSparse Attention
/// для долгосрочного прогнозирования временных рядов.
#[derive(Debug)]
pub struct InformerModel {
    /// Конфигурация
    config: InformerConfig,
    /// Token embedding
    token_embedding: TokenEmbedding,
    /// Positional encoding
    positional_encoding: Option<PositionalEncoding>,
    /// Encoder layers
    encoder_layers: Vec<EncoderLayer>,
    /// Output projection weights [flatten_dim, output_dim]
    output_w1: Array2<f64>,
    output_b1: Array1<f64>,
    output_w2: Array2<f64>,
    output_b2: Array1<f64>,
}

impl InformerModel {
    /// Создаёт новую модель Informer
    pub fn new(config: InformerConfig) -> Self {
        config.validate().expect("Invalid config");

        let token_embedding = TokenEmbedding::new(&config);
        let positional_encoding = if config.use_positional_encoding {
            Some(PositionalEncoding::new(
                config.d_model,
                config.seq_len * 2,
                config.dropout,
            ))
        } else {
            None
        };

        // Создаём encoder layers с distilling
        let mut encoder_layers = Vec::with_capacity(config.n_encoder_layers);
        for i in 0..config.n_encoder_layers {
            // Последний слой не делает distilling
            let use_distilling = config.use_distilling && (i < config.n_encoder_layers - 1);
            encoder_layers.push(EncoderLayer::new(&config, use_distilling));
        }

        // Вычисляем размер выхода после distilling
        let final_seq_len = config.final_seq_len();
        let flatten_dim = config.d_model * final_seq_len;

        // Output head
        let output_dim = match config.output_type {
            OutputType::Quantile => config.pred_len * config.quantiles.len(),
            OutputType::Direction => config.pred_len * 3,
            OutputType::Regression => config.pred_len,
        };

        let scale = (2.0 / (flatten_dim + config.d_ff) as f64).sqrt();
        let output_w1 = Array2::from_shape_fn(
            (flatten_dim, config.d_ff),
            |_| rand_normal() * scale
        );
        let output_b1 = Array1::zeros(config.d_ff);

        let scale2 = (2.0 / (config.d_ff + output_dim) as f64).sqrt();
        let output_w2 = Array2::from_shape_fn(
            (config.d_ff, output_dim),
            |_| rand_normal() * scale2
        );
        let output_b2 = Array1::zeros(output_dim);

        Self {
            config,
            token_embedding,
            positional_encoding,
            encoder_layers,
            output_w1,
            output_b1,
            output_w2,
            output_b2,
        }
    }

    /// Прямой проход модели
    ///
    /// # Arguments
    ///
    /// * `x` - Входной тензор [batch, seq_len, features]
    /// * `return_attention` - Возвращать ли веса внимания
    ///
    /// # Returns
    ///
    /// * `predictions` - Прогнозы [batch, pred_len] или [batch, pred_len, n_outputs]
    /// * `attention_weights` - Опциональные веса внимания каждого слоя
    pub fn forward(
        &self,
        x: &Array3<f64>,
        return_attention: bool,
    ) -> (Array2<f64>, Vec<AttentionWeights>) {
        let batch_size = x.dim().0;

        // Token embedding
        let mut hidden = self.token_embedding.forward(x);

        // Positional encoding
        if let Some(ref pe) = self.positional_encoding {
            hidden = pe.forward_no_dropout(&hidden);
        }

        // Encoder layers
        let mut all_attention = Vec::new();
        for layer in &self.encoder_layers {
            let (out, attn) = layer.forward(&hidden, return_attention);
            hidden = out;
            if let Some(a) = attn {
                all_attention.push(a);
            }
        }

        // Flatten and output projection
        let (_, final_seq_len, d_model) = hidden.dim();
        let flatten_dim = final_seq_len * d_model;
        let output_dim = self.output_w2.dim().1;

        let mut predictions = Array2::zeros((batch_size, output_dim));

        for b in 0..batch_size {
            // Flatten
            let mut flat = Array1::zeros(flatten_dim);
            for t in 0..final_seq_len {
                for d in 0..d_model {
                    flat[t * d_model + d] = hidden[[b, t, d]];
                }
            }

            // First linear + GELU
            let d_ff = self.output_w1.dim().1;
            let mut hidden_ff = Array1::zeros(d_ff);
            for f in 0..d_ff {
                let mut sum = self.output_b1[f];
                for i in 0..flatten_dim {
                    sum += flat[i] * self.output_w1[[i, f]];
                }
                hidden_ff[f] = gelu(sum);
            }

            // Second linear
            for o in 0..output_dim {
                let mut sum = self.output_b2[o];
                for f in 0..d_ff {
                    sum += hidden_ff[f] * self.output_w2[[f, o]];
                }
                predictions[[b, o]] = sum;
            }
        }

        (predictions, all_attention)
    }

    /// Предсказание без возврата attention weights
    pub fn predict(&self, x: &Array3<f64>) -> Array2<f64> {
        let (predictions, _) = self.forward(x, false);
        predictions
    }

    /// Возвращает конфигурацию модели
    pub fn config(&self) -> &InformerConfig {
        &self.config
    }

    /// Возвращает количество параметров модели
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        // Embedding (approximation)
        count += self.config.d_model * self.config.input_features * self.config.kernel_size;

        // Each encoder layer
        let d_model = self.config.d_model;
        let d_ff = self.config.d_ff;

        for _ in 0..self.config.n_encoder_layers {
            // Attention projections: 4 * d_model^2
            count += 4 * d_model * d_model;
            // Layer norms: 4 * d_model
            count += 4 * d_model;
            // Feed forward: d_model * d_ff + d_ff * d_model
            count += 2 * d_model * d_ff;
            // Biases
            count += d_ff + d_model;
        }

        // Output head
        let final_seq_len = self.config.final_seq_len();
        let flatten_dim = d_model * final_seq_len;
        count += flatten_dim * d_ff + d_ff;
        count += d_ff * self.output_w2.dim().1 + self.output_w2.dim().1;

        count
    }
}

/// GELU activation function
fn gelu(x: f64) -> f64 {
    use std::f64::consts::PI;
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
}

/// Генерирует случайное число из стандартного нормального распределения
fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_informer_model_creation() {
        let config = InformerConfig::small();
        let model = InformerModel::new(config);

        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_informer_forward_shape() {
        let config = InformerConfig {
            seq_len: 32,
            pred_len: 8,
            input_features: 6,
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            n_encoder_layers: 2,
            use_distilling: true,
            ..Default::default()
        };

        let model = InformerModel::new(config.clone());

        // [batch=2, seq_len=32, features=6]
        let x = Array3::from_shape_fn((2, 32, 6), |_| rand_normal());

        let predictions = model.predict(&x);

        // Output should be [batch=2, pred_len=8]
        assert_eq!(predictions.dim(), (2, 8));
    }

    #[test]
    fn test_informer_with_attention() {
        let config = InformerConfig::small();
        let model = InformerModel::new(config.clone());

        let x = Array3::from_shape_fn((1, config.seq_len, config.input_features), |_| rand_normal());

        let (predictions, attention) = model.forward(&x, true);

        assert_eq!(predictions.dim().0, 1);
        assert_eq!(attention.len(), config.n_encoder_layers);
    }

    #[test]
    fn test_informer_different_output_types() {
        // Regression
        let config = InformerConfig {
            output_type: OutputType::Regression,
            pred_len: 12,
            ..InformerConfig::small()
        };
        let model = InformerModel::new(config.clone());
        let x = Array3::from_shape_fn((1, config.seq_len, config.input_features), |_| rand_normal());
        let pred = model.predict(&x);
        assert_eq!(pred.dim(), (1, 12));

        // Direction
        let config = InformerConfig {
            output_type: OutputType::Direction,
            pred_len: 12,
            ..InformerConfig::small()
        };
        let model = InformerModel::new(config.clone());
        let x = Array3::from_shape_fn((1, config.seq_len, config.input_features), |_| rand_normal());
        let pred = model.predict(&x);
        assert_eq!(pred.dim(), (1, 12 * 3));

        // Quantile
        let config = InformerConfig {
            output_type: OutputType::Quantile,
            pred_len: 12,
            quantiles: vec![0.1, 0.5, 0.9],
            ..InformerConfig::small()
        };
        let model = InformerModel::new(config.clone());
        let x = Array3::from_shape_fn((1, config.seq_len, config.input_features), |_| rand_normal());
        let pred = model.predict(&x);
        assert_eq!(pred.dim(), (1, 12 * 3));
    }

    #[test]
    fn test_encoder_layer() {
        let config = InformerConfig::small();
        let layer = EncoderLayer::new(&config, true);

        let x = Array3::from_shape_fn((2, config.seq_len, config.d_model), |_| rand_normal());

        let (output, _) = layer.forward(&x, false);

        // With distilling, output should be half the sequence length
        assert_eq!(output.dim(), (2, config.seq_len / 2, config.d_model));
    }

    #[test]
    fn test_no_nan_in_output() {
        let config = InformerConfig::small();
        let model = InformerModel::new(config.clone());

        let x = Array3::from_shape_fn((2, config.seq_len, config.input_features), |_| rand_normal());

        let predictions = model.predict(&x);

        for val in predictions.iter() {
            assert!(!val.is_nan(), "Output contains NaN");
            assert!(!val.is_infinite(), "Output contains Inf");
        }
    }
}
