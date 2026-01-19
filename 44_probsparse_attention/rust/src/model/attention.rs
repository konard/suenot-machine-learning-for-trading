//! ProbSparse Attention Mechanism
//!
//! Реализация эффективного механизма внимания из статьи Informer
//! с вычислительной сложностью O(L·log(L)) вместо O(L²).
//!
//! Ключевая идея: не все запросы (queries) одинаково информативны.
//! "Активные" запросы имеют "острое" распределение внимания,
//! "ленивые" - близкое к равномерному.
//!
//! Метрика разреженности: M(q, K) = max(qK^T) - mean(qK^T)
//! аппроксимирует KL-дивергенцию от равномерного распределения.

use ndarray::{Array1, Array2, Array3, Array4, Axis, s};
use std::cmp::Ordering;

use crate::model::config::InformerConfig;

/// Веса внимания для интерпретации
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Временные веса внимания [batch, n_heads, seq_len, seq_len]
    pub temporal_weights: Option<Array4<f64>>,
    /// Индексы выбранных запросов [batch, n_heads, u]
    pub selected_indices: Option<Array3<usize>>,
}

impl AttentionWeights {
    pub fn new() -> Self {
        Self {
            temporal_weights: None,
            selected_indices: None,
        }
    }

    /// Возвращает средние веса внимания по всем головам
    pub fn mean_attention(&self) -> Option<Array3<f64>> {
        self.temporal_weights.as_ref().map(|w| {
            w.mean_axis(Axis(1)).unwrap()
        })
    }

    /// Возвращает топ-k позиций с наибольшим вниманием для каждой позиции
    pub fn top_k_positions(&self, k: usize) -> Vec<Vec<(usize, f64)>> {
        let mut results = Vec::new();

        if let Some(ref weights) = self.temporal_weights {
            // Усредняем по batch и heads
            let mean_weights = weights.mean_axis(Axis(0)).unwrap()
                .mean_axis(Axis(0)).unwrap();

            let seq_len = mean_weights.dim().0;

            for i in 0..seq_len {
                let mut positions: Vec<(usize, f64)> = (0..seq_len)
                    .map(|j| (j, mean_weights[[i, j]]))
                    .collect();

                positions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                positions.truncate(k);
                results.push(positions);
            }
        }

        results
    }
}

impl Default for AttentionWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// ProbSparse Self-Attention
///
/// Эффективный механизм внимания, выбирающий только top-u
/// наиболее информативных запросов для полного вычисления.
///
/// # Complexity
///
/// - Традиционное внимание: O(L²)
/// - ProbSparse: O(L·log(L))
///
/// # Algorithm
///
/// 1. Быстрый скан: вычисляем M(q, K) для всех запросов
/// 2. Выбираем top-u запросов с наибольшим M
/// 3. Полное внимание только для выбранных запросов
/// 4. Для остальных - среднее значение (lazy queries approximation)
#[derive(Debug, Clone)]
pub struct ProbSparseAttention {
    /// Query проекция [d_model, d_model]
    w_q: Array2<f64>,
    /// Key проекция [d_model, d_model]
    w_k: Array2<f64>,
    /// Value проекция [d_model, d_model]
    w_v: Array2<f64>,
    /// Output проекция [d_model, d_model]
    w_o: Array2<f64>,
    /// Количество голов
    n_heads: usize,
    /// Размерность головы
    head_dim: usize,
    /// Масштабирующий коэффициент sqrt(head_dim)
    scale: f64,
    /// Фактор разреженности
    sampling_factor: f64,
}

impl ProbSparseAttention {
    /// Создаёт новый слой ProbSparse Attention
    ///
    /// # Arguments
    ///
    /// * `config` - Конфигурация модели
    pub fn new(config: &InformerConfig) -> Self {
        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let head_dim = d_model / n_heads;

        // Xavier инициализация
        let scale_init = (2.0 / (d_model * 2) as f64).sqrt();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            n_heads,
            head_dim,
            scale: (head_dim as f64).sqrt(),
            sampling_factor: config.sampling_factor,
        }
    }

    /// Прямой проход ProbSparse Attention
    ///
    /// # Arguments
    ///
    /// * `x` - Входной тензор [batch, seq_len, d_model]
    /// * `return_attention` - Возвращать ли веса внимания
    ///
    /// # Returns
    ///
    /// * `output` - Выход [batch, seq_len, d_model]
    /// * `weights` - Опциональные веса внимания
    pub fn forward(&self, x: &Array3<f64>, return_attention: bool) -> (Array3<f64>, AttentionWeights) {
        let (batch_size, seq_len, d_model) = x.dim();

        // Линейные проекции Q, K, V
        let q = self.linear_transform(x, &self.w_q);
        let k = self.linear_transform(x, &self.w_k);
        let v = self.linear_transform(x, &self.w_v);

        // Вычисляем количество активных запросов u = c·log(L)
        let u = ((self.sampling_factor * (seq_len as f64 + 1.0).ln()).ceil() as usize)
            .max(1)
            .min(seq_len);

        let mut output = Array3::zeros((batch_size, seq_len, d_model));
        let mut all_weights = if return_attention {
            Some(Array4::zeros((batch_size, self.n_heads, seq_len, seq_len)))
        } else {
            None
        };
        let mut all_indices = if return_attention {
            Some(Array3::zeros((batch_size, self.n_heads, u)))
        } else {
            None
        };

        for b in 0..batch_size {
            for h in 0..self.n_heads {
                let h_start = h * self.head_dim;
                let h_end = (h + 1) * self.head_dim;

                // Извлекаем Q, K, V для текущей головы
                let q_h = q.slice(s![b, .., h_start..h_end]);
                let k_h = k.slice(s![b, .., h_start..h_end]);
                let v_h = v.slice(s![b, .., h_start..h_end]);

                // Шаг 1: Вычисляем sparsity measurement M(q, K)
                let (top_indices, _measurements) = self.select_top_queries(&q_h, &k_h, u);

                // Сохраняем индексы
                if let Some(ref mut indices) = all_indices {
                    for (i, &idx) in top_indices.iter().enumerate() {
                        indices[[b, h, i]] = idx;
                    }
                }

                // Шаг 2: Полное внимание только для выбранных запросов
                for &i in &top_indices {
                    // Вычисляем scores: Q[i] @ K^T / sqrt(d)
                    let mut scores = Array1::<f64>::zeros(seq_len);
                    for j in 0..seq_len {
                        let mut score = 0.0;
                        for d in 0..self.head_dim {
                            score += q_h[[i, d]] * k_h[[j, d]];
                        }
                        scores[j] = score / self.scale;
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_scores: Array1<f64> = scores.mapv(|s| (s - max_score).exp());
                    let sum_exp = exp_scores.sum();
                    let attn_probs = exp_scores.mapv(|e| e / sum_exp);

                    // Сохраняем веса внимания
                    if let Some(ref mut weights) = all_weights {
                        for j in 0..seq_len {
                            weights[[b, h, i, j]] = attn_probs[j];
                        }
                    }

                    // Вычисляем взвешенную сумму значений
                    for d in h_start..h_end {
                        let d_local = d - h_start;
                        let mut sum = 0.0;
                        for j in 0..seq_len {
                            sum += attn_probs[j] * v_h[[j, d_local]];
                        }
                        output[[b, i, d]] = sum;
                    }
                }

                // Шаг 3: Для ленивых запросов используем среднее значение V
                let mean_v = v_h.mean_axis(Axis(0)).unwrap();

                for i in 0..seq_len {
                    if !top_indices.contains(&i) {
                        for d in h_start..h_end {
                            let d_local = d - h_start;
                            output[[b, i, d]] = mean_v[d_local];
                        }
                    }
                }
            }
        }

        // Выходная проекция
        let projected = self.linear_transform(&output, &self.w_o);

        let weights = AttentionWeights {
            temporal_weights: all_weights,
            selected_indices: all_indices,
        };

        (projected, weights)
    }

    /// Линейное преобразование для 3D тензора
    fn linear_transform(&self, x: &Array3<f64>, w: &Array2<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_in) = x.dim();
        let d_out = w.dim().1;
        let mut output = Array3::zeros((batch_size, seq_len, d_out));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_o in 0..d_out {
                    let mut sum = 0.0;
                    for d_i in 0..d_in {
                        sum += x[[b, t, d_i]] * w[[d_i, d_o]];
                    }
                    output[[b, t, d_o]] = sum;
                }
            }
        }

        output
    }

    /// Выбирает top-u запросов по sparsity measurement M(q, K)
    ///
    /// M(q_i, K) = max_j(q_i · k_j^T / sqrt(d)) - mean_j(q_i · k_j^T / sqrt(d))
    ///
    /// Высокий M означает "острое" распределение внимания = важный запрос
    fn select_top_queries(
        &self,
        q: &ndarray::ArrayView2<f64>,
        k: &ndarray::ArrayView2<f64>,
        u: usize,
    ) -> (Vec<usize>, Vec<f64>) {
        let seq_len = q.dim().0;
        let head_dim = q.dim().1;

        // Семплируем подмножество ключей для эффективности
        let u_part = ((self.sampling_factor * seq_len as f64 * (seq_len as f64 + 1.0).ln()).ceil() as usize)
            .min(seq_len);

        // Генерируем случайные индексы для семплирования
        let sample_indices: Vec<usize> = if u_part < seq_len {
            use rand::seq::SliceRandom;
            let mut indices: Vec<usize> = (0..seq_len).collect();
            indices.shuffle(&mut rand::thread_rng());
            indices.truncate(u_part);
            indices
        } else {
            (0..seq_len).collect()
        };

        // Вычисляем M(q_i) для каждого запроса
        let mut measurements: Vec<(usize, f64)> = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            // Вычисляем scores относительно семплированных ключей
            let mut scores: Vec<f64> = Vec::with_capacity(sample_indices.len());

            for &j in &sample_indices {
                let mut score = 0.0;
                for d in 0..head_dim {
                    score += q[[i, d]] * k[[j, d]];
                }
                scores.push(score / self.scale);
            }

            // M(q) = max(scores) - mean(scores)
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

            measurements.push((i, max_score - mean_score));
        }

        // Сортируем по M в убывающем порядке
        measurements.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Возвращаем top-u индексов и их значения M
        let top_indices: Vec<usize> = measurements.iter().take(u).map(|(idx, _)| *idx).collect();
        let top_measurements: Vec<f64> = measurements.iter().take(u).map(|(_, m)| *m).collect();

        (top_indices, top_measurements)
    }
}

/// Attention Distilling Layer
///
/// Уменьшает длину последовательности вдвое после каждого слоя encoder,
/// извлекая наиболее важную информацию.
///
/// Структура: Conv1d -> BatchNorm -> ELU -> MaxPool(2)
///
/// После N слоёв encoder с distilling:
/// - Общая сложность: O(L + L/2 + L/4 + ...) = O(2L) = O(L)
/// - Память: значительно уменьшается с каждым слоем
#[derive(Debug, Clone)]
pub struct AttentionDistilling {
    /// Веса свёртки [d_model, d_model, kernel_size]
    conv_weights: Array3<f64>,
    /// Bias свёртки [d_model]
    conv_bias: Array1<f64>,
    /// Размер ядра
    kernel_size: usize,
    /// Padding
    padding: usize,
}

impl AttentionDistilling {
    /// Создаёт новый слой distilling
    pub fn new(d_model: usize) -> Self {
        let kernel_size = 3;
        let padding = kernel_size / 2;

        // Xavier инициализация
        let scale = (2.0 / (d_model * kernel_size) as f64).sqrt();

        let conv_weights = Array3::from_shape_fn(
            (d_model, d_model, kernel_size),
            |_| rand_normal() * scale
        );
        let conv_bias = Array1::zeros(d_model);

        Self {
            conv_weights,
            conv_bias,
            kernel_size,
            padding,
        }
    }

    /// Прямой проход distilling
    ///
    /// # Arguments
    ///
    /// * `x` - Входной тензор [batch, seq_len, d_model]
    ///
    /// # Returns
    ///
    /// * `output` - Выход [batch, seq_len/2, d_model]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();

        // Применяем padding (circular для временных рядов)
        let padded_len = seq_len + 2 * self.padding;
        let mut padded = Array3::zeros((batch_size, padded_len, d_model));

        for b in 0..batch_size {
            // Circular padding
            for p in 0..self.padding {
                for d in 0..d_model {
                    padded[[b, p, d]] = x[[b, seq_len - self.padding + p, d]];
                    padded[[b, padded_len - self.padding + p, d]] = x[[b, p, d]];
                }
            }
            for t in 0..seq_len {
                for d in 0..d_model {
                    padded[[b, self.padding + t, d]] = x[[b, t, d]];
                }
            }
        }

        // 1D Convolution
        let mut conv_out = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_out in 0..d_model {
                    let mut sum = self.conv_bias[d_out];
                    for k in 0..self.kernel_size {
                        for d_in in 0..d_model {
                            sum += padded[[b, self.padding + t - self.padding + k, d_in]]
                                * self.conv_weights[[d_out, d_in, k]];
                        }
                    }
                    // ELU activation
                    conv_out[[b, t, d_out]] = if sum > 0.0 { sum } else { sum.exp() - 1.0 };
                }
            }
        }

        // MaxPool with stride 2
        let out_len = seq_len / 2;
        let mut output = Array3::zeros((batch_size, out_len, d_model));

        for b in 0..batch_size {
            for t in 0..out_len {
                for d in 0..d_model {
                    let val1 = conv_out[[b, t * 2, d]];
                    let val2 = conv_out[[b, t * 2 + 1, d]];
                    output[[b, t, d]] = val1.max(val2);
                }
            }
        }

        output
    }
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

    fn test_config() -> InformerConfig {
        InformerConfig {
            d_model: 32,
            n_heads: 4,
            sampling_factor: 5.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_probsparse_attention_shape() {
        let config = test_config();
        let attn = ProbSparseAttention::new(&config);

        // [batch=2, seq_len=16, d_model=32]
        let x = Array3::from_shape_fn((2, 16, 32), |_| rand_normal());

        let (output, _weights) = attn.forward(&x, false);

        assert_eq!(output.dim(), (2, 16, 32));
    }

    #[test]
    fn test_probsparse_attention_weights() {
        let config = test_config();
        let attn = ProbSparseAttention::new(&config);

        let x = Array3::from_shape_fn((1, 16, 32), |_| rand_normal());

        let (_, weights) = attn.forward(&x, true);

        assert!(weights.temporal_weights.is_some());

        let tw = weights.temporal_weights.unwrap();
        assert_eq!(tw.dim(), (1, 4, 16, 16));

        // Проверяем, что выбранные запросы имеют нормализованные веса (сумма = 1)
        let u = ((5.0 * 17.0_f64.ln()).ceil() as usize).max(1).min(16);
        for h in 0..4 {
            let mut active_count = 0;
            for i in 0..16 {
                let row_sum: f64 = (0..16).map(|j| tw[[0, h, i, j]]).sum();
                if row_sum > 0.5 {  // Активный запрос
                    active_count += 1;
                    // Проверяем softmax (сумма ~ 1)
                    assert!((row_sum - 1.0).abs() < 1e-6,
                        "Softmax sum should be 1, got {}", row_sum);
                }
            }
            assert!(active_count <= u, "Should have at most {} active queries, got {}",
                u, active_count);
        }
    }

    #[test]
    fn test_attention_distilling_shape() {
        let distill = AttentionDistilling::new(32);

        // [batch=2, seq_len=16, d_model=32]
        let x = Array3::from_shape_fn((2, 16, 32), |_| rand_normal());

        let output = distill.forward(&x);

        // Output should be half the sequence length
        assert_eq!(output.dim(), (2, 8, 32));
    }

    #[test]
    fn test_attention_distilling_values() {
        let distill = AttentionDistilling::new(8);

        let x = Array3::from_shape_fn((1, 4, 8), |_| rand_normal());

        let output = distill.forward(&x);

        // Проверяем, что output не содержит NaN
        for val in output.iter() {
            assert!(!val.is_nan(), "Output contains NaN");
            assert!(!val.is_infinite(), "Output contains Inf");
        }
    }

    #[test]
    fn test_top_k_positions() {
        let mut weights = AttentionWeights::new();

        // Создаём тестовые веса с известным паттерном
        let mut tw = Array4::zeros((1, 1, 4, 4));

        // Позиция 0 обращает внимание на позицию 2
        tw[[0, 0, 0, 2]] = 0.9;
        tw[[0, 0, 0, 0]] = 0.05;
        tw[[0, 0, 0, 1]] = 0.03;
        tw[[0, 0, 0, 3]] = 0.02;

        weights.temporal_weights = Some(tw);

        let top = weights.top_k_positions(2);

        assert_eq!(top.len(), 4);
        // Первая позиция должна иметь позицию 2 как top-1
        assert_eq!(top[0][0].0, 2);
        assert!((top[0][0].1 - 0.9).abs() < 1e-6);
    }
}
