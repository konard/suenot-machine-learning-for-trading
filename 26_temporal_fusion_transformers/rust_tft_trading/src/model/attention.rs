//! Temporal Self-Attention
//!
//! Interpretable Multi-Head Attention для TFT.
//! Позволяет понять, какие временные точки важны для прогноза.

use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Multi-Head Attention (интерпретируемая версия для TFT)
#[derive(Debug)]
pub struct Attention {
    /// Размер скрытого слоя
    pub hidden_size: usize,

    /// Количество голов
    pub num_heads: usize,

    /// Размер одной головы
    pub head_size: usize,

    /// Веса для Query
    query_weights: Array2<f64>,

    /// Веса для Key
    key_weights: Array2<f64>,

    /// Веса для Value (в TFT это identity для интерпретируемости)
    value_weights: Array2<f64>,

    /// Веса для выхода
    output_weights: Array2<f64>,

    /// Dropout rate
    pub dropout: f64,
}

impl Attention {
    /// Создает новый attention слой
    pub fn new(hidden_size: usize, num_heads: usize, dropout: f64) -> Self {
        assert!(
            hidden_size % num_heads == 0,
            "hidden_size must be divisible by num_heads"
        );

        let head_size = hidden_size / num_heads;
        let mut rng = rand::thread_rng();

        let scale = (2.0 / (hidden_size * 2) as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();

        let query_weights = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.sample(normal));
        let key_weights = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.sample(normal));

        // В TFT value weights - это identity для интерпретируемости
        let value_weights = Array2::eye(hidden_size);

        let output_weights = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.sample(normal));

        Self {
            hidden_size,
            num_heads,
            head_size,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            dropout,
        }
    }

    /// Forward pass
    ///
    /// # Аргументы
    /// * `query` - Query tensor shape (seq_len, hidden_size)
    /// * `key` - Key tensor shape (seq_len, hidden_size)
    /// * `value` - Value tensor shape (seq_len, hidden_size)
    /// * `mask` - Опциональная маска для causal attention
    ///
    /// # Возвращает
    /// * (output, attention_weights)
    pub fn forward(
        &self,
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
        mask: Option<&Array2<f64>>,
    ) -> (Array2<f64>, Array2<f64>) {
        let seq_len = query.nrows();

        // Проецируем Q, K, V
        let q = query.dot(&self.query_weights.t());
        let k = key.dot(&self.key_weights.t());
        let v = value.dot(&self.value_weights.t());

        // Scaled dot-product attention
        // Для простоты делаем single-head версию
        let scale = (self.hidden_size as f64).sqrt();
        let scores = q.dot(&k.t()) / scale;

        // Применяем маску если есть
        let scores = if let Some(m) = mask {
            &scores + m
        } else {
            scores
        };

        // Softmax по последнему измерению
        let attention_weights = self.softmax_2d(&scores);

        // Взвешенная сумма values
        let context = attention_weights.dot(&v);

        // Output projection
        let output = context.dot(&self.output_weights.t());

        (output, attention_weights)
    }

    /// Softmax для 2D массива (по строкам)
    fn softmax_2d(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = x.clone();

        for mut row in result.rows_mut() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - max_val).exp());
            let sum: f64 = row.sum();
            row.mapv_inplace(|v| v / sum);
        }

        result
    }

    /// Создает causal mask (lower triangular)
    pub fn create_causal_mask(seq_len: usize) -> Array2<f64> {
        let mut mask = Array2::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[[i, j]] = f64::NEG_INFINITY;
            }
        }

        mask
    }

    /// Возвращает количество параметров
    pub fn num_parameters(&self) -> usize {
        self.query_weights.len()
            + self.key_weights.len()
            + self.value_weights.len()
            + self.output_weights.len()
    }
}

/// Interpretable Multi-Head Attention для TFT
#[derive(Debug)]
pub struct InterpretableMultiHeadAttention {
    /// Базовый attention
    attention: Attention,

    /// Сохраненные attention weights для интерпретации
    last_attention_weights: Option<Array2<f64>>,
}

impl InterpretableMultiHeadAttention {
    /// Создает новый IMHA
    pub fn new(hidden_size: usize, num_heads: usize, dropout: f64) -> Self {
        Self {
            attention: Attention::new(hidden_size, num_heads, dropout),
            last_attention_weights: None,
        }
    }

    /// Forward pass с сохранением весов
    pub fn forward(
        &mut self,
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
        mask: Option<&Array2<f64>>,
    ) -> Array2<f64> {
        let (output, weights) = self.attention.forward(query, key, value, mask);
        self.last_attention_weights = Some(weights);
        output
    }

    /// Возвращает последние attention weights
    pub fn get_attention_weights(&self) -> Option<&Array2<f64>> {
        self.last_attention_weights.as_ref()
    }

    /// Анализирует временную важность
    ///
    /// Возвращает средние веса внимания для каждого временного шага
    pub fn get_temporal_importance(&self) -> Option<Array1<f64>> {
        self.last_attention_weights
            .as_ref()
            .map(|w| w.mean_axis(Axis(0)).unwrap())
    }

    /// Находит самые важные временные точки для каждого прогноза
    pub fn get_top_important_steps(&self, top_k: usize) -> Option<Vec<Vec<usize>>> {
        self.last_attention_weights.as_ref().map(|weights| {
            weights
                .rows()
                .into_iter()
                .map(|row| {
                    let mut indexed: Vec<_> = row.iter().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                    indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
                })
                .collect()
        })
    }
}

/// Результат анализа внимания
#[derive(Debug, Clone)]
pub struct AttentionAnalysis {
    /// Матрица весов внимания
    pub weights: Array2<f64>,

    /// Средняя важность каждого временного шага
    pub temporal_importance: Array1<f64>,

    /// Топ-K важных шагов для каждого прогноза
    pub top_steps: Vec<Vec<usize>>,
}

impl AttentionAnalysis {
    /// Создает анализ из attention weights
    pub fn from_weights(weights: Array2<f64>, top_k: usize) -> Self {
        let temporal_importance = weights.mean_axis(Axis(0)).unwrap();

        let top_steps: Vec<Vec<usize>> = weights
            .rows()
            .into_iter()
            .map(|row| {
                let mut indexed: Vec<_> = row.iter().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
            })
            .collect();

        Self {
            weights,
            temporal_importance,
            top_steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_forward() {
        let attention = Attention::new(16, 4, 0.1);

        let seq_len = 10;
        let q = Array2::from_elem((seq_len, 16), 0.5);
        let k = Array2::from_elem((seq_len, 16), 0.5);
        let v = Array2::from_elem((seq_len, 16), 0.5);

        let (output, weights) = attention.forward(&q, &k, &v, None);

        assert_eq!(output.shape(), &[seq_len, 16]);
        assert_eq!(weights.shape(), &[seq_len, seq_len]);

        // Каждая строка весов должна суммироваться в 1
        for row in weights.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_causal_mask() {
        let mask = Attention::create_causal_mask(5);

        // Верхний треугольник должен быть -inf
        assert!(mask[[0, 1]].is_infinite());
        assert!(mask[[0, 4]].is_infinite());

        // Нижний треугольник и диагональ должны быть 0
        assert_eq!(mask[[1, 0]], 0.0);
        assert_eq!(mask[[2, 2]], 0.0);
    }

    #[test]
    fn test_interpretable_attention() {
        let mut imha = InterpretableMultiHeadAttention::new(16, 4, 0.1);

        let seq_len = 10;
        let q = Array2::from_elem((seq_len, 16), 0.5);
        let k = Array2::from_elem((seq_len, 16), 0.5);
        let v = Array2::from_elem((seq_len, 16), 0.5);

        let _ = imha.forward(&q, &k, &v, None);

        // Должны сохраниться веса
        assert!(imha.get_attention_weights().is_some());

        // Можем получить temporal importance
        let importance = imha.get_temporal_importance().unwrap();
        assert_eq!(importance.len(), seq_len);
    }

    #[test]
    fn test_attention_analysis() {
        let weights = Array2::from_shape_fn((5, 10), |(_i, j)| {
            if j < 5 { 0.15 } else { 0.05 }
        });

        // Нормализуем строки
        let mut normalized = weights.clone();
        for mut row in normalized.rows_mut() {
            let sum: f64 = row.sum();
            row.mapv_inplace(|v| v / sum);
        }

        let analysis = AttentionAnalysis::from_weights(normalized, 3);

        // Top steps должны содержать индексы с наибольшими весами
        for top in &analysis.top_steps {
            assert!(top.len() <= 3);
        }
    }
}
