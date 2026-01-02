//! Слои для WaveNet

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// 1D Свёртка
#[derive(Debug, Clone)]
pub struct Conv1D {
    pub weights: Vec<Vec<f64>>,  // [channels_out][channels_in * kernel_size]
    pub bias: Vec<f64>,
    pub kernel_size: usize,
    pub channels_in: usize,
    pub channels_out: usize,
}

impl Conv1D {
    /// Создать новый слой с инициализацией Xavier
    pub fn new(channels_in: usize, channels_out: usize, kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (channels_in * kernel_size + channels_out) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = (0..channels_out)
            .map(|_| {
                (0..channels_in * kernel_size)
                    .map(|_| normal.sample(&mut rng))
                    .collect()
            })
            .collect();

        let bias = vec![0.0; channels_out];

        Self {
            weights,
            bias,
            kernel_size,
            channels_in,
            channels_out,
        }
    }

    /// Прямой проход (с причинным padding)
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = input[0].len();
        let mut output = vec![vec![0.0; seq_len]; self.channels_out];

        for t in 0..seq_len {
            for c_out in 0..self.channels_out {
                let mut sum = self.bias[c_out];

                for k in 0..self.kernel_size {
                    let idx = t as i64 - k as i64;
                    if idx >= 0 {
                        for c_in in 0..self.channels_in {
                            sum += self.weights[c_out][c_in * self.kernel_size + k]
                                * input[c_in][idx as usize];
                        }
                    }
                }

                output[c_out][t] = sum;
            }
        }

        output
    }
}

/// Расширенная 1D свёртка (Dilated Convolution)
#[derive(Debug, Clone)]
pub struct DilatedConv1D {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub kernel_size: usize,
    pub dilation: usize,
    pub channels_in: usize,
    pub channels_out: usize,
}

impl DilatedConv1D {
    /// Создать новый расширенный свёрточный слой
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (channels_in * kernel_size + channels_out) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = (0..channels_out)
            .map(|_| {
                (0..channels_in * kernel_size)
                    .map(|_| normal.sample(&mut rng))
                    .collect()
            })
            .collect();

        let bias = vec![0.0; channels_out];

        Self {
            weights,
            bias,
            kernel_size,
            dilation,
            channels_in,
            channels_out,
        }
    }

    /// Прямой проход с расширением
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = input[0].len();
        let mut output = vec![vec![0.0; seq_len]; self.channels_out];

        for t in 0..seq_len {
            for c_out in 0..self.channels_out {
                let mut sum = self.bias[c_out];

                for k in 0..self.kernel_size {
                    let idx = t as i64 - (k * self.dilation) as i64;
                    if idx >= 0 {
                        for c_in in 0..self.channels_in {
                            sum += self.weights[c_out][c_in * self.kernel_size + k]
                                * input[c_in][idx as usize];
                        }
                    }
                }

                output[c_out][t] = sum;
            }
        }

        output
    }

    /// Рецептивное поле этого слоя
    pub fn receptive_field(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }
}

/// Полносвязный слой (Dense/Linear)
#[derive(Debug, Clone)]
pub struct Dense {
    pub weights: Vec<Vec<f64>>,  // [out_features][in_features]
    pub bias: Vec<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Dense {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (in_features + out_features) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = (0..out_features)
            .map(|_| (0..in_features).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        let bias = vec![0.0; out_features];

        Self {
            weights,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = self.bias.clone();

        for (i, w) in self.weights.iter().enumerate() {
            for (j, &x) in input.iter().enumerate() {
                output[i] += w[j] * x;
            }
        }

        output
    }
}

/// Layer Normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: vec![1.0; size],
            beta: vec![0.0; size],
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64;
        let std = (variance + self.eps).sqrt();

        input
            .iter()
            .zip(self.gamma.iter())
            .zip(self.beta.iter())
            .map(|((x, g), b)| (x - mean) / std * g + b)
            .collect()
    }
}

/// Dropout (для обучения)
pub struct Dropout {
    pub rate: f64,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }

    pub fn forward(&self, input: &[f64], training: bool) -> Vec<f64> {
        if !training || self.rate <= 0.0 {
            return input.to_vec();
        }

        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.rate);

        input
            .iter()
            .map(|&x| {
                if rng.gen::<f64>() > self.rate {
                    x * scale
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d() {
        let conv = Conv1D::new(1, 1, 3);
        let input = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let output = conv.forward(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 5);
    }

    #[test]
    fn test_dilated_conv1d() {
        let conv = DilatedConv1D::new(1, 1, 2, 4);
        assert_eq!(conv.receptive_field(), 5);

        let input = vec![vec![1.0; 10]];
        let output = conv.forward(&input);
        assert_eq!(output[0].len(), 10);
    }

    #[test]
    fn test_dense() {
        let dense = Dense::new(3, 2);
        let input = vec![1.0, 2.0, 3.0];
        let output = dense.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.forward(&input);

        // Проверяем, что среднее близко к 0
        let mean: f64 = output.iter().sum::<f64>() / output.len() as f64;
        assert!(mean.abs() < 1e-5);
    }
}
