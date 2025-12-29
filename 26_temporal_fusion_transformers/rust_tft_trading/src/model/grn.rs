//! Gated Residual Network (GRN)
//!
//! Основной строительный блок TFT для нелинейной обработки
//! с gating механизмом и residual connections.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Normal;

/// Gated Residual Network
///
/// GRN(x, c) = LayerNorm(x + GLU(W1 * ELU(W2 * x + W3 * c + b)))
#[derive(Debug, Clone)]
pub struct GatedResidualNetwork {
    /// Размер входа
    pub input_size: usize,

    /// Размер скрытого слоя
    pub hidden_size: usize,

    /// Размер выхода
    pub output_size: usize,

    /// Размер контекста (опционально)
    pub context_size: Option<usize>,

    /// Dropout rate
    pub dropout: f64,

    /// Веса первого слоя (input -> hidden)
    weights_1: Array2<f64>,

    /// Bias первого слоя
    bias_1: Array1<f64>,

    /// Веса второго слоя (hidden -> output*2 для GLU)
    weights_2: Array2<f64>,

    /// Bias второго слоя
    bias_2: Array1<f64>,

    /// Веса для контекста (опционально)
    context_weights: Option<Array2<f64>>,

    /// Веса для skip connection если размеры не совпадают
    skip_weights: Option<Array2<f64>>,

    /// Layer normalization параметры
    ln_gamma: Array1<f64>,
    ln_beta: Array1<f64>,
}

impl GatedResidualNetwork {
    /// Создает новый GRN
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        context_size: Option<usize>,
        dropout: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale_1 = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let scale_2 = (2.0 / (hidden_size + output_size * 2) as f64).sqrt();

        let normal_1 = Normal::new(0.0, scale_1).unwrap();
        let normal_2 = Normal::new(0.0, scale_2).unwrap();

        // Первый слой
        let weights_1 = Array2::from_shape_fn((hidden_size, input_size), |_| rng.sample(normal_1));
        let bias_1 = Array1::zeros(hidden_size);

        // Второй слой (output * 2 для GLU)
        let weights_2 =
            Array2::from_shape_fn((output_size * 2, hidden_size), |_| rng.sample(normal_2));
        let bias_2 = Array1::zeros(output_size * 2);

        // Контекстные веса
        let context_weights = context_size.map(|cs| {
            let scale = (2.0 / (cs + hidden_size) as f64).sqrt();
            let normal = Normal::new(0.0, scale).unwrap();
            Array2::from_shape_fn((hidden_size, cs), |_| rng.sample(normal))
        });

        // Skip connection если размеры не совпадают
        let skip_weights = if input_size != output_size {
            let scale = (2.0 / (input_size + output_size) as f64).sqrt();
            let normal = Normal::new(0.0, scale).unwrap();
            Some(Array2::from_shape_fn((output_size, input_size), |_| {
                rng.sample(normal)
            }))
        } else {
            None
        };

        // Layer normalization
        let ln_gamma = Array1::ones(output_size);
        let ln_beta = Array1::zeros(output_size);

        Self {
            input_size,
            hidden_size,
            output_size,
            context_size,
            dropout,
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            context_weights,
            skip_weights,
            ln_gamma,
            ln_beta,
        }
    }

    /// Forward pass для одного sample
    pub fn forward(&self, x: &Array1<f64>, context: Option<&Array1<f64>>) -> Array1<f64> {
        // Первый слой с ELU
        let mut hidden = self.weights_1.dot(x) + &self.bias_1;

        // Добавляем контекст если есть
        if let (Some(c), Some(cw)) = (context, &self.context_weights) {
            hidden = hidden + cw.dot(c);
        }

        // ELU activation
        hidden.mapv_inplace(|v| if v > 0.0 { v } else { v.exp() - 1.0 });

        // Второй слой
        let glu_input = self.weights_2.dot(&hidden) + &self.bias_2;

        // GLU (Gated Linear Unit)
        let half = self.output_size;
        let main = glu_input.slice(ndarray::s![..half]).to_owned();
        let gate = glu_input.slice(ndarray::s![half..]).to_owned();

        // Sigmoid gate
        let gate_sigmoid = gate.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let glu_output = &main * &gate_sigmoid;

        // Skip connection
        let skip = if let Some(sw) = &self.skip_weights {
            sw.dot(x)
        } else {
            x.clone()
        };

        // Residual connection
        let output = &skip + &glu_output;

        // Layer normalization
        self.layer_norm(&output)
    }

    /// Forward pass для batch
    pub fn forward_batch(
        &self,
        x: &Array2<f64>,
        context: Option<&Array2<f64>>,
    ) -> Array2<f64> {
        let batch_size = x.nrows();
        let mut output = Array2::zeros((batch_size, self.output_size));

        for i in 0..batch_size {
            let xi = x.row(i).to_owned();
            let ci = context.map(|c| c.row(i).to_owned());
            let yi = self.forward(&xi, ci.as_ref());
            output.row_mut(i).assign(&yi);
        }

        output
    }

    /// Layer normalization
    fn layer_norm(&self, x: &Array1<f64>) -> Array1<f64> {
        let mean = x.mean().unwrap_or(0.0);
        let variance = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (variance + 1e-6).sqrt();

        let normalized = x.mapv(|v| (v - mean) / std);
        &normalized * &self.ln_gamma + &self.ln_beta
    }

    /// Возвращает количество параметров
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.weights_1.len() + self.bias_1.len();
        count += self.weights_2.len() + self.bias_2.len();
        if let Some(cw) = &self.context_weights {
            count += cw.len();
        }
        if let Some(sw) = &self.skip_weights {
            count += sw.len();
        }
        count += self.ln_gamma.len() + self.ln_beta.len();
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grn_forward() {
        let grn = GatedResidualNetwork::new(16, 32, 16, None, 0.1);

        let x = Array1::from_vec(vec![0.5; 16]);
        let output = grn.forward(&x, None);

        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_grn_with_context() {
        let grn = GatedResidualNetwork::new(16, 32, 16, Some(8), 0.1);

        let x = Array1::from_vec(vec![0.5; 16]);
        let c = Array1::from_vec(vec![0.3; 8]);
        let output = grn.forward(&x, Some(&c));

        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_grn_different_sizes() {
        let grn = GatedResidualNetwork::new(16, 32, 24, None, 0.1);

        let x = Array1::from_vec(vec![0.5; 16]);
        let output = grn.forward(&x, None);

        assert_eq!(output.len(), 24);
    }

    #[test]
    fn test_grn_batch() {
        let grn = GatedResidualNetwork::new(16, 32, 16, None, 0.1);

        let x = Array2::from_elem((4, 16), 0.5);
        let output = grn.forward_batch(&x, None);

        assert_eq!(output.shape(), &[4, 16]);
    }
}
