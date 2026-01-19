//! Complete FNet Model
//!
//! Full FNet architecture for financial time series prediction.

use ndarray::{Array2, Array3, Axis};
use rand_distr::{Distribution, Normal};

use super::encoder::FNetEncoderBlock;

/// Complete FNet model for financial time series.
pub struct FNet {
    /// Input projection weights [n_features, d_model]
    input_proj: Array2<f64>,
    /// Input projection bias [d_model]
    input_bias: Vec<f64>,
    /// Positional encoding [max_seq_len, d_model]
    pos_encoding: Array2<f64>,
    /// Encoder blocks
    encoder_blocks: Vec<FNetEncoderBlock>,
    /// Output head weights [d_model, d_model/2]
    output_w1: Array2<f64>,
    /// Output head bias [d_model/2]
    output_b1: Vec<f64>,
    /// Final output weights [d_model/2, output_dim]
    output_w2: Array2<f64>,
    /// Final output bias [output_dim]
    output_b2: Vec<f64>,
    /// Model dimension
    d_model: usize,
    /// Dropout rate
    dropout: f64,
}

impl FNet {
    /// Create a new FNet model.
    ///
    /// # Arguments
    /// * `n_features` - Number of input features
    /// * `d_model` - Model dimension
    /// * `n_layers` - Number of encoder layers
    /// * `d_ff` - Feed-forward dimension
    /// * `dropout` - Dropout rate
    /// * `max_seq_len` - Maximum sequence length
    /// * `output_dim` - Output dimension
    pub fn new(
        n_features: usize,
        d_model: usize,
        n_layers: usize,
        d_ff: usize,
        dropout: f64,
        max_seq_len: usize,
        output_dim: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization for input projection
        let std = (2.0 / (n_features + d_model) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let input_proj = Array2::from_shape_fn((n_features, d_model), |_| {
            normal.sample(&mut rng)
        });
        let input_bias = vec![0.0; d_model];

        // Sinusoidal positional encoding
        let pos_encoding = create_positional_encoding(max_seq_len, d_model);

        // Encoder blocks
        let encoder_blocks = (0..n_layers)
            .map(|_| FNetEncoderBlock::new(d_model, d_ff, dropout))
            .collect();

        // Output head
        let d_hidden = d_model / 2;
        let std1 = (2.0 / (d_model + d_hidden) as f64).sqrt();
        let std2 = (2.0 / (d_hidden + output_dim) as f64).sqrt();
        let normal1 = Normal::new(0.0, std1).unwrap();
        let normal2 = Normal::new(0.0, std2).unwrap();

        let output_w1 = Array2::from_shape_fn((d_model, d_hidden), |_| {
            normal1.sample(&mut rng)
        });
        let output_b1 = vec![0.0; d_hidden];
        let output_w2 = Array2::from_shape_fn((d_hidden, output_dim), |_| {
            normal2.sample(&mut rng)
        });
        let output_b2 = vec![0.0; output_dim];

        Self {
            input_proj,
            input_bias,
            pos_encoding,
            encoder_blocks,
            output_w1,
            output_b1,
            output_w2,
            output_b2,
            d_model,
            dropout,
        }
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_features]
    /// * `training` - Whether in training mode (affects dropout)
    ///
    /// # Returns
    /// Output predictions [batch, output_dim]
    pub fn forward(&mut self, x: &Array3<f64>, training: bool) -> Array2<f64> {
        let (batch, seq_len, _) = x.dim();

        // Input projection
        let mut hidden = self.project_input(x);

        // Add positional encoding
        for b in 0..batch {
            for t in 0..seq_len {
                for d in 0..self.d_model {
                    hidden[[b, t, d]] += self.pos_encoding[[t, d]];
                }
            }
        }

        // Apply encoder blocks
        for block in &mut self.encoder_blocks {
            hidden = block.forward(&hidden, training);
        }

        // Global average pooling
        let pooled = self.global_avg_pool(&hidden);

        // Output head
        self.output_head(&pooled, training)
    }

    /// Forward pass with frequency maps for analysis.
    pub fn forward_with_frequencies(
        &mut self,
        x: &Array3<f64>,
        training: bool
    ) -> (Array2<f64>, Vec<Array3<f64>>) {
        let (batch, seq_len, _) = x.dim();

        let mut hidden = self.project_input(x);

        for b in 0..batch {
            for t in 0..seq_len {
                for d in 0..self.d_model {
                    hidden[[b, t, d]] += self.pos_encoding[[t, d]];
                }
            }
        }

        let mut freq_maps = Vec::new();
        for block in &mut self.encoder_blocks {
            let (out, freq) = block.forward_with_frequencies(&hidden, training);
            hidden = out;
            freq_maps.push(freq);
        }

        let pooled = self.global_avg_pool(&hidden);
        let output = self.output_head(&pooled, training);

        (output, freq_maps)
    }

    /// Project input features to model dimension.
    fn project_input(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, n_features) = x.dim();
        let mut output = Array3::zeros((batch, seq_len, self.d_model));

        for b in 0..batch {
            for t in 0..seq_len {
                for d in 0..self.d_model {
                    let mut sum = self.input_bias[d];
                    for f in 0..n_features {
                        sum += x[[b, t, f]] * self.input_proj[[f, d]];
                    }
                    output[[b, t, d]] = sum;
                }
            }
        }

        output
    }

    /// Global average pooling over sequence dimension.
    fn global_avg_pool(&self, x: &Array3<f64>) -> Array2<f64> {
        x.mean_axis(Axis(1)).unwrap()
    }

    /// Output head: Linear -> GELU -> Linear
    fn output_head(&self, x: &Array2<f64>, _training: bool) -> Array2<f64> {
        let (batch, _) = x.dim();
        let d_hidden = self.output_w1.dim().1;
        let output_dim = self.output_w2.dim().1;

        let mut output = Array2::zeros((batch, output_dim));

        for b in 0..batch {
            // First layer
            let mut hidden = vec![0.0; d_hidden];
            for h in 0..d_hidden {
                let mut sum = self.output_b1[h];
                for d in 0..self.d_model {
                    sum += x[[b, d]] * self.output_w1[[d, h]];
                }
                hidden[h] = gelu(sum);
            }

            // Second layer
            for o in 0..output_dim {
                let mut sum = self.output_b2[o];
                for h in 0..d_hidden {
                    sum += hidden[h] * self.output_w2[[h, o]];
                }
                output[[b, o]] = sum;
            }
        }

        output
    }

    /// Get the number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        let input_params = self.input_proj.len() + self.input_bias.len();
        let output_params = self.output_w1.len() + self.output_b1.len()
            + self.output_w2.len() + self.output_b2.len();

        // Encoder blocks have LayerNorm and FeedForward params
        let block_params = self.encoder_blocks.len() * (
            // Two LayerNorms: 2 * 2 * d_model (gamma + beta each)
            4 * self.d_model +
            // FeedForward: w1 + b1 + w2 + b2
            self.d_model * (self.d_model * 4) + (self.d_model * 4) +
            (self.d_model * 4) * self.d_model + self.d_model
        );

        input_params + output_params + block_params
    }

    /// Predict on single input.
    pub fn predict(&mut self, x: &Array3<f64>) -> Array2<f64> {
        self.forward(x, false)
    }
}

/// Create sinusoidal positional encoding.
fn create_positional_encoding(max_len: usize, d_model: usize) -> Array2<f64> {
    let mut pe = Array2::zeros((max_len, d_model));

    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
            pe[[pos, 2 * i]] = angle.sin();
            pe[[pos, 2 * i + 1]] = angle.cos();
        }
    }

    pe
}

/// GELU activation function.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnet_forward() {
        let mut model = FNet::new(7, 32, 2, 64, 0.0, 64, 1);

        let input = Array3::from_shape_fn((4, 16, 7), |(_, t, _)| {
            (t as f64 / 16.0).sin()
        });

        let output = model.forward(&input, false);

        // Check output shape
        assert_eq!(output.dim(), (4, 1));
    }

    #[test]
    fn test_fnet_with_frequencies() {
        let mut model = FNet::new(7, 32, 2, 64, 0.0, 64, 1);

        let input = Array3::from_shape_fn((2, 16, 7), |(_, t, _)| {
            (t as f64 / 16.0).sin()
        });

        let (output, freq_maps) = model.forward_with_frequencies(&input, false);

        assert_eq!(output.dim(), (2, 1));
        assert_eq!(freq_maps.len(), 2); // 2 encoder layers
    }

    #[test]
    fn test_positional_encoding() {
        let pe = create_positional_encoding(100, 32);

        // Check shape
        assert_eq!(pe.dim(), (100, 32));

        // Check values are bounded
        for val in pe.iter() {
            assert!(val.abs() <= 1.0, "Positional encoding should be in [-1, 1]");
        }
    }

    #[test]
    fn test_num_parameters() {
        let model = FNet::new(7, 64, 4, 256, 0.1, 512, 1);
        let params = model.num_parameters();

        // Should have significant number of parameters
        assert!(params > 10000);
        println!("FNet parameters: {}", params);
    }
}
