use ndarray::{Array3, Array2, Axis};

/// 1D-CNN Feature Extractor for stock price patterns.
/// Mimics the Python encoder: Conv1d -> ReLU -> GlobalAvgPool.
pub struct StockEncoder {
    pub kernel: Array3<f64>, // (out_channels, in_channels, kernel_size)
    pub bias: Array2<f64>,   // (out_channels, 1)
}

impl StockEncoder {
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let (in_channels, seq_len) = x.dim();
        let (out_channels, _, kernel_size) = self.kernel.dim();
        
        let out_len = seq_len - kernel_size + 1;
        let mut output = Array2::zeros((out_channels, out_len));

        // Simplified 1D Convolution for demonstration
        for oc in 0..out_channels {
            for t in 0..out_len {
                let mut sum = self.bias[[oc, 0]];
                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        sum += x[[ic, t + k]] * self.kernel[[oc, ic, k]];
                    }
                }
                output[[oc, t]] = sum.max(0.0); // ReLU
            }
        }

        // Global Average Pooling (Reduce time dimension)
        let pooled = output.mean_axis(Axis(1)).unwrap();
        pooled.insert_axis(Axis(0)) // Result: (1, out_channels)
    }
}

pub mod features {
    use super::*;
    use ndarray::Array3;

    pub fn get_pre_trained_encoder() -> StockEncoder {
        // Mock weights for proof-of-concept
        StockEncoder {
            kernel: Array3::from_elem((32, 1, 7), 0.05),
            bias: Array2::from_elem((32, 1), 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_forward() {
        let encoder = features::get_pre_trained_encoder();
        let input = Array2::from_elem((1, 128), 1.0); // (Channels, Time)
        let latent = encoder.forward(&input);
        
        assert_eq!(latent.shape(), &[1, 32]);
        assert!(latent[[0, 0]] >= 0.0);
    }
}
