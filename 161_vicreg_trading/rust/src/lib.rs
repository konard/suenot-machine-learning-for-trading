use ndarray::{Array3, Array2, Axis};

/// VICReg Feature Extractor for real-time inference.
/// This module implements the production-ready forward pass of the trained VICReg model.
/// It consists of a 1D-CNN Encoder followed by a 3-layer MLP Projector.
pub struct VICRegInference {
    pub cnn_weights: Array3<f64>, // (out_channels, in_channels, kernel)
    pub cnn_bias: Array2<f64>,    // (out_channels, 1)
    pub proj_w1: Array2<f64>,     // (proj_hidden, out_channels)
    pub proj_b1: Array1<f64>,
    pub proj_w2: Array2<f64>,     // (proj_hidden, proj_hidden)
    pub proj_b2: Array1<f64>,
    pub proj_w3: Array2<f64>,     // (proj_out, proj_hidden)
    pub proj_b3: Array1<f64>,
}

use ndarray::Array1;

impl VICRegInference {
    /// Extracts a high-dimensional embedding from a price window.
    pub fn extract_features(&self, windows: &Array2<f64>) -> Array1<f64> {
        let (in_channels, seq_len) = windows.dim();
        let (out_channels, _, kernel_size) = self.cnn_weights.dim();
        
        // 1. 1D Convolution
        let out_len = seq_len - kernel_size + 1;
        let mut conv_out = Array2::zeros((out_channels, out_len));

        for oc in 0..out_channels {
            for t in 0..out_len {
                let mut sum = self.cnn_bias[[oc, 0]];
                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        sum += windows[[ic, t + k]] * self.cnn_weights[[oc, ic, k]];
                    }
                }
                conv_out[[oc, t]] = sum.max(0.0); // ReLU
            }
        }

        // 2. Global Average Pooling
        let h = conv_out.mean_axis(Axis(1)).unwrap();
        
        // 3. Projector (3 Linear Layers)
        let z1 = (self.proj_w1.dot(&h) + &self.proj_b1).mapv(|x| x.max(0.0));
        let z2 = (self.proj_w2.dot(&z1) + &self.proj_b2).mapv(|x| x.max(0.0));
        let z3 = self.proj_w3.dot(&z2) + &self.proj_b3;

        z3
    }
}

pub mod production {
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    /// Simulate loading the model weights. In practice, this would load from a 
    /// serialized format after conversion from PyTorch.
    pub fn load_pretrained_model() -> VICRegInference {
        VICRegInference {
            cnn_weights: Array3::from_elem((64, 1, 7), 0.02),
            cnn_bias: Array2::from_elem((64, 1), 0.0),
            proj_w1: Array2::from_elem((256, 64), 0.01),
            proj_b1: Array1::from_elem(256, 0.0),
            proj_w2: Array2::from_elem((256, 256), 0.01),
            proj_b2: Array1::from_elem(256, 0.0),
            proj_w3: Array2::from_elem((256, 256), 0.01),
            proj_b3: Array1::from_elem(256, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_vicreg_inference_flow() {
        let model = production::load_pretrained_model();
        let price_window = Array2::from_elem((1, 128), 1.0);
        
        let embedding = model.extract_features(&price_window);
        
        assert_eq!(embedding.len(), 256);
        // Ensure no NaNs or infinities
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }
}
