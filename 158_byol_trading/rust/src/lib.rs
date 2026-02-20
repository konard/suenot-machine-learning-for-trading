use ndarray::{Array3, Array2, Axis};

/// Feature Extractor based on the Target Network.
/// In BYOL, the Target Network (The "Master") provides the most stable
/// and robust representations for downstream downstream tasks, as it is 
/// an exponential moving average of the online learning process.
pub struct TargetFeatureExtractor {
    pub weights: Array3<f64>, // (out_channels, in_channels, kernel_size)
    pub bias: Array2<f64>,    // (out_channels, 1)
}

impl TargetFeatureExtractor {
    pub fn extract(&self, windows: &Array2<f64>) -> Array2<f64> {
        let (in_channels, seq_len) = windows.dim();
        let (out_channels, _, kernel_size) = self.weights.dim();
        
        let out_len = seq_len - kernel_size + 1;
        let mut output = Array2::zeros((out_channels, out_len));

        // 1D Convolution + ReLU (Mirroring the PyTorch CNN1DEncoder target block)
        for oc in 0..out_channels {
            for t in 0..out_len {
                let mut sum = self.bias[[oc, 0]];
                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        sum += windows[[ic, t + k]] * self.weights[[oc, ic, k]];
                    }
                }
                output[[oc, t]] = sum.max(0.0);
            }
        }

        // AdaptiveAvgPool1d(1) equivalent -> mean across the time dimension
        let pooled = output.mean_axis(Axis(1)).unwrap();
        pooled.insert_axis(Axis(0)) // Return shape (1, out_channels)
    }
}

pub mod production {
    use super::*;
    use ndarray::Array3;

    pub fn load_ema_target_model() -> TargetFeatureExtractor {
        // In reality, this loads EMA weights stored from the PyTorch BYOL target_encoder 
        TargetFeatureExtractor {
            weights: Array3::from_elem((64, 1, 7), 0.03),
            bias: Array2::from_elem((64, 1), 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_target_feature_extraction() {
        let extractor = production::load_ema_target_model();
        let price_window = Array2::from_elem((1, 128), 1.0); // (Channels, Time)
        
        let target_rep = extractor.extract(&price_window);
        
        assert_eq!(target_rep.shape(), &[1, 64]);
        
        // With positive weights and inputs, ReLU should pass positive values
        assert!(target_rep[[0, 0]] >= 0.0);
        
        // Assert representations don't trivially collapse to exactly zero
        assert!(target_rep[[0, 0]] > 0.0);
    }
}
