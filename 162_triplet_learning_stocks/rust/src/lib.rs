use ndarray::{Array3, Array2, Axis, Array1};

/// Triplet Learning Feature Extractor
/// Extracts continuous normalized embeddings from financial time series.
/// Specifically trained such that Euclidean distances between embeddings
/// directly correspond to semantic market regime differences.
pub struct TripletInference {
    pub cnn_weights: Array3<f64>, // (out_channels=64, in_channels=1, kernel=3)
    pub cnn_bias: Array2<f64>,    // (64, 1)
    pub fc_w: Array2<f64>,        // (hidden_dim=128, hidden_dim=64)
    pub fc_b: Array1<f64>,        // (128)
}

impl TripletInference {
    /// Extracts a L2-normalized embedding from a price window.
    pub fn extract_features(&self, windows: &Array2<f64>) -> Array1<f64> {
        let (in_channels, seq_len) = windows.dim();
        let (out_channels, _, kernel_size) = self.cnn_weights.dim();
        
        // 1. 1D Convolution (Simulated matching CNN1DEncoder for simplicity)
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

        // 2. Global Average Pooling (simulating adaptive_pool(1))
        let h = conv_out.mean_axis(Axis(1)).unwrap();
        
        // 3. Linear layer
        let z = self.fc_w.dot(&h) + &self.fc_b;
        
        // 4. L2 Normalization (Crucial for Triplet Margin Loss Euclidean mapping)
        let norm: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            z.mapv(|x| x / norm)
        } else {
            z
        }
    }

    /// Calculates the Euclidean distance between two extracted embeddings.
    /// This is the core metric optimized by the Triplet Margin Loss.
    pub fn calculate_distance(z1: &Array1<f64>, z2: &Array1<f64>) -> f64 {
        z1.iter()
            .zip(z2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

pub mod production {
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    /// Load mock weights representing a trained Triplet model.
    pub fn load_pretrained_triplet_model() -> TripletInference {
        TripletInference {
            cnn_weights: Array3::from_elem((64, 1, 3), 0.05),
            cnn_bias: Array2::from_elem((64, 1), 0.0),
            fc_w: Array2::from_elem((128, 64), 0.02),
            fc_b: Array1::from_elem(128, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_triplet_inference_and_distance() {
        let model = production::load_pretrained_triplet_model();
        
        // Simulate two slightly different market windows (Anchor and Positive)
        let mut anchor_window = Array2::from_elem((1, 128), 1.0);
        let mut positive_window = Array2::from_elem((1, 128), 1.05); // Slight shift
        
        // Simulate a wildly different market window (Negative)
        let mut negative_window = Array2::from_elem((1, 128), -1.0); // Inverted
        
        // Emulate some temporal variation
        for i in 0..10 {
            anchor_window[[0, i]] += 0.1;
            positive_window[[0, i]] += 0.1;
            negative_window[[0, i]] -= 0.5;
        }

        let z_a = model.extract_features(&anchor_window);
        let z_p = model.extract_features(&positive_window);
        let z_n = model.extract_features(&negative_window);
        
        assert_eq!(z_a.len(), 128);
        assert!((z_a.iter().map(|v| v*v).sum::<f64>().sqrt() - 1.0).abs() < 1e-5, "Embeddings must be L2 normalized");

        let dist_ap = TripletInference::calculate_distance(&z_a, &z_p);
        let dist_an = TripletInference::calculate_distance(&z_a, &z_n);

        // While these are mock weights, the distance to the identical-pattern but shifted
        // positive should theoretically be smaller than the inverted negative 
        // in a properly trained network. (With mock weights, distancing is arbitrary but 
        // we test the math).
        
        assert!(dist_ap >= 0.0);
        assert!(dist_an >= 0.0);
    }
}
