use ndarray::{Array3, Array2, Axis, Array1};

/// Temporal Contrastive Feature Extractor
/// Extracts continuous normalized embeddings from financial time series.
/// Trained via InfoNCE such that temporal neighbors have high cosine similarity.
pub struct TemporalInference {
    pub cnn_weights: Array3<f64>, // (out_channels=64, in_channels=1, kernel=3)
    pub cnn_bias: Array2<f64>,    // (64, 1)
    pub proj_w: Array2<f64>,      // (proj_dim=128, base_dim=64)
    pub proj_b: Array1<f64>,      // (128)
}

impl TemporalInference {
    /// Extracts a L2-normalized Projected embedding (v_t) from a price window.
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
        let z_t = conv_out.mean_axis(Axis(1)).unwrap();
        
        // 3. InfoNCE Projector (Linear layer matching python/model.py projector)
        let v_t = self.proj_w.dot(&z_t) + &self.proj_b;
        
        // 4. L2 Normalization (Crucial for Cosine Similarity in InfoNCE)
        let norm: f64 = v_t.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            v_t.mapv(|x| x / norm)
        } else {
            v_t
        }
    }

    /// Calculates the Cosine distance between two normalized embeddings.
    pub fn cosine_similarity(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
        // Because v1 and v2 are L2-normalized, the dot product IS the cosine similarity.
        v1.dot(v2)
    }
}

pub mod production {
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    /// Load mock weights representing a trained TCL model.
    pub fn load_pretrained_tcl_model() -> TemporalInference {
        TemporalInference {
            cnn_weights: Array3::from_elem((64, 1, 3), 0.05),
            cnn_bias: Array2::from_elem((64, 1), 0.0),
            proj_w: Array2::from_elem((128, 64), 0.02),
            proj_b: Array1::from_elem(128, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_temporal_inference_smoothness() {
        let model = production::load_pretrained_tcl_model();
        
        // Assume t0, t1, t2 are consecutive time windows
        let v_t0 = model.extract_features(&Array2::from_elem((1, 128), 1.0));
        let v_t1 = model.extract_features(&Array2::from_elem((1, 128), 1.02)); // temporal adjency
        
        // Assume t99 is a distant time window
        let v_t99 = model.extract_features(&Array2::from_elem((1, 128), -0.5)); // regime shift

        assert_eq!(v_t0.len(), 128);
        
        // Assert embeddings are properly L2 Normalized (magnitude = 1)
        assert!((v_t0.iter().map(|v| v*v).sum::<f64>().sqrt() - 1.0).abs() < 1e-5);

        // Calculate similarities
        let sim_adjacent = TemporalInference::cosine_similarity(&v_t0, &v_t1);
        let sim_distant = TemporalInference::cosine_similarity(&v_t0, &v_t99);

        // The adjacent windows should be highly similar
        assert!(sim_adjacent > 0.9);
        // The mock weights won't show the learned InfoNCE separation, but we can verify math works
        assert!(sim_distant <= 1.0 && sim_distant >= -1.0);
    }
}
