use ndarray::{Array3, Array2, Axis};

/// Feature Extractor based on the SwAV Encoder and Prototypes layer.
/// This module implements the production inference steps for SwAV:
/// 1. Forward pass through the continuous 1D-CNN Encoder.
/// 2. L2-Normalization of continuous features.
/// 3. Cosine similarity mapping against the k discrete Prototypes (Clusters).
/// 4. Returning both the continuous embedding and the Argmax cluster ID.
pub struct SwAVClusterInference {
    pub cnn_weights: Array3<f64>, // (out_channels, in_channels, kernel_size)
    pub cnn_bias: Array2<f64>,    // (out_channels, 1)
    pub prototypes: Array2<f64>,  // (num_clusters, out_channels) - Already L2 Normalized
}

impl SwAVClusterInference {
    pub fn infer_cluster(&self, windows: &Array2<f64>) -> (Array2<f64>, usize) {
        let (in_channels, seq_len) = windows.dim();
        let (out_channels, _, kernel_size) = self.cnn_weights.dim();
        
        let out_len = seq_len - kernel_size + 1;
        let mut output = Array2::zeros((out_channels, out_len));

        // 1. Continuous feature extraction (1D Convolution + ReLU)
        for oc in 0..out_channels {
            for t in 0..out_len {
                let mut sum = self.cnn_bias[[oc, 0]];
                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        sum += windows[[ic, t + k]] * self.cnn_weights[[oc, ic, k]];
                    }
                }
                output[[oc, t]] = sum.max(0.0);
            }
        }

        // AdaptiveAvgPool1d equivalent -> mean across the time dimension
        let mut continuous_emb_1d = output.mean_axis(Axis(1)).unwrap();
        
        // 2. L2 Normalization
        let l2_norm = continuous_emb_1d.fold(0.0, |acc: f64, &x| acc + x * x).sqrt();
        let l2_norm_safe = if l2_norm < 1e-8 { 1e-8 } else { l2_norm };
        continuous_emb_1d.mapv_inplace(|x| x / l2_norm_safe);
        
        let continuous_emb = continuous_emb_1d.insert_axis(Axis(0)); // Shape (1, out_channels)

        // 3. Dot product against L2 normalized Prototypes = Cosine Similarity
        // Continuous: (1, Dim) dot Prototypes^T (Dim, K)
        let num_clusters = self.prototypes.shape()[0];
        let mut best_cluster = 0;
        let mut max_sim = f64::NEG_INFINITY;
        
        for k in 0..num_clusters {
            let mut sim = 0.0;
            for oc in 0..out_channels {
                sim += continuous_emb[[0, oc]] * self.prototypes[[k, oc]];
            }
            if sim > max_sim {
                max_sim = sim;
                best_cluster = k;
            }
        }

        (continuous_emb, best_cluster)
    }
}

pub mod production {
    use super::*;
    use ndarray::{Array2, Array3};

    pub fn load_swav_model() -> SwAVClusterInference {
        // Loads CNN weights and the Prototypes matrix from the PyTorch SwAV model
        // In reality, these are deserialized from binary or JSON
        SwAVClusterInference {
            cnn_weights: Array3::from_elem((64, 1, 7), 0.05),
            cnn_bias: Array2::from_elem((64, 1), 0.0),
            prototypes: Array2::from_elem((10, 64), 0.1), // 10 clusters, 64-dim L2 normed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_swav_cluster_inference() {
        let inference = production::load_swav_model();
        let price_window = Array2::from_elem((1, 128), 1.0); // (Channels, Time)
        
        let (continuous, cluster_id) = inference.infer_cluster(&price_window);
        
        assert_eq!(continuous.shape(), &[1, 64]);
        // With positive arbitrary weights, it should assign to a valid cluster index [0..9]
        assert!(cluster_id < 10);
        
        // Ensure L2 normalization works (~1.0 magnitude)
        let norm_sq = continuous.fold(0.0, |acc: f64, &x| acc + x * x);
        assert!((norm_sq - 1.0).abs() < 1e-5);
    }
}
