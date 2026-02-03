//! SSM-GNN Hybrid model combining temporal and graph components.

use rand::Rng;
use rand_distr::StandardNormal;

use super::ssm::DiagonalSSM;
use super::gnn::GraphAttentionLayer;

/// Trading signal from the hybrid model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Long position (+1)
    Long,
    /// Neutral / no position (0)
    Neutral,
    /// Short position (-1)
    Short,
}

impl Signal {
    /// Convert signal to numeric value.
    pub fn value(&self) -> f64 {
        match self {
            Signal::Long => 1.0,
            Signal::Neutral => 0.0,
            Signal::Short => -1.0,
        }
    }
}

/// Prediction output for a single asset.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub signal: Signal,
    pub confidence: f64,
    pub logits: Vec<f64>,
}

/// SSM-GNN Hybrid model for multi-asset trading.
///
/// Architecture:
/// 1. Per-asset SSM encoder for temporal feature extraction
/// 2. GAT layer(s) for cross-asset message passing
/// 3. Linear prediction head for signal generation
pub struct SsmGnnHybrid {
    ssm: DiagonalSSM,
    gnn_layers: Vec<GraphAttentionLayer>,
    /// Prediction weights (d_model x n_classes)
    w_pred: Vec<Vec<f64>>,
    /// Prediction bias (n_classes,)
    b_pred: Vec<f64>,
    d_model: usize,
    n_classes: usize,
}

impl SsmGnnHybrid {
    /// Create a new SSM-GNN hybrid model.
    ///
    /// # Arguments
    /// * `n_features` - Number of input features per asset per time step
    /// * `d_model` - Hidden dimension
    /// * `d_state` - SSM state dimension
    /// * `n_classes` - Number of output classes (default 3: down/flat/up)
    pub fn new(n_features: usize, d_model: usize, d_state: usize, n_classes: usize) -> Self {
        let mut rng = rand::thread_rng();

        let ssm = DiagonalSSM::new(n_features, d_model, d_state);

        let gnn_layers = vec![
            GraphAttentionLayer::new(d_model, d_model),
            GraphAttentionLayer::new(d_model, d_model),
        ];

        let w_pred = (0..d_model)
            .map(|_| {
                (0..n_classes)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
                    .collect()
            })
            .collect();

        let b_pred = vec![0.0; n_classes];

        Self {
            ssm,
            gnn_layers,
            w_pred,
            b_pred,
            d_model,
            n_classes,
        }
    }

    /// Forward pass through the hybrid model.
    ///
    /// # Arguments
    /// * `features` - (n_assets, seq_len, n_features) input features
    /// * `edge_index` - (2, n_edges) graph edge indices
    ///
    /// # Returns
    /// (n_assets, n_classes) logit matrix
    pub fn forward(
        &self,
        features: &[Vec<Vec<f64>>],
        edge_index: &[Vec<usize>],
    ) -> Vec<Vec<f64>> {
        let n_assets = features.len();

        // Step 1: SSM encoding per asset (take last time step)
        let mut node_embeddings: Vec<Vec<f64>> = Vec::with_capacity(n_assets);
        for asset_features in features.iter() {
            let ssm_output = self.ssm.forward(asset_features);
            let last = ssm_output.last().unwrap().clone();
            node_embeddings.push(last);
        }

        // Step 2: GNN message passing with residual connections
        let mut h = node_embeddings;
        for gnn_layer in &self.gnn_layers {
            let h_new = gnn_layer.forward(&h, edge_index, None);
            // Residual connection
            h = h
                .iter()
                .zip(h_new.iter())
                .map(|(old, new)| {
                    old.iter().zip(new.iter()).map(|(a, b)| a + b).collect()
                })
                .collect();
        }

        // Step 3: Prediction head
        let logits: Vec<Vec<f64>> = h
            .iter()
            .map(|node| {
                (0..self.n_classes)
                    .map(|c| {
                        let dot: f64 = (0..self.d_model)
                            .map(|m| node[m] * self.w_pred[m][c])
                            .sum();
                        dot + self.b_pred[c]
                    })
                    .collect()
            })
            .collect();

        logits
    }

    /// Generate trading signals from model predictions.
    pub fn predict_signals(
        &self,
        features: &[Vec<Vec<f64>>],
        edge_index: &[Vec<usize>],
    ) -> Vec<Prediction> {
        let logits = self.forward(features, edge_index);

        logits
            .iter()
            .map(|asset_logits| {
                // Softmax
                let max_logit = asset_logits
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let exp_logits: Vec<f64> =
                    asset_logits.iter().map(|x| (x - max_logit).exp()).collect();
                let sum: f64 = exp_logits.iter().sum();
                let probs: Vec<f64> = exp_logits.iter().map(|x| x / sum).collect();

                // Argmax
                let (pred_class, &max_prob) = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                let signal = match pred_class {
                    0 => Signal::Short,
                    1 => Signal::Neutral,
                    2 => Signal::Long,
                    _ => Signal::Neutral,
                };

                Prediction {
                    signal,
                    confidence: max_prob,
                    logits: asset_logits.clone(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_forward() {
        let model = SsmGnnHybrid::new(5, 8, 4, 3);

        // 3 assets, 10 time steps, 5 features
        let features = vec![
            vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]; 10],
            vec![vec![0.5, 0.4, 0.3, 0.2, 0.1]; 10],
            vec![vec![0.3, 0.3, 0.3, 0.3, 0.3]; 10],
        ];

        let edge_index = vec![
            vec![0, 0, 1, 1, 2, 2],
            vec![1, 2, 0, 2, 0, 1],
        ];

        let logits = model.forward(&features, &edge_index);
        assert_eq!(logits.len(), 3);
        assert_eq!(logits[0].len(), 3);
    }

    #[test]
    fn test_predict_signals() {
        let model = SsmGnnHybrid::new(5, 8, 4, 3);

        let features = vec![
            vec![vec![0.1; 5]; 10],
            vec![vec![0.2; 5]; 10],
        ];

        let edge_index = vec![vec![0, 1], vec![1, 0]];

        let predictions = model.predict_signals(&features, &edge_index);
        assert_eq!(predictions.len(), 2);
        assert!(predictions[0].confidence >= 0.0 && predictions[0].confidence <= 1.0);
    }
}
