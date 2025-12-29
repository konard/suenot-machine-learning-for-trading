//! GNN model module.

mod gcn;
mod gat;
mod layers;

pub use gcn::GCN;
pub use gat::GAT;
pub use layers::{GraphConvLayer, GraphAttentionLayer};

/// Configuration for GNN models.
#[derive(Debug, Clone)]
pub struct GNNConfig {
    /// Number of input features per node
    pub num_features: i64,
    /// Hidden dimension
    pub hidden_dim: i64,
    /// Number of output classes
    pub num_classes: i64,
    /// Number of GNN layers
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Learning rate
    pub learning_rate: f64,
}

impl Default for GNNConfig {
    fn default() -> Self {
        Self {
            num_features: 10,
            hidden_dim: 64,
            num_classes: 3, // Down, Neutral, Up
            num_layers: 3,
            dropout: 0.3,
            learning_rate: 0.001,
        }
    }
}

/// Prediction result from GNN.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Node index
    pub node_idx: usize,
    /// Symbol name
    pub symbol: String,
    /// Probability of down movement
    pub prob_down: f64,
    /// Probability of neutral
    pub prob_neutral: f64,
    /// Probability of up movement
    pub prob_up: f64,
    /// Predicted class (0: down, 1: neutral, 2: up)
    pub predicted_class: i64,
}

impl Prediction {
    /// Get confidence (max probability).
    pub fn confidence(&self) -> f64 {
        self.prob_down.max(self.prob_neutral).max(self.prob_up)
    }

    /// Get predicted direction as string.
    pub fn direction(&self) -> &'static str {
        match self.predicted_class {
            0 => "DOWN",
            1 => "NEUTRAL",
            2 => "UP",
            _ => "UNKNOWN",
        }
    }
}

/// Trait for GNN models.
pub trait GNNModel {
    /// Forward pass.
    fn forward(
        &self,
        x: &tch::Tensor,
        edge_index: &tch::Tensor,
        edge_weight: Option<&tch::Tensor>,
    ) -> tch::Tensor;

    /// Get predictions with probabilities.
    fn predict(
        &self,
        x: &tch::Tensor,
        edge_index: &tch::Tensor,
        edge_weight: Option<&tch::Tensor>,
        symbols: &[String],
    ) -> Vec<Prediction> {
        let logits = self.forward(x, edge_index, edge_weight);
        let probs = logits.softmax(-1, tch::Kind::Float);
        let predicted = probs.argmax(-1, false);

        let probs_vec: Vec<Vec<f64>> = Vec::<Vec<f64>>::try_from(probs).unwrap();
        let predicted_vec: Vec<i64> = Vec::<i64>::try_from(predicted).unwrap();

        symbols
            .iter()
            .enumerate()
            .map(|(i, symbol)| Prediction {
                node_idx: i,
                symbol: symbol.clone(),
                prob_down: probs_vec[i][0],
                prob_neutral: probs_vec[i][1],
                prob_up: probs_vec[i][2],
                predicted_class: predicted_vec[i],
            })
            .collect()
    }
}

/// Create edge index tensor from vectors.
pub fn create_edge_index(sources: &[i64], targets: &[i64], device: tch::Device) -> tch::Tensor {
    let sources_tensor = tch::Tensor::from_slice(sources).to_device(device);
    let targets_tensor = tch::Tensor::from_slice(targets).to_device(device);
    tch::Tensor::stack(&[sources_tensor, targets_tensor], 0)
}

/// Create node features tensor.
pub fn create_features(features: &[Vec<f64>], device: tch::Device) -> tch::Tensor {
    let flat: Vec<f64> = features.iter().flatten().cloned().collect();
    let n = features.len() as i64;
    let d = if n > 0 { features[0].len() as i64 } else { 0 };
    tch::Tensor::from_slice(&flat)
        .reshape(&[n, d])
        .to_device(device)
        .to_kind(tch::Kind::Float)
}

/// Create labels tensor.
pub fn create_labels(labels: &[i64], device: tch::Device) -> tch::Tensor {
    tch::Tensor::from_slice(labels).to_device(device)
}
