//! Graph Attention Network implementation.

use super::{layers::GraphAttentionLayer, GNNConfig, GNNModel};
use tch::{nn, Device, Tensor};

/// Graph Attention Network for cryptocurrency prediction.
pub struct GAT {
    /// GAT layers
    layers: Vec<GraphAttentionLayer>,
    /// Classifier layer
    classifier: nn::Linear,
    /// Dropout probability
    dropout: f64,
    /// Number of attention heads
    num_heads: i64,
    /// Variable store
    vs: nn::VarStore,
    /// Device
    device: Device,
}

impl GAT {
    /// Create a new GAT model.
    pub fn new(config: &GNNConfig, device: Device) -> Self {
        Self::with_heads(config, device, 4)
    }

    /// Create a new GAT model with specified number of attention heads.
    pub fn with_heads(config: &GNNConfig, device: Device, num_heads: i64) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let mut layers = Vec::new();

        // First layer
        layers.push(GraphAttentionLayer::new(
            &(&root / "layer_0"),
            config.num_features,
            config.hidden_dim,
            num_heads,
        ));

        // Hidden layers
        for i in 1..config.num_layers {
            layers.push(GraphAttentionLayer::new(
                &(&root / format!("layer_{}", i)),
                config.hidden_dim,
                config.hidden_dim,
                num_heads,
            ));
        }

        // Classifier
        let classifier = nn::linear(
            &root / "classifier",
            config.hidden_dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            layers,
            classifier,
            dropout: config.dropout,
            num_heads,
            vs,
            device,
        }
    }

    /// Save model to file.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        self.vs.save(path)?;
        Ok(())
    }

    /// Load model from file.
    pub fn load(&mut self, path: &str) -> anyhow::Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    /// Get trainable parameters.
    pub fn trainable_variables(&self) -> Vec<Tensor> {
        self.vs.trainable_variables()
    }

    /// Get optimizer.
    pub fn optimizer(&self, learning_rate: f64) -> nn::Optimizer {
        nn::Adam::default().build(&self.vs, learning_rate).unwrap()
    }

    /// Forward pass with attention weights.
    pub fn forward_with_attention(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
    ) -> (Tensor, Vec<Tensor>) {
        let mut h = x.shallow_clone();
        let mut attention_weights = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let (new_h, attn) = layer.forward_with_attention(&h, edge_index);
            h = new_h.elu();

            if i < self.layers.len() - 1 {
                h = h.dropout(self.dropout, true);
            }

            attention_weights.push(attn);
        }

        let logits = h.apply(&self.classifier);
        (logits, attention_weights)
    }

    /// Get node embeddings (before classification layer).
    pub fn get_embeddings(&self, x: &Tensor, edge_index: &Tensor) -> Tensor {
        let mut h = x.shallow_clone();

        for layer in &self.layers {
            h = layer.forward(&h, edge_index);
            h = h.elu();
        }

        h
    }

    /// Analyze attention weights to find influential neighbors.
    pub fn get_influential_neighbors(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        node_idx: i64,
        symbols: &[String],
    ) -> Vec<(String, f64)> {
        let (_, attention_weights) = self.forward_with_attention(x, edge_index);

        if attention_weights.is_empty() {
            return Vec::new();
        }

        // Get attention from last layer
        let last_attn = &attention_weights[attention_weights.len() - 1];

        // Find edges where target is node_idx
        let targets = edge_index.select(0, 1);
        let sources = edge_index.select(0, 0);

        let mask = targets.eq(node_idx);
        let neighbor_indices = sources.masked_select(&mask);
        let attention_values = last_attn.mean_dim(-1, false, tch::Kind::Float).masked_select(&mask);

        let neighbor_idx_vec: Vec<i64> = Vec::<i64>::try_from(neighbor_indices).unwrap();
        let attn_vec: Vec<f64> = Vec::<f64>::try_from(attention_values).unwrap();

        let mut neighbors: Vec<(String, f64)> = neighbor_idx_vec
            .into_iter()
            .zip(attn_vec)
            .filter_map(|(idx, attn)| {
                symbols.get(idx as usize).map(|s| (s.clone(), attn))
            })
            .collect();

        // Sort by attention weight descending
        neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        neighbors
    }
}

impl GNNModel for GAT {
    fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        _edge_weight: Option<&Tensor>,
    ) -> Tensor {
        let mut h = x.shallow_clone();

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, edge_index);
            h = h.elu();

            // Don't apply dropout after last layer
            if i < self.layers.len() - 1 {
                h = h.dropout(self.dropout, self.device == Device::Cpu);
            }
        }

        // Classification
        h.apply(&self.classifier)
    }
}

/// Training utilities for GAT.
pub struct GATTrainer {
    model: GAT,
    optimizer: nn::Optimizer,
}

impl GATTrainer {
    /// Create a new trainer.
    pub fn new(config: &GNNConfig, device: Device, num_heads: i64) -> Self {
        let model = GAT::with_heads(config, device, num_heads);
        let optimizer = model.optimizer(config.learning_rate);
        Self { model, optimizer }
    }

    /// Train for one epoch.
    pub fn train_epoch(
        &mut self,
        x: &Tensor,
        edge_index: &Tensor,
        labels: &Tensor,
        train_mask: Option<&Tensor>,
    ) -> f64 {
        let logits = self.model.forward(x, edge_index, None);

        // Compute loss
        let loss = if let Some(mask) = train_mask {
            let masked_logits = logits.index_select(0, mask);
            let masked_labels = labels.index_select(0, mask);
            masked_logits.cross_entropy_for_logits(&masked_labels)
        } else {
            logits.cross_entropy_for_logits(labels)
        };

        // Backward pass
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        f64::try_from(loss).unwrap()
    }

    /// Evaluate model.
    pub fn evaluate(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        labels: &Tensor,
        mask: Option<&Tensor>,
    ) -> (f64, f64) {
        let logits = self.model.forward(x, edge_index, None);
        let predictions = logits.argmax(-1, false);

        let (correct, total) = if let Some(m) = mask {
            let masked_pred = predictions.index_select(0, m);
            let masked_labels = labels.index_select(0, m);
            let correct = masked_pred.eq_tensor(&masked_labels).sum(tch::Kind::Int64);
            (
                i64::try_from(correct).unwrap() as f64,
                m.size()[0] as f64,
            )
        } else {
            let correct = predictions.eq_tensor(labels).sum(tch::Kind::Int64);
            (
                i64::try_from(correct).unwrap() as f64,
                labels.size()[0] as f64,
            )
        };

        let accuracy = correct / total;

        // Compute loss
        let loss = if let Some(m) = mask {
            let masked_logits = logits.index_select(0, m);
            let masked_labels = labels.index_select(0, m);
            f64::try_from(masked_logits.cross_entropy_for_logits(&masked_labels)).unwrap()
        } else {
            f64::try_from(logits.cross_entropy_for_logits(labels)).unwrap()
        };

        (loss, accuracy)
    }

    /// Get trained model.
    pub fn model(&self) -> &GAT {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gat_forward() {
        let config = GNNConfig {
            num_features: 10,
            hidden_dim: 32,
            num_classes: 3,
            num_layers: 2,
            dropout: 0.0,
            learning_rate: 0.01,
        };

        let model = GAT::new(&config, Device::Cpu);

        let x = Tensor::randn(&[5, 10], (tch::Kind::Float, Device::Cpu));
        let edge_index = Tensor::from_slice2(&[[0i64, 1, 2, 3], [1, 2, 3, 4]]);

        let output = model.forward(&x, &edge_index, None);
        assert_eq!(output.size(), vec![5, 3]);
    }

    #[test]
    fn test_gat_with_attention() {
        let config = GNNConfig::default();
        let model = GAT::new(&config, Device::Cpu);

        let x = Tensor::randn(&[5, 10], (tch::Kind::Float, Device::Cpu));
        let edge_index = Tensor::from_slice2(&[[0i64, 1, 2, 3], [1, 2, 3, 4]]);

        let (logits, attention) = model.forward_with_attention(&x, &edge_index);
        assert_eq!(logits.size(), vec![5, 3]);
        assert!(!attention.is_empty());
    }
}
