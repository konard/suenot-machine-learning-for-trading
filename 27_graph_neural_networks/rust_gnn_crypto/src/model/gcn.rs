//! Graph Convolutional Network implementation.

use super::{layers::GraphConvLayer, GNNConfig, GNNModel};
use tch::{nn, Device, Tensor};

/// Graph Convolutional Network for cryptocurrency prediction.
pub struct GCN {
    /// GCN layers
    layers: Vec<GraphConvLayer>,
    /// Classifier layer
    classifier: nn::Linear,
    /// Dropout probability
    dropout: f64,
    /// Variable store
    vs: nn::VarStore,
    /// Device
    device: Device,
}

impl GCN {
    /// Create a new GCN model.
    pub fn new(config: &GNNConfig, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let mut layers = Vec::new();

        // First layer
        layers.push(GraphConvLayer::new(
            &(&root / "layer_0"),
            config.num_features,
            config.hidden_dim,
        ));

        // Hidden layers
        for i in 1..config.num_layers {
            layers.push(GraphConvLayer::new(
                &(&root / format!("layer_{}", i)),
                config.hidden_dim,
                config.hidden_dim,
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

    /// Set training mode.
    pub fn set_train(&mut self, training: bool) {
        if training {
            self.vs.set_kind(tch::Kind::Float);
        }
    }

    /// Forward pass with intermediate outputs for analysis.
    pub fn forward_with_intermediates(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_weight: Option<&Tensor>,
    ) -> Vec<Tensor> {
        let mut outputs = Vec::new();
        let mut h = x.shallow_clone();

        for layer in &self.layers {
            h = layer.forward(&h, edge_index, edge_weight);
            h = h.relu();
            h = h.dropout(self.dropout, true);
            outputs.push(h.shallow_clone());
        }

        // Final classification
        let logits = h.apply(&self.classifier);
        outputs.push(logits);

        outputs
    }

    /// Get node embeddings (before classification layer).
    pub fn get_embeddings(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_weight: Option<&Tensor>,
    ) -> Tensor {
        let mut h = x.shallow_clone();

        for layer in &self.layers {
            h = layer.forward(&h, edge_index, edge_weight);
            h = h.relu();
        }

        h
    }
}

impl GNNModel for GCN {
    fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_weight: Option<&Tensor>,
    ) -> Tensor {
        let mut h = x.shallow_clone();

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, edge_index, edge_weight);
            h = h.relu();

            // Don't apply dropout after last GCN layer
            if i < self.layers.len() - 1 {
                h = h.dropout(self.dropout, self.vs.device() == Device::Cpu);
            }
        }

        // Classification
        h.apply(&self.classifier)
    }
}

/// Training utilities for GCN.
pub struct GCNTrainer {
    model: GCN,
    optimizer: nn::Optimizer,
}

impl GCNTrainer {
    /// Create a new trainer.
    pub fn new(config: &GNNConfig, device: Device) -> Self {
        let model = GCN::new(config, device);
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
        self.model.set_train(true);

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
    pub fn model(&self) -> &GCN {
        &self.model
    }

    /// Get mutable model.
    pub fn model_mut(&mut self) -> &mut GCN {
        &mut self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcn_forward() {
        let config = GNNConfig {
            num_features: 10,
            hidden_dim: 32,
            num_classes: 3,
            num_layers: 2,
            dropout: 0.0,
            learning_rate: 0.01,
        };

        let model = GCN::new(&config, Device::Cpu);

        let x = Tensor::randn(&[5, 10], (tch::Kind::Float, Device::Cpu));
        let edge_index = Tensor::from_slice2(&[[0i64, 1, 2, 3], [1, 2, 3, 4]]);

        let output = model.forward(&x, &edge_index, None);
        assert_eq!(output.size(), vec![5, 3]);
    }

    #[test]
    fn test_gcn_trainer() {
        let config = GNNConfig::default();
        let mut trainer = GCNTrainer::new(&config, Device::Cpu);

        let x = Tensor::randn(&[5, 10], (tch::Kind::Float, Device::Cpu));
        let edge_index = Tensor::from_slice2(&[[0i64, 1, 2, 3], [1, 2, 3, 4]]);
        let labels = Tensor::from_slice(&[0i64, 1, 2, 0, 1]);

        let loss = trainer.train_epoch(&x, &edge_index, &labels, None);
        assert!(loss > 0.0);

        let (eval_loss, accuracy) = trainer.evaluate(&x, &edge_index, &labels, None);
        assert!(eval_loss > 0.0);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}
