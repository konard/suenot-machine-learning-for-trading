//! Transfer learning network combining feature extraction, domain adaptation,
//! and task-specific classification.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

use super::feature_extractor::FeatureExtractor;
use super::domain_adapter::{DomainAdapter, AdaptationMethod};

/// Fine-tuning strategy controlling which layers are updated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FineTuneStrategy {
    /// Freeze all feature extractor layers; only train the classifier head
    FeatureExtraction,
    /// Freeze early layers; fine-tune later layers and classifier
    PartialFineTune,
    /// Fine-tune all layers with discriminative learning rates
    FullFineTune,
}

/// Transfer learning network for trading signal prediction.
///
/// Combines a feature extractor (shared between domains), domain adaptation,
/// and a task-specific classifier head.
#[derive(Debug, Clone)]
pub struct TransferNetwork {
    /// Shared feature extractor
    feature_extractor: FeatureExtractor,
    /// Domain adaptation module
    domain_adapter: DomainAdapter,
    /// Classifier head weights
    classifier_weights: Vec<Array2<f64>>,
    /// Classifier head biases
    classifier_biases: Vec<Array1<f64>>,
    /// Number of output classes (e.g., 3 for Buy/Sell/Hold)
    num_classes: usize,
    /// Whether domain adaptation is enabled
    use_adaptation: bool,
    /// Fine-tuning strategy
    fine_tune_strategy: FineTuneStrategy,
}

impl TransferNetwork {
    /// Create a new transfer learning network.
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    /// * `hidden_dim` - Hidden layer dimension
    /// * `feature_dim` - Feature embedding dimension
    /// * `use_adaptation` - Whether to use domain adaptation
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        feature_dim: usize,
        use_adaptation: bool,
    ) -> Self {
        let feature_extractor = FeatureExtractor::new(input_dim, hidden_dim, feature_dim);

        let domain_adapter = if use_adaptation {
            DomainAdapter::new(AdaptationMethod::MMD)
        } else {
            DomainAdapter::new(AdaptationMethod::None)
        };

        let num_classes = 3; // Buy, Sell, Hold
        let mut rng = rand::thread_rng();

        let classifier_weights = vec![
            xavier_init(&mut rng, feature_dim, 64),
            xavier_init(&mut rng, 64, num_classes),
        ];

        let classifier_biases = vec![
            Array1::zeros(64),
            Array1::zeros(num_classes),
        ];

        Self {
            feature_extractor,
            domain_adapter,
            classifier_weights,
            classifier_biases,
            num_classes,
            use_adaptation,
            fine_tune_strategy: FineTuneStrategy::PartialFineTune,
        }
    }

    /// Set the domain adaptation method.
    pub fn with_adaptation_method(mut self, method: AdaptationMethod) -> Self {
        self.domain_adapter = DomainAdapter::new(method);
        self
    }

    /// Set the fine-tuning strategy.
    pub fn with_fine_tune_strategy(mut self, strategy: FineTuneStrategy) -> Self {
        self.fine_tune_strategy = strategy;
        self
    }

    /// Set the number of output classes.
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = num_classes;
        let mut rng = rand::thread_rng();
        self.classifier_weights[1] = xavier_init(&mut rng, 64, num_classes);
        self.classifier_biases[1] = Array1::zeros(num_classes);
        self
    }

    /// Forward pass through the entire network.
    /// Returns (predictions, features) where predictions are class logits.
    pub fn forward(&self, input: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // Extract features
        let features = self.feature_extractor.forward(input);

        // Classify
        let logits = self.classify(&features);

        (logits, features)
    }

    /// Classify features using the task head.
    fn classify(&self, features: &Array2<f64>) -> Array2<f64> {
        let mut x = features.dot(&self.classifier_weights[0]);
        for mut row in x.rows_mut() {
            row += &self.classifier_biases[0];
        }
        // ReLU
        x.mapv_inplace(|v| v.max(0.0));

        let mut logits = x.dot(&self.classifier_weights[1]);
        for mut row in logits.rows_mut() {
            row += &self.classifier_biases[1];
        }

        logits
    }

    /// Predict class labels for input data.
    pub fn predict(&self, input: &Array2<f64>) -> Vec<usize> {
        let (logits, _) = self.forward(input);
        logits
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict class probabilities using softmax.
    pub fn predict_proba(&self, input: &Array2<f64>) -> Array2<f64> {
        let (logits, _) = self.forward(input);
        softmax(&logits)
    }

    /// Predict for a single sample, returning (class_index, confidence).
    pub fn predict_single(&self, input: &Array1<f64>) -> (usize, f64) {
        let batch = input.clone().insert_axis(Axis(0));
        let proba = self.predict_proba(&batch);
        let row = proba.row(0);
        let (idx, &conf) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        (idx, conf)
    }

    /// Pre-train on source domain data.
    pub fn pretrain(
        &mut self,
        features: &Array2<f64>,
        labels: &[usize],
        learning_rate: f64,
        epochs: usize,
    ) {
        for _epoch in 0..epochs {
            let (logits, _) = self.forward(features);
            let proba = softmax(&logits);

            // Compute cross-entropy gradients (simplified)
            let n = features.nrows() as f64;
            let mut grad = proba.clone();
            for (i, &label) in labels.iter().enumerate() {
                if label < self.num_classes {
                    grad[[i, label]] -= 1.0;
                }
            }
            grad /= n;

            // Update classifier weights
            let feat = self.feature_extractor.forward(features);
            let hidden = feat.dot(&self.classifier_weights[0]);
            let relu_hidden = hidden.mapv(|v| v.max(0.0));

            // Gradient for classifier layer 2
            let grad_w2 = relu_hidden.t().dot(&grad);
            let grad_b2 = grad.sum_axis(Axis(0));
            self.classifier_weights[1] = &self.classifier_weights[1] - &(&grad_w2 * learning_rate);
            self.classifier_biases[1] = &self.classifier_biases[1] - &(&grad_b2 * learning_rate);

            // Gradient for classifier layer 1
            let grad_hidden = grad.dot(&self.classifier_weights[1].t());
            let relu_mask = hidden.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            let grad_pre_relu = &grad_hidden * &relu_mask;
            let grad_w1 = feat.t().dot(&grad_pre_relu);
            let grad_b1 = grad_pre_relu.sum_axis(Axis(0));
            self.classifier_weights[0] = &self.classifier_weights[0] - &(&grad_w1 * learning_rate);
            self.classifier_biases[0] = &self.classifier_biases[0] - &(&grad_b1 * learning_rate);

            // Update batch norm stats
            self.feature_extractor.update_bn_stats(features);
        }
    }

    /// Apply fine-tuning strategy for target domain adaptation.
    pub fn apply_fine_tune_strategy(&mut self) {
        match self.fine_tune_strategy {
            FineTuneStrategy::FeatureExtraction => {
                self.feature_extractor.freeze_layers(3); // Freeze all
            }
            FineTuneStrategy::PartialFineTune => {
                self.feature_extractor.freeze_layers(1); // Freeze first layer
            }
            FineTuneStrategy::FullFineTune => {
                self.feature_extractor.unfreeze_all();
            }
        }
    }

    /// Compute the total loss (task loss + adaptation loss).
    pub fn compute_total_loss(
        &self,
        source_features: &Array2<f64>,
        target_features: &Array2<f64>,
        source_labels: &[usize],
    ) -> f64 {
        let (logits, src_feat) = self.forward(source_features);
        let tgt_feat = self.feature_extractor.forward(target_features);

        // Task loss (cross-entropy)
        let proba = softmax(&logits);
        let mut task_loss = 0.0;
        for (i, &label) in source_labels.iter().enumerate() {
            if label < self.num_classes {
                task_loss -= proba[[i, label]].max(1e-10).ln();
            }
        }
        task_loss /= source_labels.len() as f64;

        // Adaptation loss
        let adapt_loss = self.domain_adapter.compute_loss(&src_feat, &tgt_feat);

        task_loss + adapt_loss
    }

    /// Get a reference to the feature extractor.
    pub fn feature_extractor(&self) -> &FeatureExtractor {
        &self.feature_extractor
    }

    /// Get a mutable reference to the feature extractor.
    pub fn feature_extractor_mut(&mut self) -> &mut FeatureExtractor {
        &mut self.feature_extractor
    }

    /// Get a reference to the domain adapter.
    pub fn domain_adapter(&self) -> &DomainAdapter {
        &self.domain_adapter
    }

    /// Get a mutable reference to the domain adapter.
    pub fn domain_adapter_mut(&mut self) -> &mut DomainAdapter {
        &mut self.domain_adapter
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Xavier/Glorot initialization.
fn xavier_init(rng: &mut impl Rng, fan_in: usize, fan_out: usize) -> Array2<f64> {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let normal = Normal::new(0.0, std).unwrap();
    Array2::from_shape_fn((fan_in, fan_out), |_| rng.sample(normal))
}

/// Softmax over each row of a 2D array.
fn softmax(logits: &Array2<f64>) -> Array2<f64> {
    let mut result = logits.clone();
    for mut row in result.rows_mut() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum: f64 = row.iter().sum();
        row.mapv_inplace(|v| v / sum);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use rand_distr::Normal;

    #[test]
    fn test_transfer_network_creation() {
        let network = TransferNetwork::new(20, 128, 64, true);
        assert_eq!(network.num_classes(), 3);
    }

    #[test]
    fn test_forward_pass() {
        let network = TransferNetwork::new(20, 128, 64, false);
        let input = Array2::random((10, 20), Normal::new(0.0, 1.0).unwrap());
        let (logits, features) = network.forward(&input);
        assert_eq!(logits.shape(), &[10, 3]);
        assert_eq!(features.shape(), &[10, 64]);
    }

    #[test]
    fn test_predict() {
        let network = TransferNetwork::new(20, 128, 64, false);
        let input = Array2::random((10, 20), Normal::new(0.0, 1.0).unwrap());
        let predictions = network.predict(&input);
        assert_eq!(predictions.len(), 10);
        for p in &predictions {
            assert!(*p < 3);
        }
    }

    #[test]
    fn test_predict_proba() {
        let network = TransferNetwork::new(20, 128, 64, false);
        let input = Array2::random((5, 20), Normal::new(0.0, 1.0).unwrap());
        let proba = network.predict_proba(&input);
        assert_eq!(proba.shape(), &[5, 3]);
        // Each row should sum to ~1.0
        for row in proba.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_predict_single() {
        let network = TransferNetwork::new(10, 32, 16, false);
        let input = Array1::from_vec(vec![0.1; 10]);
        let (class, confidence) = network.predict_single(&input);
        assert!(class < 3);
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_fine_tune_strategies() {
        let mut network = TransferNetwork::new(20, 128, 64, false)
            .with_fine_tune_strategy(FineTuneStrategy::FeatureExtraction);
        network.apply_fine_tune_strategy();
        assert!(network.feature_extractor().is_frozen(0));
        assert!(network.feature_extractor().is_frozen(1));
        assert!(network.feature_extractor().is_frozen(2));

        let mut network = TransferNetwork::new(20, 128, 64, false)
            .with_fine_tune_strategy(FineTuneStrategy::PartialFineTune);
        network.apply_fine_tune_strategy();
        assert!(network.feature_extractor().is_frozen(0));
        assert!(!network.feature_extractor().is_frozen(1));

        let mut network = TransferNetwork::new(20, 128, 64, false)
            .with_fine_tune_strategy(FineTuneStrategy::FullFineTune);
        network.apply_fine_tune_strategy();
        assert!(!network.feature_extractor().is_frozen(0));
    }

    #[test]
    fn test_softmax() {
        let logits = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let proba = softmax(&logits);
        for row in proba.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        // For equal logits, probabilities should be equal
        let row1 = proba.row(1);
        assert!((row1[0] - row1[1]).abs() < 1e-6);
    }

    #[test]
    fn test_custom_num_classes() {
        let network = TransferNetwork::new(20, 128, 64, false).with_num_classes(5);
        let input = Array2::random((10, 20), Normal::new(0.0, 1.0).unwrap());
        let predictions = network.predict(&input);
        for p in &predictions {
            assert!(*p < 5);
        }
    }
}
