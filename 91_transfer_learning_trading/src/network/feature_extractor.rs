//! Feature extraction network for transforming raw market features into embeddings.
//!
//! The feature extractor is the shared component that transfers knowledge between
//! source and target domains. Lower layers capture general market patterns while
//! upper layers learn domain-specific representations.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// A multi-layer feature extraction network with batch normalization.
///
/// Architecture:
/// ```text
/// Input → Linear → BatchNorm → ReLU → Dropout →
///         Linear → BatchNorm → ReLU → Dropout →
///         Linear → Output (feature embedding)
/// ```
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Input dimension (number of raw features)
    input_dim: usize,
    /// Hidden layer dimension
    hidden_dim: usize,
    /// Output feature/embedding dimension
    feature_dim: usize,
    /// Weight matrices for each layer
    weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    biases: Vec<Array1<f64>>,
    /// Batch normalization running means
    bn_means: Vec<Array1<f64>>,
    /// Batch normalization running variances
    bn_vars: Vec<Array1<f64>>,
    /// Dropout rate (0.0 to 1.0)
    dropout_rate: f64,
    /// Whether each layer is frozen (not trainable)
    frozen_layers: Vec<bool>,
}

impl FeatureExtractor {
    /// Create a new feature extractor with Xavier initialization.
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    /// * `hidden_dim` - Hidden layer size
    /// * `feature_dim` - Output embedding dimension
    pub fn new(input_dim: usize, hidden_dim: usize, feature_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = vec![
            xavier_init(&mut rng, input_dim, hidden_dim),
            xavier_init(&mut rng, hidden_dim, hidden_dim),
            xavier_init(&mut rng, hidden_dim, feature_dim),
        ];

        let biases = vec![
            Array1::zeros(hidden_dim),
            Array1::zeros(hidden_dim),
            Array1::zeros(feature_dim),
        ];

        let bn_means = vec![
            Array1::zeros(hidden_dim),
            Array1::zeros(hidden_dim),
        ];

        let bn_vars = vec![
            Array1::ones(hidden_dim),
            Array1::ones(hidden_dim),
        ];

        Self {
            input_dim,
            hidden_dim,
            feature_dim,
            weights,
            biases,
            bn_means,
            bn_vars,
            dropout_rate: 0.3,
            frozen_layers: vec![false, false, false],
        }
    }

    /// Forward pass: transform input features into embeddings.
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();

        // Layer 1: Linear + BatchNorm + ReLU
        x = x.dot(&self.weights[0]);
        add_bias_inplace(&mut x, &self.biases[0]);
        batch_norm_inplace(&mut x, &self.bn_means[0], &self.bn_vars[0]);
        relu_inplace(&mut x);

        // Layer 2: Linear + BatchNorm + ReLU
        x = x.dot(&self.weights[1]);
        add_bias_inplace(&mut x, &self.biases[1]);
        batch_norm_inplace(&mut x, &self.bn_means[1], &self.bn_vars[1]);
        relu_inplace(&mut x);

        // Layer 3: Linear (output embedding)
        x = x.dot(&self.weights[2]);
        add_bias_inplace(&mut x, &self.biases[2]);

        x
    }

    /// Extract features from a single sample.
    pub fn extract(&self, input: &Array1<f64>) -> Array1<f64> {
        let batch = input.clone().insert_axis(Axis(0));
        let result = self.forward(&batch);
        result.index_axis(Axis(0), 0).to_owned()
    }

    /// Freeze layers up to the given index (0-indexed, exclusive).
    /// Frozen layers will not be updated during fine-tuning.
    pub fn freeze_layers(&mut self, up_to: usize) {
        for i in 0..self.frozen_layers.len().min(up_to) {
            self.frozen_layers[i] = true;
        }
    }

    /// Unfreeze all layers for full fine-tuning.
    pub fn unfreeze_all(&mut self) {
        for frozen in &mut self.frozen_layers {
            *frozen = false;
        }
    }

    /// Check if a specific layer is frozen.
    pub fn is_frozen(&self, layer: usize) -> bool {
        self.frozen_layers.get(layer).copied().unwrap_or(false)
    }

    /// Update weights using gradient descent (simplified).
    /// Only updates non-frozen layers.
    pub fn update_weights(
        &mut self,
        gradients: &[(Array2<f64>, Array1<f64>)],
        learning_rate: f64,
    ) {
        for (i, (grad_w, grad_b)) in gradients.iter().enumerate() {
            if i < self.weights.len() && !self.frozen_layers[i] {
                let lr = if i == 0 {
                    learning_rate * 0.1 // Lower LR for early layers
                } else if i == 1 {
                    learning_rate * 0.5 // Medium LR for middle layers
                } else {
                    learning_rate // Full LR for later layers
                };

                self.weights[i] = &self.weights[i] - &(grad_w * lr);
                self.biases[i] = &self.biases[i] - &(grad_b * lr);
            }
        }
    }

    /// Update batch normalization statistics from a batch of data.
    pub fn update_bn_stats(&mut self, input: &Array2<f64>) {
        let mut x = input.clone();
        let momentum = 0.1;

        // After layer 1
        x = x.dot(&self.weights[0]);
        add_bias_inplace(&mut x, &self.biases[0]);
        let mean = x.mean_axis(Axis(0)).unwrap();
        let var = x.var_axis(Axis(0), 0.0);
        self.bn_means[0] = &self.bn_means[0] * (1.0 - momentum) + &mean * momentum;
        self.bn_vars[0] = &self.bn_vars[0] * (1.0 - momentum) + &var * momentum;
        batch_norm_inplace(&mut x, &self.bn_means[0], &self.bn_vars[0]);
        relu_inplace(&mut x);

        // After layer 2
        x = x.dot(&self.weights[1]);
        add_bias_inplace(&mut x, &self.biases[1]);
        let mean = x.mean_axis(Axis(0)).unwrap();
        let var = x.var_axis(Axis(0), 0.0);
        self.bn_means[1] = &self.bn_means[1] * (1.0 - momentum) + &mean * momentum;
        self.bn_vars[1] = &self.bn_vars[1] * (1.0 - momentum) + &var * momentum;
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    pub fn num_layers(&self) -> usize {
        self.weights.len()
    }
}

/// Xavier/Glorot initialization for weight matrices.
fn xavier_init(rng: &mut impl Rng, fan_in: usize, fan_out: usize) -> Array2<f64> {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let normal = Normal::new(0.0, std).unwrap();
    Array2::from_shape_fn((fan_in, fan_out), |_| rng.sample(normal))
}

/// Apply ReLU activation in-place.
fn relu_inplace(x: &mut Array2<f64>) {
    x.mapv_inplace(|v| v.max(0.0));
}

/// Add bias to each row of a 2D array.
fn add_bias_inplace(x: &mut Array2<f64>, bias: &Array1<f64>) {
    for mut row in x.rows_mut() {
        row += bias;
    }
}

/// Apply batch normalization in-place.
fn batch_norm_inplace(x: &mut Array2<f64>, mean: &Array1<f64>, var: &Array1<f64>) {
    let eps = 1e-5;
    for mut row in x.rows_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            *val = (*val - mean[i]) / (var[i] + eps).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new(20, 128, 64);
        assert_eq!(extractor.input_dim(), 20);
        assert_eq!(extractor.hidden_dim(), 128);
        assert_eq!(extractor.feature_dim(), 64);
        assert_eq!(extractor.num_layers(), 3);
    }

    #[test]
    fn test_forward_pass_shape() {
        let extractor = FeatureExtractor::new(20, 128, 64);
        let input = Array2::zeros((10, 20)); // batch of 10, 20 features each
        let output = extractor.forward(&input);
        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_extract_single() {
        let extractor = FeatureExtractor::new(10, 32, 16);
        let input = Array1::zeros(10);
        let output = extractor.extract(&input);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_freeze_layers() {
        let mut extractor = FeatureExtractor::new(20, 128, 64);
        assert!(!extractor.is_frozen(0));
        assert!(!extractor.is_frozen(1));

        extractor.freeze_layers(2);
        assert!(extractor.is_frozen(0));
        assert!(extractor.is_frozen(1));
        assert!(!extractor.is_frozen(2));

        extractor.unfreeze_all();
        assert!(!extractor.is_frozen(0));
    }

    #[test]
    fn test_xavier_init_scale() {
        let mut rng = rand::thread_rng();
        let w = xavier_init(&mut rng, 100, 100);
        let mean = w.mean().unwrap();
        let std = w.std(0.0);
        // Xavier init should have mean near 0, std near sqrt(2/200) = 0.1
        assert!(mean.abs() < 0.1);
        assert!((std - 0.1).abs() < 0.05);
    }
}
