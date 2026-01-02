//! DenseNet Network Implementation
//!
//! The complete DenseNet architecture for trading signal prediction.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

use super::block::DenseBlock;
use super::config::DenseNetConfig;
use super::transition::TransitionLayer;

/// Complete DenseNet model for trading
#[derive(Debug)]
pub struct DenseNet {
    /// Model configuration
    pub config: DenseNetConfig,

    /// Initial convolution weights
    init_conv_weights: Array2<f64>,

    /// Initial convolution bias
    init_conv_bias: Array1<f64>,

    /// Initial batch norm parameters
    init_bn_gamma: Array1<f64>,
    init_bn_beta: Array1<f64>,
    init_bn_mean: Array1<f64>,
    init_bn_var: Array1<f64>,

    /// Dense blocks
    dense_blocks: Vec<DenseBlock>,

    /// Transition layers
    transitions: Vec<TransitionLayer>,

    /// Final batch norm
    final_bn_gamma: Array1<f64>,
    final_bn_beta: Array1<f64>,
    final_bn_mean: Array1<f64>,
    final_bn_var: Array1<f64>,

    /// Classifier weights
    classifier_w1: Array2<f64>,
    classifier_b1: Array1<f64>,
    classifier_w2: Array2<f64>,
    classifier_b2: Array1<f64>,

    /// Number of features at the end
    final_channels: usize,
}

impl DenseNet {
    /// Create a new DenseNet model
    pub fn new(config: DenseNetConfig) -> Self {
        config.validate().expect("Invalid configuration");

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Initial convolution: input_features -> init_channels, kernel=7
        let init_weights: Vec<f64> = (0..config.init_channels * config.input_features * 7)
            .map(|_| rng.sample(normal))
            .collect();
        let init_conv_weights = Array2::from_shape_vec(
            (config.init_channels, config.input_features * 7),
            init_weights,
        ).unwrap();
        let init_conv_bias = Array1::zeros(config.init_channels);

        // Initial batch norm
        let init_bn_gamma = Array1::ones(config.init_channels);
        let init_bn_beta = Array1::zeros(config.init_channels);
        let init_bn_mean = Array1::zeros(config.init_channels);
        let init_bn_var = Array1::ones(config.init_channels);

        // Build dense blocks and transitions
        let mut dense_blocks = Vec::new();
        let mut transitions = Vec::new();
        let mut num_features = config.init_channels;

        for (i, &num_layers) in config.block_config.iter().enumerate() {
            // Dense block
            let block = DenseBlock::new(
                num_layers,
                num_features,
                config.growth_rate,
                config.dropout,
                config.use_bottleneck,
                config.bottleneck_factor,
            );
            num_features = block.out_channels;
            dense_blocks.push(block);

            // Transition layer (except after last block)
            if i < config.block_config.len() - 1 {
                let trans = TransitionLayer::new(num_features, config.compression, 2);
                num_features = trans.out_channels;
                transitions.push(trans);
            }
        }

        let final_channels = num_features;

        // Final batch norm
        let final_bn_gamma = Array1::ones(final_channels);
        let final_bn_beta = Array1::zeros(final_channels);
        let final_bn_mean = Array1::zeros(final_channels);
        let final_bn_var = Array1::ones(final_channels);

        // Classifier: final_channels -> 128 -> num_classes
        let w1: Vec<f64> = (0..final_channels * 128).map(|_| rng.sample(normal)).collect();
        let classifier_w1 = Array2::from_shape_vec((128, final_channels), w1).unwrap();
        let classifier_b1 = Array1::zeros(128);

        let w2: Vec<f64> = (0..128 * config.num_classes).map(|_| rng.sample(normal)).collect();
        let classifier_w2 = Array2::from_shape_vec((config.num_classes, 128), w2).unwrap();
        let classifier_b2 = Array1::zeros(config.num_classes);

        Self {
            config,
            init_conv_weights,
            init_conv_bias,
            init_bn_gamma,
            init_bn_beta,
            init_bn_mean,
            init_bn_var,
            dense_blocks,
            transitions,
            final_bn_gamma,
            final_bn_beta,
            final_bn_mean,
            final_bn_var,
            classifier_w1,
            classifier_b1,
            classifier_w2,
            classifier_b2,
            final_channels,
        }
    }

    /// Forward pass through the network
    ///
    /// Input: [input_features, sequence_length]
    /// Output: TradingPrediction with probabilities and confidence
    pub fn forward(&self, x: &Array2<f64>, training: bool) -> TradingPrediction {
        // 1. Initial convolution
        let mut features = self.initial_conv(x);

        // 2. Initial batch norm + ReLU
        features = self.batch_norm(&features, &self.init_bn_gamma, &self.init_bn_beta,
                                   &self.init_bn_mean, &self.init_bn_var);
        features = features.mapv(|v| v.max(0.0));

        // 3. Initial max pooling (stride 2)
        features = self.max_pool(&features, 2);

        // 4. Dense blocks and transitions
        for (i, block) in self.dense_blocks.iter().enumerate() {
            features = block.forward(&features, training);

            if i < self.transitions.len() {
                features = self.transitions[i].forward(&features);
            }
        }

        // 5. Final batch norm + ReLU
        features = self.batch_norm(&features, &self.final_bn_gamma, &self.final_bn_beta,
                                   &self.final_bn_mean, &self.final_bn_var);
        features = features.mapv(|v| v.max(0.0));

        // 6. Global average pooling
        let pooled = self.global_avg_pool(&features);

        // 7. Classifier
        let hidden = self.classifier_w1.dot(&pooled) + &self.classifier_b1;
        let hidden = hidden.mapv(|v| v.max(0.0)); // ReLU

        // Apply dropout during training
        let hidden = if training && self.config.dropout > 0.0 {
            self.apply_dropout(&hidden)
        } else {
            hidden
        };

        let logits = self.classifier_w2.dot(&hidden) + &self.classifier_b2;

        // Softmax for probabilities
        let probs = self.softmax(&logits);

        // Calculate confidence as max probability - 1/num_classes
        let max_prob = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let baseline = 1.0 / self.config.num_classes as f64;
        let confidence = (max_prob - baseline) / (1.0 - baseline);

        TradingPrediction {
            logits: logits.to_vec(),
            probabilities: probs.to_vec(),
            predicted_class: self.argmax(&probs),
            confidence: confidence.max(0.0).min(1.0),
        }
    }

    /// Initial 7x1 convolution with stride 2
    fn initial_conv(&self, x: &Array2<f64>) -> Array2<f64> {
        let (in_ch, seq_len) = x.dim();
        let out_ch = self.config.init_channels;
        let k = 7;
        let stride = 2;
        let padding = k / 2;
        let new_len = (seq_len + 2 * padding - k) / stride + 1;

        let mut output = Array2::zeros((out_ch, new_len));

        for out_c in 0..out_ch {
            for i in 0..new_len {
                let pos = i * stride;
                let mut sum = self.init_conv_bias[out_c];

                for in_c in 0..in_ch.min(self.config.input_features) {
                    for ki in 0..k {
                        let input_pos = pos as i64 + ki as i64 - padding as i64;
                        if input_pos >= 0 && input_pos < seq_len as i64 {
                            let weight_idx = in_c * k + ki;
                            if weight_idx < self.init_conv_weights.ncols() {
                                sum += x[[in_c, input_pos as usize]]
                                    * self.init_conv_weights[[out_c, weight_idx]];
                            }
                        }
                    }
                }

                output[[out_c, i]] = sum;
            }
        }

        output
    }

    /// Batch normalization
    fn batch_norm(
        &self,
        x: &Array2<f64>,
        gamma: &Array1<f64>,
        beta: &Array1<f64>,
        mean: &Array1<f64>,
        var: &Array1<f64>
    ) -> Array2<f64> {
        let eps = 1e-5;
        let mut result = x.clone();

        for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
            if i < gamma.len() {
                row.mapv_inplace(|v| {
                    gamma[i] * (v - mean[i]) / (var[i] + eps).sqrt() + beta[i]
                });
            }
        }

        result
    }

    /// Max pooling with given stride
    fn max_pool(&self, x: &Array2<f64>, stride: usize) -> Array2<f64> {
        let (channels, seq_len) = x.dim();
        let new_len = seq_len / stride;

        if new_len == 0 {
            return x.clone();
        }

        let mut output = Array2::zeros((channels, new_len));

        for c in 0..channels {
            for i in 0..new_len {
                let start = i * stride;
                let end = (start + stride).min(seq_len);

                let max_val = (start..end)
                    .map(|j| x[[c, j]])
                    .fold(f64::NEG_INFINITY, f64::max);

                output[[c, i]] = max_val;
            }
        }

        output
    }

    /// Global average pooling
    fn global_avg_pool(&self, x: &Array2<f64>) -> Array1<f64> {
        x.mean_axis(Axis(1)).unwrap()
    }

    /// Softmax activation
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum: f64 = exp_x.sum();
        exp_x / sum
    }

    /// Apply dropout
    fn apply_dropout(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.config.dropout);

        x.mapv(|v| {
            if rng.gen::<f64>() < self.config.dropout {
                0.0
            } else {
                v * scale
            }
        })
    }

    /// Argmax
    fn argmax(&self, x: &Array1<f64>) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        // Initial conv
        count += self.init_conv_weights.len() + self.init_conv_bias.len();

        // Initial batch norm
        count += 2 * self.config.init_channels;

        // Dense blocks
        count += self.dense_blocks.iter().map(|b| b.num_parameters()).sum::<usize>();

        // Transitions
        count += self.transitions.iter().map(|t| t.num_parameters()).sum::<usize>();

        // Final batch norm
        count += 2 * self.final_channels;

        // Classifier
        count += self.classifier_w1.len() + self.classifier_b1.len();
        count += self.classifier_w2.len() + self.classifier_b2.len();

        count
    }

    /// Get model summary
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("DenseNet Trading Model\n");
        s.push_str("======================\n\n");

        s.push_str(&format!("Input: {} features x {} sequence\n",
            self.config.input_features, self.config.sequence_length));
        s.push_str(&format!("Growth rate: {}\n", self.config.growth_rate));
        s.push_str(&format!("Compression: {:.2}\n", self.config.compression));
        s.push_str(&format!("Bottleneck: {}\n\n", self.config.use_bottleneck));

        s.push_str("Architecture:\n");
        s.push_str(&format!("  Initial Conv: {} -> {} channels\n",
            self.config.input_features, self.config.init_channels));

        for (i, block) in self.dense_blocks.iter().enumerate() {
            let info = block.info();
            s.push_str(&format!("  {}\n", info));

            if i < self.transitions.len() {
                let t_info = self.transitions[i].info();
                s.push_str(&format!("  {}\n", t_info));
            }
        }

        s.push_str(&format!("\n  Classifier: {} -> 128 -> {}\n",
            self.final_channels, self.config.num_classes));

        s.push_str(&format!("\nTotal parameters: {}\n", self.num_parameters()));

        s
    }
}

/// Trading prediction output
#[derive(Debug, Clone)]
pub struct TradingPrediction {
    /// Raw logits
    pub logits: Vec<f64>,

    /// Softmax probabilities
    pub probabilities: Vec<f64>,

    /// Predicted class (0=Short, 1=Hold, 2=Long)
    pub predicted_class: usize,

    /// Confidence score [0, 1]
    pub confidence: f64,
}

impl TradingPrediction {
    /// Get trading signal
    pub fn signal(&self) -> TradingAction {
        match self.predicted_class {
            0 => TradingAction::Short,
            1 => TradingAction::Hold,
            2 => TradingAction::Long,
            _ => TradingAction::Hold,
        }
    }

    /// Check if prediction is confident enough
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAction {
    Short,
    Hold,
    Long,
}

impl std::fmt::Display for TradingAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingAction::Short => write!(f, "SHORT"),
            TradingAction::Hold => write!(f, "HOLD"),
            TradingAction::Long => write!(f, "LONG"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_densenet_creation() {
        let config = DenseNetConfig::tiny();
        let model = DenseNet::new(config);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_densenet_forward() {
        let config = DenseNetConfig::tiny();
        let model = DenseNet::new(config.clone());

        let input = Array2::ones((config.input_features, config.sequence_length));
        let output = model.forward(&input, false);

        assert_eq!(output.probabilities.len(), config.num_classes);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_prediction_signal() {
        let pred = TradingPrediction {
            logits: vec![-1.0, 0.0, 2.0],
            probabilities: vec![0.1, 0.2, 0.7],
            predicted_class: 2,
            confidence: 0.8,
        };

        assert_eq!(pred.signal(), TradingAction::Long);
        assert!(pred.is_confident(0.5));
    }

    #[test]
    fn test_model_summary() {
        let config = DenseNetConfig::small();
        let model = DenseNet::new(config);
        let summary = model.summary();
        assert!(summary.contains("DenseNet"));
        assert!(summary.contains("parameters"));
    }
}
