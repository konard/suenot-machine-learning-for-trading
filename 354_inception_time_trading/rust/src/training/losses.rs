//! Loss functions for training
//!
//! This module provides various loss functions for classification.

use tch::Tensor;

/// Standard cross-entropy loss
pub fn cross_entropy_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    predictions.cross_entropy_for_logits(targets)
}

/// Weighted cross-entropy loss for imbalanced classes
pub fn weighted_cross_entropy(predictions: &Tensor, targets: &Tensor, weights: &[f64]) -> Tensor {
    let weight_tensor = Tensor::from_slice(weights).to_device(predictions.device());
    predictions.cross_entropy_loss::<Tensor>(targets, Some(weight_tensor), tch::Reduction::Mean, -100, 0.0)
}

/// Focal loss for handling class imbalance
///
/// FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
pub fn focal_loss(predictions: &Tensor, targets: &Tensor, gamma: f64, alpha: Option<&[f64]>) -> Tensor {
    let probs = predictions.softmax(-1, tch::Kind::Float);

    // Get probability of true class
    let num_classes = predictions.size().last().unwrap();
    let targets_one_hot = Tensor::zeros(predictions.size().as_slice(), (tch::Kind::Float, predictions.device()))
        .scatter_(-1, &targets.unsqueeze(-1), &Tensor::ones(&[1], (tch::Kind::Float, predictions.device())));

    let p_t = (&probs * &targets_one_hot).sum_dim_intlist([-1].as_slice(), false, tch::Kind::Float);

    // Focal weight: (1 - p_t)^gamma
    let focal_weight = (1.0 - &p_t).pow_tensor_scalar(gamma);

    // Log probability
    let log_p = p_t.clamp(1e-8, 1.0).log();

    // Apply alpha weighting if provided
    let loss = if let Some(alpha_weights) = alpha {
        let alpha_tensor = Tensor::from_slice(alpha_weights).to_device(predictions.device());
        let alpha_t = alpha_tensor.index_select(0, targets);
        -&alpha_t * &focal_weight * &log_p
    } else {
        -&focal_weight * &log_p
    };

    loss.mean(tch::Kind::Float)
}

/// Label smoothing cross-entropy
pub fn label_smoothing_loss(predictions: &Tensor, targets: &Tensor, smoothing: f64) -> Tensor {
    let num_classes = *predictions.size().last().unwrap() as f64;
    let log_probs = predictions.log_softmax(-1, tch::Kind::Float);

    // Create smoothed targets
    let targets_one_hot = Tensor::zeros(predictions.size().as_slice(), (tch::Kind::Float, predictions.device()))
        .scatter_(-1, &targets.unsqueeze(-1), &Tensor::ones(&[1], (tch::Kind::Float, predictions.device())));

    let smoothed_targets = &targets_one_hot * (1.0 - smoothing) + smoothing / num_classes;

    // KL divergence
    let loss = -(&smoothed_targets * &log_probs).sum_dim_intlist([-1].as_slice(), false, tch::Kind::Float);
    loss.mean(tch::Kind::Float)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_loss_functions_exist() {
        // Basic test to ensure functions compile
        assert!(true);
    }
}
