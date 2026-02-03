//! Trading signal generation from SSM-GNN model output.

use crate::model::{SsmGnnHybrid, Signal};

/// Generate trading signals for multiple assets using the SSM-GNN model.
///
/// # Arguments
/// * `model` - The SSM-GNN hybrid model
/// * `features` - (n_assets, seq_len, n_features) input features
/// * `edge_index` - (2, n_edges) graph edge indices
///
/// # Returns
/// Vector of (signal, confidence) for each asset
pub fn generate_signals(
    model: &SsmGnnHybrid,
    features: &[Vec<Vec<f64>>],
    edge_index: &[Vec<usize>],
) -> Vec<(Signal, f64)> {
    let predictions = model.predict_signals(features, edge_index);
    predictions
        .iter()
        .map(|p| (p.signal, p.confidence))
        .collect()
}

/// Compute portfolio weights from signals and confidences.
///
/// # Arguments
/// * `signals` - Vector of (signal, confidence) per asset
///
/// # Returns
/// Normalized portfolio weight vector
pub fn compute_weights(signals: &[(Signal, f64)]) -> Vec<f64> {
    let raw_weights: Vec<f64> = signals
        .iter()
        .map(|(signal, confidence): &(Signal, f64)| signal.value() * confidence)
        .collect();

    let total_abs: f64 = raw_weights.iter().map(|w: &f64| w.abs()).sum();
    if total_abs < 1e-10 {
        return vec![0.0; signals.len()];
    }

    raw_weights.iter().map(|w| w / total_abs).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_weights() {
        let signals = vec![
            (Signal::Long, 0.8),
            (Signal::Short, 0.6),
            (Signal::Neutral, 0.9),
        ];

        let weights = compute_weights(&signals);
        assert_eq!(weights.len(), 3);

        // Sum of absolute weights should be ~1
        let total: f64 = weights.iter().map(|w| w.abs()).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_neutral() {
        let signals = vec![
            (Signal::Neutral, 0.5),
            (Signal::Neutral, 0.5),
        ];

        let weights = compute_weights(&signals);
        assert!(weights.iter().all(|w| *w == 0.0));
    }
}
