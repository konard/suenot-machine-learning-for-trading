//! Proximity metrics for counterfactual evaluation.

/// Proximity metric types.
#[derive(Debug, Clone, Copy)]
pub enum ProximityMetric {
    /// L1 (Manhattan) distance
    L1,
    /// L2 (Euclidean) distance
    L2,
    /// L0 (count of changes) distance
    L0,
}

impl ProximityMetric {
    /// Calculate distance between two vectors.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        match self {
            ProximityMetric::L1 => x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).abs())
                .sum(),
            ProximityMetric::L2 => x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt(),
            ProximityMetric::L0 => x
                .iter()
                .zip(y.iter())
                .filter(|(a, b)| (a - b).abs() > 0.01)
                .count() as f64,
        }
    }
}

/// Count the number of features that changed.
pub fn count_changes(x: &[f64], y: &[f64], threshold: f64) -> usize {
    x.iter()
        .zip(y.iter())
        .filter(|(a, b)| (*a - *b).abs() > threshold)
        .count()
}

/// Find which features changed.
pub fn find_changed_features(
    x: &[f64],
    y: &[f64],
    threshold: f64,
) -> Vec<(usize, f64, f64)> {
    x.iter()
        .zip(y.iter())
        .enumerate()
        .filter(|(_, (a, b))| (*a - *b).abs() > threshold)
        .map(|(i, (&a, &b))| (i, a, b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_distance() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];
        assert!((ProximityMetric::L1.distance(&x, &y) - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_l2_distance() {
        let x = vec![0.0, 0.0];
        let y = vec![3.0, 4.0];
        assert!((ProximityMetric::L2.distance(&x, &y) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_count_changes() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.5, 3.0];
        assert_eq!(count_changes(&x, &y, 0.1), 1);
    }
}
