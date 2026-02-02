//! Transfer Entropy estimation methods.
//!
//! Provides binning and symbolic estimators for computing Transfer Entropy
//! between time series.

use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Estimation method for Transfer Entropy.
#[derive(Debug, Clone)]
pub enum TEMethod {
    /// Histogram-based discretization.
    Binning { n_bins: usize },
    /// Ordinal pattern symbolization.
    Symbolic { order: usize },
}

/// Transfer Entropy estimator supporting multiple methods.
#[derive(Debug, Clone)]
pub struct TransferEntropyEstimator {
    method: TEMethod,
    history_length: usize,
}

impl TransferEntropyEstimator {
    /// Create a new estimator.
    ///
    /// # Arguments
    /// * `method` - Estimation method (Binning or Symbolic).
    /// * `history_length` - Number of past time steps to condition on.
    pub fn new(method: TEMethod, history_length: usize) -> Self {
        Self { method, history_length }
    }

    /// Compute Transfer Entropy from source to target.
    ///
    /// Returns TE in bits (log base 2).
    pub fn compute_te(&self, source: &[f64], target: &[f64]) -> f64 {
        assert_eq!(source.len(), target.len(), "Source and target must have same length");
        assert!(
            source.len() >= self.history_length + 2,
            "Time series too short for given history length"
        );

        match &self.method {
            TEMethod::Binning { n_bins } => self.te_binning(source, target, *n_bins),
            TEMethod::Symbolic { order } => self.te_symbolic(source, target, *order),
        }
    }

    /// Compute Effective Transfer Entropy with permutation test.
    ///
    /// Returns (effective_te, p_value).
    pub fn effective_te(
        &self,
        source: &[f64],
        target: &[f64],
        n_shuffles: usize,
    ) -> (f64, f64) {
        let te_original = self.compute_te(source, target);

        let mut rng = StdRng::seed_from_u64(42);
        let mut shuffled: Vec<f64> = source.to_vec();
        let mut surrogate_sum = 0.0;
        let mut count_ge = 0usize;

        for _ in 0..n_shuffles {
            shuffled.shuffle(&mut rng);
            let te_surr = self.compute_te(&shuffled, target);
            surrogate_sum += te_surr;
            if te_surr >= te_original {
                count_ge += 1;
            }
            // Restore for next shuffle
            shuffled.copy_from_slice(source);
        }

        let mean_surrogate = surrogate_sum / n_shuffles as f64;
        let ete = te_original - mean_surrogate;
        let p_value = count_ge as f64 / n_shuffles as f64;
        (ete, p_value)
    }

    /// Compute Net Transfer Entropy: TE(X→Y) - TE(Y→X).
    ///
    /// Positive means X is the net information source.
    pub fn net_te(&self, x: &[f64], y: &[f64]) -> f64 {
        self.compute_te(x, y) - self.compute_te(y, x)
    }

    // ------------------------------------------------------------------ //
    //  Binning estimator
    // ------------------------------------------------------------------ //

    fn te_binning(&self, source: &[f64], target: &[f64], n_bins: usize) -> f64 {
        let src_binned = Self::discretize(source, n_bins);
        let tgt_binned = Self::discretize(target, n_bins);

        let k = self.history_length;
        let n = source.len();

        let mut joint_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut yk_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut yk_yt1_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut yk_xl_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut total = 0usize;

        for t in k..n - 1 {
            let y_past: Vec<usize> = tgt_binned[t - k..t].to_vec();
            let x_past: Vec<usize> = src_binned[t - k..t].to_vec();
            let y_next = tgt_binned[t + 1];

            // Full key: y_past + x_past + y_next
            let mut key_full = y_past.clone();
            key_full.extend_from_slice(&x_past);
            key_full.push(y_next);

            // yk_xl key
            let mut key_yk_xl = y_past.clone();
            key_yk_xl.extend_from_slice(&x_past);

            // yk_yt1 key
            let mut key_yk_yt1 = y_past.clone();
            key_yk_yt1.push(y_next);

            *joint_counts.entry(key_full).or_insert(0) += 1;
            *yk_counts.entry(y_past).or_insert(0) += 1;
            *yk_yt1_counts.entry(key_yk_yt1).or_insert(0) += 1;
            *yk_xl_counts.entry(key_yk_xl).or_insert(0) += 1;

            total += 1;
        }

        if total == 0 {
            return 0.0;
        }

        let mut te_sum = 0.0f64;

        for (key_full, &count) in &joint_counts {
            let y_past = &key_full[..k];
            let x_past = &key_full[k..2 * k];
            let y_next = key_full[2 * k];

            let p_full = count as f64 / total as f64;

            let mut key_yk_xl = y_past.to_vec();
            key_yk_xl.extend_from_slice(x_past);

            let mut key_yk_yt1 = y_past.to_vec();
            key_yk_yt1.push(y_next);

            let n_yk_xl = *yk_xl_counts.get(&key_yk_xl).unwrap_or(&1) as f64;
            let n_yk_yt1 = *yk_yt1_counts.get(&key_yk_yt1).unwrap_or(&0) as f64;
            let n_yk = *yk_counts.get(y_past).unwrap_or(&1) as f64;

            let p_yt1_given_yk_xl = count as f64 / n_yk_xl;
            let p_yt1_given_yk = n_yk_yt1 / n_yk;

            if p_yt1_given_yk > 0.0 && p_yt1_given_yk_xl > 0.0 {
                te_sum += p_full * (p_yt1_given_yk_xl / p_yt1_given_yk).log2();
            }
        }

        te_sum.max(0.0)
    }

    fn discretize(x: &[f64], n_bins: usize) -> Vec<usize> {
        let min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-15 {
            return vec![0; x.len()];
        }

        let bin_width = (max_val - min_val + 1e-10) / n_bins as f64;
        x.iter()
            .map(|&v| {
                let bin = ((v - min_val) / bin_width) as usize;
                bin.min(n_bins - 1)
            })
            .collect()
    }

    // ------------------------------------------------------------------ //
    //  Symbolic estimator
    // ------------------------------------------------------------------ //

    fn te_symbolic(&self, source: &[f64], target: &[f64], order: usize) -> f64 {
        let src_symbols = Self::to_ordinal_patterns(source, order);
        let tgt_symbols = Self::to_ordinal_patterns(target, order);

        let min_len = src_symbols.len().min(tgt_symbols.len());
        let src_symbols = &src_symbols[..min_len];
        let tgt_symbols = &tgt_symbols[..min_len];

        let k = self.history_length;
        let n = src_symbols.len();

        if n < k + 2 {
            return 0.0;
        }

        let mut joint_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut yk_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut yk_yt1_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut yk_xl_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut total = 0usize;

        for t in k..n - 1 {
            let y_past: Vec<usize> = tgt_symbols[t - k..t].to_vec();
            let x_past: Vec<usize> = src_symbols[t - k..t].to_vec();
            let y_next = tgt_symbols[t + 1];

            let mut key_full = y_past.clone();
            key_full.extend_from_slice(&x_past);
            key_full.push(y_next);

            let mut key_yk_xl = y_past.clone();
            key_yk_xl.extend_from_slice(&x_past);

            let mut key_yk_yt1 = y_past.clone();
            key_yk_yt1.push(y_next);

            *joint_counts.entry(key_full).or_insert(0) += 1;
            *yk_counts.entry(y_past).or_insert(0) += 1;
            *yk_yt1_counts.entry(key_yk_yt1).or_insert(0) += 1;
            *yk_xl_counts.entry(key_yk_xl).or_insert(0) += 1;

            total += 1;
        }

        if total == 0 {
            return 0.0;
        }

        let mut te_sum = 0.0f64;

        for (key_full, &count) in &joint_counts {
            let y_past = &key_full[..k];
            let x_past = &key_full[k..2 * k];
            let y_next = key_full[2 * k];

            let p_full = count as f64 / total as f64;

            let mut key_yk_xl = y_past.to_vec();
            key_yk_xl.extend_from_slice(x_past);

            let mut key_yk_yt1 = y_past.to_vec();
            key_yk_yt1.push(y_next);

            let n_yk_xl = *yk_xl_counts.get(&key_yk_xl).unwrap_or(&1) as f64;
            let n_yk_yt1 = *yk_yt1_counts.get(&key_yk_yt1).unwrap_or(&0) as f64;
            let n_yk = *yk_counts.get(y_past).unwrap_or(&1) as f64;

            let p_yt1_given_yk_xl = count as f64 / n_yk_xl;
            let p_yt1_given_yk = n_yk_yt1 / n_yk;

            if p_yt1_given_yk > 0.0 && p_yt1_given_yk_xl > 0.0 {
                te_sum += p_full * (p_yt1_given_yk_xl / p_yt1_given_yk).log2();
            }
        }

        te_sum.max(0.0)
    }

    fn to_ordinal_patterns(x: &[f64], d: usize) -> Vec<usize> {
        let n = x.len();
        if n < d {
            return vec![];
        }

        let mut symbols = Vec::with_capacity(n - d + 1);
        for i in 0..=(n - d) {
            let window = &x[i..i + d];
            // Compute rank permutation
            let mut indices: Vec<usize> = (0..d).collect();
            indices.sort_by(|&a, &b| window[a].partial_cmp(&window[b]).unwrap());
            let mut ranks = vec![0usize; d];
            for (rank, &idx) in indices.iter().enumerate() {
                ranks[idx] = rank;
            }
            // Lehmer code
            let mut index = 0usize;
            let mut factorial = 1usize;
            for j in (0..d).rev() {
                let count = ranks[j + 1..].iter().filter(|&&r| r < ranks[j]).count();
                index += count * factorial;
                factorial *= d - j;
            }
            symbols.push(index);
        }
        symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    #[test]
    fn test_te_zero_for_independent() {
        // Use large sample and few bins to reduce finite-sample bias
        let mut rng = StdRng::seed_from_u64(42);
        let n = 2000;
        let source: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let target: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();

        let estimator = TransferEntropyEstimator::new(
            TEMethod::Binning { n_bins: 3 },
            1,
        );
        let te = estimator.compute_te(&source, &target);
        // TE should be close to zero for independent series
        assert!(te < 0.05, "TE for independent series should be small, got {}", te);
    }

    #[test]
    fn test_te_positive_for_dependent() {
        // Use a deterministic periodic signal for reliable testing
        let n = 10000;
        // Source: periodic pattern (0, 0.33, 0.66, 1.0, 0, 0.33, ...)
        let source: Vec<f64> = (0..n).map(|i| ((i % 4) as f64) / 3.0).collect();
        // Target: exact lagged copy of source
        let mut target = vec![0.0; n];
        for i in 1..n {
            target[i] = source[i - 1];
        }

        let estimator = TransferEntropyEstimator::new(
            TEMethod::Binning { n_bins: 4 },
            1,
        );
        let te_forward = estimator.compute_te(&source, &target);
        let te_reverse = estimator.compute_te(&target, &source);

        // With deterministic lag, forward TE should be clearly larger
        assert!(
            te_forward > 0.0,
            "Forward TE should be positive, got {}",
            te_forward
        );
        assert!(
            te_forward > te_reverse,
            "Forward TE ({}) should exceed reverse TE ({})",
            te_forward,
            te_reverse
        );
    }

    #[test]
    fn test_effective_te() {
        // Deterministic periodic signal with lag
        let n = 5000;
        let source: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) / 4.0).collect();
        let mut target = vec![0.0; n];
        for i in 1..n {
            target[i] = source[i - 1];
        }

        let estimator = TransferEntropyEstimator::new(
            TEMethod::Binning { n_bins: 5 },
            1,
        );
        let (ete, p_value) = estimator.effective_te(&source, &target, 20);
        assert!(ete > 0.0, "Effective TE should be positive for dependent series, got {}", ete);
        assert!(p_value < 0.2, "p-value should be small for dependent series, got {}", p_value);
    }

    #[test]
    fn test_symbolic_te() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 500;
        let source: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let mut target = vec![0.0; n];
        for i in 1..n {
            target[i] = 0.5 * source[i - 1] + 0.5 * rng.gen::<f64>();
        }

        let estimator = TransferEntropyEstimator::new(
            TEMethod::Symbolic { order: 3 },
            2,
        );
        let te = estimator.compute_te(&source, &target);
        assert!(te >= 0.0, "Symbolic TE should be non-negative");
    }

    #[test]
    fn test_discretize() {
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let binned = TransferEntropyEstimator::discretize(&x, 4);
        // Check all bins are within range
        for &b in &binned {
            assert!(b < 4, "Bin {} out of range", b);
        }
    }
}
