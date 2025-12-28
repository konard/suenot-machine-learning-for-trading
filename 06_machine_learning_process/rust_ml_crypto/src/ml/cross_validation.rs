//! Cross-validation utilities for model selection
//!
//! Includes:
//! - K-Fold cross-validation
//! - Time series split (forward chaining)
//! - Purged K-Fold for financial data

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Cross-validation split
#[derive(Debug, Clone)]
pub struct CVSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

/// Cross-validator
pub struct CrossValidator;

impl CrossValidator {
    /// K-Fold cross-validation splits
    ///
    /// # Arguments
    /// * `n_samples` - Total number of samples
    /// * `n_folds` - Number of folds
    /// * `shuffle` - Whether to shuffle indices
    pub fn k_fold(n_samples: usize, n_folds: usize, shuffle: bool) -> Vec<CVSplit> {
        assert!(n_folds > 1, "n_folds must be > 1");
        assert!(n_samples >= n_folds, "n_samples must be >= n_folds");

        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }

        let fold_size = n_samples / n_folds;
        let mut splits = Vec::with_capacity(n_folds);

        for i in 0..n_folds {
            let test_start = i * fold_size;
            let test_end = if i == n_folds - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices[..test_start]
                .iter()
                .chain(indices[test_end..].iter())
                .cloned()
                .collect();

            splits.push(CVSplit {
                train_indices,
                test_indices,
            });
        }

        splits
    }

    /// Time series split (forward chaining)
    /// Each fold adds more training data and tests on the next window
    ///
    /// # Arguments
    /// * `n_samples` - Total number of samples
    /// * `n_splits` - Number of splits
    /// * `test_size` - Size of each test set (None = n_samples / (n_splits + 1))
    pub fn time_series_split(
        n_samples: usize,
        n_splits: usize,
        test_size: Option<usize>,
    ) -> Vec<CVSplit> {
        assert!(n_splits > 0, "n_splits must be > 0");

        let test_size = test_size.unwrap_or(n_samples / (n_splits + 1));
        let mut splits = Vec::with_capacity(n_splits);

        for i in 0..n_splits {
            let test_start = (i + 1) * test_size;
            let test_end = (test_start + test_size).min(n_samples);

            if test_start >= n_samples {
                break;
            }

            let train_indices: Vec<usize> = (0..test_start).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            if !test_indices.is_empty() {
                splits.push(CVSplit {
                    train_indices,
                    test_indices,
                });
            }
        }

        splits
    }

    /// Purged K-Fold for financial data
    /// Removes samples from training set that are close to test set to prevent leakage
    ///
    /// # Arguments
    /// * `n_samples` - Total number of samples
    /// * `n_folds` - Number of folds
    /// * `purge_gap` - Number of samples to purge before test set
    /// * `embargo_gap` - Number of samples to embargo after test set
    pub fn purged_k_fold(
        n_samples: usize,
        n_folds: usize,
        purge_gap: usize,
        embargo_gap: usize,
    ) -> Vec<CVSplit> {
        assert!(n_folds > 1, "n_folds must be > 1");

        let fold_size = n_samples / n_folds;
        let mut splits = Vec::with_capacity(n_folds);

        for i in 0..n_folds {
            let test_start = i * fold_size;
            let test_end = if i == n_folds - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_indices: Vec<usize> = (test_start..test_end).collect();

            // Purge: remove samples before test set
            let purge_start = test_start.saturating_sub(purge_gap);

            // Embargo: remove samples after test set
            let embargo_end = (test_end + embargo_gap).min(n_samples);

            // Training indices: everything except test + purge + embargo
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&idx| idx < purge_start || idx >= embargo_end)
                .collect();

            splits.push(CVSplit {
                train_indices,
                test_indices,
            });
        }

        splits
    }

    /// Combinatorial purged cross-validation (CPCV)
    /// Generates all combinations of k groups for testing
    ///
    /// # Arguments
    /// * `n_samples` - Total number of samples
    /// * `n_groups` - Number of groups to divide data into
    /// * `n_test_groups` - Number of groups to use for testing in each split
    /// * `purge_gap` - Samples to purge between train and test
    pub fn combinatorial_purged_cv(
        n_samples: usize,
        n_groups: usize,
        n_test_groups: usize,
        purge_gap: usize,
    ) -> Vec<CVSplit> {
        assert!(n_test_groups < n_groups, "n_test_groups must be < n_groups");

        let group_size = n_samples / n_groups;
        let mut splits = Vec::new();

        // Generate combinations of test groups
        let combinations = Self::combinations(n_groups, n_test_groups);

        for test_group_indices in combinations {
            let mut test_indices = Vec::new();
            let mut purge_ranges: Vec<(usize, usize)> = Vec::new();

            for &group_idx in &test_group_indices {
                let start = group_idx * group_size;
                let end = if group_idx == n_groups - 1 {
                    n_samples
                } else {
                    (group_idx + 1) * group_size
                };

                test_indices.extend(start..end);

                // Add purge ranges around this test group
                let purge_start = start.saturating_sub(purge_gap);
                let purge_end = (end + purge_gap).min(n_samples);
                purge_ranges.push((purge_start, purge_end));
            }

            // Merge overlapping purge ranges
            purge_ranges.sort_by_key(|r| r.0);
            let merged_ranges = Self::merge_ranges(&purge_ranges);

            // Training indices: everything not in purge ranges
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&idx| !merged_ranges.iter().any(|&(s, e)| idx >= s && idx < e))
                .collect();

            splits.push(CVSplit {
                train_indices,
                test_indices,
            });
        }

        splits
    }

    /// Generate combinations of n items taken k at a time
    fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut current = Vec::with_capacity(k);
        Self::combinations_helper(0, n, k, &mut current, &mut result);
        result
    }

    fn combinations_helper(
        start: usize,
        n: usize,
        k: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }

        for i in start..n {
            current.push(i);
            Self::combinations_helper(i + 1, n, k, current, result);
            current.pop();
        }
    }

    /// Merge overlapping ranges
    fn merge_ranges(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        if ranges.is_empty() {
            return vec![];
        }

        let mut merged = vec![ranges[0]];

        for &(start, end) in &ranges[1..] {
            let last = merged.last_mut().unwrap();
            if start <= last.1 {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        }

        merged
    }

    /// Cross-validation score with a custom scoring function
    pub fn cross_val_score<F>(
        x: &Array2<f64>,
        y: &Array1<f64>,
        splits: &[CVSplit],
        scorer: F,
    ) -> Vec<f64>
    where
        F: Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>, &Array1<f64>) -> f64,
    {
        splits
            .iter()
            .map(|split| {
                let x_train = x.select(ndarray::Axis(0), &split.train_indices);
                let y_train = Array1::from_vec(
                    split.train_indices.iter().map(|&i| y[i]).collect(),
                );

                let x_test = x.select(ndarray::Axis(0), &split.test_indices);
                let y_test = Array1::from_vec(
                    split.test_indices.iter().map(|&i| y[i]).collect(),
                );

                scorer(&x_train, &y_train, &x_test, &y_test)
            })
            .collect()
    }
}

/// Summary statistics for cross-validation scores
#[derive(Debug, Clone)]
pub struct CVScores {
    pub scores: Vec<f64>,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl CVScores {
    /// Calculate summary statistics from scores
    pub fn from_scores(scores: Vec<f64>) -> Self {
        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            scores,
            mean,
            std,
            min,
            max,
        }
    }

    /// Print a summary of the scores
    pub fn summary(&self) -> String {
        format!(
            "CV Scores: mean={:.4} (+/- {:.4}), min={:.4}, max={:.4}",
            self.mean,
            self.std * 2.0,
            self.min,
            self.max
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_fold() {
        let splits = CrossValidator::k_fold(10, 5, false);

        assert_eq!(splits.len(), 5);

        // Each test fold should have 2 samples
        for split in &splits {
            assert_eq!(split.test_indices.len(), 2);
            assert_eq!(split.train_indices.len(), 8);
        }

        // All indices should be covered
        let all_test: Vec<usize> = splits.iter().flat_map(|s| s.test_indices.clone()).collect();
        assert_eq!(all_test.len(), 10);
    }

    #[test]
    fn test_time_series_split() {
        let splits = CrossValidator::time_series_split(100, 5, None);

        // Each subsequent split should have more training data
        for i in 1..splits.len() {
            assert!(splits[i].train_indices.len() > splits[i - 1].train_indices.len());
        }

        // Training and test should not overlap
        for split in &splits {
            for train_idx in &split.train_indices {
                assert!(!split.test_indices.contains(train_idx));
            }
        }
    }

    #[test]
    fn test_purged_k_fold() {
        let splits = CrossValidator::purged_k_fold(100, 5, 2, 2);

        assert_eq!(splits.len(), 5);

        // Check that purge and embargo gaps are respected
        for split in &splits {
            let test_min = *split.test_indices.iter().min().unwrap();
            let test_max = *split.test_indices.iter().max().unwrap();

            for &train_idx in &split.train_indices {
                // Training indices should not be within purge gap before test
                if train_idx < test_min {
                    assert!(train_idx < test_min.saturating_sub(2));
                }
                // Training indices should not be within embargo gap after test
                if train_idx > test_max {
                    assert!(train_idx > test_max + 2);
                }
            }
        }
    }

    #[test]
    fn test_combinations() {
        let combs = CrossValidator::combinations(4, 2);
        assert_eq!(combs.len(), 6); // C(4,2) = 6
        assert!(combs.contains(&vec![0, 1]));
        assert!(combs.contains(&vec![2, 3]));
    }
}
