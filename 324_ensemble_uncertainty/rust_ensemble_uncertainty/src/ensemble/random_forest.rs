//! Random Forest ensemble implementation.

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::SeedableRng;
use rayon::prelude::*;
use tracing::info;

use super::decision_tree::DecisionTree;
use super::traits::{EnsembleConfig, EnsembleModel, FeatureImportance, MaxFeatures};

/// Random Forest ensemble for regression
pub struct RandomForestEnsemble {
    config: EnsembleConfig,
    trees: Vec<DecisionTree>,
    oob_indices: Vec<Vec<usize>>,
    feature_importances: Option<Array1<f64>>,
    is_fitted: bool,
}

impl RandomForestEnsemble {
    /// Create a new Random Forest with specified number of trees and max depth
    pub fn new(n_estimators: usize, max_depth: Option<usize>) -> Self {
        Self {
            config: EnsembleConfig {
                n_estimators,
                max_depth,
                ..Default::default()
            },
            trees: Vec::new(),
            oob_indices: Vec::new(),
            feature_importances: None,
            is_fitted: false,
        }
    }

    /// Create with full configuration
    pub fn with_config(config: EnsembleConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            oob_indices: Vec::new(),
            feature_importances: None,
            is_fitted: false,
        }
    }

    /// Get out-of-bag predictions
    pub fn oob_predictions(&self, x: &Array2<f64>, y: &Array1<f64>) -> Option<(Array1<f64>, Array1<f64>)> {
        if !self.is_fitted || self.oob_indices.is_empty() {
            return None;
        }

        let n_samples = x.nrows();
        let mut oob_preds: Vec<Vec<f64>> = vec![Vec::new(); n_samples];

        for (tree_idx, tree) in self.trees.iter().enumerate() {
            let oob_mask = &self.oob_indices[tree_idx];
            for &sample_idx in oob_mask {
                if sample_idx < n_samples {
                    let row = x.row(sample_idx).to_owned();
                    let (pred, _) = tree.predict_single(&row);
                    oob_preds[sample_idx].push(pred);
                }
            }
        }

        let mut oob_mean = Array1::zeros(n_samples);
        let mut oob_std = Array1::zeros(n_samples);

        for i in 0..n_samples {
            if !oob_preds[i].is_empty() {
                let mean: f64 = oob_preds[i].iter().sum::<f64>() / oob_preds[i].len() as f64;
                let variance: f64 = oob_preds[i].iter()
                    .map(|&p| (p - mean).powi(2))
                    .sum::<f64>() / oob_preds[i].len() as f64;
                oob_mean[i] = mean;
                oob_std[i] = variance.sqrt();
            } else {
                oob_mean[i] = f64::NAN;
                oob_std[i] = f64::NAN;
            }
        }

        Some((oob_mean, oob_std))
    }

    /// Predict quantiles
    pub fn predict_quantiles(&self, x: &Array2<f64>, quantiles: &[f64]) -> Result<Array2<f64>> {
        if !self.is_fitted {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        let n_samples = x.nrows();
        let n_quantiles = quantiles.len();

        // Get all tree predictions
        let all_predictions: Vec<Array1<f64>> = self.trees
            .iter()
            .map(|tree| tree.predict(x))
            .collect();

        let mut result = Array2::zeros((n_samples, n_quantiles));

        for i in 0..n_samples {
            let mut sample_preds: Vec<f64> = all_predictions
                .iter()
                .map(|preds| preds[i])
                .collect();
            sample_preds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            for (j, &q) in quantiles.iter().enumerate() {
                let idx = ((sample_preds.len() as f64 - 1.0) * q).round() as usize;
                result[[i, j]] = sample_preds[idx.min(sample_preds.len() - 1)];
            }
        }

        Ok(result)
    }
}

impl EnsembleModel for RandomForestEnsemble {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let max_features = self.config.max_features.calculate(n_features);

        info!(
            "Fitting Random Forest with {} trees, max_depth={:?}, max_features={}",
            self.config.n_estimators,
            self.config.max_depth,
            max_features
        );

        let base_seed = self.config.random_state.unwrap_or(42);

        // Train trees in parallel
        let results: Vec<(DecisionTree, Vec<usize>)> = (0..self.config.n_estimators)
            .into_par_iter()
            .map(|tree_idx| {
                let seed = base_seed + tree_idx as u64;
                let mut rng = StdRng::seed_from_u64(seed);

                // Bootstrap sample
                let bootstrap_indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.gen_range(0..n_samples))
                    .collect();

                // OOB indices (samples not in bootstrap)
                let oob_indices: Vec<usize> = (0..n_samples)
                    .filter(|i| !bootstrap_indices.contains(i))
                    .collect();

                // Create training data from bootstrap
                let mut x_boot = Array2::zeros((n_samples, n_features));
                let mut y_boot = Array1::zeros(n_samples);
                for (i, &idx) in bootstrap_indices.iter().enumerate() {
                    x_boot.row_mut(i).assign(&x.row(idx));
                    y_boot[i] = y[idx];
                }

                // Train tree
                let mut tree = DecisionTree::new(
                    self.config.max_depth,
                    self.config.min_samples_split,
                    self.config.min_samples_leaf,
                    max_features,
                    Some(seed),
                );
                tree.fit(&x_boot, &y_boot).unwrap();

                (tree, oob_indices)
            })
            .collect();

        // Collect results
        self.trees.clear();
        self.oob_indices.clear();
        for (tree, oob) in results {
            self.trees.push(tree);
            self.oob_indices.push(oob);
        }

        self.is_fitted = true;
        info!("Random Forest training complete");

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        // Get predictions from all trees
        let all_predictions: Vec<Array1<f64>> = self.trees
            .par_iter()
            .map(|tree| tree.predict(x))
            .collect();

        // Average predictions
        let n_samples = x.nrows();
        let mut mean_pred = Array1::zeros(n_samples);

        for preds in &all_predictions {
            mean_pred = mean_pred + preds;
        }
        mean_pred /= self.trees.len() as f64;

        Ok(mean_pred)
    }

    fn predict_with_uncertainty(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        if !self.is_fitted {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        // Get predictions from all trees
        let all_predictions: Vec<Array1<f64>> = self.trees
            .par_iter()
            .map(|tree| tree.predict(x))
            .collect();

        let n_samples = x.nrows();
        let n_trees = self.trees.len() as f64;

        // Calculate mean
        let mut mean_pred = Array1::zeros(n_samples);
        for preds in &all_predictions {
            mean_pred = mean_pred + preds;
        }
        mean_pred /= n_trees;

        // Calculate standard deviation (uncertainty)
        let mut variance = Array1::zeros(n_samples);
        for preds in &all_predictions {
            let diff = preds - &mean_pred;
            variance = variance + &diff.mapv(|x| x.powi(2));
        }
        variance /= n_trees;
        let uncertainty = variance.mapv(|v| v.sqrt());

        Ok((mean_pred, uncertainty))
    }

    fn n_estimators(&self) -> usize {
        self.config.n_estimators
    }

    fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

impl FeatureImportance for RandomForestEnsemble {
    fn feature_importances(&self) -> Result<Array1<f64>> {
        if let Some(ref importances) = self.feature_importances {
            Ok(importances.clone())
        } else {
            Err(anyhow::anyhow!("Feature importances not computed"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_random_forest_fit_predict() {
        let n_samples = 100;
        let n_features = 5;

        let x = Array2::random((n_samples, n_features), Uniform::new(-1.0, 1.0));
        let y: Array1<f64> = x.column(0).to_owned() * 2.0
            + x.column(1).to_owned()
            + Array1::random(n_samples, Uniform::new(-0.1, 0.1));

        let mut rf = RandomForestEnsemble::new(10, Some(5));
        rf.fit(&x, &y).unwrap();

        assert!(rf.is_fitted());
        assert_eq!(rf.n_estimators(), 10);

        let predictions = rf.predict(&x).unwrap();
        assert_eq!(predictions.len(), n_samples);

        let (preds, uncertainties) = rf.predict_with_uncertainty(&x).unwrap();
        assert_eq!(preds.len(), n_samples);
        assert_eq!(uncertainties.len(), n_samples);
        assert!(uncertainties.iter().all(|&u| u >= 0.0));
    }
}
