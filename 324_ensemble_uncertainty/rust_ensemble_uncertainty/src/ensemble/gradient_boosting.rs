//! Gradient Boosting ensemble implementation.

use anyhow::Result;
use ndarray::{Array1, Array2};
use tracing::info;

use super::decision_tree::DecisionTree;
use super::traits::{EnsembleConfig, EnsembleModel, MaxFeatures};

/// Gradient Boosting ensemble for regression
pub struct GradientBoostingEnsemble {
    config: GradientBoostingConfig,
    trees: Vec<DecisionTree>,
    initial_prediction: f64,
    quantile_trees_lower: Vec<DecisionTree>,
    quantile_trees_upper: Vec<DecisionTree>,
    is_fitted: bool,
}

/// Configuration for Gradient Boosting
#[derive(Debug, Clone)]
pub struct GradientBoostingConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub subsample: f64,
    pub use_quantile: bool,
    pub lower_quantile: f64,
    pub upper_quantile: f64,
    pub random_state: Option<u64>,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            use_quantile: true,
            lower_quantile: 0.1,
            upper_quantile: 0.9,
            random_state: Some(42),
        }
    }
}

impl GradientBoostingEnsemble {
    /// Create a new Gradient Boosting ensemble
    pub fn new(n_estimators: usize, use_quantile: bool) -> Self {
        Self {
            config: GradientBoostingConfig {
                n_estimators,
                use_quantile,
                ..Default::default()
            },
            trees: Vec::new(),
            initial_prediction: 0.0,
            quantile_trees_lower: Vec::new(),
            quantile_trees_upper: Vec::new(),
            is_fitted: false,
        }
    }

    /// Create with full configuration
    pub fn with_config(config: GradientBoostingConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            initial_prediction: 0.0,
            quantile_trees_lower: Vec::new(),
            quantile_trees_upper: Vec::new(),
            is_fitted: false,
        }
    }

    /// Predict confidence interval
    pub fn predict_interval(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        if !self.is_fitted {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        let median = self.predict(x)?;

        if self.config.use_quantile && !self.quantile_trees_lower.is_empty() {
            let lower = self.predict_with_trees(x, &self.quantile_trees_lower);
            let upper = self.predict_with_trees(x, &self.quantile_trees_upper);
            Ok((median, lower, upper))
        } else {
            // Fallback: estimate interval from staged predictions
            let (_, uncertainty) = self.predict_with_uncertainty(x)?;
            let lower = &median - &uncertainty * 1.28; // ~80% CI
            let upper = &median + &uncertainty * 1.28;
            Ok((median, lower, upper))
        }
    }

    fn predict_with_trees(&self, x: &Array2<f64>, trees: &[DecisionTree]) -> Array1<f64> {
        let n_samples = x.nrows();
        let mut predictions = Array1::from_elem(n_samples, self.initial_prediction);

        for tree in trees {
            let tree_preds = tree.predict(x);
            predictions = predictions + &tree_preds * self.config.learning_rate;
        }

        predictions
    }

    /// Fit quantile trees for uncertainty estimation
    fn fit_quantile_trees(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.ncols();
        let max_features = (n_features as f64).sqrt().ceil() as usize;

        // Fit lower quantile trees
        info!("Fitting lower quantile trees (q={})...", self.config.lower_quantile);
        self.quantile_trees_lower = self.fit_quantile_boosting(
            x, y,
            self.config.lower_quantile,
            max_features,
        )?;

        // Fit upper quantile trees
        info!("Fitting upper quantile trees (q={})...", self.config.upper_quantile);
        self.quantile_trees_upper = self.fit_quantile_boosting(
            x, y,
            self.config.upper_quantile,
            max_features,
        )?;

        Ok(())
    }

    fn fit_quantile_boosting(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        quantile: f64,
        max_features: usize,
    ) -> Result<Vec<DecisionTree>> {
        let n_samples = x.nrows();
        let mut trees = Vec::with_capacity(self.config.n_estimators);

        // Initial prediction (quantile of y)
        let mut sorted_y: Vec<f64> = y.to_vec();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let initial_pred = sorted_y[(quantile * (n_samples as f64 - 1.0)) as usize];

        let mut predictions = Array1::from_elem(n_samples, initial_pred);

        for tree_idx in 0..self.config.n_estimators {
            // Compute pseudo-residuals for quantile loss
            let residuals: Array1<f64> = y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| {
                    if yi > pi {
                        quantile
                    } else {
                        quantile - 1.0
                    }
                })
                .collect();

            // Fit tree to residuals
            let mut tree = DecisionTree::new(
                self.config.max_depth,
                self.config.min_samples_split,
                self.config.min_samples_leaf,
                max_features,
                self.config.random_state.map(|s| s + tree_idx as u64 + 1000),
            );
            tree.fit(x, &residuals)?;

            // Update predictions
            let tree_preds = tree.predict(x);
            predictions = predictions + &tree_preds * self.config.learning_rate;

            trees.push(tree);
        }

        Ok(trees)
    }
}

impl EnsembleModel for GradientBoostingEnsemble {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let max_features = (n_features as f64).sqrt().ceil() as usize;

        info!(
            "Fitting Gradient Boosting with {} trees, lr={}, max_depth={:?}",
            self.config.n_estimators,
            self.config.learning_rate,
            self.config.max_depth
        );

        // Initial prediction is the mean
        self.initial_prediction = y.mean().unwrap_or(0.0);
        let mut predictions = Array1::from_elem(n_samples, self.initial_prediction);

        self.trees.clear();

        for tree_idx in 0..self.config.n_estimators {
            // Compute residuals (negative gradient for MSE loss)
            let residuals = y - &predictions;

            // Fit tree to residuals
            let mut tree = DecisionTree::new(
                self.config.max_depth,
                self.config.min_samples_split,
                self.config.min_samples_leaf,
                max_features,
                self.config.random_state.map(|s| s + tree_idx as u64),
            );
            tree.fit(x, &residuals)?;

            // Update predictions
            let tree_preds = tree.predict(x);
            predictions = predictions + &tree_preds * self.config.learning_rate;

            self.trees.push(tree);
        }

        // Fit quantile trees if enabled
        if self.config.use_quantile {
            self.fit_quantile_trees(x, y)?;
        }

        self.is_fitted = true;
        info!("Gradient Boosting training complete");

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        Ok(self.predict_with_trees(x, &self.trees))
    }

    fn predict_with_uncertainty(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        if !self.is_fitted {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        let predictions = self.predict(x)?;

        let uncertainty = if self.config.use_quantile && !self.quantile_trees_lower.is_empty() {
            let lower = self.predict_with_trees(x, &self.quantile_trees_lower);
            let upper = self.predict_with_trees(x, &self.quantile_trees_upper);
            // Estimate std from IQR (80% CI -> divide by 2.56 for std)
            (&upper - &lower) / 2.56
        } else {
            // Estimate from staged predictions variance
            let n_samples = x.nrows();
            let n_trees = self.trees.len();
            let last_n = (n_trees / 10).max(5).min(n_trees);

            let mut staged_preds = Vec::new();
            let mut current_pred = Array1::from_elem(n_samples, self.initial_prediction);

            for (i, tree) in self.trees.iter().enumerate() {
                let tree_pred = tree.predict(x);
                current_pred = current_pred + &tree_pred * self.config.learning_rate;

                if i >= n_trees - last_n {
                    staged_preds.push(current_pred.clone());
                }
            }

            // Compute std of last N staged predictions
            let mut variance = Array1::zeros(n_samples);
            let mean = &predictions;

            for staged in &staged_preds {
                let diff = staged - mean;
                variance = variance + &diff.mapv(|x| x.powi(2));
            }
            variance /= staged_preds.len() as f64;
            variance.mapv(|v| v.sqrt())
        };

        Ok((predictions, uncertainty))
    }

    fn n_estimators(&self) -> usize {
        self.config.n_estimators
    }

    fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_gradient_boosting() {
        let n_samples = 100;
        let n_features = 5;

        let x = Array2::random((n_samples, n_features), Uniform::new(-1.0, 1.0));
        let y: Array1<f64> = x.column(0).to_owned() * 2.0
            + x.column(1).to_owned()
            + Array1::random(n_samples, Uniform::new(-0.1, 0.1));

        let mut gb = GradientBoostingEnsemble::new(20, true);
        gb.fit(&x, &y).unwrap();

        assert!(gb.is_fitted());

        let predictions = gb.predict(&x).unwrap();
        assert_eq!(predictions.len(), n_samples);

        let (preds, uncertainties) = gb.predict_with_uncertainty(&x).unwrap();
        assert_eq!(preds.len(), n_samples);
        assert_eq!(uncertainties.len(), n_samples);
    }
}
