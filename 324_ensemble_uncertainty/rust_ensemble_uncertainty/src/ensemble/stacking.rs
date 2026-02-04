//! Stacking ensemble implementation.

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use tracing::info;

use super::random_forest::RandomForestEnsemble;
use super::gradient_boosting::GradientBoostingEnsemble;
use super::traits::EnsembleModel;

/// Stacking ensemble that combines multiple base models
pub struct StackingEnsemble {
    rf_model: RandomForestEnsemble,
    gb_model: GradientBoostingEnsemble,
    meta_model: RandomForestEnsemble,
    is_fitted: bool,
}

impl StackingEnsemble {
    /// Create a new stacking ensemble
    pub fn new(n_estimators: usize) -> Self {
        Self {
            rf_model: RandomForestEnsemble::new(n_estimators / 2, Some(10)),
            gb_model: GradientBoostingEnsemble::new(n_estimators / 2, false),
            meta_model: RandomForestEnsemble::new(n_estimators / 4, Some(5)),
            is_fitted: false,
        }
    }

    /// Get predictions from base models
    fn get_base_predictions(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let rf_preds = self.rf_model.predict(x)?;
        let gb_preds = self.gb_model.predict(x)?;

        // Stack predictions
        let n_samples = x.nrows();
        let mut meta_features = Array2::zeros((n_samples, 2));
        meta_features.column_mut(0).assign(&rf_preds);
        meta_features.column_mut(1).assign(&gb_preds);

        Ok(meta_features)
    }

    /// Get uncertainty from base model disagreement
    fn get_base_uncertainty(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let rf_preds = self.rf_model.predict(x)?;
        let gb_preds = self.gb_model.predict(x)?;

        // Uncertainty from disagreement
        let disagreement = (&rf_preds - &gb_preds).mapv(|d| d.abs());

        // Also get individual uncertainties
        let (_, rf_unc) = self.rf_model.predict_with_uncertainty(x)?;
        let (_, gb_unc) = self.gb_model.predict_with_uncertainty(x)?;

        // Combine uncertainties
        let avg_model_unc = (&rf_unc + &gb_unc) / 2.0;
        let total_unc = (disagreement.mapv(|d| d.powi(2)) + avg_model_unc.mapv(|u| u.powi(2)))
            .mapv(|v| v.sqrt());

        Ok(total_unc)
    }
}

impl EnsembleModel for StackingEnsemble {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        info!("Fitting Stacking ensemble...");

        // Fit base models
        info!("Training Random Forest base model...");
        self.rf_model.fit(x, y)?;

        info!("Training Gradient Boosting base model...");
        self.gb_model.fit(x, y)?;

        // Get base predictions for meta-learner
        let meta_features = self.get_base_predictions(x)?;

        // Fit meta-learner
        info!("Training meta-learner...");
        self.meta_model.fit(&meta_features, y)?;

        self.is_fitted = true;
        info!("Stacking ensemble training complete");

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        let meta_features = self.get_base_predictions(x)?;
        self.meta_model.predict(&meta_features)
    }

    fn predict_with_uncertainty(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        if !self.is_fitted {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        let meta_features = self.get_base_predictions(x)?;

        // Meta-model prediction and uncertainty
        let (meta_pred, meta_unc) = self.meta_model.predict_with_uncertainty(&meta_features)?;

        // Base model disagreement uncertainty
        let base_unc = self.get_base_uncertainty(x)?;

        // Combined uncertainty
        let total_unc = (meta_unc.mapv(|u| u.powi(2)) + base_unc.mapv(|u| u.powi(2)))
            .mapv(|v| v.sqrt());

        Ok((meta_pred, total_unc))
    }

    fn n_estimators(&self) -> usize {
        self.rf_model.n_estimators() + self.gb_model.n_estimators()
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
    fn test_stacking_ensemble() {
        let n_samples = 100;
        let n_features = 5;

        let x = Array2::random((n_samples, n_features), Uniform::new(-1.0, 1.0));
        let y: Array1<f64> = x.column(0).to_owned() * 2.0
            + x.column(1).to_owned()
            + Array1::random(n_samples, Uniform::new(-0.1, 0.1));

        let mut stack = StackingEnsemble::new(20);
        stack.fit(&x, &y).unwrap();

        assert!(stack.is_fitted());

        let predictions = stack.predict(&x).unwrap();
        assert_eq!(predictions.len(), n_samples);

        let (preds, uncertainties) = stack.predict_with_uncertainty(&x).unwrap();
        assert_eq!(preds.len(), n_samples);
        assert_eq!(uncertainties.len(), n_samples);
    }
}
