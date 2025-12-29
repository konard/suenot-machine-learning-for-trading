//! PCA Analysis implementation

use super::decomposition::{covariance_matrix, EigenDecomposition};
use crate::data::Returns;
use ndarray::{s, Array1, Array2, Axis};

/// PCA Analysis results
#[derive(Debug, Clone)]
pub struct PCAAnalysis {
    /// Number of components retained
    pub n_components: usize,
    /// Principal components (eigenvectors as columns)
    pub components: Array2<f64>,
    /// Explained variance for each component
    pub explained_variance: Array1<f64>,
    /// Explained variance ratio (percentage)
    pub explained_variance_ratio: Array1<f64>,
    /// Cumulative explained variance ratio
    pub cumulative_variance_ratio: Array1<f64>,
    /// Mean of original data (for centering)
    pub mean: Array1<f64>,
    /// Original feature names (if available)
    pub feature_names: Vec<String>,
}

impl PCAAnalysis {
    /// Fit PCA on returns data
    pub fn fit(returns: &Returns, n_components: Option<usize>) -> Self {
        let data = &returns.returns;
        let feature_names = returns.symbols.clone();

        Self::fit_matrix(data, n_components, feature_names)
    }

    /// Fit PCA on a raw matrix
    pub fn fit_matrix(
        data: &Array2<f64>,
        n_components: Option<usize>,
        feature_names: Vec<String>,
    ) -> Self {
        let (n_samples, n_features) = data.dim();
        let n_components = n_components.unwrap_or(n_features).min(n_features).min(n_samples);

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean;

        // Compute covariance matrix
        let cov = covariance_matrix(&centered);

        // Eigendecomposition
        let eigen = EigenDecomposition::from_symmetric(&cov);

        // Take only n_components
        let components = eigen
            .eigenvectors
            .slice(s![.., ..n_components])
            .to_owned();
        let explained_variance = eigen.eigenvalues.slice(s![..n_components]).to_owned();

        // Calculate variance ratios
        let total_variance = eigen.eigenvalues.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            &explained_variance / total_variance
        } else {
            Array1::zeros(n_components)
        };

        // Cumulative variance ratio
        let mut cumulative = Array1::zeros(n_components);
        let mut cum_sum = 0.0;
        for i in 0..n_components {
            cum_sum += explained_variance_ratio[i];
            cumulative[i] = cum_sum;
        }

        Self {
            n_components,
            components,
            explained_variance,
            explained_variance_ratio,
            cumulative_variance_ratio: cumulative,
            mean,
            feature_names,
        }
    }

    /// Fit PCA and automatically select components to explain target variance
    pub fn fit_with_variance_threshold(
        returns: &Returns,
        variance_threshold: f64,
    ) -> Self {
        let full_pca = Self::fit(returns, None);

        // Find number of components needed
        let n_components = full_pca
            .cumulative_variance_ratio
            .iter()
            .position(|&v| v >= variance_threshold)
            .map(|i| i + 1)
            .unwrap_or(full_pca.n_components);

        Self::fit(returns, Some(n_components))
    }

    /// Transform data into principal component space
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let centered = data - &self.mean;
        centered.dot(&self.components)
    }

    /// Transform returns data
    pub fn transform_returns(&self, returns: &Returns) -> Array2<f64> {
        self.transform(&returns.returns)
    }

    /// Inverse transform from PC space back to original space
    pub fn inverse_transform(&self, transformed: &Array2<f64>) -> Array2<f64> {
        transformed.dot(&self.components.t()) + &self.mean
    }

    /// Calculate reconstruction error
    pub fn reconstruction_error(&self, data: &Array2<f64>) -> f64 {
        let transformed = self.transform(data);
        let reconstructed = self.inverse_transform(&transformed);

        let diff = data - &reconstructed;
        let mse: f64 = diff.iter().map(|x| x * x).sum::<f64>() / diff.len() as f64;
        mse.sqrt()
    }

    /// Get loadings matrix (correlation between original features and PCs)
    pub fn loadings(&self) -> Array2<f64> {
        let n_features = self.components.nrows();
        let mut loadings = Array2::zeros((n_features, self.n_components));

        for j in 0..self.n_components {
            let std_pc = self.explained_variance[j].sqrt();
            for i in 0..n_features {
                loadings[[i, j]] = self.components[[i, j]] * std_pc;
            }
        }

        loadings
    }

    /// Get feature contributions to each component
    pub fn feature_contributions(&self) -> Vec<Vec<(String, f64)>> {
        let mut contributions = Vec::new();

        for pc_idx in 0..self.n_components {
            let mut pc_contributions: Vec<(String, f64)> = self
                .feature_names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), self.components[[i, pc_idx]]))
                .collect();

            // Sort by absolute contribution
            pc_contributions.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            contributions.push(pc_contributions);
        }

        contributions
    }

    /// Print summary of PCA results
    pub fn summary(&self) {
        println!("\n=== PCA Summary ===");
        println!("Number of components: {}", self.n_components);
        println!("Number of features: {}", self.feature_names.len());
        println!();

        println!("Explained Variance Ratio:");
        println!("{:-<50}", "");
        println!("{:>5} {:>12} {:>12} {:>12}", "PC", "Variance", "Ratio", "Cumulative");
        println!("{:-<50}", "");

        for i in 0..self.n_components.min(10) {
            println!(
                "{:>5} {:>12.6} {:>11.2}% {:>11.2}%",
                i + 1,
                self.explained_variance[i],
                self.explained_variance_ratio[i] * 100.0,
                self.cumulative_variance_ratio[i] * 100.0
            );
        }

        if self.n_components > 10 {
            println!("... and {} more components", self.n_components - 10);
        }

        println!();
        println!("Top feature contributions to PC1:");
        if let Some(contributions) = self.feature_contributions().first() {
            for (name, weight) in contributions.iter().take(5) {
                println!("  {:>10}: {:>8.4}", name, weight);
            }
        }
    }

    /// Find optimal number of components using elbow method
    pub fn find_elbow(&self) -> usize {
        if self.n_components <= 2 {
            return self.n_components;
        }

        // Calculate second derivative of cumulative variance
        let mut max_curvature = 0.0;
        let mut elbow_idx = 1;

        for i in 1..(self.n_components - 1) {
            let prev = self.cumulative_variance_ratio[i - 1];
            let curr = self.cumulative_variance_ratio[i];
            let next = self.cumulative_variance_ratio[i + 1];

            // Second derivative approximation
            let curvature = (prev + next - 2.0 * curr).abs();

            if curvature > max_curvature {
                max_curvature = curvature;
                elbow_idx = i + 1;
            }
        }

        elbow_idx
    }
}

/// Calculate principal component scores for time series
pub fn calculate_pc_scores(pca: &PCAAnalysis, returns: &Returns) -> Array2<f64> {
    pca.transform_returns(returns)
}

/// Risk factor analysis using PCA
#[derive(Debug, Clone)]
pub struct RiskFactorAnalysis {
    /// PCA results
    pub pca: PCAAnalysis,
    /// Factor returns (PC scores over time)
    pub factor_returns: Array2<f64>,
    /// Factor loadings (how each asset loads on factors)
    pub factor_loadings: Array2<f64>,
    /// Timestamps
    pub timestamps: Vec<i64>,
}

impl RiskFactorAnalysis {
    /// Create risk factor analysis from returns
    pub fn from_returns(returns: &Returns, n_factors: Option<usize>) -> Self {
        let pca = PCAAnalysis::fit(returns, n_factors);
        let factor_returns = pca.transform_returns(returns);
        let factor_loadings = pca.components.clone();

        Self {
            pca,
            factor_returns,
            factor_loadings,
            timestamps: returns.timestamps.clone(),
        }
    }

    /// Get factor return time series
    pub fn get_factor_returns(&self, factor_idx: usize) -> Option<Array1<f64>> {
        if factor_idx >= self.pca.n_components {
            return None;
        }
        Some(self.factor_returns.column(factor_idx).to_owned())
    }

    /// Calculate factor exposures for a given asset
    pub fn get_asset_exposures(&self, asset_idx: usize) -> Option<Array1<f64>> {
        if asset_idx >= self.pca.feature_names.len() {
            return None;
        }
        Some(self.factor_loadings.row(asset_idx).to_owned())
    }

    /// Calculate correlation between factors (should be ~0)
    pub fn factor_correlations(&self) -> Array2<f64> {
        let n = self.pca.n_components;
        let mut corr = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let fi = self.factor_returns.column(i);
                let fj = self.factor_returns.column(j);

                let mean_i = fi.mean().unwrap_or(0.0);
                let mean_j = fj.mean().unwrap_or(0.0);

                let mut cov = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;

                for k in 0..fi.len() {
                    let di = fi[k] - mean_i;
                    let dj = fj[k] - mean_j;
                    cov += di * dj;
                    var_i += di * di;
                    var_j += dj * dj;
                }

                if var_i > 0.0 && var_j > 0.0 {
                    corr[[i, j]] = cov / (var_i.sqrt() * var_j.sqrt());
                } else if i == j {
                    corr[[i, j]] = 1.0;
                }
            }
        }

        corr
    }

    /// Print summary
    pub fn summary(&self) {
        println!("\n=== Risk Factor Analysis ===");
        self.pca.summary();

        println!("\nFactor Correlations (should be ~0 for off-diagonal):");
        let corr = self.factor_correlations();
        for i in 0..corr.nrows().min(5) {
            for j in 0..corr.ncols().min(5) {
                print!("{:>8.4} ", corr[[i, j]]);
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pca_fit() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        let pca = PCAAnalysis::fit_matrix(
            &data,
            Some(2),
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        );

        assert_eq!(pca.n_components, 2);
        assert_eq!(pca.components.shape(), &[3, 2]);

        // Variance ratios should sum to <= 1
        assert!(pca.explained_variance_ratio.sum() <= 1.0 + 1e-10);

        // Cumulative should be monotonically increasing
        assert!(pca.cumulative_variance_ratio[1] >= pca.cumulative_variance_ratio[0]);
    }

    #[test]
    fn test_transform_inverse_transform() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let pca = PCAAnalysis::fit_matrix(&data, None, vec!["A".to_string(), "B".to_string()]);

        let transformed = pca.transform(&data);
        let reconstructed = pca.inverse_transform(&transformed);

        // Reconstruction should be very close to original
        let error: f64 = (&data - &reconstructed).iter().map(|x| x.abs()).sum();
        assert!(error < 1e-10);
    }
}
