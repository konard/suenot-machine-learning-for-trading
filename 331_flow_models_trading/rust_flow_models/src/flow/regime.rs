//! Market regime detection using latent space clustering

use crate::flow::model::NormalizingFlow;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Simple K-means clustering for regime detection
#[derive(Debug, Clone)]
pub struct KMeans {
    pub k: usize,
    pub centers: Option<Array2<f64>>,
    pub max_iters: usize,
}

impl KMeans {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            centers: None,
            max_iters: 100,
        }
    }

    /// Fit K-means on data
    pub fn fit(&mut self, data: &Array2<f64>) {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Initialize centers randomly from data points
        let mut rng = rand::thread_rng();
        use rand::seq::SliceRandom;
        let indices: Vec<usize> = (0..n_samples).collect();
        let selected: Vec<usize> = indices
            .choose_multiple(&mut rng, self.k)
            .cloned()
            .collect();

        let mut centers = Array2::zeros((self.k, n_features));
        for (i, &idx) in selected.iter().enumerate() {
            centers.row_mut(i).assign(&data.row(idx));
        }

        // Iterate
        for _ in 0..self.max_iters {
            // Assign points to nearest center
            let assignments = self.assign(data, &centers);

            // Update centers
            let mut new_centers = Array2::zeros((self.k, n_features));
            let mut counts = vec![0usize; self.k];

            for (i, &cluster) in assignments.iter().enumerate() {
                new_centers.row_mut(cluster).scaled_add(1.0, &data.row(i));
                counts[cluster] += 1;
            }

            for (k, count) in counts.iter().enumerate() {
                if *count > 0 {
                    new_centers.row_mut(k).mapv_inplace(|v| v / *count as f64);
                }
            }

            // Check convergence
            let diff: f64 = (&new_centers - &centers).mapv(|v| v * v).sum();
            centers = new_centers;

            if diff < 1e-6 {
                break;
            }
        }

        self.centers = Some(centers);
    }

    /// Assign points to nearest cluster
    fn assign(&self, data: &Array2<f64>, centers: &Array2<f64>) -> Vec<usize> {
        data.rows()
            .into_iter()
            .map(|row| {
                let distances: Vec<f64> = centers
                    .rows()
                    .into_iter()
                    .map(|center| {
                        (&row - &center).mapv(|v| v * v).sum()
                    })
                    .collect();

                distances
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict cluster for new data
    pub fn predict(&self, data: &Array2<f64>) -> Vec<usize> {
        match &self.centers {
            Some(centers) => self.assign(data, centers),
            None => vec![0; data.nrows()],
        }
    }

    /// Get distances to all cluster centers
    pub fn distances(&self, data: &Array2<f64>) -> Array2<f64> {
        match &self.centers {
            Some(centers) => {
                let n_samples = data.nrows();
                let mut distances = Array2::zeros((n_samples, self.k));

                for (i, row) in data.rows().into_iter().enumerate() {
                    for (j, center) in centers.rows().into_iter().enumerate() {
                        distances[[i, j]] = (&row - &center).mapv(|v| v * v).sum().sqrt();
                    }
                }

                distances
            }
            None => Array2::zeros((data.nrows(), self.k)),
        }
    }
}

/// Market regime detector using flow latent space
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    pub n_regimes: usize,
    pub kmeans: KMeans,
    pub regime_labels: Vec<String>,
    pub regime_stats: HashMap<usize, RegimeStats>,
}

/// Statistics for a regime
#[derive(Debug, Clone)]
pub struct RegimeStats {
    pub mean_return: f64,
    pub volatility: f64,
    pub sharpe: f64,
    pub count: usize,
}

impl RegimeDetector {
    /// Create new regime detector
    pub fn new(n_regimes: usize) -> Self {
        Self {
            n_regimes,
            kmeans: KMeans::new(n_regimes),
            regime_labels: (0..n_regimes).map(|i| format!("Regime_{}", i)).collect(),
            regime_stats: HashMap::new(),
        }
    }

    /// Fit regimes on latent representations
    pub fn fit(&mut self, model: &mut NormalizingFlow, data: &Array2<f64>) {
        // Transform to latent space
        let (z, _) = model.forward(data);

        // Fit K-means clustering
        self.kmeans.fit(&z);

        log::info!("Fitted {} regimes on {} samples", self.n_regimes, data.nrows());
    }

    /// Fit regimes with return data for labeling
    pub fn fit_with_returns(
        &mut self,
        model: &mut NormalizingFlow,
        data: &Array2<f64>,
        returns: &Array1<f64>,
    ) {
        // First fit the clusters
        self.fit(model, data);

        // Transform data and get assignments
        let (z, _) = model.forward(data);
        let assignments = self.kmeans.predict(&z);

        // Compute statistics per regime
        for regime in 0..self.n_regimes {
            let regime_returns: Vec<f64> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &r)| r == regime)
                .map(|(i, _)| returns[i])
                .collect();

            if !regime_returns.is_empty() {
                let n = regime_returns.len();
                let mean: f64 = regime_returns.iter().sum::<f64>() / n as f64;
                let variance: f64 = regime_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / n as f64;
                let std = variance.sqrt();
                let sharpe = if std > 0.0 { mean / std } else { 0.0 };

                self.regime_stats.insert(
                    regime,
                    RegimeStats {
                        mean_return: mean,
                        volatility: std,
                        sharpe,
                        count: n,
                    },
                );
            }
        }

        // Label regimes based on characteristics
        self.label_regimes();
    }

    /// Automatically label regimes based on statistics
    fn label_regimes(&mut self) {
        let mut regime_info: Vec<(usize, f64, f64)> = self
            .regime_stats
            .iter()
            .map(|(&regime, stats)| (regime, stats.volatility, stats.mean_return))
            .collect();

        // Sort by volatility then by return
        regime_info.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Assign labels
        let median_vol = if !regime_info.is_empty() {
            let vols: Vec<f64> = regime_info.iter().map(|(_, v, _)| *v).collect();
            let mid = vols.len() / 2;
            vols[mid]
        } else {
            0.0
        };

        for (regime, vol, ret) in &regime_info {
            let vol_label = if *vol > median_vol { "High Vol" } else { "Low Vol" };
            let trend_label = if *ret > 0.0 { "Bull" } else { "Bear" };
            self.regime_labels[*regime] = format!("{} {}", vol_label, trend_label);
        }
    }

    /// Detect current regime
    pub fn detect(&self, model: &mut NormalizingFlow, data: &Array2<f64>) -> (Vec<usize>, Array2<f64>) {
        let (z, _) = model.forward(data);
        let regimes = self.kmeans.predict(&z);
        let distances = self.kmeans.distances(&z);
        (regimes, distances)
    }

    /// Get regime label for a sample
    pub fn get_label(&self, regime: usize) -> &str {
        self.regime_labels.get(regime).map(|s| s.as_str()).unwrap_or("Unknown")
    }

    /// Get regime statistics
    pub fn get_stats(&self, regime: usize) -> Option<&RegimeStats> {
        self.regime_stats.get(&regime)
    }

    /// Compute regime confidence from distances
    pub fn confidence(&self, distances: &Array1<f64>) -> f64 {
        let min_dist = distances.iter().cloned().fold(f64::INFINITY, f64::min);
        1.0 / (1.0 + min_dist)
    }
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::config::FlowConfig;

    #[test]
    fn test_kmeans() {
        let mut kmeans = KMeans::new(3);
        let data = Array2::from_shape_fn((100, 4), |(i, j)| {
            ((i % 3) as f64) + (j as f64) * 0.1
        });

        kmeans.fit(&data);
        assert!(kmeans.centers.is_some());

        let predictions = kmeans.predict(&data);
        assert_eq!(predictions.len(), 100);
    }

    #[test]
    fn test_regime_detector() {
        let config = FlowConfig::default().with_input_dim(4).with_num_layers(2);
        let mut model = NormalizingFlow::new(config);

        let data = Array2::from_shape_fn((100, 4), |(i, j)| {
            ((i + j) as f64) * 0.01
        });

        let mut detector = RegimeDetector::new(4);
        detector.fit(&mut model, &data);

        let (regimes, distances) = detector.detect(&mut model, &data);
        assert_eq!(regimes.len(), 100);
        assert_eq!(distances.shape(), &[100, 4]);
    }
}
