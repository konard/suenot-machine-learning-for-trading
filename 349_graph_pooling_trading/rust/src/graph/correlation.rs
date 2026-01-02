//! Correlation computation for building market graphs

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Method for computing correlations
#[derive(Debug, Clone, Copy)]
pub enum CorrelationMethod {
    /// Pearson correlation coefficient
    Pearson,
    /// Spearman rank correlation
    Spearman,
    /// Kendall tau correlation
    Kendall,
}

/// Correlation-based graph construction
pub struct CorrelationGraph {
    /// Correlation matrix
    pub correlation_matrix: Array2<f64>,

    /// Adjacency matrix (thresholded correlations)
    pub adjacency_matrix: Array2<f64>,

    /// Asset symbols
    pub symbols: Vec<String>,

    /// Correlation threshold used
    pub threshold: f64,
}

impl CorrelationGraph {
    /// Build a correlation graph from return series
    ///
    /// # Arguments
    /// * `returns` - HashMap of symbol -> return series
    /// * `threshold` - Minimum absolute correlation to include edge
    /// * `method` - Correlation method to use
    pub fn from_returns(
        returns: &HashMap<String, Vec<f64>>,
        threshold: f64,
        method: CorrelationMethod,
    ) -> Self {
        let symbols: Vec<String> = returns.keys().cloned().collect();
        let n = symbols.len();

        // Build return matrix [time, assets]
        let min_len = returns.values().map(|v| v.len()).min().unwrap_or(0);

        let mut return_matrix = Array2::<f64>::zeros((min_len, n));
        for (i, symbol) in symbols.iter().enumerate() {
            if let Some(ret) = returns.get(symbol) {
                for (t, &r) in ret.iter().take(min_len).enumerate() {
                    return_matrix[[t, i]] = r;
                }
            }
        }

        // Compute correlation matrix
        let correlation_matrix = match method {
            CorrelationMethod::Pearson => Self::pearson_correlation(&return_matrix),
            CorrelationMethod::Spearman => Self::spearman_correlation(&return_matrix),
            CorrelationMethod::Kendall => Self::kendall_correlation(&return_matrix),
        };

        // Build adjacency matrix (threshold + remove self-loops)
        let mut adjacency_matrix = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j && correlation_matrix[[i, j]].abs() >= threshold {
                    adjacency_matrix[[i, j]] = correlation_matrix[[i, j]].abs();
                }
            }
        }

        Self {
            correlation_matrix,
            adjacency_matrix,
            symbols,
            threshold,
        }
    }

    /// Compute Pearson correlation matrix
    fn pearson_correlation(data: &Array2<f64>) -> Array2<f64> {
        let n_assets = data.ncols();
        let n_time = data.nrows();

        // Compute means
        let means: Vec<f64> = (0..n_assets)
            .map(|i| data.column(i).sum() / n_time as f64)
            .collect();

        // Compute standard deviations
        let stds: Vec<f64> = (0..n_assets)
            .map(|i| {
                let col = data.column(i);
                let mean = means[i];
                let variance: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_time as f64;
                variance.sqrt()
            })
            .collect();

        // Compute correlation matrix
        let mut corr = Array2::<f64>::zeros((n_assets, n_assets));

        for i in 0..n_assets {
            for j in i..n_assets {
                if i == j {
                    corr[[i, j]] = 1.0;
                } else {
                    let cov: f64 = (0..n_time)
                        .map(|t| (data[[t, i]] - means[i]) * (data[[t, j]] - means[j]))
                        .sum::<f64>()
                        / n_time as f64;

                    let r = if stds[i] > 0.0 && stds[j] > 0.0 {
                        cov / (stds[i] * stds[j])
                    } else {
                        0.0
                    };

                    corr[[i, j]] = r;
                    corr[[j, i]] = r;
                }
            }
        }

        corr
    }

    /// Compute Spearman rank correlation matrix
    fn spearman_correlation(data: &Array2<f64>) -> Array2<f64> {
        let n_assets = data.ncols();
        let n_time = data.nrows();

        // Convert to ranks
        let mut ranked_data = Array2::<f64>::zeros((n_time, n_assets));

        for j in 0..n_assets {
            let col: Vec<f64> = data.column(j).to_vec();
            let ranks = Self::rank_data(&col);
            for (t, &r) in ranks.iter().enumerate() {
                ranked_data[[t, j]] = r;
            }
        }

        // Apply Pearson to ranks
        Self::pearson_correlation(&ranked_data)
    }

    /// Compute Kendall tau correlation matrix (simplified version)
    fn kendall_correlation(data: &Array2<f64>) -> Array2<f64> {
        let n_assets = data.ncols();
        let n_time = data.nrows();

        let mut corr = Array2::<f64>::zeros((n_assets, n_assets));

        for i in 0..n_assets {
            for j in i..n_assets {
                if i == j {
                    corr[[i, j]] = 1.0;
                } else {
                    let mut concordant = 0i64;
                    let mut discordant = 0i64;

                    for t1 in 0..n_time {
                        for t2 in (t1 + 1)..n_time {
                            let sign_i = (data[[t2, i]] - data[[t1, i]]).signum();
                            let sign_j = (data[[t2, j]] - data[[t1, j]]).signum();

                            if sign_i * sign_j > 0.0 {
                                concordant += 1;
                            } else if sign_i * sign_j < 0.0 {
                                discordant += 1;
                            }
                        }
                    }

                    let total = concordant + discordant;
                    let tau = if total > 0 {
                        (concordant - discordant) as f64 / total as f64
                    } else {
                        0.0
                    };

                    corr[[i, j]] = tau;
                    corr[[j, i]] = tau;
                }
            }
        }

        corr
    }

    /// Convert data to ranks
    fn rank_data(data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranks = vec![0.0; n];
        for (rank, (idx, _)) in indexed.into_iter().enumerate() {
            ranks[idx] = rank as f64 + 1.0;
        }

        ranks
    }

    /// Get the degree of each node (sum of edge weights)
    pub fn degrees(&self) -> Array1<f64> {
        self.adjacency_matrix.sum_axis(Axis(1))
    }

    /// Get the normalized Laplacian matrix
    pub fn normalized_laplacian(&self) -> Array2<f64> {
        let n = self.adjacency_matrix.nrows();
        let degrees = self.degrees();

        let mut laplacian = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    laplacian[[i, j]] = 1.0;
                } else if self.adjacency_matrix[[i, j]] > 0.0 {
                    let di = degrees[i].sqrt();
                    let dj = degrees[j].sqrt();
                    if di > 0.0 && dj > 0.0 {
                        laplacian[[i, j]] = -self.adjacency_matrix[[i, j]] / (di * dj);
                    }
                }
            }
        }

        laplacian
    }

    /// Get graph density (ratio of edges to possible edges)
    pub fn density(&self) -> f64 {
        let n = self.adjacency_matrix.nrows();
        if n < 2 {
            return 0.0;
        }

        let n_edges: usize = self
            .adjacency_matrix
            .iter()
            .filter(|&&x| x > 0.0)
            .count();

        n_edges as f64 / (n * (n - 1)) as f64
    }

    /// Get clustering coefficient for a node
    pub fn clustering_coefficient(&self, node: usize) -> f64 {
        let n = self.adjacency_matrix.nrows();
        if node >= n {
            return 0.0;
        }

        // Get neighbors
        let neighbors: Vec<usize> = (0..n)
            .filter(|&j| self.adjacency_matrix[[node, j]] > 0.0)
            .collect();

        let k = neighbors.len();
        if k < 2 {
            return 0.0;
        }

        // Count edges between neighbors
        let mut triangles = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                if self.adjacency_matrix[[neighbors[i], neighbors[j]]] > 0.0 {
                    triangles += 1;
                }
            }
        }

        2.0 * triangles as f64 / (k * (k - 1)) as f64
    }

    /// Get average clustering coefficient
    pub fn avg_clustering_coefficient(&self) -> f64 {
        let n = self.adjacency_matrix.nrows();
        if n == 0 {
            return 0.0;
        }

        let sum: f64 = (0..n).map(|i| self.clustering_coefficient(i)).sum();
        sum / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        let mut returns = HashMap::new();
        returns.insert("A".to_string(), vec![0.01, 0.02, -0.01, 0.03, -0.02]);
        returns.insert("B".to_string(), vec![0.01, 0.02, -0.01, 0.03, -0.02]); // Same as A
        returns.insert("C".to_string(), vec![-0.01, -0.02, 0.01, -0.03, 0.02]); // Opposite of A

        let graph = CorrelationGraph::from_returns(&returns, 0.5, CorrelationMethod::Pearson);

        // A and B should have correlation 1.0
        // A and C should have correlation -1.0
        assert!(graph.correlation_matrix[[0, 0]] > 0.99);
    }

    #[test]
    fn test_graph_density() {
        let mut returns = HashMap::new();
        returns.insert("A".to_string(), vec![0.01, 0.02, -0.01]);
        returns.insert("B".to_string(), vec![0.01, 0.02, -0.01]);

        let graph = CorrelationGraph::from_returns(&returns, 0.0, CorrelationMethod::Pearson);
        assert!(graph.density() > 0.0);
    }
}
