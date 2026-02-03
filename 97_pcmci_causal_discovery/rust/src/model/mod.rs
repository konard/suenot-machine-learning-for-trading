//! PCMCI Causal Discovery Model
//!
//! Implements the PCMCI algorithm for causal discovery in time series data.
//! The algorithm has two phases:
//! 1. PC-stable phase: iterative condition selection via conditional independence tests
//! 2. MCI phase: momentary conditional independence test to validate causal links

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Independence test type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum IndependenceTest {
    /// Partial correlation test (linear dependence)
    ParCorr,
    /// Conditional mutual information (nonlinear dependence)
    CMI,
}

/// Configuration for the PCMCI algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCMCIConfig {
    /// Maximum time lag to consider
    pub max_lag: usize,
    /// Significance level (p-value threshold)
    pub significance_level: f64,
    /// Type of independence test
    pub test_type: IndependenceTest,
    /// Maximum conditioning set dimension (None = unlimited)
    pub max_conds_dim: Option<usize>,
}

impl Default for PCMCIConfig {
    fn default() -> Self {
        Self {
            max_lag: 3,
            significance_level: 0.05,
            test_type: IndependenceTest::ParCorr,
            max_conds_dim: None,
        }
    }
}

/// A discovered causal link between two variables at a specific lag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    /// Source variable index
    pub source: usize,
    /// Target variable index
    pub target: usize,
    /// Time lag (source precedes target by this many steps)
    pub lag: usize,
    /// Strength of the causal link (partial correlation or CMI value)
    pub strength: f64,
    /// P-value from the independence test
    pub p_value: f64,
}

/// Causal graph discovered by PCMCI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    /// Number of variables
    pub n_vars: usize,
    /// Maximum lag considered
    pub max_lag: usize,
    /// Adjacency tensor: val_matrix[target][source][lag] = strength
    pub val_matrix: Vec<Vec<Vec<f64>>>,
    /// P-value tensor: p_matrix[target][source][lag] = p_value
    pub p_matrix: Vec<Vec<Vec<f64>>>,
    /// All discovered causal links
    pub links: Vec<CausalLink>,
}

impl CausalGraph {
    /// Create a new empty causal graph
    pub fn new(n_vars: usize, max_lag: usize) -> Self {
        let val_matrix = vec![vec![vec![0.0; max_lag + 1]; n_vars]; n_vars];
        let p_matrix = vec![vec![vec![1.0; max_lag + 1]; n_vars]; n_vars];
        Self {
            n_vars,
            max_lag,
            val_matrix,
            p_matrix,
            links: Vec::new(),
        }
    }

    /// Set a link value and p-value
    pub fn set_link(&mut self, source: usize, target: usize, lag: usize, strength: f64, p_value: f64) {
        if source < self.n_vars && target < self.n_vars && lag <= self.max_lag {
            self.val_matrix[target][source][lag] = strength;
            self.p_matrix[target][source][lag] = p_value;
            self.links.push(CausalLink {
                source,
                target,
                lag,
                strength,
                p_value,
            });
        }
    }

    /// Get all significant causal links below a p-value threshold
    pub fn get_significant_links(&self, threshold: f64) -> Vec<CausalLink> {
        self.links
            .iter()
            .filter(|link| link.p_value < threshold && link.lag > 0)
            .cloned()
            .collect()
    }

    /// Get the parents of a given variable (significant incoming links)
    pub fn get_parents(&self, target: usize, threshold: f64) -> Vec<CausalLink> {
        self.links
            .iter()
            .filter(|link| link.target == target && link.p_value < threshold && link.lag > 0)
            .cloned()
            .collect()
    }

    /// Get the strongest causal link for a given target
    pub fn get_strongest_cause(&self, target: usize) -> Option<CausalLink> {
        self.links
            .iter()
            .filter(|link| link.target == target && link.lag > 0)
            .max_by(|a, b| a.strength.abs().partial_cmp(&b.strength.abs()).unwrap())
            .cloned()
    }

    /// Get summary statistics of the causal graph
    pub fn summary(&self) -> CausalGraphSummary {
        let significant = self.get_significant_links(0.05);
        let n_significant = significant.len();
        let avg_strength = if n_significant > 0 {
            significant.iter().map(|l| l.strength.abs()).sum::<f64>() / n_significant as f64
        } else {
            0.0
        };
        let max_strength = significant
            .iter()
            .map(|l| l.strength.abs())
            .fold(0.0_f64, f64::max);

        CausalGraphSummary {
            n_vars: self.n_vars,
            max_lag: self.max_lag,
            n_significant_links: n_significant,
            avg_link_strength: avg_strength,
            max_link_strength: max_strength,
        }
    }
}

/// Summary statistics for a causal graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraphSummary {
    pub n_vars: usize,
    pub max_lag: usize,
    pub n_significant_links: usize,
    pub avg_link_strength: f64,
    pub max_link_strength: f64,
}

/// Type alias for parents map: target -> Vec<(source, lag)>
type ParentsMap = HashMap<usize, Vec<(usize, usize)>>;

/// PCMCI Causal Discovery engine
#[derive(Debug)]
pub struct PCMCICausalDiscovery {
    config: PCMCIConfig,
}

impl PCMCICausalDiscovery {
    /// Create a new PCMCI instance with the given configuration
    pub fn new(config: PCMCIConfig) -> Self {
        Self { config }
    }

    /// Compute partial correlation between x and y, conditioning on z
    ///
    /// Uses the formula: r_xy|z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    /// For multiple conditioning variables, iteratively partials out each one.
    pub fn partial_correlation(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &[Array1<f64>],
    ) -> f64 {
        if z.is_empty() {
            return Self::pearson_correlation(x, y);
        }

        // Residualize x and y with respect to z using OLS
        let x_resid = Self::residualize(x, z);
        let y_resid = Self::residualize(y, z);

        Self::pearson_correlation(&x_resid, &y_resid)
    }

    /// Compute Pearson correlation between two arrays
    fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 3.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-15 {
            return 0.0;
        }

        (cov / denom).clamp(-1.0, 1.0)
    }

    /// Residualize a variable with respect to conditioning variables using OLS
    fn residualize(x: &Array1<f64>, z: &[Array1<f64>]) -> Array1<f64> {
        let n = x.len();
        let p = z.len();

        if p == 0 || n == 0 {
            return x.clone();
        }

        // Build design matrix Z: n x p
        let mut z_matrix = Array2::<f64>::zeros((n, p));
        for (j, zj) in z.iter().enumerate() {
            for i in 0..n {
                z_matrix[[i, j]] = zj[i];
            }
        }

        // Compute Z^T Z
        let ztz = z_matrix.t().dot(&z_matrix);

        // Compute Z^T x
        let ztx = z_matrix.t().dot(x);

        // Solve for beta using simple Cholesky-like approach
        // For small conditioning sets, use direct matrix inverse
        let beta = Self::solve_linear_system(&ztz, &ztx);

        // Compute residual: x - Z * beta
        let predicted = z_matrix.dot(&beta);
        x - &predicted
    }

    /// Solve a linear system A * x = b using Gaussian elimination
    fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = a.nrows();
        if n == 0 {
            return Array1::zeros(0);
        }

        // Augmented matrix [A | b]
        let mut aug = Array2::<f64>::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_val = aug[[col, col]].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[[row, col]].abs() > max_val {
                    max_val = aug[[row, col]].abs();
                    max_row = row;
                }
            }

            // Swap rows
            if max_row != col {
                for j in 0..=n {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            let pivot = aug[[col, col]];
            if pivot.abs() < 1e-15 {
                continue; // Singular, skip
            }

            // Eliminate below
            for row in (col + 1)..n {
                let factor = aug[[row, col]] / pivot;
                for j in col..=n {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum -= aug[[i, j]] * x[j];
            }
            if aug[[i, i]].abs() > 1e-15 {
                x[i] = sum / aug[[i, i]];
            }
        }

        x
    }

    /// Compute p-value from partial correlation using Fisher's z-transform
    fn p_value_from_parcorr(r: f64, n: usize, n_conds: usize) -> f64 {
        let effective_n = n as f64 - n_conds as f64 - 2.0;
        if effective_n <= 0.0 {
            return 1.0;
        }

        // Fisher's z-transform
        let r_clamped = r.clamp(-0.9999, 0.9999);
        let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();

        // Standard error
        let se = 1.0 / (effective_n - 1.0).max(1.0).sqrt();

        // Two-sided p-value using normal approximation
        let z_score = (z / se).abs();
        2.0 * normal_cdf(-z_score)
    }

    /// Compute conditional mutual information (Gaussian approximation)
    ///
    /// CMI(X; Y | Z) = -0.5 * ln(1 - r_xy|z^2)
    fn conditional_mutual_information(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &[Array1<f64>],
    ) -> f64 {
        let parcorr = self.partial_correlation(x, y, z);
        let r2 = parcorr * parcorr;
        if r2 >= 1.0 {
            return f64::INFINITY;
        }
        -0.5 * (1.0 - r2).ln()
    }

    /// Run the PC-stable phase: iterative condition selection
    ///
    /// For each target variable and each potential parent (source, lag),
    /// tests conditional independence with increasing conditioning set sizes.
    /// Removes links that are found to be conditionally independent.
    pub fn pc_stable_phase(&self, data: &Array2<f64>) -> ParentsMap {
        let n_vars = data.ncols();
        let n_samples = data.nrows();
        let max_lag = self.max_lag;

        // Initialize: all lagged variables are potential parents
        let mut parents: ParentsMap = HashMap::new();
        for target in 0..n_vars {
            let mut target_parents = Vec::new();
            for source in 0..n_vars {
                for lag in 1..=max_lag {
                    target_parents.push((source, lag));
                }
            }
            parents.insert(target, target_parents);
        }

        // Iteratively increase conditioning set size
        let max_conds = self.config.max_conds_dim.unwrap_or(n_vars * max_lag);

        for cond_dim in 0..=max_conds {
            let mut any_removed = false;

            for target in 0..n_vars {
                let current_parents = parents.get(&target).cloned().unwrap_or_default();

                if current_parents.len() <= cond_dim {
                    continue;
                }

                let mut to_remove = Vec::new();

                for &(source, lag) in &current_parents {
                    // Build target and source time series (aligned for lag)
                    if lag >= n_samples {
                        continue;
                    }
                    let effective_n = n_samples - max_lag;
                    if effective_n < 5 {
                        continue;
                    }

                    let y_series = self.extract_series(data, target, 0, max_lag, effective_n);
                    let x_series = self.extract_series(data, source, lag, max_lag, effective_n);

                    // Select conditioning set (other parents, excluding current link)
                    let other_parents: Vec<(usize, usize)> = current_parents
                        .iter()
                        .filter(|&&(s, l)| !(s == source && l == lag))
                        .cloned()
                        .collect();

                    // Test with subsets of size cond_dim
                    let cond_subsets = self.get_conditioning_subsets(&other_parents, cond_dim);

                    let mut is_independent = false;
                    for subset in &cond_subsets {
                        let z_series: Vec<Array1<f64>> = subset
                            .iter()
                            .map(|&(s, l)| self.extract_series(data, s, l, max_lag, effective_n))
                            .collect();

                        let (test_stat, p_val) = self.run_independence_test(
                            &x_series,
                            &y_series,
                            &z_series,
                            effective_n,
                        );

                        let _ = test_stat; // Used for debugging

                        if p_val > self.config.significance_level {
                            is_independent = true;
                            break;
                        }
                    }

                    if is_independent {
                        to_remove.push((source, lag));
                        any_removed = true;
                    }
                }

                // Remove links found to be independent
                if let Some(p) = parents.get_mut(&target) {
                    p.retain(|link| !to_remove.contains(link));
                }
            }

            if !any_removed {
                break;
            }
        }

        parents
    }

    /// Run the MCI (Momentary Conditional Independence) test
    ///
    /// For each remaining link from the PC-stable phase, tests:
    /// X(t-lag) _|_ Y(t) | Parents(Y)\{X(t-lag)}, Parents(X)
    pub fn mci_test(&self, data: &Array2<f64>, parents: &ParentsMap) -> CausalGraph {
        let n_vars = data.ncols();
        let n_samples = data.nrows();
        let max_lag = self.max_lag;
        let effective_n = n_samples - max_lag;

        let mut graph = CausalGraph::new(n_vars, max_lag);

        if effective_n < 5 {
            return graph;
        }

        for target in 0..n_vars {
            let target_parents = parents.get(&target).cloned().unwrap_or_default();

            for source in 0..n_vars {
                for lag in 1..=max_lag {
                    let y_series = self.extract_series(data, target, 0, max_lag, effective_n);
                    let x_series = self.extract_series(data, source, lag, max_lag, effective_n);

                    // Build conditioning set: Parents(Y) \ {X(t-lag)} + Parents(X)
                    let mut cond_set: Vec<(usize, usize)> = Vec::new();

                    // Add parents of target (excluding the tested link)
                    for &(s, l) in &target_parents {
                        if !(s == source && l == lag) {
                            cond_set.push((s, l));
                        }
                    }

                    // Add parents of source
                    if let Some(source_parents) = parents.get(&source) {
                        for &(s, l) in source_parents {
                            let adjusted_lag = l + lag;
                            if adjusted_lag <= max_lag {
                                let entry = (s, adjusted_lag);
                                if !cond_set.contains(&entry) {
                                    cond_set.push(entry);
                                }
                            }
                        }
                    }

                    // Build conditioning variable series
                    let z_series: Vec<Array1<f64>> = cond_set
                        .iter()
                        .filter_map(|&(s, l)| {
                            if l < n_samples {
                                Some(self.extract_series(data, s, l, max_lag, effective_n))
                            } else {
                                None
                            }
                        })
                        .collect();

                    let (test_stat, p_val) = self.run_independence_test(
                        &x_series,
                        &y_series,
                        &z_series,
                        effective_n,
                    );

                    graph.set_link(source, target, lag, test_stat, p_val);
                }
            }
        }

        graph
    }

    /// Run the full PCMCI algorithm on multivariate time series data
    ///
    /// # Arguments
    /// * `data` - Array2<f64> with shape (n_samples, n_vars)
    ///
    /// # Returns
    /// A CausalGraph containing all discovered causal links with strengths and p-values
    pub fn fit(&mut self, data: &Array2<f64>) -> CausalGraph {
        log::info!(
            "Running PCMCI with {} variables, {} samples, max_lag={}",
            data.ncols(),
            data.nrows(),
            self.max_lag
        );

        // Phase 1: PC-stable condition selection
        log::info!("Phase 1: PC-stable condition selection");
        let parents = self.pc_stable_phase(data);

        let total_parents: usize = parents.values().map(|v| v.len()).sum();
        log::info!("PC-stable phase found {} parent links", total_parents);

        // Phase 2: MCI test
        log::info!("Phase 2: MCI testing");
        let graph = self.mci_test(data, &parents);

        let n_significant = graph.get_significant_links(self.config.significance_level).len();
        log::info!("MCI phase found {} significant links", n_significant);

        graph
    }

    /// Extract a time series for a given variable at a given lag
    fn extract_series(
        &self,
        data: &Array2<f64>,
        var: usize,
        lag: usize,
        max_lag: usize,
        effective_n: usize,
    ) -> Array1<f64> {
        let col = data.column(var);
        let start = max_lag - lag;
        let end = start + effective_n;
        let end = end.min(col.len());
        col.slice(ndarray::s![start..end]).to_owned()
    }

    /// Run the configured independence test
    fn run_independence_test(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &[Array1<f64>],
        n: usize,
    ) -> (f64, f64) {
        match self.config.test_type {
            IndependenceTest::ParCorr => {
                let r = self.partial_correlation(x, y, z);
                let p = Self::p_value_from_parcorr(r, n, z.len());
                (r, p)
            }
            IndependenceTest::CMI => {
                let cmi = self.conditional_mutual_information(x, y, z);
                // Approximate p-value using chi-squared with 1 df
                let chi2 = 2.0 * n as f64 * cmi;
                let p = 1.0 - chi_squared_cdf(chi2, 1.0);
                (cmi, p)
            }
        }
    }

    /// Generate subsets of a given size from a list of parents
    fn get_conditioning_subsets(
        &self,
        parents: &[(usize, usize)],
        size: usize,
    ) -> Vec<Vec<(usize, usize)>> {
        if size == 0 {
            return vec![vec![]];
        }
        if parents.len() < size {
            return vec![];
        }

        // For efficiency, limit the number of subsets
        let max_subsets = 20;
        let mut subsets = Vec::new();
        let mut indices = (0..size).collect::<Vec<_>>();

        loop {
            let subset: Vec<(usize, usize)> = indices.iter().map(|&i| parents[i]).collect();
            subsets.push(subset);

            if subsets.len() >= max_subsets {
                break;
            }

            // Generate next combination
            if !next_combination(&mut indices, parents.len()) {
                break;
            }
        }

        subsets
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &PCMCIConfig {
        &self.config
    }
}

/// Standard normal CDF approximation (Abramowitz and Stegun)
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Chi-squared CDF approximation for small degrees of freedom
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Use incomplete gamma function approximation
    // For df=1: CDF = 2 * Phi(sqrt(x)) - 1
    if (df - 1.0).abs() < 0.01 {
        let z = x.sqrt();
        return 2.0 * normal_cdf(z) - 1.0;
    }

    // General case: series approximation
    let k = df / 2.0;
    let x_half = x / 2.0;
    regularized_lower_gamma(k, x_half)
}

/// Regularized lower incomplete gamma function approximation
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    // Series expansion for small x
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        for n in 1..200 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }
        return sum * (-x + a * x.ln() - ln_gamma(a)).exp();
    }

    // Continued fraction for large x
    1.0 - regularized_upper_gamma(a, x)
}

/// Regularized upper incomplete gamma function using continued fraction
fn regularized_upper_gamma(a: f64, x: f64) -> f64 {
    let mut f = 1.0 + x - a;
    if f.abs() < 1e-30 {
        f = 1e-30;
    }
    let mut c = 1.0 / f;
    let mut d = 1.0;

    for i in 1..200 {
        let an = i as f64 * (a - i as f64);
        let bn = 2.0 * i as f64 + 1.0 + x - a;
        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }

    (-x + a * x.ln() - ln_gamma(a)).exp() / f
}

/// Log-gamma function approximation (Stirling)
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Lanczos approximation
    let g = 7.0;
    let coefficients = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0;
    let mut sum = coefficients[0];
    for (i, &c) in coefficients.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }

    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t).ln() * (x + 0.5) - t + sum.ln()
}

/// Generate next combination in lexicographic order
fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    if k == 0 {
        return false;
    }

    let mut i = k;
    while i > 0 {
        i -= 1;
        if indices[i] != i + n - k {
            break;
        }
        if i == 0 && indices[0] == n - k {
            return false;
        }
    }

    indices[i] += 1;
    for j in (i + 1)..k {
        indices[j] = indices[j - 1] + 1;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_pearson_correlation() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let r = PCMCICausalDiscovery::pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Perfect positive correlation expected, got {}", r);
    }

    #[test]
    fn test_partial_correlation_no_conditioning() {
        let config = PCMCIConfig::default();
        let pcmci = PCMCICausalDiscovery::new(config);

        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![2.0, 4.0, 5.0, 4.0, 5.0]);

        let r = pcmci.partial_correlation(&x, &y, &[]);
        assert!(r > 0.0, "Positive correlation expected, got {}", r);
        assert!(r < 1.0, "Non-perfect correlation expected, got {}", r);
    }

    #[test]
    fn test_partial_correlation_with_conditioning() {
        let config = PCMCIConfig::default();
        let pcmci = PCMCICausalDiscovery::new(config);

        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let z_var = Array1::from(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
        // y is mostly determined by z, with some noise
        let y = &z_var * 2.0 + 0.1;

        let r = pcmci.partial_correlation(&x, &y, &[z_var]);
        // After conditioning on z, the correlation between x and y should be much reduced
        assert!(
            r.abs() < 0.95,
            "Partial correlation should be reduced after conditioning, got {}",
            r
        );
    }

    #[test]
    fn test_causal_graph() {
        let mut graph = CausalGraph::new(3, 2);
        graph.set_link(0, 1, 1, 0.5, 0.01);
        graph.set_link(1, 2, 2, -0.3, 0.03);
        graph.set_link(0, 2, 1, 0.1, 0.8); // Not significant

        let significant = graph.get_significant_links(0.05);
        assert_eq!(significant.len(), 2);

        let parents = graph.get_parents(1, 0.05);
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0].source, 0);
    }

    #[test]
    fn test_pcmci_synthetic() {
        use rand::Rng;

        let n_samples = 200;
        let n_vars = 3;
        let mut rng = rand::thread_rng();

        // Generate data with known causal structure: X0(t-1) -> X1(t)
        let mut data = Array2::<f64>::zeros((n_samples, n_vars));

        // Random noise for X0
        for i in 0..n_samples {
            data[[i, 0]] = rng.gen::<f64>() - 0.5;
        }

        // X1 depends on X0 with lag 1
        for i in 1..n_samples {
            data[[i, 1]] = 0.7 * data[[i - 1, 0]] + 0.3 * (rng.gen::<f64>() - 0.5);
        }

        // X2 is independent noise
        for i in 0..n_samples {
            data[[i, 2]] = rng.gen::<f64>() - 0.5;
        }

        let config = PCMCIConfig {
            max_lag: 2,
            significance_level: 0.05,
            test_type: IndependenceTest::ParCorr,
            max_conds_dim: Some(2),
        };

        let mut pcmci = PCMCICausalDiscovery::new(config);
        let graph = pcmci.fit(&data);

        // Check that the link X0 -> X1 at lag 1 is significant
        let link = &graph.val_matrix[1][0][1];
        assert!(
            link.abs() > 0.3,
            "Expected strong link X0->X1 at lag 1, got strength {}",
            link
        );
    }

    #[test]
    fn test_normal_cdf() {
        let val = normal_cdf(0.0);
        assert!((val - 0.5).abs() < 1e-6);

        let val = normal_cdf(1.96);
        assert!((val - 0.975).abs() < 0.01);
    }

    #[test]
    fn test_next_combination() {
        let mut indices = vec![0, 1];
        assert!(next_combination(&mut indices, 4));
        assert_eq!(indices, vec![0, 2]);

        assert!(next_combination(&mut indices, 4));
        assert_eq!(indices, vec![0, 3]);

        assert!(next_combination(&mut indices, 4));
        assert_eq!(indices, vec![1, 2]);
    }
}
