//! Rational-Quadratic Spline implementation
//!
//! This module implements the monotonic rational-quadratic spline transformation
//! from "Neural Spline Flows" (Durkan et al., 2019).

use ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;

/// Spline parameters for a single dimension
#[derive(Debug, Clone)]
pub struct SplineParams {
    /// Bin widths (K values, sum to 2*bound)
    pub widths: Array1<f64>,
    /// Bin heights (K values, sum to 2*bound)
    pub heights: Array1<f64>,
    /// Derivatives at knots (K+1 values, all positive)
    pub derivatives: Array1<f64>,
}

impl SplineParams {
    /// Create new spline parameters
    pub fn new(widths: Array1<f64>, heights: Array1<f64>, derivatives: Array1<f64>) -> Self {
        Self {
            widths,
            heights,
            derivatives,
        }
    }

    /// Create default spline parameters (identity-like)
    pub fn identity(num_bins: usize, bound: f64) -> Self {
        let width = 2.0 * bound / num_bins as f64;
        let widths = Array1::from_elem(num_bins, width);
        let heights = Array1::from_elem(num_bins, width);
        let derivatives = Array1::ones(num_bins + 1);

        Self {
            widths,
            heights,
            derivatives,
        }
    }

    /// Number of bins
    pub fn num_bins(&self) -> usize {
        self.widths.len()
    }
}

/// Rational-Quadratic Spline transformation
#[derive(Debug, Clone)]
pub struct RationalQuadraticSpline {
    /// Number of bins
    num_bins: usize,
    /// Spline boundary
    bound: f64,
    /// Minimum derivative value
    min_derivative: f64,
}

impl Default for RationalQuadraticSpline {
    fn default() -> Self {
        Self::new(8, 3.0, 1e-3)
    }
}

impl RationalQuadraticSpline {
    /// Create a new rational-quadratic spline
    ///
    /// # Arguments
    ///
    /// * `num_bins` - Number of spline bins
    /// * `bound` - Spline boundary ([-bound, bound])
    /// * `min_derivative` - Minimum derivative for numerical stability
    pub fn new(num_bins: usize, bound: f64, min_derivative: f64) -> Self {
        Self {
            num_bins,
            bound,
            min_derivative,
        }
    }

    /// Get number of parameters per dimension
    pub fn params_per_dim(&self) -> usize {
        3 * self.num_bins + 1
    }

    /// Convert raw parameters to spline parameters
    pub fn params_from_raw(&self, raw: &Array1<f64>) -> SplineParams {
        assert_eq!(raw.len(), self.params_per_dim());

        let widths_raw = raw.slice(ndarray::s![..self.num_bins]);
        let heights_raw = raw.slice(ndarray::s![self.num_bins..2 * self.num_bins]);
        let derivatives_raw = raw.slice(ndarray::s![2 * self.num_bins..]);

        // Softmax for widths and heights
        let widths = softmax(&widths_raw.to_owned()) * 2.0 * self.bound;
        let heights = softmax(&heights_raw.to_owned()) * 2.0 * self.bound;

        // Softplus + min for derivatives
        let derivatives = derivatives_raw.mapv(|x| softplus(x) + self.min_derivative);

        SplineParams {
            widths,
            heights,
            derivatives,
        }
    }

    /// Forward transformation: x -> y
    ///
    /// Returns (y, log_det) where log_det is the log determinant of the Jacobian
    pub fn forward(&self, x: f64, params: &SplineParams) -> (f64, f64) {
        // Handle out-of-bounds (identity outside spline region)
        if x <= -self.bound {
            return (x, 0.0);
        }
        if x >= self.bound {
            return (x, 0.0);
        }

        // Compute cumulative widths and heights
        let cumwidths = cumsum_with_start(&params.widths, -self.bound);
        let cumheights = cumsum_with_start(&params.heights, -self.bound);

        // Find bin index
        let bin_idx = find_bin(x, &cumwidths);
        let bin_idx = bin_idx.min(self.num_bins - 1);

        // Get bin parameters
        let x_k = cumwidths[bin_idx];
        let x_k_plus_1 = cumwidths[bin_idx + 1];
        let y_k = cumheights[bin_idx];
        let y_k_plus_1 = cumheights[bin_idx + 1];
        let w_k = params.widths[bin_idx];
        let h_k = params.heights[bin_idx];
        let d_k = params.derivatives[bin_idx];
        let d_k_plus_1 = params.derivatives[bin_idx + 1];

        // Compute delta (slope)
        let delta = h_k / w_k;

        // Compute xi (normalized position within bin)
        let xi = (x - x_k) / w_k;

        // Compute spline value
        let xi_sq = xi * xi;
        let one_minus_xi = 1.0 - xi;
        let one_minus_xi_sq = one_minus_xi * one_minus_xi;

        let numerator = h_k * (delta * xi_sq + d_k * xi * one_minus_xi);
        let denominator = delta + (d_k + d_k_plus_1 - 2.0 * delta) * xi * one_minus_xi;

        let y = y_k + numerator / denominator;

        // Compute log derivative
        let derivative_numerator =
            delta * delta * (d_k_plus_1 * xi_sq + 2.0 * delta * xi * one_minus_xi + d_k * one_minus_xi_sq);
        let log_det = derivative_numerator.ln() - 2.0 * denominator.ln();

        (y, log_det)
    }

    /// Inverse transformation: y -> x
    ///
    /// Returns (x, log_det) where log_det is the log determinant (negative of forward)
    pub fn inverse(&self, y: f64, params: &SplineParams) -> (f64, f64) {
        // Handle out-of-bounds
        if y <= -self.bound {
            return (y, 0.0);
        }
        if y >= self.bound {
            return (y, 0.0);
        }

        // Compute cumulative widths and heights
        let cumwidths = cumsum_with_start(&params.widths, -self.bound);
        let cumheights = cumsum_with_start(&params.heights, -self.bound);

        // Find bin index (search in cumheights for inverse)
        let bin_idx = find_bin(y, &cumheights);
        let bin_idx = bin_idx.min(self.num_bins - 1);

        // Get bin parameters
        let x_k = cumwidths[bin_idx];
        let y_k = cumheights[bin_idx];
        let y_k_plus_1 = cumheights[bin_idx + 1];
        let w_k = params.widths[bin_idx];
        let h_k = params.heights[bin_idx];
        let d_k = params.derivatives[bin_idx];
        let d_k_plus_1 = params.derivatives[bin_idx + 1];

        // Compute delta
        let delta = h_k / w_k;

        // Solve quadratic for xi
        let y_minus_y_k = y - y_k;

        let a = h_k * (delta - d_k) + y_minus_y_k * (d_k + d_k_plus_1 - 2.0 * delta);
        let b = h_k * d_k - y_minus_y_k * (d_k + d_k_plus_1 - 2.0 * delta);
        let c = -delta * y_minus_y_k;

        // Quadratic formula: xi = 2c / (-b - sqrt(b^2 - 4ac))
        let discriminant = b * b - 4.0 * a * c;
        let xi = if a.abs() < 1e-10 {
            // Linear case
            -c / b
        } else {
            2.0 * c / (-b - discriminant.sqrt())
        };

        let x = xi * w_k + x_k;

        // Compute log derivative (negative of forward)
        let xi_sq = xi * xi;
        let one_minus_xi = 1.0 - xi;
        let one_minus_xi_sq = one_minus_xi * one_minus_xi;

        let denominator = delta + (d_k + d_k_plus_1 - 2.0 * delta) * xi * one_minus_xi;
        let derivative_numerator =
            delta * delta * (d_k_plus_1 * xi_sq + 2.0 * delta * xi * one_minus_xi + d_k * one_minus_xi_sq);
        let log_det = -(derivative_numerator.ln() - 2.0 * denominator.ln());

        (x, log_det)
    }

    /// Batch forward transformation
    pub fn forward_batch(&self, x: &Array1<f64>, params: &[SplineParams]) -> (Array1<f64>, f64) {
        assert_eq!(x.len(), params.len());

        let mut y = Array1::zeros(x.len());
        let mut total_log_det = 0.0;

        for (i, (xi, pi)) in x.iter().zip(params.iter()).enumerate() {
            let (yi, log_det) = self.forward(*xi, pi);
            y[i] = yi;
            total_log_det += log_det;
        }

        (y, total_log_det)
    }

    /// Batch inverse transformation
    pub fn inverse_batch(&self, y: &Array1<f64>, params: &[SplineParams]) -> (Array1<f64>, f64) {
        assert_eq!(y.len(), params.len());

        let mut x = Array1::zeros(y.len());
        let mut total_log_det = 0.0;

        for (i, (yi, pi)) in y.iter().zip(params.iter()).enumerate() {
            let (xi, log_det) = self.inverse(*yi, pi);
            x[i] = xi;
            total_log_det += log_det;
        }

        (x, total_log_det)
    }
}

/// Compute softmax of a vector
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum_exp = exp_x.sum();
    exp_x / sum_exp
}

/// Compute softplus: log(1 + exp(x))
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Compute cumulative sum with a starting value
fn cumsum_with_start(arr: &Array1<f64>, start: f64) -> Array1<f64> {
    let mut result = Array1::zeros(arr.len() + 1);
    result[0] = start;
    for i in 0..arr.len() {
        result[i + 1] = result[i] + arr[i];
    }
    result
}

/// Find bin index for a value in sorted cumulative array
fn find_bin(value: f64, cumulative: &Array1<f64>) -> usize {
    for i in 0..cumulative.len() - 1 {
        if value < cumulative[i + 1] {
            return i;
        }
    }
    cumulative.len() - 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_identity_spline() {
        let spline = RationalQuadraticSpline::new(8, 3.0, 1e-3);
        let params = SplineParams::identity(8, 3.0);

        // Identity spline should approximately preserve values
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let (y, _) = spline.forward(x, &params);
            assert_abs_diff_eq!(x, y, epsilon = 0.1);
        }
    }

    #[test]
    fn test_invertibility() {
        let spline = RationalQuadraticSpline::new(8, 3.0, 1e-3);
        let params = SplineParams::identity(8, 3.0);

        // Forward then inverse should recover original
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let (y, _) = spline.forward(x, &params);
            let (x_recovered, _) = spline.inverse(y, &params);
            assert_abs_diff_eq!(x, x_recovered, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = softmax(&x);
        assert_abs_diff_eq!(result.sum(), 1.0, epsilon = 1e-10);
    }
}
