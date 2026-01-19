//! Fourier Transform Layer
//!
//! Replaces self-attention with Fast Fourier Transform for O(n log n) complexity.

use ndarray::{Array2, Array3};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

/// Fourier Transform layer that replaces self-attention.
///
/// Uses FFT to mix tokens across sequence and feature dimensions.
/// This layer has no learnable parameters.
pub struct FourierLayer {
    planner: FftPlanner<f64>,
}

impl FourierLayer {
    /// Create a new Fourier layer.
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }

    /// Apply 2D Fourier Transform to input.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    ///
    /// # Returns
    /// Real part of FFT output [batch, seq_len, d_model]
    pub fn forward(&mut self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::<f64>::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            // Get the 2D slice for this batch
            let slice = x.slice(ndarray::s![b, .., ..]);

            // Apply 2D FFT
            let fft_result = self.fft_2d(&slice.to_owned());

            // Store real part
            for i in 0..seq_len {
                for j in 0..d_model {
                    output[[b, i, j]] = fft_result[[i, j]].re;
                }
            }
        }

        output
    }

    /// Perform 2D FFT on a matrix.
    fn fft_2d(&mut self, input: &Array2<f64>) -> Array2<Complex<f64>> {
        let (rows, cols) = input.dim();

        // Convert to complex
        let mut complex_data: Vec<Vec<Complex<f64>>> = input
            .outer_iter()
            .map(|row| row.iter().map(|&x| Complex::new(x, 0.0)).collect())
            .collect();

        // FFT along rows
        let fft_row = self.planner.plan_fft_forward(cols);
        for row in &mut complex_data {
            fft_row.process(row);
        }

        // Transpose
        let mut transposed: Vec<Vec<Complex<f64>>> = (0..cols)
            .map(|j| (0..rows).map(|i| complex_data[i][j]).collect())
            .collect();

        // FFT along columns (now rows after transpose)
        let fft_col = self.planner.plan_fft_forward(rows);
        for col in &mut transposed {
            fft_col.process(col);
        }

        // Transpose back and create output array
        let mut output = Array2::<Complex<f64>>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                output[[i, j]] = transposed[j][i];
            }
        }

        output
    }

    /// Forward pass returning both output and magnitude spectrum.
    pub fn forward_with_magnitudes(&mut self, x: &Array3<f64>) -> (Array3<f64>, Array3<f64>) {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::<f64>::zeros((batch_size, seq_len, d_model));
        let mut magnitude = Array3::<f64>::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            let slice = x.slice(ndarray::s![b, .., ..]);
            let fft_result = self.fft_2d(&slice.to_owned());

            for i in 0..seq_len {
                for j in 0..d_model {
                    output[[b, i, j]] = fft_result[[i, j]].re;
                    magnitude[[b, i, j]] = fft_result[[i, j]].norm();
                }
            }
        }

        (output, magnitude)
    }

    /// Get magnitude spectrum for analysis.
    pub fn get_magnitude_spectrum(&mut self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut magnitude = Array3::<f64>::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            let slice = x.slice(ndarray::s![b, .., ..]);
            let fft_result = self.fft_2d(&slice.to_owned());

            for i in 0..seq_len {
                for j in 0..d_model {
                    magnitude[[b, i, j]] = fft_result[[i, j]].norm();
                }
            }
        }

        magnitude
    }
}

impl Default for FourierLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate Discrete Fourier Transform manually (for reference).
///
/// This is O(n²) - use FFT for O(n log n).
pub fn dft_1d(signal: &[f64]) -> Vec<Complex<f64>> {
    let n = signal.len();
    let mut result = Vec::with_capacity(n);

    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (t, &x) in signal.iter().enumerate() {
            let angle = -2.0 * PI * (k * t) as f64 / n as f64;
            sum += Complex::new(x * angle.cos(), x * angle.sin());
        }
        result.push(sum);
    }

    result
}

/// Inverse DFT for reconstruction.
pub fn idft_1d(spectrum: &[Complex<f64>]) -> Vec<f64> {
    let n = spectrum.len();
    let mut result = Vec::with_capacity(n);

    for t in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (k, &x) in spectrum.iter().enumerate() {
            let angle = 2.0 * PI * (k * t) as f64 / n as f64;
            sum += x * Complex::new(angle.cos(), angle.sin());
        }
        result.push(sum.re / n as f64);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use approx::assert_relative_eq;

    #[test]
    fn test_fourier_layer_forward() {
        let mut layer = FourierLayer::new();

        // Create test input [batch=2, seq_len=8, d_model=4]
        let input = Array3::<f64>::from_shape_fn((2, 8, 4), |(b, i, j)| {
            (b * 100 + i * 10 + j) as f64 / 100.0
        });

        let output = layer.forward(&input);

        // Check output shape matches input
        assert_eq!(output.dim(), input.dim());

        // Output should not be all zeros
        assert!(output.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_dft_idft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = dft_1d(&signal);
        let recovered = idft_1d(&spectrum);

        for (original, rec) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(original, rec, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_magnitude_spectrum() {
        let mut layer = FourierLayer::new();

        let input = Array3::<f64>::from_shape_fn((1, 16, 8), |(_, i, _)| {
            // Create a simple sinusoidal pattern
            (2.0 * PI * i as f64 / 16.0).sin()
        });

        let magnitude = layer.get_magnitude_spectrum(&input);

        // All magnitudes should be non-negative
        assert!(magnitude.iter().all(|&x| x >= 0.0));
    }
}
