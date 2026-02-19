use rustfft::{FftPlanner, num_complex::Complex};

pub mod spectral_engine {
    use super::*;

    /// Low-level Rust executing a 1D Fourier Neural Operator spectral layer.
    /// This bypasses Python completely for live market-making (LOB updates)
    /// representing sequence trajectories as PDE operators.
    pub fn perform_fno_1d_pass(
        market_sequence: &[f64], 
        kept_modes: usize
    ) -> Vec<f64> {
        let n = market_sequence.len();
        if n == 0 { return vec![]; }

        let mut planner = FftPlanner::new();
        
        // 1. Convert real price data into complex domain for FFT
        let mut buffer: Vec<Complex<f64>> = market_sequence
            .iter()
            .map(|&val| Complex { re: val, im: 0.0 })
            .collect();

        // Let's plan a forward Discrete Fourier Transform (C -> C mapped over reals)
        let fft_forward = planner.plan_fft_forward(n);
        fft_forward.process(&mut buffer);

        // 2. FNO Spectral Truncation (The secret sauce of Fourier Operators)
        // We simulate zeroing out frequencies higher than `kept_modes`.
        // This instantly destroys highly-volatile micro-texture and retains 
        // the core foundational trend of the PDE limit book curve.
        for i in kept_modes..n {
            // Except for the real symmetry ends if keeping complex structure
            buffer[i] = Complex { re: 0.0, im: 0.0 };
        }

        // Simulating the Parametric Complex Matrix Multiplication (W * F(x))
        // We double the amplitude of the lowest frequencies as a mock feature map.
        for i in 0..kept_modes {
            buffer[i].re *= 1.5; 
            buffer[i].im *= 1.5;
        }

        // 3. Inverse Fast Fourier Transform (Back to Time Domain)
        let fft_inverse = planner.plan_fft_inverse(n);
        fft_inverse.process(&mut buffer);

        // Map back to real space and normalize due to IDFT scaling
        let scaling_factor = 1.0 / (n as f64);
        buffer.iter().map(|c| c.re * scaling_factor).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_filtering() {
        let dummy_prices = vec![1.0, 2.0, -1.0, 4.0, 0.5, 3.0, 2.0, 1.0];
        
        // Keeping only 2 low-frequency modes, discarding high frequency jitter
        let filtered = spectral_engine::perform_fno_1d_pass(&dummy_prices, 2);
        
        assert_eq!(filtered.len(), 8);
        assert!(!filtered.iter().any(|&x| x.is_nan())); // Ensure math didn't explode
    }
}
