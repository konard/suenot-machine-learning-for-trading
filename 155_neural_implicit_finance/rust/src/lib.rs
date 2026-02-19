use ndarray::Array2;

/// A single SIREN layer implementation in Rust for high-speed inference.
/// This matches the Python implementation's activation: sin(omega_0 * (Wx + b)).
pub struct SirenLayer {
    pub weights: Array2<f64>,
    pub bias: Array2<f64>,
    pub omega_0: f64,
}

impl SirenLayer {
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut out = x.dot(&self.weights) + &self.bias;
        out.mapv_inplace(|val| (self.omega_0 * val).sin());
        out
    }
}

pub struct SirenNet {
    pub layers: Vec<SirenLayer>,
    pub final_weights: Array2<f64>,
    pub final_bias: Array2<f64>,
}

impl SirenNet {
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        // Final linear layer forward
        current.dot(&self.final_weights) + &self.final_bias
    }
}

pub mod calibration {
    use super::*;
    use ndarray::array;

    /// Mock function to get a calibrated SIREN model for production inferencing.
    pub fn get_mock_model() -> SirenNet {
        // In reality, weights would be loaded from a binary file exported by PyTorch
        SirenNet {
            layers: vec![
                SirenLayer {
                    weights: Array2::from_elem((2, 64), 0.1),
                    bias: Array2::from_elem((1, 64), 0.0),
                    omega_0: 30.0,
                }
            ],
            final_weights: Array2::from_elem((64, 1), 0.01),
            final_bias: array![[0.2]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_siren_forward() {
        let model = calibration::get_mock_model();
        let input = array![[0.0, 0.5]];
        let output = model.forward(&input);
        
        assert_eq!(output.shape(), &[1, 1]);
        assert!(output[[0, 0]] > 0.0);
    }
}
