//! Dilated Convolutions Module
//!
//! This module implements dilated (atrous) convolutions for time series processing.
//!
//! ## Key Concepts
//!
//! - **Dilation**: Skip between kernel elements
//! - **Causal**: Only uses past information
//! - **Receptive Field**: How far back the model can see
//!
//! ## Example
//!
//! ```rust
//! use dilated_conv_trading::conv::{DilatedConv1D, DilatedConvStack};
//!
//! // Create a single dilated convolution layer
//! let conv = DilatedConv1D::new(5, 32, 3, 4); // 5 inputs, 32 outputs, kernel=3, dilation=4
//!
//! // Create a stack of dilated convolutions (WaveNet-style)
//! let stack = DilatedConvStack::new(5, 32, &[1, 2, 4, 8, 16, 32]);
//! ```

pub mod causal;
pub mod dilated;
pub mod wavenet;

pub use causal::CausalConv1D;
pub use dilated::DilatedConv1D;
pub use wavenet::{DilatedConvStack, DilatedResidualBlock, GatedActivation};

/// Calculate receptive field for a stack of dilated convolutions
///
/// # Arguments
/// - `kernel_size` - Size of convolution kernel
/// - `dilation_rates` - Slice of dilation rates for each layer
///
/// # Returns
/// - Total receptive field in timesteps
pub fn calculate_receptive_field(kernel_size: usize, dilation_rates: &[usize]) -> usize {
    1 + dilation_rates
        .iter()
        .map(|d| (kernel_size - 1) * d)
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receptive_field_calculation() {
        // WaveNet-style: kernel=2, dilations 1,2,4,8,16,32
        let rf = calculate_receptive_field(2, &[1, 2, 4, 8, 16, 32]);
        assert_eq!(rf, 64); // 1 + (1+2+4+8+16+32) = 64

        // Larger kernel: kernel=3, dilations 1,2,4,8
        let rf = calculate_receptive_field(3, &[1, 2, 4, 8]);
        assert_eq!(rf, 31); // 1 + 2*(1+2+4+8) = 31
    }

    #[test]
    fn test_single_layer() {
        let rf = calculate_receptive_field(3, &[1]);
        assert_eq!(rf, 3); // 1 + 2*1 = 3
    }
}
