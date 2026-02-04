//! Image stack generation for multi-channel inputs
//!
//! Combines multiple timeframes or transformation types into RGB images.

use ndarray::{Array2, Array3};

use super::gaf::GAFTransformer;
use super::mtf::MTFTransformer;

/// Image stack generator for creating multi-channel images
#[derive(Debug, Clone)]
pub struct ImageStackGenerator {
    /// Output image size
    pub image_size: usize,
    /// GAF transformer
    gaf: GAFTransformer,
    /// MTF transformer
    mtf: MTFTransformer,
}

impl ImageStackGenerator {
    /// Create a new image stack generator
    pub fn new(image_size: usize) -> Self {
        Self {
            image_size,
            gaf: GAFTransformer::new(image_size),
            mtf: MTFTransformer::new(8, image_size),
        }
    }

    /// Create multi-transform stack (GASF, GADF, MTF as RGB channels)
    ///
    /// # Arguments
    /// * `prices` - Input price series
    ///
    /// # Returns
    /// 3D array of shape (image_size, image_size, 3)
    pub fn create_multi_transform_stack(&self, prices: &[f64]) -> Array3<f64> {
        let gasf = self.gaf.transform_gasf(prices);
        let gadf = self.gaf.transform_gadf(prices);
        let mtf = self.mtf.transform(prices);

        // Normalize each channel to [0, 1]
        let gasf_norm = self.normalize_channel(&gasf);
        let gadf_norm = self.normalize_channel(&gadf);
        let mtf_norm = self.normalize_channel(&mtf);

        // Stack into RGB image
        self.stack_channels(&gasf_norm, &gadf_norm, &mtf_norm)
    }

    /// Create timeframe stack from multiple price series
    ///
    /// # Arguments
    /// * `prices_1m` - 1-minute close prices
    /// * `prices_5m` - 5-minute close prices
    /// * `prices_15m` - 15-minute close prices
    ///
    /// # Returns
    /// 3D array of shape (image_size, image_size, 3)
    pub fn create_timeframe_stack(
        &self,
        prices_1m: &[f64],
        prices_5m: &[f64],
        prices_15m: &[f64],
    ) -> Array3<f64> {
        let ch1 = self.gaf.transform_gasf(prices_1m);
        let ch2 = self.gaf.transform_gasf(prices_5m);
        let ch3 = self.gaf.transform_gasf(prices_15m);

        let ch1_norm = self.normalize_channel(&ch1);
        let ch2_norm = self.normalize_channel(&ch2);
        let ch3_norm = self.normalize_channel(&ch3);

        self.stack_channels(&ch1_norm, &ch2_norm, &ch3_norm)
    }

    /// Create single-channel GASF image as grayscale RGB
    pub fn create_gasf_rgb(&self, prices: &[f64]) -> Array3<f64> {
        let gasf = self.gaf.transform_gasf(prices);
        let gasf_norm = self.normalize_channel(&gasf);

        self.stack_channels(&gasf_norm, &gasf_norm, &gasf_norm)
    }

    /// Normalize a channel to [0, 1] range
    fn normalize_channel(&self, channel: &Array2<f64>) -> Array2<f64> {
        let min_val = channel.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = channel.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max_val - min_val;
        if range < 1e-10 {
            return Array2::zeros(channel.dim());
        }

        channel.mapv(|x| (x - min_val) / range)
    }

    /// Stack three 2D arrays into a 3D RGB array
    fn stack_channels(
        &self,
        red: &Array2<f64>,
        green: &Array2<f64>,
        blue: &Array2<f64>,
    ) -> Array3<f64> {
        let (h, w) = red.dim();
        let mut stacked = Array3::zeros((h, w, 3));

        for i in 0..h {
            for j in 0..w {
                stacked[[i, j, 0]] = red[[i, j]];
                stacked[[i, j, 1]] = green[[i, j]];
                stacked[[i, j, 2]] = blue[[i, j]];
            }
        }

        stacked
    }

    /// Convert image to bytes for saving
    pub fn to_bytes(&self, image: &Array3<f64>) -> Vec<u8> {
        let (h, w, c) = image.dim();
        let mut bytes = Vec::with_capacity(h * w * c);

        for i in 0..h {
            for j in 0..w {
                for k in 0..c {
                    let value = (image[[i, j, k]] * 255.0).clamp(0.0, 255.0) as u8;
                    bytes.push(value);
                }
            }
        }

        bytes
    }
}

impl Default for ImageStackGenerator {
    fn default() -> Self {
        Self::new(224)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_transform_stack() {
        let prices = vec![100.0, 105.0, 103.0, 110.0, 108.0];
        let generator = ImageStackGenerator::new(64);

        let image = generator.create_multi_transform_stack(&prices);

        assert_eq!(image.dim(), (64, 64, 3));

        // Check values are in [0, 1]
        for &val in image.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_timeframe_stack() {
        let prices_1m = vec![100.0, 105.0, 103.0, 110.0, 108.0];
        let prices_5m = vec![100.0, 106.0, 108.0, 107.0];
        let prices_15m = vec![100.0, 107.0, 109.0];

        let generator = ImageStackGenerator::new(64);
        let image = generator.create_timeframe_stack(&prices_1m, &prices_5m, &prices_15m);

        assert_eq!(image.dim(), (64, 64, 3));
    }

    #[test]
    fn test_to_bytes() {
        let generator = ImageStackGenerator::new(32);
        let prices = vec![100.0, 105.0, 103.0, 110.0, 108.0];
        let image = generator.create_multi_transform_stack(&prices);

        let bytes = generator.to_bytes(&image);

        assert_eq!(bytes.len(), 32 * 32 * 3);
    }
}
