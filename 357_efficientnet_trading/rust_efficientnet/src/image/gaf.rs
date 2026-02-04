//! Gramian Angular Field (GAF) transformation
//!
//! Converts time series data to polar coordinates and creates angular field images.

use ndarray::{Array1, Array2};

/// GAF field type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GAFType {
    /// Gramian Angular Summation Field
    Summation,
    /// Gramian Angular Difference Field
    Difference,
}

/// Gramian Angular Field transformer
#[derive(Debug, Clone)]
pub struct GAFTransformer {
    /// Output image size
    pub image_size: usize,
    /// Field type (summation or difference)
    pub field_type: GAFType,
}

impl GAFTransformer {
    /// Create a new GAF transformer with default summation field
    pub fn new(image_size: usize) -> Self {
        Self {
            image_size,
            field_type: GAFType::Summation,
        }
    }

    /// Create a GASF (summation) transformer
    pub fn summation(image_size: usize) -> Self {
        Self {
            image_size,
            field_type: GAFType::Summation,
        }
    }

    /// Create a GADF (difference) transformer
    pub fn difference(image_size: usize) -> Self {
        Self {
            image_size,
            field_type: GAFType::Difference,
        }
    }

    /// Transform time series to GAF image
    ///
    /// # Arguments
    /// * `series` - Input time series (e.g., close prices)
    ///
    /// # Returns
    /// 2D array of shape (image_size, image_size)
    pub fn transform(&self, series: &[f64]) -> Array2<f64> {
        // Normalize to [-1, 1]
        let scaled = self.normalize(series);

        // Convert to angular representation
        let phi = self.to_angular(&scaled);

        // Create GAF matrix
        let gaf = self.create_gaf(&phi);

        // Resize to target size
        self.resize(&gaf)
    }

    /// Transform to GASF
    pub fn transform_gasf(&self, series: &[f64]) -> Array2<f64> {
        let mut transformer = self.clone();
        transformer.field_type = GAFType::Summation;
        transformer.transform(series)
    }

    /// Transform to GADF
    pub fn transform_gadf(&self, series: &[f64]) -> Array2<f64> {
        let mut transformer = self.clone();
        transformer.field_type = GAFType::Difference;
        transformer.transform(series)
    }

    /// Normalize series to [-1, 1]
    fn normalize(&self, series: &[f64]) -> Vec<f64> {
        if series.is_empty() {
            return Vec::new();
        }

        let min_val = series.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max_val - min_val;
        if range < 1e-10 {
            return vec![0.0; series.len()];
        }

        series
            .iter()
            .map(|&x| {
                let scaled = 2.0 * (x - min_val) / range - 1.0;
                scaled.clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Convert normalized values to angular representation
    fn to_angular(&self, scaled: &[f64]) -> Array1<f64> {
        Array1::from_vec(
            scaled
                .iter()
                .map(|&x| x.clamp(-1.0, 1.0).acos())
                .collect(),
        )
    }

    /// Create the GAF matrix
    fn create_gaf(&self, phi: &Array1<f64>) -> Array2<f64> {
        let n = phi.len();
        let mut gaf = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                gaf[[i, j]] = match self.field_type {
                    GAFType::Summation => (phi[i] + phi[j]).cos(),
                    GAFType::Difference => (phi[i] - phi[j]).sin(),
                };
            }
        }

        gaf
    }

    /// Resize image using bilinear interpolation
    fn resize(&self, image: &Array2<f64>) -> Array2<f64> {
        let (h, w) = image.dim();

        if h == self.image_size && w == self.image_size {
            return image.clone();
        }

        let mut resized = Array2::zeros((self.image_size, self.image_size));

        for i in 0..self.image_size {
            for j in 0..self.image_size {
                let orig_i = if self.image_size > 1 {
                    i as f64 * (h - 1) as f64 / (self.image_size - 1) as f64
                } else {
                    0.0
                };
                let orig_j = if self.image_size > 1 {
                    j as f64 * (w - 1) as f64 / (self.image_size - 1) as f64
                } else {
                    0.0
                };

                // Bilinear interpolation
                let i0 = orig_i.floor() as usize;
                let i1 = (i0 + 1).min(h - 1);
                let j0 = orig_j.floor() as usize;
                let j1 = (j0 + 1).min(w - 1);

                let di = orig_i - i0 as f64;
                let dj = orig_j - j0 as f64;

                resized[[i, j]] = image[[i0, j0]] * (1.0 - di) * (1.0 - dj)
                    + image[[i1, j0]] * di * (1.0 - dj)
                    + image[[i0, j1]] * (1.0 - di) * dj
                    + image[[i1, j1]] * di * dj;
            }
        }

        resized
    }
}

impl Default for GAFTransformer {
    fn default() -> Self {
        Self::new(224)
    }
}

/// Normalize image to [0, 1] range
pub fn normalize_image(image: &Array2<f64>) -> Array2<f64> {
    let min_val = image.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let range = max_val - min_val;
    if range < 1e-10 {
        return Array2::zeros(image.dim());
    }

    image.mapv(|x| (x - min_val) / range)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaf_transform() {
        let prices = vec![100.0, 105.0, 103.0, 110.0, 108.0];
        let transformer = GAFTransformer::new(64);

        let gaf = transformer.transform(&prices);

        assert_eq!(gaf.dim(), (64, 64));
    }

    #[test]
    fn test_gasf_vs_gadf() {
        let prices = vec![100.0, 105.0, 103.0, 110.0, 108.0];
        let transformer = GAFTransformer::new(32);

        let gasf = transformer.transform_gasf(&prices);
        let gadf = transformer.transform_gadf(&prices);

        // GASF and GADF should produce different results
        assert_ne!(gasf[[0, 0]], gadf[[0, 0]]);
    }

    #[test]
    fn test_normalize() {
        let transformer = GAFTransformer::new(64);
        let series = vec![0.0, 50.0, 100.0];

        let normalized = transformer.normalize(&series);

        assert_eq!(normalized.len(), 3);
        assert!((normalized[0] - (-1.0)).abs() < 1e-10);
        assert!((normalized[1] - 0.0).abs() < 1e-10);
        assert!((normalized[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_series() {
        let transformer = GAFTransformer::new(64);
        let gaf = transformer.transform(&[]);

        assert_eq!(gaf.dim(), (64, 64));
    }
}
