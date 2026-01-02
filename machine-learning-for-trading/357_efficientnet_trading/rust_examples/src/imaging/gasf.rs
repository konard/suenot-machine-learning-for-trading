//! Gramian Angular Summation Field (GASF) renderer
//!
//! Converts time series data into images using polar coordinate transformation.

use crate::imaging::{value_to_heatmap_color, ImageConfig};
use image::RgbImage;

/// GASF/GADF image renderer
pub struct GasfRenderer {
    config: ImageConfig,
    use_summation: bool, // true for GASF, false for GADF
}

impl GasfRenderer {
    /// Create a GASF renderer
    pub fn gasf(width: u32, height: u32) -> Self {
        Self {
            config: ImageConfig {
                width,
                height,
                ..Default::default()
            },
            use_summation: true,
        }
    }

    /// Create a GADF renderer
    pub fn gadf(width: u32, height: u32) -> Self {
        Self {
            config: ImageConfig {
                width,
                height,
                ..Default::default()
            },
            use_summation: false,
        }
    }

    /// Render time series to GASF/GADF image
    pub fn render(&self, series: &[f64]) -> RgbImage {
        let n = series.len();

        if n == 0 {
            return RgbImage::from_pixel(
                self.config.width,
                self.config.height,
                self.config.background,
            );
        }

        // Normalize series to [-1, 1]
        let normalized = self.normalize(series);

        // Convert to polar coordinates (arccos)
        let phi: Vec<f64> = normalized
            .iter()
            .map(|&x| x.clamp(-1.0, 1.0).acos())
            .collect();

        // Compute Gramian matrix
        let gramian = self.compute_gramian(&phi);

        // Render to image
        self.gramian_to_image(&gramian)
    }

    /// Render from candle close prices
    pub fn render_from_closes(&self, closes: &[f64]) -> RgbImage {
        self.render(closes)
    }

    fn normalize(&self, series: &[f64]) -> Vec<f64> {
        if series.is_empty() {
            return Vec::new();
        }

        let min = series.iter().cloned().fold(f64::MAX, f64::min);
        let max = series.iter().cloned().fold(f64::MIN, f64::max);
        let range = max - min;

        if range == 0.0 {
            return vec![0.0; series.len()];
        }

        // Normalize to [-1, 1]
        series
            .iter()
            .map(|&x| 2.0 * (x - min) / range - 1.0)
            .collect()
    }

    fn compute_gramian(&self, phi: &[f64]) -> Vec<Vec<f64>> {
        let n = phi.len();
        let mut gramian = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                gramian[i][j] = if self.use_summation {
                    // GASF: cos(φ_i + φ_j)
                    (phi[i] + phi[j]).cos()
                } else {
                    // GADF: sin(φ_i - φ_j)
                    (phi[i] - phi[j]).sin()
                };
            }
        }

        gramian
    }

    fn gramian_to_image(&self, gramian: &[Vec<f64>]) -> RgbImage {
        let n = gramian.len();
        let mut img = RgbImage::new(self.config.width, self.config.height);

        if n == 0 {
            return img;
        }

        // Find min/max for normalization
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;

        for row in gramian {
            for &val in row {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        let range = max_val - min_val;

        // Scale factors
        let scale_x = n as f64 / self.config.width as f64;
        let scale_y = n as f64 / self.config.height as f64;

        for py in 0..self.config.height {
            for px in 0..self.config.width {
                let i = (py as f64 * scale_y) as usize;
                let j = (px as f64 * scale_x) as usize;

                let i = i.min(n - 1);
                let j = j.min(n - 1);

                let value = if range > 0.0 {
                    (gramian[i][j] - min_val) / range
                } else {
                    0.5
                };

                let color = value_to_heatmap_color(value);
                img.put_pixel(px, py, color);
            }
        }

        img
    }
}

/// Compute GASF matrix directly (useful for feature extraction)
pub fn compute_gasf(series: &[f64]) -> Vec<Vec<f64>> {
    if series.is_empty() {
        return Vec::new();
    }

    // Normalize to [-1, 1]
    let min = series.iter().cloned().fold(f64::MAX, f64::min);
    let max = series.iter().cloned().fold(f64::MIN, f64::max);
    let range = max - min;

    let normalized: Vec<f64> = if range > 0.0 {
        series.iter().map(|&x| 2.0 * (x - min) / range - 1.0).collect()
    } else {
        vec![0.0; series.len()]
    };

    // Convert to polar coordinates
    let phi: Vec<f64> = normalized
        .iter()
        .map(|&x| x.clamp(-1.0, 1.0).acos())
        .collect();

    // Compute GASF matrix
    let n = phi.len();
    let mut gasf = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            gasf[i][j] = (phi[i] + phi[j]).cos();
        }
    }

    gasf
}

/// Compute GADF matrix directly
pub fn compute_gadf(series: &[f64]) -> Vec<Vec<f64>> {
    if series.is_empty() {
        return Vec::new();
    }

    // Normalize to [-1, 1]
    let min = series.iter().cloned().fold(f64::MAX, f64::min);
    let max = series.iter().cloned().fold(f64::MIN, f64::max);
    let range = max - min;

    let normalized: Vec<f64> = if range > 0.0 {
        series.iter().map(|&x| 2.0 * (x - min) / range - 1.0).collect()
    } else {
        vec![0.0; series.len()]
    };

    // Convert to polar coordinates
    let phi: Vec<f64> = normalized
        .iter()
        .map(|&x| x.clamp(-1.0, 1.0).acos())
        .collect();

    // Compute GADF matrix
    let n = phi.len();
    let mut gadf = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            gadf[i][j] = (phi[i] - phi[j]).sin();
        }
    }

    gadf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gasf_render() {
        let renderer = GasfRenderer::gasf(64, 64);
        let series = vec![1.0, 2.0, 3.0, 2.5, 3.5, 4.0, 3.8];
        let img = renderer.render(&series);

        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_gadf_render() {
        let renderer = GasfRenderer::gadf(64, 64);
        let series = vec![1.0, 2.0, 3.0, 2.5, 3.5, 4.0, 3.8];
        let img = renderer.render(&series);

        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_compute_gasf() {
        let series = vec![0.0, 0.5, 1.0];
        let gasf = compute_gasf(&series);

        assert_eq!(gasf.len(), 3);
        assert_eq!(gasf[0].len(), 3);

        // Diagonal should be cos(2*phi_i)
        // For normalized 0: phi = acos(-1) = pi, cos(2*pi) = 1
        assert!((gasf[0][0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_series() {
        let renderer = GasfRenderer::gasf(64, 64);
        let img = renderer.render(&[]);

        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }
}
