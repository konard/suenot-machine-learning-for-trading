//! Markov Transition Field (MTF) transformation
//!
//! Encodes transition dynamics between quantile bins.

use ndarray::Array2;

/// Markov Transition Field transformer
#[derive(Debug, Clone)]
pub struct MTFTransformer {
    /// Number of quantile bins
    pub n_bins: usize,
    /// Output image size
    pub image_size: usize,
}

impl MTFTransformer {
    /// Create a new MTF transformer
    pub fn new(n_bins: usize, image_size: usize) -> Self {
        Self { n_bins, image_size }
    }

    /// Transform time series to MTF image
    ///
    /// # Arguments
    /// * `series` - Input time series
    ///
    /// # Returns
    /// 2D array of shape (image_size, image_size)
    pub fn transform(&self, series: &[f64]) -> Array2<f64> {
        if series.is_empty() {
            return Array2::zeros((self.image_size, self.image_size));
        }

        // Discretize into quantile bins
        let digitized = self.discretize(series);

        // Build transition matrix
        let trans_matrix = self.build_transition_matrix(&digitized);

        // Create MTF
        let mtf = self.create_mtf(&digitized, &trans_matrix);

        // Resize to target size
        self.resize(&mtf)
    }

    /// Discretize series into quantile bins
    fn discretize(&self, series: &[f64]) -> Vec<usize> {
        // Calculate quantile boundaries
        let mut sorted = series.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut boundaries = Vec::with_capacity(self.n_bins + 1);
        for i in 0..=self.n_bins {
            let idx = (i * (sorted.len() - 1)) / self.n_bins;
            boundaries.push(sorted[idx]);
        }

        // Assign each value to a bin
        series
            .iter()
            .map(|&x| {
                let mut bin = 0;
                for i in 1..boundaries.len() {
                    if x <= boundaries[i] {
                        bin = i - 1;
                        break;
                    }
                    bin = self.n_bins - 1;
                }
                bin.min(self.n_bins - 1)
            })
            .collect()
    }

    /// Build transition probability matrix
    fn build_transition_matrix(&self, digitized: &[usize]) -> Array2<f64> {
        let mut trans_matrix = Array2::zeros((self.n_bins, self.n_bins));

        // Count transitions
        for i in 0..digitized.len().saturating_sub(1) {
            let from = digitized[i];
            let to = digitized[i + 1];
            trans_matrix[[from, to]] += 1.0;
        }

        // Normalize rows
        for i in 0..self.n_bins {
            let row_sum: f64 = trans_matrix.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..self.n_bins {
                    trans_matrix[[i, j]] /= row_sum;
                }
            }
        }

        trans_matrix
    }

    /// Create the MTF matrix
    fn create_mtf(&self, digitized: &[usize], trans_matrix: &Array2<f64>) -> Array2<f64> {
        let n = digitized.len();
        let mut mtf = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                mtf[[i, j]] = trans_matrix[[digitized[i], digitized[j]]];
            }
        }

        mtf
    }

    /// Resize image using bilinear interpolation
    fn resize(&self, image: &Array2<f64>) -> Array2<f64> {
        let (h, w) = image.dim();

        if h == self.image_size && w == self.image_size {
            return image.clone();
        }

        if h == 0 || w == 0 {
            return Array2::zeros((self.image_size, self.image_size));
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

impl Default for MTFTransformer {
    fn default() -> Self {
        Self::new(8, 224)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtf_transform() {
        let prices = vec![100.0, 105.0, 103.0, 110.0, 108.0, 112.0, 115.0, 113.0];
        let transformer = MTFTransformer::new(4, 64);

        let mtf = transformer.transform(&prices);

        assert_eq!(mtf.dim(), (64, 64));
    }

    #[test]
    fn test_discretize() {
        let transformer = MTFTransformer::new(4, 64);
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let digitized = transformer.discretize(&series);

        assert_eq!(digitized.len(), 8);
        // First values should be in lower bins
        assert!(digitized[0] <= digitized[7]);
    }

    #[test]
    fn test_empty_series() {
        let transformer = MTFTransformer::new(8, 64);
        let mtf = transformer.transform(&[]);

        assert_eq!(mtf.dim(), (64, 64));
    }
}
