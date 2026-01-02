//! Recurrence Plot renderer
//!
//! Visualizes phase space trajectories of time series data.

use crate::imaging::{colors, ImageConfig};
use image::RgbImage;

/// Recurrence plot renderer
pub struct RecurrencePlot {
    config: ImageConfig,
    embedding_dim: usize,
    time_delay: usize,
    threshold: Option<f64>, // If None, use adaptive threshold
}

impl RecurrencePlot {
    /// Create a new recurrence plot renderer
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            config: ImageConfig {
                width,
                height,
                ..Default::default()
            },
            embedding_dim: 3,
            time_delay: 1,
            threshold: None,
        }
    }

    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim.max(1);
        self
    }

    /// Set time delay
    pub fn time_delay(mut self, delay: usize) -> Self {
        self.time_delay = delay.max(1);
        self
    }

    /// Set threshold for recurrence
    pub fn threshold(mut self, thresh: f64) -> Self {
        self.threshold = Some(thresh);
        self
    }

    /// Render time series to recurrence plot
    pub fn render(&self, series: &[f64]) -> RgbImage {
        let mut img = RgbImage::from_pixel(
            self.config.width,
            self.config.height,
            self.config.background,
        );

        if series.len() < self.embedding_dim * self.time_delay {
            return img;
        }

        // Create embedded vectors
        let embedded = self.embed(series);
        let n = embedded.len();

        if n == 0 {
            return img;
        }

        // Compute distance matrix
        let distances = self.compute_distances(&embedded);

        // Determine threshold
        let threshold = self.threshold.unwrap_or_else(|| self.adaptive_threshold(&distances));

        // Render recurrence plot
        let scale_x = n as f64 / self.config.width as f64;
        let scale_y = n as f64 / self.config.height as f64;

        for py in 0..self.config.height {
            for px in 0..self.config.width {
                let i = (py as f64 * scale_y) as usize;
                let j = (px as f64 * scale_x) as usize;

                let i = i.min(n - 1);
                let j = j.min(n - 1);

                if distances[i][j] <= threshold {
                    img.put_pixel(px, py, colors::WHITE);
                }
            }
        }

        img
    }

    /// Render with gradient coloring based on distance
    pub fn render_gradient(&self, series: &[f64]) -> RgbImage {
        let mut img = RgbImage::from_pixel(
            self.config.width,
            self.config.height,
            self.config.background,
        );

        if series.len() < self.embedding_dim * self.time_delay {
            return img;
        }

        let embedded = self.embed(series);
        let n = embedded.len();

        if n == 0 {
            return img;
        }

        let distances = self.compute_distances(&embedded);

        // Find max distance for normalization
        let max_dist: f64 = distances
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0, f64::max);

        if max_dist == 0.0 {
            return img;
        }

        let scale_x = n as f64 / self.config.width as f64;
        let scale_y = n as f64 / self.config.height as f64;

        for py in 0..self.config.height {
            for px in 0..self.config.width {
                let i = (py as f64 * scale_y) as usize;
                let j = (px as f64 * scale_x) as usize;

                let i = i.min(n - 1);
                let j = j.min(n - 1);

                // Color based on distance (closer = brighter)
                let normalized = 1.0 - (distances[i][j] / max_dist);
                let intensity = (normalized * 255.0) as u8;

                img.put_pixel(px, py, image::Rgb([intensity, intensity, intensity]));
            }
        }

        img
    }

    fn embed(&self, series: &[f64]) -> Vec<Vec<f64>> {
        let n = series.len();
        let embed_len = n - (self.embedding_dim - 1) * self.time_delay;

        if embed_len == 0 {
            return Vec::new();
        }

        (0..embed_len)
            .map(|i| {
                (0..self.embedding_dim)
                    .map(|d| series[i + d * self.time_delay])
                    .collect()
            })
            .collect()
    }

    fn compute_distances(&self, embedded: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = embedded.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                let dist = self.euclidean_distance(&embedded[i], &embedded[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn adaptive_threshold(&self, distances: &[Vec<f64>]) -> f64 {
        // Use 10th percentile of distances
        let mut all_distances: Vec<f64> = distances
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .filter(|&d| d > 0.0)
            .collect();

        all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if all_distances.is_empty() {
            return 0.0;
        }

        let percentile_idx = (all_distances.len() as f64 * 0.1) as usize;
        all_distances[percentile_idx]
    }
}

impl Default for RecurrencePlot {
    fn default() -> Self {
        Self::new(224, 224)
    }
}

/// Compute recurrence quantification analysis metrics
pub struct RQAMetrics {
    /// Recurrence rate
    pub recurrence_rate: f64,
    /// Determinism
    pub determinism: f64,
    /// Average diagonal line length
    pub avg_diagonal_length: f64,
    /// Laminarity
    pub laminarity: f64,
}

impl RQAMetrics {
    /// Compute RQA metrics from a recurrence matrix
    pub fn from_recurrence_matrix(matrix: &[Vec<bool>]) -> Self {
        let n = matrix.len();
        if n == 0 {
            return Self {
                recurrence_rate: 0.0,
                determinism: 0.0,
                avg_diagonal_length: 0.0,
                laminarity: 0.0,
            };
        }

        // Count recurrence points
        let mut recurrence_count = 0;
        for row in matrix {
            for &val in row {
                if val {
                    recurrence_count += 1;
                }
            }
        }

        let recurrence_rate = recurrence_count as f64 / (n * n) as f64;

        // Count diagonal lines
        let mut diagonal_lengths: Vec<usize> = Vec::new();
        for offset in (1 - n as i32)..(n as i32) {
            let mut current_length = 0;
            for i in 0..n {
                let j = (i as i32 + offset) as usize;
                if j < n && matrix[i][j] {
                    current_length += 1;
                } else if current_length > 1 {
                    diagonal_lengths.push(current_length);
                    current_length = 0;
                } else {
                    current_length = 0;
                }
            }
            if current_length > 1 {
                diagonal_lengths.push(current_length);
            }
        }

        let determinism = if recurrence_count > 0 {
            let diagonal_points: usize = diagonal_lengths.iter().sum();
            diagonal_points as f64 / recurrence_count as f64
        } else {
            0.0
        };

        let avg_diagonal_length = if !diagonal_lengths.is_empty() {
            diagonal_lengths.iter().sum::<usize>() as f64 / diagonal_lengths.len() as f64
        } else {
            0.0
        };

        // Count vertical lines for laminarity
        let mut vertical_lengths: Vec<usize> = Vec::new();
        for j in 0..n {
            let mut current_length = 0;
            for i in 0..n {
                if matrix[i][j] {
                    current_length += 1;
                } else if current_length > 1 {
                    vertical_lengths.push(current_length);
                    current_length = 0;
                } else {
                    current_length = 0;
                }
            }
            if current_length > 1 {
                vertical_lengths.push(current_length);
            }
        }

        let laminarity = if recurrence_count > 0 {
            let vertical_points: usize = vertical_lengths.iter().sum();
            vertical_points as f64 / recurrence_count as f64
        } else {
            0.0
        };

        Self {
            recurrence_rate,
            determinism,
            avg_diagonal_length,
            laminarity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recurrence_plot() {
        let renderer = RecurrencePlot::new(64, 64);
        let series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let img = renderer.render(&series);

        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_gradient_render() {
        let renderer = RecurrencePlot::new(64, 64);
        let series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let img = renderer.render_gradient(&series);

        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_short_series() {
        let renderer = RecurrencePlot::new(64, 64).embedding_dim(5);
        let series = vec![1.0, 2.0, 3.0]; // Too short for embedding
        let img = renderer.render(&series);

        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }
}
