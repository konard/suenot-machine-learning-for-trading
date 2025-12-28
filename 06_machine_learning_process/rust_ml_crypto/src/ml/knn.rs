//! K-Nearest Neighbors classifier and regressor
//!
//! A simple yet effective ML algorithm that classifies/predicts
//! based on the k closest training examples.

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Distance metric for KNN
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2)
    Euclidean,
    /// Manhattan distance (L1)
    Manhattan,
    /// Minkowski distance with parameter p
    Minkowski(f64),
}

/// KNN Classifier
#[derive(Debug, Clone)]
pub struct KNNClassifier {
    k: usize,
    metric: DistanceMetric,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    weights: String, // "uniform" or "distance"
}

impl KNNClassifier {
    /// Create a new KNN classifier
    ///
    /// # Arguments
    /// * `k` - Number of neighbors to consider
    pub fn new(k: usize) -> Self {
        Self {
            k,
            metric: DistanceMetric::Euclidean,
            x_train: None,
            y_train: None,
            weights: "uniform".to_string(),
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the weighting scheme
    /// - "uniform": All neighbors have equal weight
    /// - "distance": Weight by inverse of distance
    pub fn with_weights(mut self, weights: &str) -> Self {
        self.weights = weights.to_string();
        self
    }

    /// Fit the classifier to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        assert_eq!(x.nrows(), y.len(), "X and y must have same number of samples");
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
    }

    /// Predict class labels for samples
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");

        let mut predictions = Vec::with_capacity(x.nrows());

        for sample_idx in 0..x.nrows() {
            let sample = x.row(sample_idx);

            // Calculate distances to all training points
            let mut distances: Vec<(usize, f64)> = x_train
                .rows()
                .into_iter()
                .enumerate()
                .map(|(i, train_sample)| {
                    let dist = self.calculate_distance(&sample.to_vec(), &train_sample.to_vec());
                    (i, dist)
                })
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Get k nearest neighbors
            let neighbors: Vec<(usize, f64)> = distances.into_iter().take(self.k).collect();

            // Weighted voting
            let prediction = if self.weights == "distance" {
                self.weighted_vote(&neighbors, y_train)
            } else {
                self.uniform_vote(&neighbors, y_train)
            };

            predictions.push(prediction);
        }

        Array1::from_vec(predictions)
    }

    /// Predict probabilities for each class
    pub fn predict_proba(&self, x: &Array2<f64>) -> Vec<HashMap<i64, f64>> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");

        let mut probas = Vec::with_capacity(x.nrows());

        for sample_idx in 0..x.nrows() {
            let sample = x.row(sample_idx);

            // Calculate distances
            let mut distances: Vec<(usize, f64)> = x_train
                .rows()
                .into_iter()
                .enumerate()
                .map(|(i, train_sample)| {
                    let dist = self.calculate_distance(&sample.to_vec(), &train_sample.to_vec());
                    (i, dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let neighbors: Vec<(usize, f64)> = distances.into_iter().take(self.k).collect();

            // Calculate probabilities
            let mut class_weights: HashMap<i64, f64> = HashMap::new();
            let mut total_weight = 0.0;

            for (idx, dist) in &neighbors {
                let label = y_train[*idx] as i64;
                let weight = if self.weights == "distance" && *dist > 0.0 {
                    1.0 / dist
                } else {
                    1.0
                };
                *class_weights.entry(label).or_insert(0.0) += weight;
                total_weight += weight;
            }

            // Normalize to probabilities
            if total_weight > 0.0 {
                for weight in class_weights.values_mut() {
                    *weight /= total_weight;
                }
            }

            probas.push(class_weights);
        }

        probas
    }

    /// Calculate distance between two points
    fn calculate_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
            DistanceMetric::Manhattan => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum()
            }
            DistanceMetric::Minkowski(p) => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs().powf(p))
                    .sum::<f64>()
                    .powf(1.0 / p)
            }
        }
    }

    /// Uniform voting (majority vote)
    fn uniform_vote(&self, neighbors: &[(usize, f64)], y_train: &Array1<f64>) -> f64 {
        let mut votes: HashMap<i64, usize> = HashMap::new();

        for (idx, _) in neighbors {
            let label = y_train[*idx] as i64;
            *votes.entry(label).or_insert(0) += 1;
        }

        votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label as f64)
            .unwrap_or(0.0)
    }

    /// Distance-weighted voting
    fn weighted_vote(&self, neighbors: &[(usize, f64)], y_train: &Array1<f64>) -> f64 {
        let mut votes: HashMap<i64, f64> = HashMap::new();

        for (idx, dist) in neighbors {
            let label = y_train[*idx] as i64;
            let weight = if *dist > 0.0 { 1.0 / dist } else { 1e10 };
            *votes.entry(label).or_insert(0.0) += weight;
        }

        votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label as f64)
            .unwrap_or(0.0)
    }

    /// Get the number of neighbors
    pub fn get_k(&self) -> usize {
        self.k
    }

    /// Set the number of neighbors
    pub fn set_k(&mut self, k: usize) {
        self.k = k;
    }
}

/// KNN Regressor
#[derive(Debug, Clone)]
pub struct KNNRegressor {
    k: usize,
    metric: DistanceMetric,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    weights: String,
}

impl KNNRegressor {
    /// Create a new KNN regressor
    pub fn new(k: usize) -> Self {
        Self {
            k,
            metric: DistanceMetric::Euclidean,
            x_train: None,
            y_train: None,
            weights: "uniform".to_string(),
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the weighting scheme
    pub fn with_weights(mut self, weights: &str) -> Self {
        self.weights = weights.to_string();
        self
    }

    /// Fit the regressor to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        assert_eq!(x.nrows(), y.len(), "X and y must have same number of samples");
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
    }

    /// Predict values for samples
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");

        let mut predictions = Vec::with_capacity(x.nrows());

        for sample_idx in 0..x.nrows() {
            let sample = x.row(sample_idx);

            // Calculate distances
            let mut distances: Vec<(usize, f64)> = x_train
                .rows()
                .into_iter()
                .enumerate()
                .map(|(i, train_sample)| {
                    let dist = self.calculate_distance(&sample.to_vec(), &train_sample.to_vec());
                    (i, dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let neighbors: Vec<(usize, f64)> = distances.into_iter().take(self.k).collect();

            // Weighted average
            let prediction = if self.weights == "distance" {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                for (idx, dist) in &neighbors {
                    let weight = if *dist > 0.0 { 1.0 / dist } else { 1e10 };
                    sum += y_train[*idx] * weight;
                    weight_sum += weight;
                }
                if weight_sum > 0.0 { sum / weight_sum } else { 0.0 }
            } else {
                neighbors.iter().map(|(idx, _)| y_train[*idx]).sum::<f64>() / self.k as f64
            };

            predictions.push(prediction);
        }

        Array1::from_vec(predictions)
    }

    /// Calculate distance between two points
    fn calculate_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
            DistanceMetric::Manhattan => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum()
            }
            DistanceMetric::Minkowski(p) => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs().powf(p))
                    .sum::<f64>()
                    .powf(1.0 / p)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_knn_classifier() {
        let x_train = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0]
        ];
        let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut knn = KNNClassifier::new(3);
        knn.fit(&x_train, &y_train);

        let x_test = array![[1.5, 1.5], [5.5, 5.5]];
        let predictions = knn.predict(&x_test);

        assert_eq!(predictions[0], 0.0);
        assert_eq!(predictions[1], 1.0);
    }

    #[test]
    fn test_knn_regressor() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_train = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let mut knn = KNNRegressor::new(2);
        knn.fit(&x_train, &y_train);

        let x_test = array![[2.5], [3.5]];
        let predictions = knn.predict(&x_test);

        // Should predict average of 2 nearest neighbors
        assert!((predictions[0] - 5.0).abs() < 1e-10); // avg of 4 and 6
        assert!((predictions[1] - 7.0).abs() < 1e-10); // avg of 6 and 8
    }
}
