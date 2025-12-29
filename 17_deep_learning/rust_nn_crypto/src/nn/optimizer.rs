//! Optimization Algorithms
//!
//! Implements various optimization algorithms for training neural networks:
//! - SGD (Stochastic Gradient Descent)
//! - SGD with Momentum
//! - Adam (Adaptive Moment Estimation)

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Optimizer trait for weight updates
pub trait Optimizer: Send + Sync {
    /// Update weights given gradients
    fn update_weights(&mut self, weights: &mut Array2<f64>, gradients: &Array2<f64>);

    /// Update biases given gradients
    fn update_biases(&mut self, biases: &mut Array1<f64>, gradients: &Array1<f64>);

    /// Reset optimizer state (for new training run)
    fn reset(&mut self);

    /// Clone the optimizer for each layer
    fn clone_box(&self) -> Box<dyn Optimizer>;
}

/// Stochastic Gradient Descent with optional momentum
#[derive(Clone, Serialize, Deserialize)]
pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
    #[serde(skip)]
    velocity_w: Option<Array2<f64>>,
    #[serde(skip)]
    velocity_b: Option<Array1<f64>>,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            velocity_w: None,
            velocity_b: None,
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for SGD {
    fn update_weights(&mut self, weights: &mut Array2<f64>, gradients: &Array2<f64>) {
        if self.momentum > 0.0 {
            let v = self.velocity_w.get_or_insert_with(|| Array2::zeros(weights.dim()));
            *v = &*v * self.momentum - gradients * self.learning_rate;
            *weights = &*weights + &*v;
        } else {
            *weights = &*weights - &(gradients * self.learning_rate);
        }
    }

    fn update_biases(&mut self, biases: &mut Array1<f64>, gradients: &Array1<f64>) {
        if self.momentum > 0.0 {
            let v = self.velocity_b.get_or_insert_with(|| Array1::zeros(biases.len()));
            *v = &*v * self.momentum - gradients * self.learning_rate;
            *biases = &*biases + &*v;
        } else {
            *biases = &*biases - &(gradients * self.learning_rate);
        }
    }

    fn reset(&mut self) {
        self.velocity_w = None;
        self.velocity_b = None;
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
#[derive(Clone, Serialize, Deserialize)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    #[serde(skip)]
    t: usize,
    #[serde(skip)]
    m_w: Option<Array2<f64>>,
    #[serde(skip)]
    v_w: Option<Array2<f64>>,
    #[serde(skip)]
    m_b: Option<Array1<f64>>,
    #[serde(skip)]
    v_b: Option<Array1<f64>>,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m_w: None,
            v_w: None,
            m_b: None,
            v_b: None,
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for Adam {
    fn update_weights(&mut self, weights: &mut Array2<f64>, gradients: &Array2<f64>) {
        self.t += 1;

        // Initialize moments if needed
        let m = self.m_w.get_or_insert_with(|| Array2::zeros(weights.dim()));
        let v = self.v_w.get_or_insert_with(|| Array2::zeros(weights.dim()));

        // Update biased first moment estimate
        *m = &*m * self.beta1 + gradients * (1.0 - self.beta1);

        // Update biased second moment estimate
        *v = &*v * self.beta2 + &(gradients * gradients) * (1.0 - self.beta2);

        // Compute bias-corrected estimates
        let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

        // Update weights
        *weights = &*weights - &(&m_hat * self.learning_rate / &(v_hat.mapv(f64::sqrt) + self.epsilon));
    }

    fn update_biases(&mut self, biases: &mut Array1<f64>, gradients: &Array1<f64>) {
        // Initialize moments if needed
        let m = self.m_b.get_or_insert_with(|| Array1::zeros(biases.len()));
        let v = self.v_b.get_or_insert_with(|| Array1::zeros(biases.len()));

        // Update biased first moment estimate
        *m = &*m * self.beta1 + gradients * (1.0 - self.beta1);

        // Update biased second moment estimate
        *v = &*v * self.beta2 + &(gradients * gradients) * (1.0 - self.beta2);

        // Compute bias-corrected estimates
        let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

        // Update biases
        *biases = &*biases - &(&m_hat * self.learning_rate / &(v_hat.mapv(f64::sqrt) + self.epsilon));
    }

    fn reset(&mut self) {
        self.t = 0;
        self.m_w = None;
        self.v_w = None;
        self.m_b = None;
        self.v_b = None;
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_update() {
        let mut optimizer = SGD::new(0.01);
        let mut weights = Array2::ones((3, 2));
        let gradients = Array2::ones((3, 2));
        optimizer.update_weights(&mut weights, &gradients);

        assert!((weights[[0, 0]] - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_adam_update() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Array2::ones((3, 2));
        let gradients = Array2::ones((3, 2));

        // Multiple updates
        for _ in 0..10 {
            optimizer.update_weights(&mut weights, &gradients);
        }

        // Weights should have decreased
        assert!(weights[[0, 0]] < 1.0);
    }
}
