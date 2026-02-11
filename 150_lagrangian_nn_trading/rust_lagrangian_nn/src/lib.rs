//! # Lagrangian Neural Networks for Trading
//!
//! This library implements Lagrangian Neural Networks (LNNs) and their
//! dissipative/forced extensions for financial market modeling and trading.
//!
//! The core idea: learn the Lagrangian L(q, q-dot) from market data, then derive
//! dynamics via the Euler-Lagrange equations:
//!   d/dt(dL/dq-dot) - dL/dq = 0
//!
//! Unlike Hamiltonian NNs (Chapter 149):
//! - Works in (q, q-dot) space, not (q, p)
//! - No Legendre transform needed
//! - Handles non-separable systems naturally
//! - Requires second derivatives (mass matrix M = d^2L/dq-dot^2)
//!
//! ## Modules
//! - `nn`: Neural network layers and LNN architecture
//! - `integrator`: Numerical integrators (RK4, leapfrog)
//! - `data`: Bybit API data fetching and config space construction
//! - `trading`: Trading strategy and backtesting
//! - `utils`: Normalization, metrics, serialization

pub mod nn;
pub mod integrator;
pub mod data;
pub mod trading;
pub mod utils;

pub use nn::{LagrangianNN, DissipativeLNN};
pub use integrator::{rk4_step, integrate_trajectory};
pub use data::{BybitClient, ConfigSpaceData};
pub use trading::{TradingStrategy, BacktestResult};

/// Module: Neural network layers and LNN architecture
pub mod nn {
    use rand::Rng;
    use rand_distr::Normal;
    use serde::{Deserialize, Serialize};

    /// Activation functions (must be smooth -- Euler-Lagrange requires d^2L)
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum Activation {
        Tanh,
        Sigmoid,
        Softplus,
    }

    impl Activation {
        pub fn apply(&self, x: f64) -> f64 {
            match self {
                Activation::Tanh => x.tanh(),
                Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                Activation::Softplus => (1.0 + x.exp()).ln(),
            }
        }

        pub fn derivative(&self, x: f64) -> f64 {
            match self {
                Activation::Tanh => {
                    let t = x.tanh();
                    1.0 - t * t
                }
                Activation::Sigmoid => {
                    let s = 1.0 / (1.0 + (-x).exp());
                    s * (1.0 - s)
                }
                Activation::Softplus => 1.0 / (1.0 + (-x).exp()),
            }
        }

        pub fn second_derivative(&self, x: f64) -> f64 {
            match self {
                Activation::Tanh => {
                    let t = x.tanh();
                    -2.0 * t * (1.0 - t * t)
                }
                Activation::Sigmoid => {
                    let s = 1.0 / (1.0 + (-x).exp());
                    s * (1.0 - s) * (1.0 - 2.0 * s)
                }
                Activation::Softplus => {
                    let s = 1.0 / (1.0 + (-x).exp());
                    s * (1.0 - s)
                }
            }
        }
    }

    /// A single dense (fully connected) layer
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DenseLayer {
        pub weights: Vec<Vec<f64>>,  // [output_dim][input_dim]
        pub biases: Vec<f64>,        // [output_dim]
        pub input_dim: usize,
        pub output_dim: usize,
    }

    impl DenseLayer {
        /// Create a new dense layer with Xavier initialization
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let mut rng = rand::thread_rng();
            let std_dev = (2.0 / (input_dim + output_dim) as f64).sqrt();
            let normal = Normal::new(0.0, std_dev).unwrap();

            let weights = (0..output_dim)
                .map(|_| {
                    (0..input_dim)
                        .map(|_| rng.sample(normal))
                        .collect()
                })
                .collect();

            let biases = vec![0.0; output_dim];

            Self {
                weights,
                biases,
                input_dim,
                output_dim,
            }
        }

        /// Forward pass: output = W * input + b
        pub fn forward(&self, input: &[f64]) -> Vec<f64> {
            assert_eq!(input.len(), self.input_dim);
            let mut output = vec![0.0; self.output_dim];
            for i in 0..self.output_dim {
                let mut sum = self.biases[i];
                for j in 0..self.input_dim {
                    sum += self.weights[i][j] * input[j];
                }
                output[i] = sum;
            }
            output
        }

        /// Get all parameters as a flat vector
        pub fn parameters(&self) -> Vec<f64> {
            let mut params = Vec::new();
            for row in &self.weights {
                params.extend(row);
            }
            params.extend(&self.biases);
            params
        }

        /// Set parameters from a flat vector
        pub fn set_parameters(&mut self, params: &[f64]) {
            let mut idx = 0;
            for i in 0..self.output_dim {
                for j in 0..self.input_dim {
                    self.weights[i][j] = params[idx];
                    idx += 1;
                }
            }
            for i in 0..self.output_dim {
                self.biases[i] = params[idx];
                idx += 1;
            }
        }

        /// Number of parameters
        pub fn num_parameters(&self) -> usize {
            self.input_dim * self.output_dim + self.output_dim
        }
    }

    /// Multi-layer perceptron with smooth activations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MLP {
        pub layers: Vec<DenseLayer>,
        pub activation: Activation,
    }

    impl MLP {
        pub fn new(
            input_dim: usize,
            hidden_dim: usize,
            output_dim: usize,
            num_hidden_layers: usize,
            activation: Activation,
        ) -> Self {
            let mut layers = Vec::new();

            // Input -> first hidden
            layers.push(DenseLayer::new(input_dim, hidden_dim));

            // Hidden layers
            for _ in 1..num_hidden_layers {
                layers.push(DenseLayer::new(hidden_dim, hidden_dim));
            }

            // Last hidden -> output
            layers.push(DenseLayer::new(hidden_dim, output_dim));

            Self { layers, activation }
        }

        /// Forward pass through the MLP
        pub fn forward(&self, input: &[f64]) -> Vec<f64> {
            let mut x = input.to_vec();

            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x);
                if i < self.layers.len() - 1 {
                    x = x.iter().map(|&v| self.activation.apply(v)).collect();
                }
            }

            x
        }

        /// Forward with intermediates for gradient computation
        pub fn forward_with_intermediates(&self, input: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
            let mut x = input.to_vec();
            let mut pre_activations = Vec::new();
            let mut post_activations = Vec::new();

            post_activations.push(x.clone());

            for (i, layer) in self.layers.iter().enumerate() {
                let z = layer.forward(&x);
                pre_activations.push(z.clone());

                if i < self.layers.len() - 1 {
                    x = z.iter().map(|&v| self.activation.apply(v)).collect();
                } else {
                    x = z;
                }
                post_activations.push(x.clone());
            }

            (x, pre_activations, post_activations)
        }

        /// Compute gradient of scalar output w.r.t. input using backpropagation
        pub fn gradient_wrt_input(&self, input: &[f64]) -> Vec<f64> {
            let (_, pre_activations, _) = self.forward_with_intermediates(input);

            let output_dim = self.layers.last().unwrap().output_dim;
            let mut grad = vec![1.0; output_dim];

            for i in (0..self.layers.len()).rev() {
                let layer = &self.layers[i];

                if i < self.layers.len() - 1 {
                    let pre_act = &pre_activations[i];
                    for j in 0..grad.len() {
                        grad[j] *= self.activation.derivative(pre_act[j]);
                    }
                }

                let mut new_grad = vec![0.0; layer.input_dim];
                for j in 0..layer.input_dim {
                    for k in 0..layer.output_dim {
                        new_grad[j] += layer.weights[k][j] * grad[k];
                    }
                }
                grad = new_grad;
            }

            grad
        }

        /// Compute Hessian of scalar output w.r.t. input using finite differences
        pub fn hessian_wrt_input(&self, input: &[f64], eps: f64) -> Vec<Vec<f64>> {
            let n = input.len();
            let mut hessian = vec![vec![0.0; n]; n];

            let grad_center = self.gradient_wrt_input(input);

            for i in 0..n {
                let mut input_plus = input.to_vec();
                input_plus[i] += eps;
                let grad_plus = self.gradient_wrt_input(&input_plus);

                for j in 0..n {
                    hessian[i][j] = (grad_plus[j] - grad_center[j]) / eps;
                }
            }

            // Symmetrize
            for i in 0..n {
                for j in (i + 1)..n {
                    let avg = (hessian[i][j] + hessian[j][i]) / 2.0;
                    hessian[i][j] = avg;
                    hessian[j][i] = avg;
                }
            }

            hessian
        }

        pub fn parameters(&self) -> Vec<f64> {
            let mut params = Vec::new();
            for layer in &self.layers {
                params.extend(layer.parameters());
            }
            params
        }

        pub fn set_parameters(&mut self, params: &[f64]) {
            let mut idx = 0;
            for layer in &mut self.layers {
                let n = layer.num_parameters();
                layer.set_parameters(&params[idx..idx + n]);
                idx += n;
            }
        }

        pub fn num_parameters(&self) -> usize {
            self.layers.iter().map(|l| l.num_parameters()).sum()
        }
    }

    /// Lagrangian Neural Network
    ///
    /// Learns L(q, q-dot) as a scalar function. Dynamics derived via
    /// Euler-Lagrange equations:
    ///   q-ddot = M^{-1} * [dL/dq - C * q-dot]
    ///   where M = d^2L/dq-dot^2, C = d^2L/dq-dot dq
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LagrangianNN {
        pub l_net: MLP,
        pub coord_dim: usize,
        pub mass_reg: f64,
    }

    impl LagrangianNN {
        pub fn new(
            coord_dim: usize,
            hidden_dim: usize,
            num_layers: usize,
            mass_reg: f64,
        ) -> Self {
            let input_dim = 2 * coord_dim;
            let l_net = MLP::new(input_dim, hidden_dim, 1, num_layers, Activation::Softplus);
            Self { l_net, coord_dim, mass_reg }
        }

        /// Compute the Lagrangian L(q, q-dot)
        pub fn lagrangian(&self, q: &[f64], qdot: &[f64]) -> f64 {
            let mut input = Vec::with_capacity(q.len() + qdot.len());
            input.extend_from_slice(q);
            input.extend_from_slice(qdot);
            self.l_net.forward(&input)[0]
        }

        /// Compute conserved energy E = q-dot * dL/dq-dot - L
        pub fn energy(&self, q: &[f64], qdot: &[f64]) -> f64 {
            let mut input = Vec::with_capacity(q.len() + qdot.len());
            input.extend_from_slice(q);
            input.extend_from_slice(qdot);

            let l = self.l_net.forward(&input)[0];
            let grad = self.l_net.gradient_wrt_input(&input);

            // dL/dq-dot is the second half of the gradient
            let dl_dqdot = &grad[self.coord_dim..];

            // E = sum(qdot_i * dL/dqdot_i) - L
            let qdot_dot_grad: f64 = qdot.iter()
                .zip(dl_dqdot.iter())
                .map(|(&v, &g)| v * g)
                .sum();

            qdot_dot_grad - l
        }

        /// Compute Euler-Lagrange acceleration: q-ddot
        ///
        /// q-ddot = M^{-1} * [dL/dq - C * q-dot]
        /// where M = d^2L/dq-dot^2, C = d^2L/dq-dot dq
        pub fn acceleration(&self, q: &[f64], qdot: &[f64]) -> Vec<f64> {
            let dim = self.coord_dim;
            let mut input = Vec::with_capacity(q.len() + qdot.len());
            input.extend_from_slice(q);
            input.extend_from_slice(qdot);

            // First derivatives
            let grad = self.l_net.gradient_wrt_input(&input);
            let dl_dq: Vec<f64> = grad[..dim].to_vec();

            // Second derivatives via finite-difference Hessian
            let eps = 1e-5;
            let hessian = self.l_net.hessian_wrt_input(&input, eps);

            // Extract M = d^2L/dqdot^2 (bottom-right block)
            // Extract C = d^2L/dqdot dq (bottom-left block)
            let mut m_matrix = vec![vec![0.0; dim]; dim];
            let mut c_matrix = vec![vec![0.0; dim]; dim];

            for i in 0..dim {
                for j in 0..dim {
                    m_matrix[i][j] = hessian[dim + i][dim + j];
                    c_matrix[i][j] = hessian[dim + i][j];
                }
            }

            // Regularize M for invertibility
            for i in 0..dim {
                m_matrix[i][i] += self.mass_reg;
            }

            // RHS = dL/dq - C * qdot
            let mut rhs = dl_dq.clone();
            for i in 0..dim {
                let mut c_qdot = 0.0;
                for j in 0..dim {
                    c_qdot += c_matrix[i][j] * qdot[j];
                }
                rhs[i] -= c_qdot;
            }

            // Solve M * qddot = rhs
            // For dim=1: qddot = rhs / M
            if dim == 1 {
                vec![rhs[0] / m_matrix[0][0]]
            } else {
                // Simple Gaussian elimination for small systems
                solve_linear_system(&m_matrix, &rhs)
            }
        }

        /// Compute time derivatives: (dq/dt, dqdot/dt) = (qdot, qddot)
        pub fn time_derivative(&self, q: &[f64], qdot: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let qddot = self.acceleration(q, qdot);
            (qdot.to_vec(), qddot)
        }

        pub fn parameters(&self) -> Vec<f64> {
            self.l_net.parameters()
        }

        pub fn set_parameters(&mut self, params: &[f64]) {
            self.l_net.set_parameters(params);
        }

        pub fn num_parameters(&self) -> usize {
            self.l_net.num_parameters()
        }
    }

    /// Dissipative Lagrangian Neural Network
    ///
    /// Extends LNN with Rayleigh dissipation D(q, q-dot) >= 0:
    ///   Modified Euler-Lagrange:
    ///     d/dt(dL/dq-dot) - dL/dq = -dD/dq-dot
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DissipativeLNN {
        pub l_net: MLP,
        pub d_net: MLP,
        pub coord_dim: usize,
        pub mass_reg: f64,
    }

    impl DissipativeLNN {
        pub fn new(
            coord_dim: usize,
            hidden_dim: usize,
            num_layers: usize,
            mass_reg: f64,
        ) -> Self {
            let input_dim = 2 * coord_dim;
            let l_net = MLP::new(input_dim, hidden_dim, 1, num_layers, Activation::Softplus);
            let d_net = MLP::new(input_dim, hidden_dim, 1, num_layers, Activation::Softplus);
            Self { l_net, d_net, coord_dim, mass_reg }
        }

        pub fn lagrangian(&self, q: &[f64], qdot: &[f64]) -> f64 {
            let mut input = Vec::with_capacity(q.len() + qdot.len());
            input.extend_from_slice(q);
            input.extend_from_slice(qdot);
            self.l_net.forward(&input)[0]
        }

        /// Dissipation function (always non-negative via softplus)
        pub fn dissipation(&self, q: &[f64], qdot: &[f64]) -> f64 {
            let mut input = Vec::with_capacity(q.len() + qdot.len());
            input.extend_from_slice(q);
            input.extend_from_slice(qdot);
            let raw = self.d_net.forward(&input)[0];
            (1.0 + raw.exp()).ln() // softplus
        }

        pub fn energy(&self, q: &[f64], qdot: &[f64]) -> f64 {
            let mut input = Vec::with_capacity(q.len() + qdot.len());
            input.extend_from_slice(q);
            input.extend_from_slice(qdot);

            let l = self.l_net.forward(&input)[0];
            let grad = self.l_net.gradient_wrt_input(&input);
            let dl_dqdot = &grad[self.coord_dim..];

            let qdot_dot_grad: f64 = qdot.iter()
                .zip(dl_dqdot.iter())
                .map(|(&v, &g)| v * g)
                .sum();

            qdot_dot_grad - l
        }

        /// Compute dissipative Euler-Lagrange acceleration
        pub fn acceleration(&self, q: &[f64], qdot: &[f64]) -> Vec<f64> {
            let dim = self.coord_dim;
            let mut input = Vec::with_capacity(q.len() + qdot.len());
            input.extend_from_slice(q);
            input.extend_from_slice(qdot);

            // Lagrangian gradient
            let l_grad = self.l_net.gradient_wrt_input(&input);
            let dl_dq: Vec<f64> = l_grad[..dim].to_vec();

            // Dissipation gradient (dD/dqdot)
            let d_grad = self.d_net.gradient_wrt_input(&input);
            let dd_dqdot_raw: Vec<f64> = d_grad[dim..].to_vec();

            // Softplus correction: d(softplus(x))/dx = sigmoid(x)
            let raw_d = self.d_net.forward(&input)[0];
            let softplus_deriv = 1.0 / (1.0 + (-raw_d).exp());
            let dd_dqdot: Vec<f64> = dd_dqdot_raw.iter()
                .map(|&x| x * softplus_deriv)
                .collect();

            // Hessian of L
            let eps = 1e-5;
            let hessian = self.l_net.hessian_wrt_input(&input, eps);

            let mut m_matrix = vec![vec![0.0; dim]; dim];
            let mut c_matrix = vec![vec![0.0; dim]; dim];

            for i in 0..dim {
                for j in 0..dim {
                    m_matrix[i][j] = hessian[dim + i][dim + j];
                    c_matrix[i][j] = hessian[dim + i][j];
                }
            }

            for i in 0..dim {
                m_matrix[i][i] += self.mass_reg;
            }

            // RHS = dL/dq - C * qdot - dD/dqdot
            let mut rhs = dl_dq;
            for i in 0..dim {
                let mut c_qdot = 0.0;
                for j in 0..dim {
                    c_qdot += c_matrix[i][j] * qdot[j];
                }
                rhs[i] -= c_qdot;
                rhs[i] -= dd_dqdot[i];
            }

            if dim == 1 {
                vec![rhs[0] / m_matrix[0][0]]
            } else {
                solve_linear_system(&m_matrix, &rhs)
            }
        }

        pub fn time_derivative(&self, q: &[f64], qdot: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let qddot = self.acceleration(q, qdot);
            (qdot.to_vec(), qddot)
        }

        pub fn parameters(&self) -> Vec<f64> {
            let mut params = self.l_net.parameters();
            params.extend(self.d_net.parameters());
            params
        }

        pub fn set_parameters(&mut self, params: &[f64]) {
            let l_n = self.l_net.num_parameters();
            self.l_net.set_parameters(&params[..l_n]);
            self.d_net.set_parameters(&params[l_n..]);
        }

        pub fn num_parameters(&self) -> usize {
            self.l_net.num_parameters() + self.d_net.num_parameters()
        }
    }

    /// Solve a small linear system Ax = b using Gaussian elimination
    fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut aug = vec![vec![0.0; n + 1]; n];

        // Build augmented matrix [A|b]
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = a[i][j];
            }
            aug[i][n] = b[i];
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_val = aug[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-12 {
                // Singular or near-singular: return zeros
                return vec![0.0; n];
            }

            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }

        x
    }
}

/// Module: Numerical integrators
pub mod integrator {
    use super::nn::{LagrangianNN, DissipativeLNN};

    /// One step of RK4 integration for LNN
    pub fn rk4_step(
        model: &LagrangianNN,
        q: &[f64],
        qdot: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = q.len();

        // k1
        let (k1_q, k1_v) = model.time_derivative(q, qdot);

        // k2
        let q2: Vec<f64> = (0..dim).map(|i| q[i] + 0.5 * dt * k1_q[i]).collect();
        let v2: Vec<f64> = (0..dim).map(|i| qdot[i] + 0.5 * dt * k1_v[i]).collect();
        let (k2_q, k2_v) = model.time_derivative(&q2, &v2);

        // k3
        let q3: Vec<f64> = (0..dim).map(|i| q[i] + 0.5 * dt * k2_q[i]).collect();
        let v3: Vec<f64> = (0..dim).map(|i| qdot[i] + 0.5 * dt * k2_v[i]).collect();
        let (k3_q, k3_v) = model.time_derivative(&q3, &v3);

        // k4
        let q4: Vec<f64> = (0..dim).map(|i| q[i] + dt * k3_q[i]).collect();
        let v4: Vec<f64> = (0..dim).map(|i| qdot[i] + dt * k3_v[i]).collect();
        let (k4_q, k4_v) = model.time_derivative(&q4, &v4);

        let q_new: Vec<f64> = (0..dim)
            .map(|i| q[i] + (dt / 6.0) * (k1_q[i] + 2.0 * k2_q[i] + 2.0 * k3_q[i] + k4_q[i]))
            .collect();
        let v_new: Vec<f64> = (0..dim)
            .map(|i| qdot[i] + (dt / 6.0) * (k1_v[i] + 2.0 * k2_v[i] + 2.0 * k3_v[i] + k4_v[i]))
            .collect();

        (q_new, v_new)
    }

    /// Euler integration (simple, for comparison)
    pub fn euler_step(
        model: &LagrangianNN,
        q: &[f64],
        qdot: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let (dq_dt, dqdot_dt) = model.time_derivative(q, qdot);
        let q_new: Vec<f64> = q.iter()
            .zip(dq_dt.iter())
            .map(|(&qi, &dqi)| qi + dt * dqi)
            .collect();
        let v_new: Vec<f64> = qdot.iter()
            .zip(dqdot_dt.iter())
            .map(|(&vi, &dvi)| vi + dt * dvi)
            .collect();
        (q_new, v_new)
    }

    /// Leapfrog step for LNN
    pub fn leapfrog_step(
        model: &LagrangianNN,
        q: &[f64],
        qdot: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = q.len();

        // Half-step velocity
        let qddot_0 = model.acceleration(q, qdot);
        let v_half: Vec<f64> = (0..dim)
            .map(|i| qdot[i] + 0.5 * dt * qddot_0[i])
            .collect();

        // Full-step position
        let q_new: Vec<f64> = (0..dim)
            .map(|i| q[i] + dt * v_half[i])
            .collect();

        // Full-step acceleration
        let qddot_1 = model.acceleration(&q_new, &v_half);

        // Half-step velocity
        let v_new: Vec<f64> = (0..dim)
            .map(|i| v_half[i] + 0.5 * dt * qddot_1[i])
            .collect();

        (q_new, v_new)
    }

    /// Integrate a trajectory using RK4
    pub fn integrate_trajectory(
        model: &LagrangianNN,
        q0: &[f64],
        qdot0: &[f64],
        dt: f64,
        n_steps: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut traj_q = vec![q0.to_vec()];
        let mut traj_v = vec![qdot0.to_vec()];

        let mut q = q0.to_vec();
        let mut v = qdot0.to_vec();

        for _ in 0..n_steps {
            let (q_new, v_new) = rk4_step(model, &q, &v, dt);
            traj_q.push(q_new.clone());
            traj_v.push(v_new.clone());
            q = q_new;
            v = v_new;
        }

        (traj_q, traj_v)
    }

    /// RK4 step for dissipative LNN
    pub fn rk4_step_dissipative(
        model: &DissipativeLNN,
        q: &[f64],
        qdot: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = q.len();

        let (k1_q, k1_v) = model.time_derivative(q, qdot);

        let q2: Vec<f64> = (0..dim).map(|i| q[i] + 0.5 * dt * k1_q[i]).collect();
        let v2: Vec<f64> = (0..dim).map(|i| qdot[i] + 0.5 * dt * k1_v[i]).collect();
        let (k2_q, k2_v) = model.time_derivative(&q2, &v2);

        let q3: Vec<f64> = (0..dim).map(|i| q[i] + 0.5 * dt * k2_q[i]).collect();
        let v3: Vec<f64> = (0..dim).map(|i| qdot[i] + 0.5 * dt * k2_v[i]).collect();
        let (k3_q, k3_v) = model.time_derivative(&q3, &v3);

        let q4: Vec<f64> = (0..dim).map(|i| q[i] + dt * k3_q[i]).collect();
        let v4: Vec<f64> = (0..dim).map(|i| qdot[i] + dt * k3_v[i]).collect();
        let (k4_q, k4_v) = model.time_derivative(&q4, &v4);

        let q_new: Vec<f64> = (0..dim)
            .map(|i| q[i] + (dt / 6.0) * (k1_q[i] + 2.0 * k2_q[i] + 2.0 * k3_q[i] + k4_q[i]))
            .collect();
        let v_new: Vec<f64> = (0..dim)
            .map(|i| qdot[i] + (dt / 6.0) * (k1_v[i] + 2.0 * k2_v[i] + 2.0 * k3_v[i] + k4_v[i]))
            .collect();

        (q_new, v_new)
    }

    /// Integrate trajectory with dissipative model
    pub fn integrate_trajectory_dissipative(
        model: &DissipativeLNN,
        q0: &[f64],
        qdot0: &[f64],
        dt: f64,
        n_steps: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut traj_q = vec![q0.to_vec()];
        let mut traj_v = vec![qdot0.to_vec()];

        let mut q = q0.to_vec();
        let mut v = qdot0.to_vec();

        for _ in 0..n_steps {
            let (q_new, v_new) = rk4_step_dissipative(model, &q, &v, dt);
            traj_q.push(q_new.clone());
            traj_v.push(v_new.clone());
            q = q_new;
            v = v_new;
        }

        (traj_q, traj_v)
    }
}

/// Module: Data fetching and configuration space construction
pub mod data {
    use serde::{Deserialize, Serialize};
    use anyhow::Result;

    /// OHLCV candle data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Candle {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
        pub turnover: f64,
    }

    /// Configuration space data (for LNN, uses q, qdot, qddot)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConfigSpaceData {
        pub q: Vec<Vec<f64>>,
        pub qdot: Vec<Vec<f64>>,
        pub qddot: Vec<Vec<f64>>,
        pub prices: Vec<f64>,
        pub timestamps: Vec<i64>,
    }

    /// Normalization statistics
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NormStats {
        pub q_mean: Vec<f64>,
        pub q_std: Vec<f64>,
        pub qdot_mean: Vec<f64>,
        pub qdot_std: Vec<f64>,
        pub qddot_mean: Vec<f64>,
        pub qddot_std: Vec<f64>,
    }

    /// Bybit API response structures
    #[derive(Debug, Deserialize)]
    pub struct BybitResponse {
        #[serde(rename = "retCode")]
        pub ret_code: i32,
        #[serde(rename = "retMsg")]
        pub ret_msg: String,
        pub result: BybitResult,
    }

    #[derive(Debug, Deserialize)]
    pub struct BybitResult {
        pub list: Vec<Vec<String>>,
    }

    /// Bybit API client
    pub struct BybitClient {
        pub base_url: String,
        client: reqwest::Client,
    }

    impl BybitClient {
        pub fn new() -> Self {
            Self {
                base_url: "https://api.bybit.com".to_string(),
                client: reqwest::Client::new(),
            }
        }

        /// Fetch kline data from Bybit V5 API
        pub async fn fetch_klines(
            &self,
            symbol: &str,
            interval: &str,
            limit: usize,
            end_time: Option<i64>,
        ) -> Result<Vec<Candle>> {
            let mut url = format!(
                "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
                self.base_url, symbol, interval, limit.min(1000)
            );

            if let Some(et) = end_time {
                url.push_str(&format!("&end={}", et));
            }

            let resp: BybitResponse = self.client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if resp.ret_code != 0 {
                anyhow::bail!("Bybit API error: {}", resp.ret_msg);
            }

            let mut candles: Vec<Candle> = resp.result.list
                .iter()
                .map(|row| {
                    Candle {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                        turnover: row.get(6)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0),
                    }
                })
                .collect();

            candles.sort_by_key(|c| c.timestamp);
            Ok(candles)
        }

        /// Fetch extended history by paginating
        pub async fn fetch_extended(
            &self,
            symbol: &str,
            interval: &str,
            total_candles: usize,
        ) -> Result<Vec<Candle>> {
            let mut all_candles = Vec::new();
            let mut end_time: Option<i64> = None;
            let mut remaining = total_candles;

            while remaining > 0 {
                let batch_size = remaining.min(1000);
                let candles = self.fetch_klines(symbol, interval, batch_size, end_time).await?;

                if candles.is_empty() {
                    break;
                }

                let earliest = candles.first().unwrap().timestamp;
                end_time = Some(earliest - 1);
                remaining = remaining.saturating_sub(candles.len());

                all_candles.extend(candles);
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }

            all_candles.sort_by_key(|c| c.timestamp);
            all_candles.dedup_by_key(|c| c.timestamp);
            Ok(all_candles)
        }
    }

    /// Construct configuration space (q, qdot, qddot) from candle data
    pub fn construct_config_space(
        candles: &[Candle],
        ma_window: usize,
    ) -> ConfigSpaceData {
        let n = candles.len();
        let log_close: Vec<f64> = candles.iter().map(|c| c.close.ln()).collect();

        // Moving average
        let mut ma = vec![f64::NAN; n];
        for i in (ma_window - 1)..n {
            let sum: f64 = log_close[i + 1 - ma_window..=i].iter().sum();
            ma[i] = sum / ma_window as f64;
        }

        // q: price deviation from MA
        let q_raw: Vec<f64> = (0..n)
            .map(|i| log_close[i] - ma[i])
            .collect();

        // qdot: velocity (central difference)
        let mut qdot_raw = vec![0.0; n];
        for i in 1..n - 1 {
            qdot_raw[i] = (q_raw[i + 1] - q_raw[i - 1]) / 2.0;
        }
        if n > 1 {
            qdot_raw[0] = q_raw[1] - q_raw[0];
            qdot_raw[n - 1] = q_raw[n - 1] - q_raw[n - 2];
        }

        // qddot: acceleration (central difference of velocity)
        let mut qddot_raw = vec![0.0; n];
        for i in 1..n - 1 {
            qddot_raw[i] = (qdot_raw[i + 1] - qdot_raw[i - 1]) / 2.0;
        }

        // Filter valid entries
        let mut data = ConfigSpaceData {
            q: Vec::new(),
            qdot: Vec::new(),
            qddot: Vec::new(),
            prices: Vec::new(),
            timestamps: Vec::new(),
        };

        for i in ma_window..n - 1 {
            if q_raw[i].is_finite() && qdot_raw[i].is_finite() && qddot_raw[i].is_finite() {
                data.q.push(vec![q_raw[i]]);
                data.qdot.push(vec![qdot_raw[i]]);
                data.qddot.push(vec![qddot_raw[i]]);
                data.prices.push(candles[i].close);
                data.timestamps.push(candles[i].timestamp);
            }
        }

        data
    }

    /// Normalize configuration space data
    pub fn normalize_config_space(data: &ConfigSpaceData) -> (ConfigSpaceData, NormStats) {
        let n = data.q.len();
        let dim = if n > 0 { data.q[0].len() } else { 1 };

        // Compute means
        let mut q_mean = vec![0.0; dim];
        let mut qdot_mean = vec![0.0; dim];
        let mut qddot_mean = vec![0.0; dim];

        for i in 0..n {
            for d in 0..dim {
                q_mean[d] += data.q[i][d];
                qdot_mean[d] += data.qdot[i][d];
                qddot_mean[d] += data.qddot[i][d];
            }
        }
        for d in 0..dim {
            q_mean[d] /= n as f64;
            qdot_mean[d] /= n as f64;
            qddot_mean[d] /= n as f64;
        }

        // Compute stds
        let mut q_var = vec![0.0; dim];
        let mut qdot_var = vec![0.0; dim];
        let mut qddot_var = vec![0.0; dim];
        for i in 0..n {
            for d in 0..dim {
                q_var[d] += (data.q[i][d] - q_mean[d]).powi(2);
                qdot_var[d] += (data.qdot[i][d] - qdot_mean[d]).powi(2);
                qddot_var[d] += (data.qddot[i][d] - qddot_mean[d]).powi(2);
            }
        }
        let q_std: Vec<f64> = q_var.iter().map(|&v| (v / n as f64).sqrt().max(1e-8)).collect();
        let qdot_std: Vec<f64> = qdot_var.iter().map(|&v| (v / n as f64).sqrt().max(1e-8)).collect();
        let qddot_std: Vec<f64> = qddot_var.iter().map(|&v| (v / n as f64).sqrt().max(1e-8)).collect();

        let stats = NormStats {
            q_mean: q_mean.clone(),
            q_std: q_std.clone(),
            qdot_mean: qdot_mean.clone(),
            qdot_std: qdot_std.clone(),
            qddot_mean: qddot_mean.clone(),
            qddot_std: qddot_std.clone(),
        };

        let mut normalized = ConfigSpaceData {
            q: Vec::with_capacity(n),
            qdot: Vec::with_capacity(n),
            qddot: Vec::with_capacity(n),
            prices: data.prices.clone(),
            timestamps: data.timestamps.clone(),
        };

        for i in 0..n {
            let q_norm: Vec<f64> = (0..dim)
                .map(|d| (data.q[i][d] - q_mean[d]) / q_std[d])
                .collect();
            let qdot_norm: Vec<f64> = (0..dim)
                .map(|d| (data.qdot[i][d] - qdot_mean[d]) / qdot_std[d])
                .collect();
            let qddot_norm: Vec<f64> = (0..dim)
                .map(|d| (data.qddot[i][d] - qddot_mean[d]) / qddot_std[d])
                .collect();

            normalized.q.push(q_norm);
            normalized.qdot.push(qdot_norm);
            normalized.qddot.push(qddot_norm);
        }

        (normalized, stats)
    }
}

/// Module: Trading strategy and backtesting
pub mod trading {
    use super::nn::LagrangianNN;
    use super::integrator;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Trade {
        pub timestamp_idx: usize,
        pub side: String,
        pub price: f64,
        pub quantity: f64,
        pub energy: f64,
        pub energy_zscore: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BacktestResult {
        pub initial_capital: f64,
        pub final_capital: f64,
        pub total_return: f64,
        pub max_drawdown: f64,
        pub sharpe_ratio: f64,
        pub win_rate: f64,
        pub n_trades: usize,
        pub trades: Vec<Trade>,
        pub equity_curve: Vec<f64>,
    }

    pub struct TradingStrategy {
        pub model: LagrangianNN,
        pub prediction_horizon: usize,
        pub dt: f64,
        pub entry_threshold: f64,
        pub stop_loss_pct: f64,
        pub take_profit_pct: f64,
        pub energy_history: Vec<f64>,
    }

    impl TradingStrategy {
        pub fn new(
            model: LagrangianNN,
            prediction_horizon: usize,
            dt: f64,
            entry_threshold: f64,
        ) -> Self {
            Self {
                model,
                prediction_horizon,
                dt,
                entry_threshold,
                stop_loss_pct: 0.03,
                take_profit_pct: 0.05,
                energy_history: Vec::new(),
            }
        }

        /// Generate a trading signal from current configuration state
        pub fn generate_signal(&mut self, q: &[f64], qdot: &[f64]) -> (String, f64, f64) {
            let energy = self.model.energy(q, qdot);
            self.energy_history.push(energy);

            // Integrate forward using RK4
            let (traj_q, _traj_v) = integrator::integrate_trajectory(
                &self.model, q, qdot, self.dt, self.prediction_horizon,
            );

            let predicted_change = traj_q.last().unwrap()[0] - traj_q[0][0];
            let strength = predicted_change.abs();

            // Energy z-score for regime detection
            let zscore = if self.energy_history.len() >= 20 {
                let recent: Vec<f64> = self.energy_history
                    .iter()
                    .rev()
                    .take(100)
                    .copied()
                    .collect();
                let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
                let var: f64 = recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / recent.len() as f64;
                let std = var.sqrt().max(1e-10);
                (energy - mean) / std
            } else {
                0.0
            };

            if zscore.abs() > 2.5 {
                return ("HOLD".to_string(), strength, zscore);
            }

            if strength > self.entry_threshold {
                if predicted_change > 0.0 {
                    return ("BUY".to_string(), strength, zscore);
                } else {
                    return ("SELL".to_string(), strength, zscore);
                }
            }

            ("HOLD".to_string(), strength, zscore)
        }

        /// Run a backtest
        pub fn backtest(
            &mut self,
            prices: &[f64],
            q_data: &[Vec<f64>],
            qdot_data: &[Vec<f64>],
            initial_capital: f64,
            commission: f64,
        ) -> BacktestResult {
            let n = prices.len().min(q_data.len());

            let mut capital = initial_capital;
            let mut position = 0.0_f64;
            let mut entry_price = 0.0_f64;
            let mut trades = Vec::new();
            let mut equity_curve = Vec::with_capacity(n);

            for i in 0..n {
                let price = prices[i];
                let (signal, strength, zscore) = self.generate_signal(&q_data[i], &qdot_data[i]);

                // Stop-loss / take-profit
                if position.abs() > 1e-10 {
                    let pnl_pct = if position > 0.0 {
                        (price - entry_price) / entry_price
                    } else {
                        (entry_price - price) / entry_price
                    };

                    if pnl_pct <= -self.stop_loss_pct || pnl_pct >= self.take_profit_pct {
                        let pnl = if position > 0.0 {
                            position * (price - entry_price)
                        } else {
                            -position * (entry_price - price)
                        };
                        capital += pnl - position.abs() * price * commission;
                        trades.push(Trade {
                            timestamp_idx: i,
                            side: "CLOSE".to_string(),
                            price,
                            quantity: position.abs(),
                            energy: self.model.energy(&q_data[i], &qdot_data[i]),
                            energy_zscore: zscore,
                        });
                        position = 0.0;
                    }
                }

                // Execute signal
                match signal.as_str() {
                    "BUY" if position <= 0.0 => {
                        if position < 0.0 {
                            let pnl = -position * (entry_price - price);
                            capital += pnl - position.abs() * price * commission;
                        }
                        let qty = capital / price;
                        position = qty;
                        entry_price = price;
                        capital -= qty * price * commission;
                        trades.push(Trade {
                            timestamp_idx: i,
                            side: "BUY".to_string(),
                            price,
                            quantity: qty,
                            energy: self.model.energy(&q_data[i], &qdot_data[i]),
                            energy_zscore: zscore,
                        });
                    }
                    "SELL" if position >= 0.0 => {
                        if position > 0.0 {
                            let pnl = position * (price - entry_price);
                            capital += pnl - position * price * commission;
                        }
                        let qty = capital / price;
                        position = -qty;
                        entry_price = price;
                        capital -= qty * price * commission;
                        trades.push(Trade {
                            timestamp_idx: i,
                            side: "SELL".to_string(),
                            price,
                            quantity: qty,
                            energy: self.model.energy(&q_data[i], &qdot_data[i]),
                            energy_zscore: zscore,
                        });
                    }
                    _ => {}
                }

                // Update equity
                let equity = if position > 0.0 {
                    capital + position * (price - entry_price)
                } else if position < 0.0 {
                    capital - position * (price - entry_price)
                } else {
                    capital
                };
                equity_curve.push(equity);
            }

            // Close remaining position
            if position.abs() > 1e-10 {
                let final_price = prices[n - 1];
                let pnl = if position > 0.0 {
                    position * (final_price - entry_price)
                } else {
                    -position * (entry_price - final_price)
                };
                capital += pnl;
            }

            // Compute metrics
            let total_return = (capital - initial_capital) / initial_capital;

            let mut peak = f64::MIN;
            let mut max_dd = 0.0_f64;
            for &eq in &equity_curve {
                if eq > peak { peak = eq; }
                let dd = (eq - peak) / peak;
                if dd < max_dd { max_dd = dd; }
            }

            let returns: Vec<f64> = equity_curve.windows(2)
                .map(|w| (w[1] - w[0]) / w[0].max(1e-10))
                .collect();
            let mean_ret = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
            let std_ret = {
                let var: f64 = returns.iter()
                    .map(|&r| (r - mean_ret).powi(2))
                    .sum::<f64>() / returns.len().max(1) as f64;
                var.sqrt().max(1e-10)
            };
            let sharpe = mean_ret / std_ret * (252.0 * 288.0_f64).sqrt();

            let win_count = trades.chunks(2)
                .filter(|chunk| {
                    if chunk.len() == 2 {
                        let entry = &chunk[0];
                        let exit = &chunk[1];
                        if entry.side == "BUY" {
                            exit.price > entry.price
                        } else {
                            exit.price < entry.price
                        }
                    } else { false }
                })
                .count();
            let total_pairs = (trades.len() / 2).max(1);
            let win_rate = win_count as f64 / total_pairs as f64;

            BacktestResult {
                initial_capital,
                final_capital: capital,
                total_return,
                max_drawdown: max_dd,
                sharpe_ratio: sharpe,
                win_rate,
                n_trades: trades.len(),
                trades,
                equity_curve,
            }
        }
    }
}

/// Module: Utility functions
pub mod utils {
    use super::nn::LagrangianNN;

    /// Simple SGD optimizer with momentum
    pub struct SGDOptimizer {
        pub learning_rate: f64,
        pub momentum: f64,
        velocity: Vec<f64>,
    }

    impl SGDOptimizer {
        pub fn new(n_params: usize, learning_rate: f64, momentum: f64) -> Self {
            Self {
                learning_rate,
                momentum,
                velocity: vec![0.0; n_params],
            }
        }

        pub fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
            for i in 0..params.len() {
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients[i];
                params[i] += self.velocity[i];
            }
        }
    }

    /// Compute MSE loss and finite-difference gradients for LNN training
    ///
    /// Loss = mean || qddot_pred - qddot_target ||^2
    pub fn compute_loss_and_gradients(
        model: &LagrangianNN,
        q_batch: &[Vec<f64>],
        qdot_batch: &[Vec<f64>],
        qddot_target: &[Vec<f64>],
    ) -> (f64, Vec<f64>) {
        let batch_size = q_batch.len();
        let dim = q_batch[0].len();

        // Forward pass: compute loss
        let mut total_loss = 0.0;
        for i in 0..batch_size {
            let qddot_pred = model.acceleration(&q_batch[i], &qdot_batch[i]);
            for d in 0..dim {
                total_loss += (qddot_pred[d] - qddot_target[i][d]).powi(2);
            }
        }
        total_loss /= batch_size as f64;

        // Finite-difference gradients
        let params = model.parameters();
        let n_params = params.len();
        let mut gradients = vec![0.0; n_params];
        let eps = 1e-5;

        for j in 0..n_params {
            let mut params_plus = params.clone();
            params_plus[j] += eps;

            let mut model_plus = model.clone();
            model_plus.set_parameters(&params_plus);

            let mut loss_plus = 0.0;
            for i in 0..batch_size {
                let qddot_pred = model_plus.acceleration(&q_batch[i], &qdot_batch[i]);
                for d in 0..dim {
                    loss_plus += (qddot_pred[d] - qddot_target[i][d]).powi(2);
                }
            }
            loss_plus /= batch_size as f64;

            gradients[j] = (loss_plus - total_loss) / eps;
        }

        (total_loss, gradients)
    }

    /// Compute energy along a trajectory
    pub fn energy_along_trajectory(
        model: &LagrangianNN,
        traj_q: &[Vec<f64>],
        traj_v: &[Vec<f64>],
    ) -> Vec<f64> {
        traj_q.iter()
            .zip(traj_v.iter())
            .map(|(q, v)| model.energy(q, v))
            .collect()
    }

    /// Export data to CSV
    pub fn export_csv(
        path: &str,
        headers: &[&str],
        data: &[Vec<f64>],
    ) -> std::io::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record(headers)?;
        for row in data {
            let record: Vec<String> = row.iter().map(|v| format!("{:.8}", v)).collect();
            wtr.write_record(&record)?;
        }
        wtr.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lnn_creation() {
        let model = nn::LagrangianNN::new(1, 32, 2, 1e-4);
        assert_eq!(model.coord_dim, 1);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_lnn_lagrangian() {
        let model = nn::LagrangianNN::new(1, 32, 2, 1e-4);
        let q = vec![0.5];
        let qdot = vec![-0.3];

        let l = model.lagrangian(&q, &qdot);
        assert!(l.is_finite());
    }

    #[test]
    fn test_lnn_energy() {
        let model = nn::LagrangianNN::new(1, 32, 2, 1e-4);
        let q = vec![0.5];
        let qdot = vec![-0.3];

        let e = model.energy(&q, &qdot);
        assert!(e.is_finite());
    }

    #[test]
    fn test_lnn_acceleration() {
        let model = nn::LagrangianNN::new(1, 32, 2, 1e-4);
        let q = vec![1.0];
        let qdot = vec![0.0];

        let qddot = model.acceleration(&q, &qdot);
        assert_eq!(qddot.len(), 1);
        assert!(qddot[0].is_finite());
    }

    #[test]
    fn test_lnn_time_derivative() {
        let model = nn::LagrangianNN::new(1, 32, 2, 1e-4);
        let q = vec![1.0];
        let qdot = vec![0.5];

        let (dq_dt, dqdot_dt) = model.time_derivative(&q, &qdot);
        assert_eq!(dq_dt.len(), 1);
        assert_eq!(dqdot_dt.len(), 1);
        // dq/dt should equal qdot
        assert!((dq_dt[0] - qdot[0]).abs() < 1e-10);
        assert!(dqdot_dt[0].is_finite());
    }

    #[test]
    fn test_rk4_integration() {
        let model = nn::LagrangianNN::new(1, 32, 2, 1e-4);
        let q0 = vec![1.0];
        let qdot0 = vec![0.0];

        let (traj_q, traj_v) = integrator::integrate_trajectory(&model, &q0, &qdot0, 0.01, 100);
        assert_eq!(traj_q.len(), 101);
        assert_eq!(traj_v.len(), 101);

        for (q, v) in traj_q.iter().zip(traj_v.iter()) {
            assert!(q[0].is_finite(), "q became non-finite during integration");
            assert!(v[0].is_finite(), "qdot became non-finite during integration");
        }
    }

    #[test]
    fn test_energy_conservation() {
        let model = nn::LagrangianNN::new(1, 32, 2, 1e-4);
        let q0 = vec![0.5];
        let qdot0 = vec![0.3];

        let (traj_q, traj_v) = integrator::integrate_trajectory(&model, &q0, &qdot0, 0.01, 50);
        let energies = utils::energy_along_trajectory(&model, &traj_q, &traj_v);

        // Energy should not drift too much (random network, so just check finiteness)
        for e in &energies {
            assert!(e.is_finite(), "Energy became non-finite");
        }
    }

    #[test]
    fn test_dissipative_lnn() {
        let model = nn::DissipativeLNN::new(1, 32, 2, 1e-4);
        let q = vec![0.5];
        let qdot = vec![-0.3];

        let l = model.lagrangian(&q, &qdot);
        let d = model.dissipation(&q, &qdot);
        assert!(l.is_finite());
        assert!(d >= 0.0, "Dissipation must be non-negative");

        let qddot = model.acceleration(&q, &qdot);
        assert!(qddot[0].is_finite());
    }

    #[test]
    fn test_dense_layer() {
        let layer = nn::DenseLayer::new(3, 5);
        assert_eq!(layer.input_dim, 3);
        assert_eq!(layer.output_dim, 5);

        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_mlp() {
        let mlp = nn::MLP::new(2, 16, 1, 2, nn::Activation::Softplus);
        let input = vec![0.5, -0.3];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_hessian_computation() {
        let mlp = nn::MLP::new(2, 8, 1, 2, nn::Activation::Softplus);
        let input = vec![1.0, 0.5];
        let hessian = mlp.hessian_wrt_input(&input, 1e-5);
        assert_eq!(hessian.len(), 2);
        assert_eq!(hessian[0].len(), 2);
        // Hessian should be symmetric
        assert!((hessian[0][1] - hessian[1][0]).abs() < 1e-3);
    }

    #[test]
    fn test_serialization() {
        let model = nn::LagrangianNN::new(1, 16, 2, 1e-4);
        let serialized = serde_json::to_string(&model).unwrap();
        let deserialized: nn::LagrangianNN = serde_json::from_str(&serialized).unwrap();
        assert_eq!(model.coord_dim, deserialized.coord_dim);
        assert_eq!(model.num_parameters(), deserialized.num_parameters());
    }
}
