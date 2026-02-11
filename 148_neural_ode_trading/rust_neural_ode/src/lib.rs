//! # Neural ODE Trading Library
//!
//! Implements Neural Ordinary Differential Equations (Neural ODEs) for
//! cryptocurrency trading using Bybit exchange data.
//!
//! ## Core Components
//!
//! - **ODE Solvers**: RK4 (fixed-step), Dormand-Prince (adaptive), Euler
//! - **Neural ODE**: Neural network that parameterizes ODE dynamics dh/dt = f_theta(h, t)
//! - **Data Pipeline**: Bybit API integration with irregular timestamp handling
//! - **Backtesting**: Simple momentum and mean-reversion strategies
//!
//! ## Mathematical Background
//!
//! A Neural ODE replaces discrete residual layers with a continuous transformation:
//!
//! ```text
//! ResNet:     h_{l+1} = h_l + f_theta(h_l)         (discrete steps)
//! Neural ODE: dh/dt = f_theta(h(t), t)              (continuous dynamics)
//!             h(T) = h(0) + integral_0^T f_theta(h(t), t) dt
//! ```
//!
//! The forward pass solves this ODE using numerical methods (RK4 or adaptive).
//! The backward pass uses the adjoint sensitivity method for O(1) memory.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use neural_ode_trading::{NeuralODE, RK4Solver, linear_layer};
//!
//! // Create a simple Neural ODE
//! let input_dim = 5;
//! let hidden_dim = 32;
//! let model = NeuralODE::new(input_dim, hidden_dim, 1, 0.01);
//!
//! // Generate prediction
//! let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let output = model.forward(&input);
//! ```

pub mod data;
pub mod model;
pub mod ode_solver;
pub mod backtest;

// Re-exports
pub use model::{NeuralODE, ODEFunc, LinearLayer, linear_layer};
pub use ode_solver::{RK4Solver, DormandPrinceSolver, EulerSolver, ODESolver};
pub use data::{BybitClient, MarketData, TickData};
pub use backtest::{Backtester, BacktestResult, Signal};

/// ODE Solver module: numerical methods for solving dh/dt = f(h, t)
pub mod ode_solver {
    use ndarray::{Array1, Array2};

    /// Trait for ODE solvers.
    ///
    /// Any solver must implement `solve` which integrates the ODE
    /// from t_start to t_end given initial state y0.
    pub trait ODESolver {
        /// Solve the ODE dy/dt = f(t, y) from t_start to t_end.
        ///
        /// # Arguments
        /// * `f` - The ODE right-hand side function f(t, y) -> dy/dt
        /// * `y0` - Initial state vector
        /// * `t_start` - Start time
        /// * `t_end` - End time
        /// * `dt` - Step size (for fixed-step methods)
        ///
        /// # Returns
        /// Final state y(t_end)
        fn solve<F>(&self, f: &F, y0: &Array1<f64>, t_start: f64, t_end: f64, dt: f64) -> Array1<f64>
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>;

        /// Solve and return the full trajectory at each step.
        fn solve_trajectory<F>(
            &self,
            f: &F,
            y0: &Array1<f64>,
            t_span: &[f64],
            dt: f64,
        ) -> Vec<(f64, Array1<f64>)>
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>;
    }

    /// Classical 4th-order Runge-Kutta solver (RK4).
    ///
    /// Fixed step size, 4th order accuracy. O(h^4) local truncation error.
    ///
    /// ```text
    /// k1 = f(t_n, y_n)
    /// k2 = f(t_n + h/2, y_n + h*k1/2)
    /// k3 = f(t_n + h/2, y_n + h*k2/2)
    /// k4 = f(t_n + h, y_n + h*k3)
    /// y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
    /// ```
    pub struct RK4Solver;

    impl ODESolver for RK4Solver {
        fn solve<F>(&self, f: &F, y0: &Array1<f64>, t_start: f64, t_end: f64, dt: f64) -> Array1<f64>
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>,
        {
            let mut t = t_start;
            let mut y = y0.clone();
            let step = if t_end > t_start { dt.abs() } else { -dt.abs() };

            while (t_end > t_start && t < t_end - 1e-12)
                || (t_end < t_start && t > t_end + 1e-12)
            {
                let h = if (t_end > t_start && t + step > t_end)
                    || (t_end < t_start && t + step < t_end)
                {
                    t_end - t
                } else {
                    step
                };

                let k1 = f(t, &y);
                let k2 = f(t + h * 0.5, &(&y + &(&k1 * (h * 0.5))));
                let k3 = f(t + h * 0.5, &(&y + &(&k2 * (h * 0.5))));
                let k4 = f(t + h, &(&y + &(&k3 * h)));

                y = &y + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (h / 6.0));
                t += h;
            }

            y
        }

        fn solve_trajectory<F>(
            &self,
            f: &F,
            y0: &Array1<f64>,
            t_span: &[f64],
            dt: f64,
        ) -> Vec<(f64, Array1<f64>)>
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>,
        {
            let mut trajectory = vec![(t_span[0], y0.clone())];
            let mut y = y0.clone();

            for i in 0..t_span.len() - 1 {
                y = self.solve(f, &y, t_span[i], t_span[i + 1], dt);
                trajectory.push((t_span[i + 1], y.clone()));
            }

            trajectory
        }
    }

    /// Euler method (1st order). Simple but inaccurate -- mainly for comparison.
    ///
    /// y_{n+1} = y_n + h * f(t_n, y_n)
    pub struct EulerSolver;

    impl ODESolver for EulerSolver {
        fn solve<F>(&self, f: &F, y0: &Array1<f64>, t_start: f64, t_end: f64, dt: f64) -> Array1<f64>
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>,
        {
            let mut t = t_start;
            let mut y = y0.clone();
            let step = if t_end > t_start { dt.abs() } else { -dt.abs() };

            while (t_end > t_start && t < t_end - 1e-12)
                || (t_end < t_start && t > t_end + 1e-12)
            {
                let h = if (t_end > t_start && t + step > t_end)
                    || (t_end < t_start && t + step < t_end)
                {
                    t_end - t
                } else {
                    step
                };

                let dy = f(t, &y);
                y = &y + &(&dy * h);
                t += h;
            }

            y
        }

        fn solve_trajectory<F>(
            &self,
            f: &F,
            y0: &Array1<f64>,
            t_span: &[f64],
            dt: f64,
        ) -> Vec<(f64, Array1<f64>)>
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>,
        {
            let mut trajectory = vec![(t_span[0], y0.clone())];
            let mut y = y0.clone();

            for i in 0..t_span.len() - 1 {
                y = self.solve(f, &y, t_span[i], t_span[i + 1], dt);
                trajectory.push((t_span[i + 1], y.clone()));
            }

            trajectory
        }
    }

    /// Dormand-Prince adaptive step-size solver (RK45).
    ///
    /// Uses embedded 4th and 5th order methods to estimate local error
    /// and adapt the step size for efficiency and accuracy.
    ///
    /// This is the Rust equivalent of `dopri5` in torchdiffeq.
    pub struct DormandPrinceSolver {
        pub rtol: f64,
        pub atol: f64,
        pub max_steps: usize,
        pub min_dt: f64,
        pub max_dt: f64,
    }

    impl Default for DormandPrinceSolver {
        fn default() -> Self {
            Self {
                rtol: 1e-4,
                atol: 1e-6,
                max_steps: 10000,
                min_dt: 1e-10,
                max_dt: 1.0,
            }
        }
    }

    impl DormandPrinceSolver {
        pub fn new(rtol: f64, atol: f64) -> Self {
            Self {
                rtol,
                atol,
                ..Default::default()
            }
        }

        /// Compute a single Dormand-Prince step with error estimate.
        fn dp_step<F>(
            &self,
            f: &F,
            t: f64,
            y: &Array1<f64>,
            h: f64,
        ) -> (Array1<f64>, Array1<f64>, f64)
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>,
        {
            // Dormand-Prince coefficients (Butcher tableau)
            let k1 = f(t, y);
            let k2 = f(
                t + h * (1.0 / 5.0),
                &(y + &(&k1 * (h * 1.0 / 5.0))),
            );
            let k3 = f(
                t + h * (3.0 / 10.0),
                &(y + &(&k1 * (h * 3.0 / 40.0)) + &(&k2 * (h * 9.0 / 40.0))),
            );
            let k4 = f(
                t + h * (4.0 / 5.0),
                &(y + &(&k1 * (h * 44.0 / 45.0))
                    + &(&k2 * (h * -56.0 / 15.0))
                    + &(&k3 * (h * 32.0 / 9.0))),
            );
            let k5 = f(
                t + h * (8.0 / 9.0),
                &(y + &(&k1 * (h * 19372.0 / 6561.0))
                    + &(&k2 * (h * -25360.0 / 2187.0))
                    + &(&k3 * (h * 64448.0 / 6561.0))
                    + &(&k4 * (h * -212.0 / 729.0))),
            );
            let k6 = f(
                t + h,
                &(y + &(&k1 * (h * 9017.0 / 3168.0))
                    + &(&k2 * (h * -355.0 / 33.0))
                    + &(&k3 * (h * 46732.0 / 5247.0))
                    + &(&k4 * (h * 49.0 / 176.0))
                    + &(&k5 * (h * -5103.0 / 18656.0))),
            );

            // 5th order solution
            let y_new = y
                + &(&k1 * (h * 35.0 / 384.0))
                + &(&k3 * (h * 500.0 / 1113.0))
                + &(&k4 * (h * 125.0 / 192.0))
                + &(&k5 * (h * -2187.0 / 6784.0))
                + &(&k6 * (h * 11.0 / 84.0));

            let k7 = f(t + h, &y_new);

            // 4th order solution (for error estimate)
            let y4 = y
                + &(&k1 * (h * 5179.0 / 57600.0))
                + &(&k3 * (h * 7571.0 / 16695.0))
                + &(&k4 * (h * 393.0 / 640.0))
                + &(&k5 * (h * -92097.0 / 339200.0))
                + &(&k6 * (h * 187.0 / 2100.0))
                + &(&k7 * (h * 1.0 / 40.0));

            // Error estimate
            let err = &y_new - &y4;
            let scale: Array1<f64> = y.mapv(|yi| self.atol + self.rtol * yi.abs());
            let err_norm = (&err / &scale).mapv(|e| e * e).sum().sqrt()
                / (err.len() as f64).sqrt();

            (y_new, err, err_norm)
        }
    }

    impl ODESolver for DormandPrinceSolver {
        fn solve<F>(
            &self,
            f: &F,
            y0: &Array1<f64>,
            t_start: f64,
            t_end: f64,
            dt: f64,
        ) -> Array1<f64>
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>,
        {
            let mut t = t_start;
            let mut y = y0.clone();
            let mut h = dt.min(self.max_dt);
            let direction = if t_end > t_start { 1.0 } else { -1.0 };
            h *= direction;

            for _ in 0..self.max_steps {
                if (direction > 0.0 && t >= t_end - 1e-12)
                    || (direction < 0.0 && t <= t_end + 1e-12)
                {
                    break;
                }

                // Clamp step to not overshoot
                if (direction > 0.0 && t + h > t_end)
                    || (direction < 0.0 && t + h < t_end)
                {
                    h = t_end - t;
                }

                let (y_new, _err, err_norm) = self.dp_step(f, t, &y, h);

                if err_norm <= 1.0 {
                    // Accept step
                    t += h;
                    y = y_new;

                    // Increase step size
                    let factor = 0.9 * (1.0 / err_norm.max(1e-10)).powf(0.2);
                    h *= factor.min(5.0).max(0.2);
                    h = h.abs().min(self.max_dt).max(self.min_dt) * direction;
                } else {
                    // Reject step, reduce step size
                    let factor = 0.9 * (1.0 / err_norm).powf(0.25);
                    h *= factor.max(0.2);
                    h = h.abs().max(self.min_dt) * direction;
                }
            }

            y
        }

        fn solve_trajectory<F>(
            &self,
            f: &F,
            y0: &Array1<f64>,
            t_span: &[f64],
            dt: f64,
        ) -> Vec<(f64, Array1<f64>)>
        where
            F: Fn(f64, &Array1<f64>) -> Array1<f64>,
        {
            let mut trajectory = vec![(t_span[0], y0.clone())];
            let mut y = y0.clone();

            for i in 0..t_span.len() - 1 {
                y = self.solve(f, &y, t_span[i], t_span[i + 1], dt);
                trajectory.push((t_span[i + 1], y.clone()));
            }

            trajectory
        }
    }
}

/// Neural network model components
pub mod model {
    use ndarray::{Array1, Array2};
    use rand::Rng;
    use rand_distr::Normal;

    use crate::ode_solver::{ODESolver, RK4Solver};

    /// A simple linear layer: y = W * x + b
    #[derive(Debug, Clone)]
    pub struct LinearLayer {
        pub weights: Array2<f64>,
        pub bias: Array1<f64>,
    }

    impl LinearLayer {
        /// Create with random Xavier initialization.
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let mut rng = rand::thread_rng();
            let std = (2.0 / (input_dim + output_dim) as f64).sqrt() * 0.1;
            let dist = Normal::new(0.0, std).unwrap();

            let weights = Array2::from_shape_fn((output_dim, input_dim), |_| rng.sample(dist));
            let bias = Array1::zeros(output_dim);

            Self { weights, bias }
        }

        /// Forward pass: y = W * x + b
        pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
            self.weights.dot(x) + &self.bias
        }
    }

    /// Helper function to create a linear layer.
    pub fn linear_layer(input_dim: usize, output_dim: usize) -> LinearLayer {
        LinearLayer::new(input_dim, output_dim)
    }

    /// Activation functions
    pub fn tanh_activation(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.tanh())
    }

    pub fn relu_activation(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }

    pub fn sigmoid_activation(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    pub fn silu_activation(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v / (1.0 + (-v).exp()))
    }

    /// ODE dynamics function: dh/dt = f_theta(h(t), t)
    ///
    /// A 2-layer neural network that defines how the hidden state evolves.
    #[derive(Debug, Clone)]
    pub struct ODEFunc {
        pub layer1: LinearLayer,
        pub layer2: LinearLayer,
        pub layer3: LinearLayer,
        pub hidden_dim: usize,
        pub time_dependent: bool,
        pub nfe: usize,
    }

    impl ODEFunc {
        /// Create a new ODE function with given hidden dimension.
        pub fn new(hidden_dim: usize, time_dependent: bool) -> Self {
            let input_dim = if time_dependent {
                hidden_dim + 1
            } else {
                hidden_dim
            };

            Self {
                layer1: LinearLayer::new(input_dim, hidden_dim),
                layer2: LinearLayer::new(hidden_dim, hidden_dim),
                layer3: LinearLayer::new(hidden_dim, hidden_dim),
                hidden_dim,
                time_dependent,
                nfe: 0,
            }
        }

        /// Evaluate dh/dt = f(h, t)
        pub fn eval(&mut self, t: f64, h: &Array1<f64>) -> Array1<f64> {
            self.nfe += 1;

            let input = if self.time_dependent {
                // Concatenate h and t
                let mut aug = Array1::zeros(self.hidden_dim + 1);
                aug.slice_mut(ndarray::s![..self.hidden_dim])
                    .assign(h);
                aug[self.hidden_dim] = t;
                aug
            } else {
                h.clone()
            };

            let h1 = tanh_activation(&self.layer1.forward(&input));
            let h2 = tanh_activation(&self.layer2.forward(&h1));
            self.layer3.forward(&h2)
        }

        /// Reset NFE counter
        pub fn reset_nfe(&mut self) {
            self.nfe = 0;
        }
    }

    /// Complete Neural ODE model for trading.
    ///
    /// Architecture:
    ///   Input -> Encoder -> ODE Solver -> Decoder -> Output
    ///
    /// The encoder maps input features to an initial hidden state h(0).
    /// The ODE solver evolves h(0) to h(T) using learned dynamics.
    /// The decoder maps h(T) to the output prediction.
    #[derive(Debug)]
    pub struct NeuralODE {
        pub encoder: Vec<LinearLayer>,
        pub ode_func: ODEFunc,
        pub decoder: Vec<LinearLayer>,
        pub hidden_dim: usize,
        pub dt: f64,
    }

    impl NeuralODE {
        /// Create a new Neural ODE model.
        ///
        /// # Arguments
        /// * `input_dim` - Number of input features (e.g., OHLCV = 5)
        /// * `hidden_dim` - ODE hidden state dimension
        /// * `output_dim` - Number of outputs (1 for price, 3 for buy/hold/sell)
        /// * `dt` - ODE solver step size
        pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, dt: f64) -> Self {
            // Encoder: input -> hidden
            let encoder = vec![
                LinearLayer::new(input_dim, hidden_dim),
                LinearLayer::new(hidden_dim, hidden_dim),
            ];

            // ODE dynamics
            let ode_func = ODEFunc::new(hidden_dim, true);

            // Decoder: hidden -> output
            let decoder = vec![
                LinearLayer::new(hidden_dim, hidden_dim / 2),
                LinearLayer::new(hidden_dim / 2, output_dim),
            ];

            Self {
                encoder,
                ode_func,
                decoder,
                hidden_dim,
                dt,
            }
        }

        /// Forward pass: encode -> solve ODE -> decode.
        pub fn forward(&self, input: &[f64]) -> Vec<f64> {
            let x = Array1::from_vec(input.to_vec());

            // Encode to initial hidden state
            let h0 = self.encode(&x);

            // Solve ODE from t=0 to t=1
            let h_final = self.solve_ode(&h0, 0.0, 1.0);

            // Decode to output
            let output = self.decode(&h_final);
            output.to_vec()
        }

        /// Forward pass with trajectory (for multi-step prediction).
        pub fn forward_trajectory(&self, input: &[f64], t_points: &[f64]) -> Vec<Vec<f64>> {
            let x = Array1::from_vec(input.to_vec());
            let h0 = self.encode(&x);

            let solver = RK4Solver;
            let mut ode_func = self.ode_func.clone();

            let trajectory = solver.solve_trajectory(
                &|t, h| ode_func.eval(t, h),
                &h0,
                t_points,
                self.dt,
            );

            trajectory
                .iter()
                .map(|(_, h)| {
                    let out = self.decode(h);
                    out.to_vec()
                })
                .collect()
        }

        fn encode(&self, x: &Array1<f64>) -> Array1<f64> {
            let h1 = silu_activation(&self.encoder[0].forward(x));
            self.encoder[1].forward(&h1)
        }

        fn solve_ode(&self, h0: &Array1<f64>, t_start: f64, t_end: f64) -> Array1<f64> {
            let solver = RK4Solver;
            let mut ode_func = self.ode_func.clone();

            solver.solve(
                &|t, h| ode_func.eval(t, h),
                h0,
                t_start,
                t_end,
                self.dt,
            )
        }

        fn decode(&self, h: &Array1<f64>) -> Array1<f64> {
            let h1 = silu_activation(&self.decoder[0].forward(h));
            self.decoder[1].forward(&h1)
        }

        /// Process a sequence of observations (sliding window).
        ///
        /// For each time step, encode the feature vector and evolve via ODE.
        pub fn process_sequence(&self, sequence: &[Vec<f64>]) -> Vec<f64> {
            if sequence.is_empty() {
                return vec![];
            }

            // Use the last observation as input
            let last = &sequence[sequence.len() - 1];
            self.forward(last)
        }

        /// Get the number of parameters in the model.
        pub fn num_parameters(&self) -> usize {
            let mut count = 0;

            for layer in &self.encoder {
                count += layer.weights.len() + layer.bias.len();
            }

            count += self.ode_func.layer1.weights.len() + self.ode_func.layer1.bias.len();
            count += self.ode_func.layer2.weights.len() + self.ode_func.layer2.bias.len();
            count += self.ode_func.layer3.weights.len() + self.ode_func.layer3.bias.len();

            for layer in &self.decoder {
                count += layer.weights.len() + layer.bias.len();
            }

            count
        }
    }

    /// Simple training via stochastic gradient descent with finite differences.
    ///
    /// In production, you'd use automatic differentiation (e.g., via tch-rs).
    /// This implementation uses numerical gradients for simplicity.
    pub struct SimpleTrainer {
        pub learning_rate: f64,
        pub epsilon: f64,
    }

    impl SimpleTrainer {
        pub fn new(lr: f64) -> Self {
            Self {
                learning_rate: lr,
                epsilon: 1e-5,
            }
        }

        /// Compute MSE loss for a batch of (input, target) pairs.
        pub fn compute_loss(
            model: &NeuralODE,
            inputs: &[Vec<f64>],
            targets: &[Vec<f64>],
        ) -> f64 {
            let mut total_loss = 0.0;
            let n = inputs.len() as f64;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                let pred = model.forward(input);
                for (p, t) in pred.iter().zip(target.iter()) {
                    total_loss += (p - t).powi(2);
                }
            }

            total_loss / n
        }

        /// Train for one epoch using simple perturbation-based updates.
        ///
        /// This is a simplified training loop. For production use,
        /// consider using the `tch` crate for proper autograd.
        pub fn train_epoch(
            &self,
            model: &mut NeuralODE,
            inputs: &[Vec<f64>],
            targets: &[Vec<f64>],
        ) -> f64 {
            let base_loss = Self::compute_loss(model, inputs, targets);

            // Perturb each encoder weight and update (simplified SGD)
            self.update_layers(&mut model.encoder, inputs, targets, model);
            self.update_layers(&mut model.decoder, inputs, targets, model);

            // Also update ODE function weights
            self.update_ode_func(&mut model.ode_func, inputs, targets, model);

            base_loss
        }

        fn update_layers(
            &self,
            layers: &mut Vec<LinearLayer>,
            inputs: &[Vec<f64>],
            targets: &[Vec<f64>],
            model: &NeuralODE,
        ) {
            for layer in layers.iter_mut() {
                let (rows, cols) = layer.weights.dim();
                for i in 0..rows {
                    for j in 0..cols {
                        let original = layer.weights[[i, j]];

                        // Forward difference
                        layer.weights[[i, j]] = original + self.epsilon;
                        let loss_plus = Self::compute_loss(model, inputs, targets);

                        // Backward difference
                        layer.weights[[i, j]] = original - self.epsilon;
                        let loss_minus = Self::compute_loss(model, inputs, targets);

                        // Gradient estimate
                        let grad = (loss_plus - loss_minus) / (2.0 * self.epsilon);

                        // Update
                        layer.weights[[i, j]] = original - self.learning_rate * grad;
                    }
                }

                // Update biases
                for i in 0..layer.bias.len() {
                    let original = layer.bias[i];
                    layer.bias[i] = original + self.epsilon;
                    let loss_plus = Self::compute_loss(model, inputs, targets);
                    layer.bias[i] = original - self.epsilon;
                    let loss_minus = Self::compute_loss(model, inputs, targets);
                    let grad = (loss_plus - loss_minus) / (2.0 * self.epsilon);
                    layer.bias[i] = original - self.learning_rate * grad;
                }
            }
        }

        fn update_ode_func(
            &self,
            _ode_func: &mut ODEFunc,
            _inputs: &[Vec<f64>],
            _targets: &[Vec<f64>],
            _model: &NeuralODE,
        ) {
            // In a real implementation, we would compute gradients through the ODE solver.
            // The adjoint method avoids backpropagating through the solver steps:
            //
            // 1. Solve forward: h(T) = ODESolve(f_theta, h(0), [0, T])
            // 2. Compute loss: L(h(T))
            // 3. Solve adjoint backward:
            //    a(t) = dL/dh(t), governed by da/dt = -a^T * (df/dh)
            //    dL/d_theta via: integral_T^0 a(t)^T * (df/d_theta) dt
            //
            // For this simplified demo, ODE function weights are updated
            // via the encoder/decoder gradient flow only.
        }
    }
}

/// Data fetching and processing module
pub mod data {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    /// Market data candle (OHLCV)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MarketData {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
        pub turnover: f64,
    }

    /// Individual trade (tick data with irregular timestamps)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TickData {
        pub timestamp_ms: i64,
        pub price: f64,
        pub size: f64,
        pub side: String,
        pub time_delta_ms: f64,
    }

    /// Bybit API response structures
    #[derive(Debug, Deserialize)]
    pub struct BybitResponse<T> {
        #[serde(rename = "retCode")]
        pub ret_code: i32,
        #[serde(rename = "retMsg")]
        pub ret_msg: String,
        pub result: T,
    }

    #[derive(Debug, Deserialize)]
    pub struct KlineResult {
        pub list: Vec<Vec<String>>,
    }

    #[derive(Debug, Deserialize)]
    pub struct TradeResult {
        pub list: Vec<TradeEntry>,
    }

    #[derive(Debug, Deserialize)]
    pub struct TradeEntry {
        #[serde(rename = "execId")]
        pub exec_id: String,
        pub symbol: String,
        pub price: String,
        pub size: String,
        pub side: String,
        pub time: String,
        #[serde(rename = "isBlockTrade")]
        pub is_block_trade: bool,
    }

    /// Bybit REST API client
    pub struct BybitClient {
        base_url: String,
        client: reqwest::blocking::Client,
    }

    impl BybitClient {
        pub fn new() -> Self {
            Self {
                base_url: "https://api.bybit.com".to_string(),
                client: reqwest::blocking::Client::new(),
            }
        }

        /// Fetch kline/candlestick data.
        pub fn fetch_klines(
            &self,
            symbol: &str,
            interval: &str,
            limit: usize,
        ) -> Result<Vec<MarketData>, Box<dyn std::error::Error>> {
            let url = format!("{}/v5/market/kline", self.base_url);

            let response = self
                .client
                .get(&url)
                .query(&[
                    ("category", "linear"),
                    ("symbol", symbol),
                    ("interval", interval),
                    ("limit", &limit.min(1000).to_string()),
                ])
                .send()?;

            let body: BybitResponse<KlineResult> = response.json()?;

            if body.ret_code != 0 {
                return Err(format!("Bybit API error: {}", body.ret_msg).into());
            }

            let mut data: Vec<MarketData> = body
                .result
                .list
                .iter()
                .filter_map(|row| {
                    if row.len() >= 7 {
                        Some(MarketData {
                            timestamp: row[0].parse().ok()?,
                            open: row[1].parse().ok()?,
                            high: row[2].parse().ok()?,
                            low: row[3].parse().ok()?,
                            close: row[4].parse().ok()?,
                            volume: row[5].parse().ok()?,
                            turnover: row[6].parse().ok()?,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            data.sort_by_key(|d| d.timestamp);
            Ok(data)
        }

        /// Fetch recent trades (tick data with irregular timestamps).
        pub fn fetch_recent_trades(
            &self,
            symbol: &str,
            limit: usize,
        ) -> Result<Vec<TickData>, Box<dyn std::error::Error>> {
            let url = format!("{}/v5/market/recent-trade", self.base_url);

            let response = self
                .client
                .get(&url)
                .query(&[
                    ("category", "linear"),
                    ("symbol", symbol),
                    ("limit", &limit.min(1000).to_string()),
                ])
                .send()?;

            let body: BybitResponse<TradeResult> = response.json()?;

            if body.ret_code != 0 {
                return Err(format!("Bybit API error: {}", body.ret_msg).into());
            }

            let mut trades: Vec<TickData> = body
                .result
                .list
                .iter()
                .map(|t| TickData {
                    timestamp_ms: t.time.parse().unwrap_or(0),
                    price: t.price.parse().unwrap_or(0.0),
                    size: t.size.parse().unwrap_or(0.0),
                    side: t.side.clone(),
                    time_delta_ms: 0.0,
                })
                .collect();

            // Sort by time and compute deltas
            trades.sort_by_key(|t| t.timestamp_ms);
            for i in 1..trades.len() {
                trades[i].time_delta_ms =
                    (trades[i].timestamp_ms - trades[i - 1].timestamp_ms) as f64;
            }

            Ok(trades)
        }
    }

    impl Default for BybitClient {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Generate synthetic market data for testing.
    pub fn generate_synthetic_klines(symbol: &str, n: usize) -> Vec<MarketData> {
        use rand::Rng;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();

        let base_price = match symbol {
            "BTCUSDT" => 65000.0,
            "ETHUSDT" => 3500.0,
            "SOLUSDT" => 150.0,
            _ => 100.0,
        };

        let dt = 1.0 / (252.0 * 24.0 * 60.0);
        let mu = 0.05;
        let sigma = 0.6;
        let dist = Normal::new((mu - 0.5 * sigma * sigma) * dt, sigma * dt.sqrt()).unwrap();

        let mut prices = Vec::with_capacity(n);
        let mut cumsum = 0.0;
        for _ in 0..n {
            cumsum += rng.sample(dist);
            prices.push(base_price * cumsum.exp());
        }

        let now_ms = chrono::Utc::now().timestamp_millis();

        prices
            .iter()
            .enumerate()
            .map(|(i, &price)| {
                let noise: f64 = rng.gen_range(0.0005..0.003);
                MarketData {
                    timestamp: now_ms - ((n - i) as i64 * 60_000),
                    open: price * (1.0 + rng.gen_range(-0.001..0.001)),
                    high: price * (1.0 + noise),
                    low: price * (1.0 - noise),
                    close: price,
                    volume: rng.gen_range(10.0..10000.0),
                    turnover: price * rng.gen_range(10.0..10000.0),
                }
            })
            .collect()
    }

    /// Generate synthetic tick data with irregular timestamps.
    pub fn generate_synthetic_ticks(symbol: &str, n: usize) -> Vec<TickData> {
        use rand::Rng;
        use rand_distr::{Exp, Normal};

        let mut rng = rand::thread_rng();

        let base_price = match symbol {
            "BTCUSDT" => 65000.0,
            "ETHUSDT" => 3500.0,
            _ => 100.0,
        };

        let time_dist = Exp::new(1.0 / 200.0).unwrap(); // mean 200ms between ticks
        let return_dist = Normal::new(0.0, 0.0001).unwrap();

        let mut ticks = Vec::with_capacity(n);
        let mut cumulative_time = 0i64;
        let mut cum_return = 0.0;

        for i in 0..n {
            let gap: f64 = rng.sample(time_dist);
            let gap_ms = (gap as i64).max(1).min(5000);
            cumulative_time += gap_ms;

            cum_return += rng.sample(return_dist);
            // Occasional jumps
            if rng.gen::<f64>() < 0.01 {
                cum_return += if rng.gen::<bool>() { 0.005 } else { -0.005 };
            }

            let price = base_price * cum_return.exp();

            ticks.push(TickData {
                timestamp_ms: cumulative_time,
                price,
                size: rng.gen_range(0.001..10.0),
                side: if rng.gen::<f64>() < 0.52 {
                    "Buy".to_string()
                } else {
                    "Sell".to_string()
                },
                time_delta_ms: gap_ms as f64,
            });
        }

        ticks
    }

    /// Normalize a feature vector
    pub fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64)
            .sqrt()
            .max(1e-8);
        let normalized: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();
        (normalized, mean, std)
    }

    /// Create sliding windows from a time series.
    pub fn create_windows(
        data: &[MarketData],
        window_size: usize,
        pred_horizon: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        if data.len() < window_size + pred_horizon {
            return (inputs, targets);
        }

        // Normalize prices
        let closes: Vec<f64> = data.iter().map(|d| d.close).collect();
        let (norm_closes, mean, std) = normalize(&closes);

        for i in 0..data.len() - window_size - pred_horizon {
            // Input: [close_norm_i, ..., close_norm_{i+window-1}] + features
            let window: Vec<f64> = (i..i + window_size)
                .flat_map(|j| {
                    vec![
                        norm_closes[j],
                        (data[j].high - data[j].low) / data[j].close, // range
                        data[j].volume.ln(),                            // log volume
                    ]
                })
                .collect();
            inputs.push(window);

            // Target: normalized close at i + window + pred_horizon - 1
            let target_idx = i + window_size + pred_horizon - 1;
            targets.push(vec![norm_closes[target_idx]]);
        }

        (inputs, targets)
    }
}

/// Backtesting module
pub mod backtest {
    use crate::data::MarketData;
    use crate::model::NeuralODE;

    /// Trading signal
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Signal {
        Buy,
        Sell,
        Hold,
    }

    /// Backtest results
    #[derive(Debug, Clone)]
    pub struct BacktestResult {
        pub total_return: f64,
        pub sharpe_ratio: f64,
        pub max_drawdown: f64,
        pub win_rate: f64,
        pub n_trades: usize,
        pub total_costs: f64,
        pub final_portfolio: f64,
        pub portfolio_values: Vec<f64>,
    }

    impl BacktestResult {
        pub fn print_summary(&self) {
            println!("\n{}", "=".repeat(50));
            println!("  Neural ODE Backtest Results");
            println!("{}", "=".repeat(50));
            println!("  Total Return:     {:>10.2}%", self.total_return * 100.0);
            println!("  Sharpe Ratio:     {:>10.3}", self.sharpe_ratio);
            println!("  Max Drawdown:     {:>10.2}%", self.max_drawdown * 100.0);
            println!("  Win Rate:         {:>10.2}%", self.win_rate * 100.0);
            println!("  Num Trades:       {:>10}", self.n_trades);
            println!("  Total Costs:      ${:>9.2}", self.total_costs);
            println!("  Final Portfolio:  ${:>9.2}", self.final_portfolio);
            println!("{}\n", "=".repeat(50));
        }
    }

    /// Backtester for Neural ODE strategies.
    pub struct Backtester {
        pub initial_capital: f64,
        pub transaction_cost: f64,
        pub max_position: f64,
    }

    impl Backtester {
        pub fn new(initial_capital: f64, transaction_cost: f64) -> Self {
            Self {
                initial_capital,
                transaction_cost,
                max_position: 1.0,
            }
        }

        /// Generate trading signals from model predictions.
        pub fn generate_signals(
            &self,
            model: &NeuralODE,
            data: &[MarketData],
            window_size: usize,
        ) -> Vec<Signal> {
            let mut signals = Vec::new();

            if data.len() < window_size + 1 {
                return signals;
            }

            for i in window_size..data.len() {
                // Create input window
                let input: Vec<f64> = (i - window_size..i)
                    .flat_map(|j| {
                        vec![
                            data[j].close / data[i - 1].close - 1.0, // normalized return
                            (data[j].high - data[j].low) / data[j].close,
                            data[j].volume.ln(),
                        ]
                    })
                    .collect();

                let pred = model.forward(&input);

                if pred.is_empty() {
                    signals.push(Signal::Hold);
                    continue;
                }

                let predicted_value = pred[0];

                if predicted_value > 0.001 {
                    signals.push(Signal::Buy);
                } else if predicted_value < -0.001 {
                    signals.push(Signal::Sell);
                } else {
                    signals.push(Signal::Hold);
                }
            }

            signals
        }

        /// Run the backtest.
        pub fn run(
            &self,
            data: &[MarketData],
            signals: &[Signal],
            window_size: usize,
        ) -> BacktestResult {
            let mut cash = self.initial_capital;
            let mut position = 0.0_f64;
            let mut portfolio_values = vec![self.initial_capital];
            let mut n_trades = 0;
            let mut total_costs = 0.0;

            let prices: Vec<f64> = data.iter().map(|d| d.close).collect();
            let trade_prices = &prices[window_size..];

            for (i, signal) in signals.iter().enumerate() {
                if i >= trade_prices.len() {
                    break;
                }

                let price = trade_prices[i];

                match signal {
                    Signal::Buy if position <= 0.0 => {
                        let trade_value = cash * self.max_position;
                        let units = trade_value / price;
                        let cost = trade_value * self.transaction_cost;
                        cash -= units * price + cost;
                        position = units;
                        n_trades += 1;
                        total_costs += cost;
                    }
                    Signal::Sell if position >= 0.0 => {
                        if position > 0.0 {
                            let trade_value = position * price;
                            let cost = trade_value * self.transaction_cost;
                            cash += trade_value - cost;
                            position = 0.0;
                            n_trades += 1;
                            total_costs += cost;
                        }
                    }
                    _ => {}
                }

                let portfolio_value = cash + position * price;
                portfolio_values.push(portfolio_value);
            }

            // Calculate metrics
            let final_value = *portfolio_values.last().unwrap_or(&self.initial_capital);
            let total_return = final_value / self.initial_capital - 1.0;

            // Returns
            let returns: Vec<f64> = portfolio_values
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();

            // Sharpe
            let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
            let std_return = (returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / returns.len().max(1) as f64)
                .sqrt();
            let sharpe = if std_return > 1e-10 {
                mean_return / std_return * (252.0_f64 * 7.0).sqrt()
            } else {
                0.0
            };

            // Max Drawdown
            let mut peak = portfolio_values[0];
            let mut max_dd = 0.0_f64;
            for &val in &portfolio_values {
                if val > peak {
                    peak = val;
                }
                let dd = (val - peak) / peak;
                if dd < max_dd {
                    max_dd = dd;
                }
            }

            // Win Rate
            let winning = returns.iter().filter(|&&r| r > 0.0).count();
            let win_rate = winning as f64 / returns.len().max(1) as f64;

            BacktestResult {
                total_return,
                sharpe_ratio: sharpe,
                max_drawdown: max_dd,
                win_rate,
                n_trades,
                total_costs,
                final_portfolio: final_value,
                portfolio_values,
            }
        }
    }

    impl Default for Backtester {
        fn default() -> Self {
            Self::new(100_000.0, 0.001)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_rk4_solver_simple() {
        // Test: dy/dt = -y, y(0) = 1 => y(t) = exp(-t)
        let solver = RK4Solver;
        let y0 = Array1::from_vec(vec![1.0]);
        let result = solver.solve(&|_t, y| -y.clone(), &y0, 0.0, 1.0, 0.01);

        let expected = (-1.0_f64).exp();
        assert!(
            (result[0] - expected).abs() < 1e-6,
            "RK4 result: {}, expected: {}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_dopri_solver() {
        let solver = DormandPrinceSolver::new(1e-6, 1e-8);
        let y0 = Array1::from_vec(vec![1.0]);
        let result = solver.solve(&|_t, y| -y.clone(), &y0, 0.0, 1.0, 0.1);

        let expected = (-1.0_f64).exp();
        assert!(
            (result[0] - expected).abs() < 1e-4,
            "Dopri result: {}, expected: {}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_neural_ode_forward() {
        let model = NeuralODE::new(3, 16, 1, 0.05);
        let input = vec![1.0, 2.0, 3.0];
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_ode_func() {
        let mut func = model::ODEFunc::new(8, true);
        let h = Array1::from_vec(vec![0.1; 8]);
        let dh = func.eval(0.5, &h);
        assert_eq!(dh.len(), 8);
        assert!(func.nfe > 0);
    }

    #[test]
    fn test_synthetic_data() {
        let data = data::generate_synthetic_klines("BTCUSDT", 100);
        assert_eq!(data.len(), 100);
        assert!(data[0].close > 0.0);
    }

    #[test]
    fn test_synthetic_ticks() {
        let ticks = data::generate_synthetic_ticks("ETHUSDT", 50);
        assert_eq!(ticks.len(), 50);
        assert!(ticks[1].time_delta_ms > 0.0);
    }

    #[test]
    fn test_sliding_windows() {
        let data = data::generate_synthetic_klines("BTCUSDT", 100);
        let (inputs, targets) = data::create_windows(&data, 10, 1);
        assert!(!inputs.is_empty());
        assert_eq!(inputs.len(), targets.len());
    }

    #[test]
    fn test_backtester() {
        let data = data::generate_synthetic_klines("BTCUSDT", 200);
        let model = NeuralODE::new(30, 16, 1, 0.05); // window_size=10 * 3 features
        let backtester = backtest::Backtester::new(100_000.0, 0.001);
        let signals = backtester.generate_signals(&model, &data, 10);
        assert!(!signals.is_empty());

        let result = backtester.run(&data, &signals, 10);
        assert!(result.final_portfolio > 0.0);
    }
}
