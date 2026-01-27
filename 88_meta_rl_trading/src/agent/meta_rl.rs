//! Meta-RL agent using GRU-based RL^2 framework.
//!
//! The agent encodes a learning algorithm within its recurrent hidden state,
//! enabling rapid adaptation to new trading environments without gradient steps.
//!
//! Reference: Duan et al., 2016. "RL^2: Fast Reinforcement Learning via Slow
//! Reinforcement Learning"

use rand_distr::{Distribution, Normal};

/// GRU (Gated Recurrent Unit) cell
#[derive(Debug, Clone)]
pub struct GRUCell {
    /// Update gate weights: W_z (input) and U_z (hidden)
    wz: Vec<Vec<f64>>,
    uz: Vec<Vec<f64>>,
    bz: Vec<f64>,
    /// Reset gate weights: W_r and U_r
    wr: Vec<Vec<f64>>,
    ur: Vec<Vec<f64>>,
    br: Vec<f64>,
    /// Candidate weights: W_h and U_h
    wh: Vec<Vec<f64>>,
    uh: Vec<Vec<f64>>,
    bh: Vec<f64>,
    input_size: usize,
    hidden_size: usize,
}

impl GRUCell {
    /// Create a new GRU cell with Xavier initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std_in = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let std_hid = (2.0 / (hidden_size + hidden_size) as f64).sqrt();
        let normal_in = Normal::new(0.0, std_in).unwrap();
        let normal_hid = Normal::new(0.0, std_hid).unwrap();

        let init_w = |rows: usize, cols: usize, normal: &Normal<f64>| -> Vec<Vec<f64>> {
            (0..rows).map(|_| (0..cols).map(|_| normal.sample(&mut rng)).collect()).collect()
        };

        Self {
            wz: init_w(hidden_size, input_size, &normal_in),
            uz: init_w(hidden_size, hidden_size, &normal_hid),
            bz: vec![0.0; hidden_size],
            wr: init_w(hidden_size, input_size, &normal_in),
            ur: init_w(hidden_size, hidden_size, &normal_hid),
            br: vec![0.0; hidden_size],
            wh: init_w(hidden_size, input_size, &normal_in),
            uh: init_w(hidden_size, hidden_size, &normal_hid),
            bh: vec![0.0; hidden_size],
            input_size,
            hidden_size,
        }
    }

    /// Forward pass through GRU cell
    pub fn forward(&self, input: &[f64], hidden: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);
        assert_eq!(hidden.len(), self.hidden_size);

        let mut z = vec![0.0; self.hidden_size];
        let mut r = vec![0.0; self.hidden_size];
        let mut h_candidate = vec![0.0; self.hidden_size];

        for i in 0..self.hidden_size {
            // Update gate: z = sigmoid(Wz*x + Uz*h + bz)
            let mut val = self.bz[i];
            for j in 0..self.input_size {
                val += self.wz[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                val += self.uz[i][j] * hidden[j];
            }
            z[i] = sigmoid(val);

            // Reset gate: r = sigmoid(Wr*x + Ur*h + br)
            val = self.br[i];
            for j in 0..self.input_size {
                val += self.wr[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                val += self.ur[i][j] * hidden[j];
            }
            r[i] = sigmoid(val);
        }

        for i in 0..self.hidden_size {
            // Candidate: h~ = tanh(Wh*x + Uh*(r*h) + bh)
            let mut val = self.bh[i];
            for j in 0..self.input_size {
                val += self.wh[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                val += self.uh[i][j] * (r[j] * hidden[j]);
            }
            h_candidate[i] = val.tanh();
        }

        // New hidden: h_new = (1 - z) * h + z * h~
        let mut new_hidden = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            new_hidden[i] = (1.0 - z[i]) * hidden[i] + z[i] * h_candidate[i];
        }

        new_hidden
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let w_params = self.input_size * self.hidden_size * 3;
        let u_params = self.hidden_size * self.hidden_size * 3;
        let b_params = self.hidden_size * 3;
        w_params + u_params + b_params
    }

    /// Get all parameters as a flat vector
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_parameters());
        for matrices in [&self.wz, &self.wr, &self.wh] {
            for row in matrices {
                params.extend(row.iter());
            }
        }
        for matrices in [&self.uz, &self.ur, &self.uh] {
            for row in matrices {
                params.extend(row.iter());
            }
        }
        params.extend(&self.bz);
        params.extend(&self.br);
        params.extend(&self.bh);
        params
    }

    /// Set parameters from a flat vector
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut idx = 0;
        for matrices in [&mut self.wz, &mut self.wr, &mut self.wh] {
            for row in matrices.iter_mut() {
                for val in row.iter_mut() {
                    *val = params[idx];
                    idx += 1;
                }
            }
        }
        for matrices in [&mut self.uz, &mut self.ur, &mut self.uh] {
            for row in matrices.iter_mut() {
                for val in row.iter_mut() {
                    *val = params[idx];
                    idx += 1;
                }
            }
        }
        for bias in [&mut self.bz, &mut self.br, &mut self.bh] {
            for val in bias.iter_mut() {
                *val = params[idx];
                idx += 1;
            }
        }
    }
}

/// Dense layer
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / (input_size + output_size) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        Self {
            weights,
            biases: vec![0.0; output_size],
            input_size,
            output_size,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            output[i] = self.biases[i];
            for j in 0..self.input_size {
                output[i] += self.weights[i][j] * input[j];
            }
        }
        output
    }

    pub fn forward_relu(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input).iter().map(|x| x.max(0.0)).collect()
    }

    pub fn num_parameters(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }

    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_parameters());
        for row in &self.weights {
            params.extend(row.iter());
        }
        params.extend(&self.biases);
        params
    }

    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut idx = 0;
        for row in self.weights.iter_mut() {
            for val in row.iter_mut() {
                *val = params[idx];
                idx += 1;
            }
        }
        for val in self.biases.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
    }
}

/// Meta-RL Agent using RL^2 framework
///
/// The agent consists of:
/// - An encoder layer that processes (state, prev_action, prev_reward, done)
/// - A GRU cell whose hidden state encodes the learning algorithm
/// - A policy head (outputs action logits)
/// - A value head (outputs state value)
#[derive(Debug, Clone)]
pub struct MetaRLAgent {
    encoder: DenseLayer,
    gru: GRUCell,
    policy_head: DenseLayer,
    value_head: DenseLayer,
    hidden_state: Vec<f64>,
    state_dim: usize,
    action_dim: usize,
    hidden_size: usize,
}

impl MetaRLAgent {
    /// Create a new Meta-RL agent
    pub fn new(state_dim: usize, action_dim: usize, hidden_size: usize) -> Self {
        // Input: state + prev_action (one-hot) + prev_reward + done
        let input_size = state_dim + action_dim + 2;

        Self {
            encoder: DenseLayer::new(input_size, hidden_size),
            gru: GRUCell::new(hidden_size, hidden_size),
            policy_head: DenseLayer::new(hidden_size, action_dim),
            value_head: DenseLayer::new(hidden_size, 1),
            hidden_state: vec![0.0; hidden_size],
            state_dim,
            action_dim,
            hidden_size,
        }
    }

    /// Reset hidden state to zeros
    pub fn reset_hidden(&mut self) {
        self.hidden_state = vec![0.0; self.hidden_size];
    }

    /// Forward pass: returns (action_probs, value)
    pub fn forward(
        &mut self,
        state: &[f64],
        prev_action: usize,
        prev_reward: f64,
        done: bool,
    ) -> (Vec<f64>, f64) {
        // Build input
        let mut input = Vec::with_capacity(self.state_dim + self.action_dim + 2);
        input.extend_from_slice(state);

        // One-hot prev_action
        for i in 0..self.action_dim {
            input.push(if i == prev_action { 1.0 } else { 0.0 });
        }
        input.push(prev_reward);
        input.push(if done { 1.0 } else { 0.0 });

        // Encode
        let encoded = self.encoder.forward_relu(&input);

        // GRU step
        self.hidden_state = self.gru.forward(&encoded, &self.hidden_state);

        // Policy (softmax)
        let logits = self.policy_head.forward(&self.hidden_state);
        let action_probs = softmax(&logits);

        // Value
        let value = self.value_head.forward(&self.hidden_state)[0];

        (action_probs, value)
    }

    /// Select action by sampling from the policy
    pub fn get_action(
        &mut self,
        state: &[f64],
        prev_action: usize,
        prev_reward: f64,
        done: bool,
    ) -> (usize, f64) {
        let (probs, value) = self.forward(state, prev_action, prev_reward, done);

        // Sample from categorical distribution
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(0.0, 1.0);
        let sample: f64 = uniform.sample(&mut rng);

        let mut cumulative = 0.0;
        let mut action = 0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if sample < cumulative {
                action = i;
                break;
            }
        }

        (action, value)
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.encoder.num_parameters()
            + self.gru.num_parameters()
            + self.policy_head.num_parameters()
            + self.value_head.num_parameters()
    }

    /// Get all parameters as a flat vector
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_parameters());
        params.extend(self.encoder.get_parameters());
        params.extend(self.gru.get_parameters());
        params.extend(self.policy_head.get_parameters());
        params.extend(self.value_head.get_parameters());
        params
    }

    /// Set parameters from flat vector
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut idx = 0;

        let n = self.encoder.num_parameters();
        self.encoder.set_parameters(&params[idx..idx + n]);
        idx += n;

        let n = self.gru.num_parameters();
        self.gru.set_parameters(&params[idx..idx + n]);
        idx += n;

        let n = self.policy_head.num_parameters();
        self.policy_head.set_parameters(&params[idx..idx + n]);
        idx += n;

        let n = self.value_head.num_parameters();
        self.value_head.set_parameters(&params[idx..idx + n]);
    }

    /// Clone the agent
    pub fn clone_agent(&self) -> Self {
        self.clone()
    }

    /// SGD update on all parameters
    pub fn sgd_step(&mut self, gradients: &[f64], learning_rate: f64) {
        let mut params = self.get_parameters();
        for (p, g) in params.iter_mut().zip(gradients.iter()) {
            *p -= learning_rate * g;
        }
        self.set_parameters(&params);
    }

    pub fn state_dim(&self) -> usize { self.state_dim }
    pub fn action_dim(&self) -> usize { self.action_dim }
    pub fn hidden_size(&self) -> usize { self.hidden_size }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = logits.iter().map(|x| (x - max_logit).exp()).sum();
    logits.iter().map(|x| (x - max_logit).exp() / exp_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let agent = MetaRLAgent::new(14, 3, 64);
        assert_eq!(agent.state_dim(), 14);
        assert_eq!(agent.action_dim(), 3);
        assert_eq!(agent.hidden_size(), 64);
        assert!(agent.num_parameters() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let mut agent = MetaRLAgent::new(4, 3, 16);
        let state = vec![0.1, 0.2, 0.3, 0.4];
        let (probs, value) = agent.forward(&state, 0, 0.0, false);

        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(value.is_finite());
    }

    #[test]
    fn test_hidden_state_updates() {
        let mut agent = MetaRLAgent::new(4, 3, 16);
        let state = vec![0.1, 0.2, 0.3, 0.4];

        let hidden_before = agent.hidden_state.clone();
        agent.forward(&state, 0, 0.0, false);
        let hidden_after = agent.hidden_state.clone();

        // Hidden state should change after forward pass
        let changed = hidden_before.iter().zip(hidden_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed);
    }

    #[test]
    fn test_reset_hidden() {
        let mut agent = MetaRLAgent::new(4, 3, 16);
        let state = vec![0.1, 0.2, 0.3, 0.4];
        agent.forward(&state, 0, 0.0, false);

        agent.reset_hidden();
        assert!(agent.hidden_state.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_parameter_round_trip() {
        let agent = MetaRLAgent::new(4, 3, 8);
        let params = agent.get_parameters();
        assert_eq!(params.len(), agent.num_parameters());

        let mut agent2 = MetaRLAgent::new(4, 3, 8);
        agent2.set_parameters(&params);
        let params2 = agent2.get_parameters();

        for (a, b) in params.iter().zip(params2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_get_action() {
        let mut agent = MetaRLAgent::new(4, 3, 16);
        let state = vec![0.1, 0.2, 0.3, 0.4];
        let (action, value) = agent.get_action(&state, 0, 0.0, false);

        assert!(action < 3);
        assert!(value.is_finite());
    }
}
