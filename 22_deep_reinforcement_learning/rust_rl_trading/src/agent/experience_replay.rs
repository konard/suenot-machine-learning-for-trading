//! Experience replay buffer for DQN.

use crate::environment::{TradingAction, TradingState};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::VecDeque;

/// Single experience tuple
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: TradingState,
    pub action: TradingAction,
    pub reward: f64,
    pub next_state: TradingState,
    pub done: bool,
}

impl Experience {
    pub fn new(
        state: TradingState,
        action: TradingAction,
        reward: f64,
        next_state: TradingState,
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}

/// Replay buffer for storing and sampling experiences
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add an experience to the buffer
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample a batch of experiences randomly
    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(batch_size.min(self.buffer.len()))
            .map(|i| self.buffer[i].clone())
            .collect()
    }

    /// Get the current size of the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if buffer has enough samples for training
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.buffer.len() >= batch_size
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Prioritized replay buffer (simplified version)
pub struct PrioritizedReplayBuffer {
    buffer: VecDeque<(Experience, f64)>, // (experience, priority)
    capacity: usize,
    alpha: f64, // Priority exponent
    beta: f64,  // Importance sampling exponent
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            alpha,
            beta,
        }
    }

    /// Add experience with initial max priority
    pub fn push(&mut self, experience: Experience) {
        let max_priority = self
            .buffer
            .iter()
            .map(|(_, p)| *p)
            .fold(1.0, f64::max);

        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back((experience, max_priority));
    }

    /// Update priority for an experience
    pub fn update_priority(&mut self, index: usize, td_error: f64) {
        if index < self.buffer.len() {
            let priority = (td_error.abs() + 1e-6).powf(self.alpha);
            self.buffer[index].1 = priority;
        }
    }

    /// Sample with priorities
    pub fn sample(&self, batch_size: usize) -> Vec<(usize, Experience, f64)> {
        let total_priority: f64 = self.buffer.iter().map(|(_, p)| *p).sum();
        let mut rng = rand::thread_rng();

        let mut samples = Vec::with_capacity(batch_size);
        let n = self.buffer.len() as f64;

        for _ in 0..batch_size.min(self.buffer.len()) {
            // Sample based on probability
            let sample_val: f64 = rand::Rng::gen(&mut rng);
            let target = sample_val * total_priority;

            let mut cumsum = 0.0;
            for (i, (exp, priority)) in self.buffer.iter().enumerate() {
                cumsum += priority;
                if cumsum >= target {
                    // Calculate importance sampling weight
                    let prob = priority / total_priority;
                    let weight = (n * prob).powf(-self.beta);
                    samples.push((i, exp.clone(), weight));
                    break;
                }
            }
        }

        // Normalize weights
        let max_weight = samples.iter().map(|(_, _, w)| *w).fold(0.0, f64::max);
        if max_weight > 0.0 {
            for (_, _, w) in &mut samples {
                *w /= max_weight;
            }
        }

        samples
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.buffer.len() >= batch_size
    }

    /// Increase beta towards 1.0
    pub fn update_beta(&mut self, step: usize, total_steps: usize) {
        self.beta = 0.4 + 0.6 * (step as f64 / total_steps as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn create_dummy_experience() -> Experience {
        let state = TradingState::new(Array1::zeros(7), 0.0, 0.0, 0.0, 0.0);
        let next_state = TradingState::new(Array1::zeros(7), 1.0, 0.01, 0.1, 0.1);
        Experience::new(state, TradingAction::Long, 0.5, next_state, false)
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);

        for _ in 0..50 {
            buffer.push(create_dummy_experience());
        }

        assert_eq!(buffer.len(), 50);
        assert!(buffer.can_sample(32));

        let samples = buffer.sample(32);
        assert_eq!(samples.len(), 32);
    }

    #[test]
    fn test_buffer_capacity() {
        let mut buffer = ReplayBuffer::new(10);

        for _ in 0..20 {
            buffer.push(create_dummy_experience());
        }

        assert_eq!(buffer.len(), 10);
    }
}
