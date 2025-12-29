//! # Agent Module
//!
//! Реализация RL агентов для оптимального исполнения.

mod traits;
mod q_learning;
mod dqn;
mod replay_buffer;
mod neural_network;

pub use traits::Agent;
pub use q_learning::QLearningAgent;
pub use dqn::{DQNAgent, DQNConfig};
pub use replay_buffer::ReplayBuffer;
pub use neural_network::{NeuralNetwork, Layer};
