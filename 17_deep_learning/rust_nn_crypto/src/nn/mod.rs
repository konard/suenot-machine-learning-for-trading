//! Neural Network Module
//!
//! Provides building blocks for feedforward neural networks:
//! - Activation functions (ReLU, Sigmoid, Tanh, Softmax)
//! - Dense layers with forward and backward propagation
//! - Full network with training capabilities

mod activation;
mod layer;
mod network;
mod optimizer;

pub use activation::{Activation, ActivationType};
pub use layer::DenseLayer;
pub use network::NeuralNetwork;
pub use optimizer::{Optimizer, SGD, Adam};
