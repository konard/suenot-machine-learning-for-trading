//! Bayesian Neural Network module.

pub mod config;
pub mod layers;
pub mod model;
pub mod uncertainty;

pub use config::BNNConfig;
pub use layers::BayesianLinear;
pub use model::BayesianNN;
pub use uncertainty::{UncertaintyEstimate, UncertaintyType};
