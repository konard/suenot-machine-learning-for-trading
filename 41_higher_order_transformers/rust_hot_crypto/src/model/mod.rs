//! Model module for Higher Order Transformer
//!
//! Contains the full HOT model architecture for price prediction

mod transformer;
mod predictor;

pub use transformer::HOTModel;
pub use predictor::{HOTPredictor, PredictionResult, MovementClass};
