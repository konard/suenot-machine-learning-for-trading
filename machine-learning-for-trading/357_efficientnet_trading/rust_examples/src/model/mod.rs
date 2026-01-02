//! EfficientNet model module
//!
//! Provides model architecture definitions and inference capabilities.

mod efficientnet;
mod blocks;
mod inference;

pub use efficientnet::{EfficientNetConfig, EfficientNetVariant};
pub use blocks::{MBConvConfig, SwishActivation};
pub use inference::{ModelPredictor, PredictionResult};
