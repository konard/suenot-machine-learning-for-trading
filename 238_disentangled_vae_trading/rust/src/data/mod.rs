//! Data loading and preprocessing module
//!
//! Provides tools for loading market data and computing features
//! suitable for disentangled VAE training.

mod dataset;
mod features;
mod loader;

pub use dataset::Dataset;
pub use features::Features;
pub use loader::DataLoader;
