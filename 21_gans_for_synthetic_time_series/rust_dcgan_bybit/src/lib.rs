//! # DCGAN for Cryptocurrency Time Series
//!
//! This crate provides a modular implementation of Deep Convolutional Generative
//! Adversarial Networks (DCGAN) for generating synthetic cryptocurrency price data.
//!
//! ## Modules
//!
//! - `data`: Data fetching from Bybit API and preprocessing
//! - `model`: DCGAN architecture (Generator and Discriminator)
//! - `training`: Training loop and loss functions
//! - `utils`: Helper functions and configuration

pub mod data;
pub mod model;
pub mod training;
pub mod utils;

pub use data::{BybitClient, OHLCVData, OHLCVDataset, DataLoader, NormalizationParams};
pub use data::{normalize_data, denormalize_data, create_sequences};
pub use model::{Generator, Discriminator, DCGAN};
pub use training::{Trainer, TrainingConfig, TrainingMetrics};
pub use utils::{Config, save_checkpoint, load_checkpoint};
