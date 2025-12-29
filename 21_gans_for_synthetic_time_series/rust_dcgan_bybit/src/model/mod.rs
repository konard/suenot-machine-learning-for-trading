//! Model module containing GAN architecture components
//!
//! This module provides:
//! - Generator network for creating synthetic time series
//! - Discriminator network for distinguishing real from fake
//! - DCGAN wrapper combining both networks

mod generator;
mod discriminator;
mod dcgan;

pub use generator::Generator;
pub use discriminator::Discriminator;
pub use dcgan::DCGAN;
