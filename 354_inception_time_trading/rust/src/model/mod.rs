//! InceptionTime model architecture
//!
//! This module provides:
//! - Inception module for multi-scale feature extraction
//! - Full InceptionTime network with residual connections
//! - Ensemble methods for robust predictions

mod ensemble;
mod inception;
mod network;

pub use ensemble::InceptionTimeEnsemble;
pub use inception::InceptionModule;
pub use network::InceptionTimeNetwork;
