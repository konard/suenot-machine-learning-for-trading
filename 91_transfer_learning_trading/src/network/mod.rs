//! Neural network components for transfer learning
//!
//! This module contains:
//! - Feature extraction networks for transforming raw market data
//! - Domain adaptation methods (MMD, CORAL, adversarial)
//! - Transfer learning network combining feature extraction and adaptation

mod feature_extractor;
mod domain_adapter;
mod transfer;

pub use feature_extractor::FeatureExtractor;
pub use domain_adapter::{DomainAdapter, AdaptationMethod};
pub use transfer::{TransferNetwork, FineTuneStrategy};
