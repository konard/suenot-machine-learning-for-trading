//! Disentangled VAE model module
//!
//! Implements variational autoencoders with disentangled latent representations
//! for learning interpretable market factors.

mod config;
mod decoder;
mod encoder;
mod vae;

pub use config::{DisentangledVAEConfig, DisentanglementMethod};
pub use decoder::Decoder;
pub use encoder::Encoder;
pub use vae::DisentangledVAE;
