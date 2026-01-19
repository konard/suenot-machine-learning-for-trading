//! FNet model components
//!
//! This module contains the core FNet architecture:
//! - Fourier Transform layer
//! - Encoder blocks
//! - Complete FNet model

pub mod fourier;
pub mod encoder;
pub mod fnet;

pub use fourier::FourierLayer;
pub use encoder::FNetEncoderBlock;
pub use fnet::FNet;
