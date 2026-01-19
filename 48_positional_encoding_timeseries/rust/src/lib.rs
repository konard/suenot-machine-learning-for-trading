//! Positional Encoding for Time Series
//!
//! This crate provides various positional encoding implementations
//! for time series transformer models, with a focus on financial data.
//!
//! # Encoding Types
//!
//! - **Sinusoidal**: Classic sine/cosine based encoding
//! - **Learned**: Trainable position embeddings
//! - **Relative**: Distance-based position encoding
//! - **Rotary (RoPE)**: Rotation-based positional encoding
//! - **Calendar**: Temporal features (day, month, etc.)
//! - **Market Session**: Trading session encodings
//!
//! # Example
//!
//! ```rust
//! use positional_encoding::{SinusoidalEncoding, PositionalEncoding};
//!
//! let encoding = SinusoidalEncoding::new(512, 1000);
//! let positions = encoding.encode(&[0, 1, 2, 3, 4]);
//! ```

pub mod encoding;
pub mod calendar;
pub mod data;
pub mod strategy;

pub use encoding::*;
pub use calendar::*;
pub use data::*;
pub use strategy::*;
