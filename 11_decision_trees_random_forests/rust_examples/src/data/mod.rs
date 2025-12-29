//! Data structures and preprocessing module
//!
//! Provides core data types for market data and ML datasets.

mod candle;
mod dataset;

pub use candle::Candle;
pub use dataset::{Dataset, Split};
