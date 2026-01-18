//! # HOT Crypto
//!
//! Higher Order Transformer library for cryptocurrency trading.
//!
//! This library implements Higher Order Attention mechanisms
//! for predicting cryptocurrency price movements using data from Bybit.
//!
//! ## Modules
//!
//! - `data`: Bybit API client and data types
//! - `attention`: Standard and higher-order attention mechanisms
//! - `tensor`: Tensor operations and CP decomposition
//! - `model`: HOT transformer model
//! - `strategy`: Trading signal generation
//! - `utils`: Configuration and utilities

pub mod data;
pub mod attention;
pub mod tensor;
pub mod model;
pub mod strategy;
pub mod utils;

pub use data::{BybitClient, Candle, PriceSeries};
pub use attention::{StandardAttention, HigherOrderAttention, KernelAttention};
pub use tensor::{Tensor3D, CPDecomposition};
pub use model::{HOTPredictor, PredictionResult};
pub use strategy::{Signal, SignalGenerator};
pub use utils::Config;

/// Default cryptocurrency universe for trading
pub fn get_crypto_universe() -> Vec<&'static str> {
    vec![
        // Major
        "BTCUSDT",
        "ETHUSDT",
        // Large cap
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        // Mid cap
        "AVAXUSDT",
        "DOTUSDT",
        "LINKUSDT",
        "MATICUSDT",
        // Small cap
        "ATOMUSDT",
        "NEARUSDT",
    ]
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
