//! # Mamba Trading
//!
//! A Rust implementation of the Mamba architecture for trading applications.
//!
//! ## Features
//!
//! - Efficient Mamba (Selective State Space) model implementation
//! - Data loading from Bybit cryptocurrency exchange
//! - Feature engineering for trading
//! - High-performance inference
//!
//! ## Example
//!
//! ```rust,no_run
//! use mamba_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load data from Bybit
//!     let loader = BybitClient::new();
//!     let data = loader.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     // Create model
//!     let config = MambaConfig::default();
//!     let model = MambaTrading::new(config);
//!
//!     // Get predictions
//!     let predictions = model.predict(&data)?;
//!     println!("Signal: {:?}", predictions);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::bybit::BybitClient;
    pub use crate::data::{features::FeatureEngineer, loader::DataLoader, OhlcvBar};
    pub use crate::model::{mamba::MambaBlock, trading::MambaTrading, MambaConfig};
}

// Re-exports
pub use api::bybit::BybitClient;
pub use data::{features::FeatureEngineer, loader::DataLoader, OhlcvBar};
pub use model::{mamba::MambaBlock, trading::MambaTrading, MambaConfig};
