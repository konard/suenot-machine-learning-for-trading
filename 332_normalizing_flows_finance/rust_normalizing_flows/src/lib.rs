//! # Normalizing Flows for Finance
//!
//! This library provides implementations of normalizing flow models
//! for financial risk management and density estimation.
//!
//! ## Core Concepts
//!
//! - **Normalizing Flows**: Generative models that learn complex distributions
//! - **Density Estimation**: Accurate probability density for financial returns
//! - **Risk Metrics**: VaR, CVaR calculated from learned distributions
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `flows` - Normalizing flow implementations
//! - `risk` - Risk metric calculations (VaR, CVaR)
//! - `utils` - Utility functions and helpers
//!
//! ## Example
//!
//! ```rust,no_run
//! use normalizing_flows_finance::prelude::*;
//!
//! fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines_sync("BTCUSDT", "1d", 500)?;
//!
//!     // Calculate returns
//!     let returns = calculate_log_returns(&klines);
//!
//!     // Create and train flow
//!     let config = FlowConfig::default();
//!     let mut flow = RealNVP::new(config);
//!     flow.train(&returns, 100)?;
//!
//!     // Compute risk metrics
//!     let var_95 = compute_var(&flow, 0.05, 100000);
//!     let cvar_95 = compute_cvar(&flow, 0.05, 100000);
//!
//!     println!("95% VaR: {:.4}", var_95);
//!     println!("95% CVaR: {:.4}", cvar_95);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod flows;
pub mod risk;
pub mod utils;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, Ticker};
pub use flows::config::FlowConfig;
pub use flows::realnvp::RealNVP;
pub use flows::traits::NormalizingFlow;
pub use risk::metrics::{compute_cvar, compute_risk_metrics, compute_var};
pub use utils::returns::calculate_log_returns;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default trading symbols for examples
pub const DEFAULT_SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT",
];

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::client::BybitClient;
    pub use crate::api::types::{Kline, Ticker};
    pub use crate::flows::config::FlowConfig;
    pub use crate::flows::realnvp::RealNVP;
    pub use crate::flows::traits::NormalizingFlow;
    pub use crate::risk::metrics::{compute_cvar, compute_risk_metrics, compute_var};
    pub use crate::utils::returns::calculate_log_returns;
    pub use crate::DEFAULT_SYMBOLS;
}
