//! # LLM Compliance Check
//!
//! A Rust library for checking trading activity compliance using Large Language Models.
//!
//! ## Features
//!
//! - Pre-trade and post-trade compliance checking
//! - Support for multiple jurisdictions (US, EU, crypto)
//! - RAG-based regulation retrieval
//! - LLM-powered analysis with explainable decisions
//! - Comprehensive audit trail generation
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_compliance_check::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//!     // Create a sample trading activity
//!     let activity = TradingActivity::new_order(
//!         "AAPL",
//!         1000.0,
//!         TradeSide::Buy,
//!         Some(150.0),
//!     );
//!
//!     // Create compliance checker
//!     let checker = ComplianceChecker::new("your-api-key".to_string());
//!
//!     // Check compliance
//!     let result = checker.check(&activity).await?;
//!
//!     println!("Status: {:?}", result.status);
//!     println!("Explanation: {}", result.explanation);
//!     Ok(())
//! }
//! ```

pub mod compliance;
pub mod data;
pub mod regulations;
pub mod reports;

pub mod error;

pub use error::{Error, Result};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::compliance::*;
    pub use crate::data::*;
    pub use crate::error::{Error, Result};
    pub use crate::regulations::*;
    pub use crate::reports::*;
}
