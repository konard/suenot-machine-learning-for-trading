//! Portfolio management modules

pub mod optimizer;
pub mod types;

// Re-export commonly used types
pub use types::{Asset, AssetClass, AssetScore, Confidence, MarketData, Portfolio, PortfolioConstraints};
