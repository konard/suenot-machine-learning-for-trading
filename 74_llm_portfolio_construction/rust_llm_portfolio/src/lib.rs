//! LLM Portfolio Construction Library
//!
//! This library provides tools for building and managing investment portfolios
//! using Large Language Model analysis.

pub mod data;
pub mod llm;
pub mod portfolio;

pub use data::bybit::BybitClient;
pub use data::stock::StockClient;
pub use llm::engine::LLMPortfolioEngine;
pub use portfolio::optimizer::MeanVarianceOptimizer;
pub use portfolio::types::{Asset, AssetClass, AssetScore, Portfolio};
