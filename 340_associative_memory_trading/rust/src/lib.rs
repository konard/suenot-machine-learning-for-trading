//! Associative Memory Trading Library
//!
//! This library provides tools for pattern-based trading using Dense Associative Memory
//! networks on cryptocurrency data from Bybit exchange.
//!
//! # Modules
//!
//! - `data`: Data fetching from Bybit API and OHLCV handling
//! - `features`: Feature engineering for pattern construction
//! - `memory`: Associative memory implementations (Dense AM, Hopfield)
//! - `strategy`: Trading strategy and signal generation
//!
//! # Example
//!
//! ```no_run
//! use associative_memory_trading::data::BybitClient;
//! use associative_memory_trading::features::PatternBuilder;
//! use associative_memory_trading::memory::DenseAssociativeMemory;
//!
//! // Fetch data
//! let client = BybitClient::public();
//! let data = client.get_klines("BTCUSDT", "60", 1000, None, None).unwrap();
//!
//! // Build patterns
//! let builder = PatternBuilder::new(20);
//! let patterns = builder.build_patterns(&data);
//!
//! // Create and query memory
//! let mut memory = DenseAssociativeMemory::new(patterns.len(), 10, 1.0);
//! memory.store(&patterns, &outcomes);
//! let (prediction, confidence) = memory.predict(&current_pattern);
//! ```

pub mod data;
pub mod features;
pub mod memory;
pub mod strategy;

pub use data::*;
pub use features::*;
pub use memory::*;
pub use strategy::*;
