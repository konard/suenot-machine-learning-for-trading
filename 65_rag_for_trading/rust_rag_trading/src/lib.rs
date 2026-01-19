//! # RAG Trading Library
//!
//! A Rust implementation of Retrieval-Augmented Generation (RAG) for trading applications.
//!
//! This library provides:
//! - Document retrieval with semantic search
//! - Trading signal generation
//! - Backtesting framework
//! - Market data loading (Yahoo Finance, Bybit)
//!
//! ## Example
//!
//! ```rust,ignore
//! use rag_trading::{Document, DocumentRetriever, RAGSignalGenerator};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut retriever = DocumentRetriever::new();
//!
//!     retriever.add_document(Document::new(
//!         "Tesla reported record deliveries...",
//!         Some("TSLA".to_string()),
//!         "Reuters",
//!         chrono::Utc::now()
//!     ));
//!
//!     let results = retriever.search("Tesla delivery", 5, None);
//!     println!("Found {} results", results.len());
//!
//!     Ok(())
//! }
//! ```

pub mod document;
pub mod retriever;
pub mod signals;
pub mod backtest;
pub mod data;
pub mod error;

// Re-exports for convenience
pub use document::{Document, DocumentType};
pub use retriever::{DocumentRetriever, SearchResult};
pub use signals::{SignalDirection, TradingSignal, RAGSignalGenerator};
pub use backtest::{BacktestResult, Trade, RAGBacktester};
pub use data::{MarketData, YahooFinanceLoader, BybitLoader, DataLoader};
pub use error::{RagError, Result};
