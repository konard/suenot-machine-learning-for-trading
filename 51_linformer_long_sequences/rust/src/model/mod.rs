//! Linformer model implementation in Rust.
//!
//! Provides linear-complexity attention mechanism for processing
//! long sequences efficiently.

pub mod attention;
pub mod config;
pub mod linformer;

pub use attention::LinformerAttention;
pub use config::LinformerConfig;
pub use linformer::Linformer;
