//! # Bybit API Module
//!
//! This module provides clients for interacting with the Bybit exchange API.
//!
//! ## Features
//!
//! - REST API client for order book and trade data
//! - WebSocket client for real-time market data
//! - Automatic rate limiting and retry logic
//! - Support for both testnet and mainnet

pub mod bybit;
pub mod websocket;

pub use bybit::BybitClient;
pub use websocket::BybitWebSocket;

/// API endpoints
pub mod endpoints {
    /// Bybit mainnet base URL
    pub const MAINNET_REST: &str = "https://api.bybit.com";

    /// Bybit testnet base URL
    pub const TESTNET_REST: &str = "https://api-testnet.bybit.com";

    /// Bybit mainnet WebSocket URL
    pub const MAINNET_WS: &str = "wss://stream.bybit.com/v5/public/spot";

    /// Bybit testnet WebSocket URL
    pub const TESTNET_WS: &str = "wss://stream-testnet.bybit.com/v5/public/spot";
}

/// API error types
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    ApiResponse { code: i32, message: String },

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    #[error("Connection timeout")]
    Timeout,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),
}
