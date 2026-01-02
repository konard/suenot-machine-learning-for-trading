//! Bybit API integration module
//!
//! Provides REST and WebSocket clients for fetching market data from Bybit exchange.

mod bybit;
mod websocket;

pub use bybit::{BybitClient, BybitConfig, BybitError};
pub use websocket::{BybitWebSocket, WebSocketMessage, StreamType};

/// API response wrapper
#[derive(Debug, Clone)]
pub struct ApiResponse<T> {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: T,
    pub time: u64,
}

impl<T> ApiResponse<T> {
    /// Check if the response is successful
    pub fn is_success(&self) -> bool {
        self.ret_code == 0
    }
}
