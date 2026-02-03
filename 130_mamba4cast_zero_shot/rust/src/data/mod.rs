//! Data loading utilities.

mod bybit;

pub use bybit::{BybitClient, KlineData, OrderbookData};

/// Stock data structure.
#[derive(Debug, Clone)]
pub struct StockData {
    pub timestamp: Vec<chrono::DateTime<chrono::Utc>>,
    pub open: Vec<f32>,
    pub high: Vec<f32>,
    pub low: Vec<f32>,
    pub close: Vec<f32>,
    pub volume: Vec<f32>,
}

impl StockData {
    /// Get close prices as a slice.
    pub fn close_prices(&self) -> &[f32] {
        &self.close
    }

    /// Get the latest price.
    pub fn latest_price(&self) -> Option<f32> {
        self.close.last().copied()
    }

    /// Get data length.
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if data is empty.
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}
