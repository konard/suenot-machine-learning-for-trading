//! Market module - Order book and market environment
//!
//! This module provides the core market infrastructure including
//! the limit order book and market environment.

mod order_book;

pub use order_book::{OrderBook, Order, OrderType, Side, OrderResult};
