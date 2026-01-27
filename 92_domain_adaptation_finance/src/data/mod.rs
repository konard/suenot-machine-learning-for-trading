//! # Data Module
//!
//! Provides data fetching and feature engineering capabilities for domain adaptation
//! in financial markets.
//!
//! ## Submodules
//!
//! - [`bybit`]: Bybit API client for fetching OHLCV kline data (with simulation fallback).
//! - [`features`]: Feature engineering pipeline that computes technical indicators
//!   and normalizes them for model consumption.

pub mod bybit;
pub mod features;
