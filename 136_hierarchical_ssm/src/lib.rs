//! Hierarchical State Space Models (HiSS) for Trading
//!
//! This crate implements hierarchical SSMs that process financial data
//! at multiple temporal resolutions for improved prediction of returns,
//! direction, and volatility.
//!
//! Based on: Bhirangi et al. (2024) - "Hierarchical State Space Models
//! for Continuous Sequence-to-Sequence Modeling" (arXiv:2402.10211)

pub mod backtest;
pub mod data;
pub mod model;
pub mod trading;
