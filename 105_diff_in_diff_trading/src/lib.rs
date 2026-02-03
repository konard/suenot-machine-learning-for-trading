//! Difference-in-Differences (DiD) for Trading
//!
//! This crate implements DiD estimation methods for financial markets,
//! including event studies, causal effect estimation, and trading strategies.
//!
//! Based on: Goodman-Bacon (2021) - "Difference-in-Differences with Variation
//! in Treatment Timing" and Roth et al. (2023) - "What's Trending in
//! Difference-in-Differences?"

pub mod backtest;
pub mod data;
pub mod model;
pub mod trading;
