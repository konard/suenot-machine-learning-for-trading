//! # SSM-GNN Hybrid for Trading
//!
//! Combines State Space Models (temporal per-asset encoding) with
//! Graph Neural Networks (cross-asset message passing) for multi-asset
//! trading signal generation.
//!
//! ## Architecture
//!
//! 1. **SSM Encoder**: Diagonal state space model processes each asset's
//!    time series independently to capture temporal patterns.
//! 2. **GNN Layer**: Graph Attention Network propagates information across
//!    the asset correlation graph.
//! 3. **Prediction Head**: Maps fused representations to trading signals.

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;
