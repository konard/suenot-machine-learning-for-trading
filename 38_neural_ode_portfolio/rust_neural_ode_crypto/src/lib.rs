//! # Neural ODE for Cryptocurrency Portfolio Optimization
//!
//! This crate provides a complete implementation of Neural Ordinary Differential
//! Equations (Neural ODE) for portfolio optimization using cryptocurrency data
//! from Bybit exchange.
//!
//! ## Features
//!
//! - **Bybit API Client**: Fetch historical klines and market data
//! - **ODE Solvers**: Euler, RK4, and Dopri5 numerical integrators
//! - **Neural ODE Models**: Portfolio dynamics modeling
//! - **Trading Strategies**: Continuous rebalancing with cost optimization
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use neural_ode_crypto::{
//!     data::BybitClient,
//!     model::NeuralODEPortfolio,
//!     strategy::ContinuousRebalancer,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Create and train model
//!     let model = NeuralODEPortfolio::new(3, 10, 64);
//!
//!     // Use for rebalancing
//!     let rebalancer = ContinuousRebalancer::new(model, 0.02);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Modules
//!
//! - [`data`]: Data fetching and preprocessing
//! - [`ode`]: ODE solvers (Euler, RK4, Dopri5)
//! - [`model`]: Neural network and portfolio models
//! - [`strategy`]: Trading strategies and backtesting

pub mod data;
pub mod model;
pub mod ode;
pub mod strategy;

// Re-exports for convenience
pub use data::{BybitClient, Candle, Features};
pub use model::{NeuralODEPortfolio, PortfolioState};
pub use ode::{ODESolver, Dopri5Solver, RK4Solver, EulerSolver};
pub use strategy::{ContinuousRebalancer, BacktestResult};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::data::{BybitClient, Candle, Features, Symbol};
    pub use crate::model::{NeuralODEPortfolio, PortfolioState, ODEFunc};
    pub use crate::ode::{ODESolver, Dopri5Solver, RK4Solver, EulerSolver};
    pub use crate::strategy::{ContinuousRebalancer, BacktestResult, RebalanceDecision};
}
