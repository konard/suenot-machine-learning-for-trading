//! # Neural Spline Flows for Trading
//!
//! This crate provides a complete implementation of Neural Spline Flows (NSF)
//! for cryptocurrency trading on the Bybit exchange.
//!
//! ## Features
//!
//! - **Neural Spline Flows**: Flexible density estimation using rational-quadratic splines
//! - **Bybit Integration**: Real-time data fetching from Bybit API
//! - **Trading Signals**: Signal generation based on probability density
//! - **Risk Management**: VaR/CVaR computation from learned distributions
//! - **Backtesting**: Comprehensive backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use neural_spline_flows::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Create and train NSF model
//!     let config = NSFConfig::default();
//!     let mut model = NeuralSplineFlow::new(config);
//!
//!     // Extract features and train
//!     let features = extract_features(&candles);
//!     model.fit(&features)?;
//!
//!     // Generate trading signal
//!     let current_state = features.last().unwrap();
//!     let signal = model.generate_signal(current_state)?;
//!
//!     println!("Signal: {:?}", signal);
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod flow;
pub mod trading;
pub mod utils;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::bybit::BybitClient;
    pub use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult};
    pub use crate::flow::{
        CouplingLayer, NSFConfig, NeuralSplineFlow, RationalQuadraticSpline, SplineParams,
    };
    pub use crate::trading::{RiskManager, SignalGenerator, TradingSignal};
    pub use crate::utils::{extract_features, normalize_features, Candle, FeatureVector};
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
