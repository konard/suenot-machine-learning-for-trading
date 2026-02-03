//! # Feature Attribution Trading
//!
//! This crate implements feature attribution methods for explaining trading model predictions.
//! Feature attribution helps traders understand which features are most important for
//! generating buy/sell signals.
//!
//! ## Features
//!
//! - **Permutation Importance**: Measure feature importance by permuting feature values
//! - **Gradient-based Attribution**: Compute attributions using model gradients
//! - **SHAP-like Values**: Approximate Shapley values for feature contributions
//! - **Bybit API Integration**: Fetch real cryptocurrency market data
//! - **Backtesting Engine**: Test trading strategies with attribution tracking
//!
//! ## Example
//!
//! ```rust,no_run
//! use feature_attribution_trading::{
//!     FeatureAttributor, PermutationImportance, GradientAttribution,
//!     TradingModel, MarketData, FeatureEngineering,
//! };
//! use ndarray::Array1;
//!
//! // Create a trading model
//! let model = TradingModel::new(10, vec![64, 32], 1);
//!
//! // Create attributors
//! let perm_importance = PermutationImportance::new(100);
//! let gradient_attr = GradientAttribution::new(200);
//!
//! // Compute attributions
//! let input = Array1::from_vec(vec![0.1; 10]);
//! let attrs = gradient_attr.explain(&model, &input, None);
//!
//! println!("Feature attributions: {:?}", attrs);
//! ```

pub mod model;
pub mod data;
pub mod backtest;

// Re-export main types
pub use model::{
    TradingModel, NeuralNetwork, Activation,
    attribution::{
        FeatureAttributor, PermutationImportance, GradientAttribution,
        ShapleyApproximation, AttributionResult, FeatureImportance,
    },
};

pub use data::{
    MarketData, FeatureEngineering, create_target,
    bybit::{BybitClient, Candle, Interval, Ticker},
};

pub use backtest::{
    Backtester, BacktestResults, Trade, TradingSignal,
    AttributionBacktester, AttributionAnalysis,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Feature names for the default feature set
pub const FEATURE_NAMES: &[&str] = &[
    "returns_1d",
    "returns_5d",
    "returns_20d",
    "rsi_14",
    "macd_histogram",
    "bollinger_position",
    "atr_normalized",
    "momentum_5d",
    "momentum_20d",
    "volume_ratio",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_feature_names() {
        assert_eq!(FEATURE_NAMES.len(), 10);
    }
}
