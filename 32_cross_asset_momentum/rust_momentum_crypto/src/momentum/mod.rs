//! Модуль для расчёта моментума
//!
//! Этот модуль содержит:
//! - Time-series momentum (absolute momentum)
//! - Cross-sectional momentum (relative momentum)
//! - Dual momentum (комбинация обоих подходов)

pub mod crosssection;
pub mod dual;
pub mod timeseries;

pub use crosssection::{
    rank_values, ranks_to_percentiles, zscore_normalize, CrossSectionalMomentum,
    CrossSectionalMomentumConfig, RankedAsset,
};
pub use dual::{quick_dual_momentum_signal, DualMomentum, DualMomentumConfig, DualMomentumResult};
pub use timeseries::{
    momentum_skip, rolling_momentum, simple_momentum, TimeSeriesMomentum,
    TimeSeriesMomentumConfig,
};
