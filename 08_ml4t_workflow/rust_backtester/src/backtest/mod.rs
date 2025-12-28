//! Backtesting engine module.

mod engine;
mod broker;
mod result;

pub use engine::{BacktestConfig, BacktestEngine};
pub use broker::{BrokerConfig, SimulatedBroker};
pub use result::BacktestResult;
