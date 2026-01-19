//! Trading strategy modules

pub mod news_strategy;
pub mod risk;
pub mod position;

pub use news_strategy::{NewsStrategy, StrategyConfig, StrategySignal};
pub use risk::{RiskManager, RiskConfig, RiskLevel};
pub use position::{Position, PositionSizer, PositionSide};
