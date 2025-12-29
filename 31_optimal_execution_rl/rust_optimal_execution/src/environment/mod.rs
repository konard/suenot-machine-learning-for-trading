//! # Execution Environment
//!
//! Среда для обучения с подкреплением в задаче оптимального исполнения ордеров.

mod state;
mod env;
mod simulator;

pub use state::{ExecutionState, ExecutionAction};
pub use env::{ExecutionEnv, StepResult, EnvConfig};
pub use simulator::MarketSimulator;
