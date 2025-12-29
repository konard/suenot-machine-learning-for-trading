//! # Baseline Execution Algorithms
//!
//! Классические алгоритмы исполнения для сравнения с RL агентами.

mod twap;
mod vwap;
mod almgren_chriss;
mod schedule;

pub use twap::TWAPExecutor;
pub use vwap::VWAPExecutor;
pub use almgren_chriss::AlmgrenChrissExecutor;
pub use schedule::ExecutionSchedule;
