//! Simulation Module
//!
//! Core simulation engine and result types.

mod engine;
mod metrics;

pub use engine::{SimulationEngine, SimulationResult, AgentResult};
pub use metrics::{calculate_performance_metrics, PerformanceMetrics, detect_bubble, BubbleInfo};
