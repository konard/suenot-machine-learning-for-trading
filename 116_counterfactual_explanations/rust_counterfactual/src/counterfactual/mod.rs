//! Counterfactual explanation generation.

mod generator;
mod metrics;
mod optimizer;

pub use generator::CounterfactualGenerator;
pub use metrics::ProximityMetric;
pub use optimizer::{CounterfactualOptimizer, CounterfactualResult};
