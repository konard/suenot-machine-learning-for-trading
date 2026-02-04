//! Conformal Prediction module
//!
//! Provides conformal prediction algorithms for uncertainty quantification.

pub mod predictor;
pub mod scores;
pub mod adaptive;

pub use predictor::SplitConformalPredictor;
pub use scores::NonConformityScore;
pub use adaptive::AdaptiveConformalPredictor;
