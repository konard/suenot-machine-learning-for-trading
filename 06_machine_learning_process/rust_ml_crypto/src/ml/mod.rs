//! Machine learning algorithms and utilities

pub mod knn;
pub mod cross_validation;
pub mod metrics;
pub mod bias_variance;

pub use knn::KNNClassifier;
pub use cross_validation::CrossValidator;
pub use metrics::Metrics;
pub use bias_variance::BiasVarianceAnalyzer;
