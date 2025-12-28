//! Bayesian statistics module.
//!
//! This module provides:
//! - Probability distributions (Beta, Normal, Student-t)
//! - Conjugate prior updates
//! - MCMC inference (Metropolis-Hastings)
//! - Bayesian linear regression

pub mod distributions;
pub mod inference;
pub mod linear_regression;
pub mod sharpe;
pub mod volatility;

pub use distributions::{Beta, Normal, StudentT};
pub use inference::{ConjugatePrior, MCMC, MetropolisHastings};
pub use linear_regression::BayesianLinearRegression;
pub use sharpe::BayesianSharpe;
pub use volatility::StochasticVolatility;
