//! Factor discovery module
//!
//! Provides LLM-based factor generation and refinement capabilities.

mod generator;
mod llm_client;
mod prompt_builder;
mod refiner;

pub use generator::{FactorCandidate, FactorGenerator, GenerationConfig};
pub use llm_client::{LlmClient, LlmConfig, LlmError, LlmResponse};
pub use prompt_builder::PromptBuilder;
pub use refiner::FactorRefiner;

use crate::evaluation::{BacktestConfig, BacktestResult, Backtester};
use crate::parser::FactorExpr;
use anyhow::Result;
use tracing::info;

/// Main factor discovery system that combines LLM generation with backtesting
pub struct FactorDiscoverySystem {
    llm_client: LlmClient,
    generator: FactorGenerator,
    refiner: FactorRefiner,
}

impl FactorDiscoverySystem {
    /// Create a new factor discovery system
    pub fn new(llm_client: LlmClient) -> Self {
        let generator = FactorGenerator::new(llm_client.clone());
        let refiner = FactorRefiner::new(llm_client.clone());

        Self {
            llm_client,
            generator,
            refiner,
        }
    }

    /// Generate factor candidates based on a hypothesis
    pub async fn generate_factors(&self, hypothesis: &str) -> Result<Vec<FactorCandidate>> {
        info!("Generating factors for hypothesis: {}", hypothesis);
        self.generator.generate(hypothesis).await
    }

    /// Refine factors based on backtest results
    pub async fn refine_factors(
        &self,
        factors: &[FactorCandidate],
        results: &[BacktestResult],
    ) -> Result<Vec<FactorCandidate>> {
        info!("Refining {} factors based on results", factors.len());
        self.refiner.refine(factors, results).await
    }

    /// Run backtests on factor candidates
    pub async fn backtest(
        &self,
        factors: &[FactorCandidate],
        config: &BacktestConfig,
    ) -> Result<Vec<BacktestResult>> {
        info!("Backtesting {} factors", factors.len());
        let backtester = Backtester::new(config.clone());

        let mut results = Vec::new();
        for factor in factors {
            if let Ok(expr) = FactorExpr::parse(&factor.expression) {
                match backtester.run(&expr).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        tracing::warn!("Backtest failed for {}: {}", factor.name, e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Full discovery cycle: generate, backtest, refine
    pub async fn discover(
        &self,
        hypothesis: &str,
        config: &BacktestConfig,
        iterations: usize,
    ) -> Result<Vec<(FactorCandidate, BacktestResult)>> {
        let mut best_factors = Vec::new();

        // Initial generation
        let mut factors = self.generate_factors(hypothesis).await?;

        for iteration in 0..iterations {
            info!("Discovery iteration {}/{}", iteration + 1, iterations);

            // Backtest current factors
            let results = self.backtest(&factors, config).await?;

            // Keep factors that meet threshold
            for (factor, result) in factors.iter().zip(results.iter()) {
                if result.ic.abs() >= crate::DEFAULT_IC_THRESHOLD
                    && result.ic_ir.abs() >= crate::DEFAULT_IC_IR_THRESHOLD
                {
                    best_factors.push((factor.clone(), result.clone()));
                }
            }

            // Refine for next iteration
            if iteration < iterations - 1 {
                factors = self.refine_factors(&factors, &results).await?;
            }
        }

        // Sort by IC_IR and deduplicate
        best_factors.sort_by(|a, b| {
            b.1.ic_ir
                .partial_cmp(&a.1.ic_ir)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(best_factors)
    }
}
