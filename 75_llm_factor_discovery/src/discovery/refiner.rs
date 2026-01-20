//! Factor refiner using LLM
//!
//! Refines and improves alpha factors based on backtest feedback.

use super::generator::FactorCandidate;
use super::llm_client::{ChatMessage, LlmClient};
use super::prompt_builder::PromptBuilder;
use crate::evaluation::BacktestResult;
use anyhow::Result;
use tracing::{debug, info, warn};

/// Factor refiner that uses LLM to improve factors based on results
pub struct FactorRefiner {
    llm_client: LlmClient,
    prompt_builder: PromptBuilder,
}

impl FactorRefiner {
    /// Create a new factor refiner
    pub fn new(llm_client: LlmClient) -> Self {
        Self {
            llm_client,
            prompt_builder: PromptBuilder::new(),
        }
    }

    /// Refine factors based on backtest results
    pub async fn refine(
        &self,
        factors: &[FactorCandidate],
        results: &[BacktestResult],
    ) -> Result<Vec<FactorCandidate>> {
        // Combine factors and results
        let factor_results: Vec<(String, String, f64, f64)> = factors
            .iter()
            .zip(results.iter())
            .map(|(f, r)| (f.name.clone(), f.expression.clone(), r.ic, r.ic_ir))
            .collect();

        // Analyze performance to determine improvement direction
        let improvement_direction = self.analyze_performance(&factor_results);

        let system_prompt = self.prompt_builder.system_prompt();
        let refinement_prompt =
            self.prompt_builder.refinement_prompt(&factor_results, &improvement_direction);

        let messages = vec![
            ChatMessage::system(&system_prompt),
            ChatMessage::user(&refinement_prompt),
        ];

        let response = self.llm_client.chat(&messages).await?;

        // Parse refined factors
        let mut refined = self.parse_factors(&response.content)?;

        // Increment iteration counter
        let max_iteration = factors.iter().map(|f| f.iteration).max().unwrap_or(0);
        for factor in &mut refined {
            factor.iteration = max_iteration + 1;
        }

        info!("Refined {} factors to {} candidates", factors.len(), refined.len());

        Ok(refined)
    }

    /// Analyze factor performance to guide refinement
    fn analyze_performance(&self, results: &[(String, String, f64, f64)]) -> String {
        let mut insights = Vec::new();

        // Calculate average metrics
        let avg_ic: f64 = results.iter().map(|(_, _, ic, _)| ic).sum::<f64>() / results.len() as f64;
        let avg_ic_ir: f64 = results.iter().map(|(_, _, _, ir)| ir).sum::<f64>() / results.len() as f64;

        // Identify best and worst performers
        let best = results
            .iter()
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        let worst = results
            .iter()
            .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        insights.push(format!(
            "Average IC: {:.4}, Average IC_IR: {:.4}",
            avg_ic, avg_ic_ir
        ));

        if let Some((name, _, ic, ic_ir)) = best {
            insights.push(format!(
                "Best performer: {} (IC: {:.4}, IC_IR: {:.4})",
                name, ic, ic_ir
            ));
        }

        if let Some((name, _, ic, ic_ir)) = worst {
            insights.push(format!(
                "Worst performer: {} (IC: {:.4}, IC_IR: {:.4})",
                name, ic, ic_ir
            ));
        }

        // Generate improvement suggestions
        if avg_ic.abs() < 0.02 {
            insights.push("IC is low overall. Consider using stronger signals or different lookback periods.".to_string());
        }

        if avg_ic_ir < 0.3 {
            insights.push("IC consistency is low. Consider smoothing factors or using longer windows.".to_string());
        }

        // Check for sign consistency
        let positive_ic = results.iter().filter(|(_, _, ic, _)| *ic > 0.0).count();
        let negative_ic = results.iter().filter(|(_, _, ic, _)| *ic < 0.0).count();

        if positive_ic > 0 && negative_ic > 0 {
            insights.push("Mixed IC signs suggest factors may need direction adjustment.".to_string());
        }

        insights.join("\n")
    }

    /// Parse factor candidates from LLM response
    fn parse_factors(&self, content: &str) -> Result<Vec<FactorCandidate>> {
        // Try to extract JSON from the response
        let json_content = self.extract_json(content)?;

        // Parse the JSON
        let candidates: Vec<FactorCandidate> = serde_json::from_str(&json_content)
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {}", e))?;

        // Validate expressions
        let valid_candidates: Vec<FactorCandidate> = candidates
            .into_iter()
            .filter(|c| {
                match crate::parser::FactorExpr::parse(&c.expression) {
                    Ok(_) => true,
                    Err(e) => {
                        warn!("Invalid factor expression '{}': {}", c.expression, e);
                        false
                    }
                }
            })
            .collect();

        Ok(valid_candidates)
    }

    /// Extract JSON from LLM response
    fn extract_json(&self, content: &str) -> Result<String> {
        // Try to find JSON in markdown code block
        if let Some(start) = content.find("```json") {
            let start = start + 7;
            if let Some(end) = content[start..].find("```") {
                return Ok(content[start..start + end].trim().to_string());
            }
        }

        // Try to find JSON in generic code block
        if let Some(start) = content.find("```") {
            let start = start + 3;
            let start = content[start..]
                .find('\n')
                .map(|i| start + i + 1)
                .unwrap_or(start);

            if let Some(end) = content[start..].find("```") {
                return Ok(content[start..start + end].trim().to_string());
            }
        }

        // Try to find raw JSON array
        if let Some(start) = content.find('[') {
            if let Some(end) = content.rfind(']') {
                return Ok(content[start..=end].to_string());
            }
        }

        Err(anyhow::anyhow!("No JSON found in response"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_performance() {
        // Create a mock refiner (can't test without actual client)
        // The logic is tested through unit test scenarios
    }
}
