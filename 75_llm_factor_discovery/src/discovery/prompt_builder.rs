//! Prompt builder for factor generation
//!
//! Constructs prompts for LLM-based factor discovery following
//! Alpha-GPT and Chain-of-Alpha methodologies.

use serde::{Deserialize, Serialize};

/// Configuration for prompt building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    /// Asset class being traded
    pub asset_class: AssetClass,
    /// Available data fields
    pub available_fields: Vec<String>,
    /// Maximum factor complexity
    pub max_complexity: usize,
    /// Number of factors to generate
    pub num_factors: usize,
    /// Include reasoning in output
    pub include_reasoning: bool,
}

/// Supported asset classes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AssetClass {
    Equities,
    Crypto,
    Forex,
    Commodities,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            asset_class: AssetClass::Crypto,
            available_fields: vec![
                "open".to_string(),
                "high".to_string(),
                "low".to_string(),
                "close".to_string(),
                "volume".to_string(),
                "vwap".to_string(),
            ],
            max_complexity: 5,
            num_factors: 5,
            include_reasoning: true,
        }
    }
}

/// Prompt builder for factor generation
pub struct PromptBuilder {
    config: PromptConfig,
}

impl PromptBuilder {
    /// Create a new prompt builder with default configuration
    pub fn new() -> Self {
        Self {
            config: PromptConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PromptConfig) -> Self {
        Self { config }
    }

    /// Build the system prompt for factor generation
    pub fn system_prompt(&self) -> String {
        let asset_class = match self.config.asset_class {
            AssetClass::Equities => "stocks",
            AssetClass::Crypto => "cryptocurrencies",
            AssetClass::Forex => "forex pairs",
            AssetClass::Commodities => "commodities",
        };

        let fields = self.config.available_fields.join(", ");

        format!(
            r#"You are an expert quantitative researcher specializing in alpha factor discovery for {asset_class}.

Your task is to generate novel alpha factors that can predict future returns.

## Available Functions

### Cross-sectional Operations (applied across all assets at each time point):
- `rank(x)` - Cross-sectional rank, normalized to [-1, 1]
- `zscore(x)` - Cross-sectional z-score standardization
- `scale(x)` - Scale to sum to 1
- `demean(x)` - Subtract cross-sectional mean

### Time Series Operations (applied to each asset's history):
- `ts_rank(x, d)` - Rank of current value in last d periods
- `ts_sum(x, d)` - Sum over last d periods
- `ts_mean(x, d)` - Mean over last d periods
- `ts_std(x, d)` - Standard deviation over last d periods
- `ts_min(x, d)` - Minimum over last d periods
- `ts_max(x, d)` - Maximum over last d periods
- `ts_argmax(x, d)` - Days since maximum in last d periods
- `ts_argmin(x, d)` - Days since minimum in last d periods
- `ts_delay(x, d)` - Value d periods ago
- `ts_delta(x, d)` - Change from d periods ago
- `ts_decay(x, d)` - Linear decay weighted sum

### Price Operations:
- `returns(x, d)` - Simple returns over d periods: (x[t] - x[t-d]) / x[t-d]
- `log_returns(x, d)` - Log returns: ln(x[t] / x[t-d])
- `volatility(x, d)` - Rolling volatility

### Statistical Operations:
- `correlation(x, y, d)` - Rolling correlation between x and y
- `covariance(x, y, d)` - Rolling covariance

### Mathematical Operations:
- `abs(x)`, `sign(x)`, `log(x)`, `sqrt(x)`, `power(x, n)`
- `max(x, y)`, `min(x, y)`
- `add(x, y)`, `sub(x, y)`, `mul(x, y)`, `div(x, y)`

## Available Data Fields
{fields}

## Factor Design Principles
1. **Economic Intuition**: Each factor should have a clear economic rationale
2. **Robustness**: Factors should work across different market conditions
3. **Low Turnover**: Prefer stable factors that don't require frequent rebalancing
4. **Orthogonality**: Try to create factors uncorrelated with common risk factors
5. **Simplicity**: Simpler factors are often more robust out-of-sample

## Output Format
For each factor, provide:
1. Name: A descriptive name
2. Expression: The factor formula using the syntax above
3. Rationale: Why this factor should predict returns
4. Expected IC: Rough estimate of Information Coefficient magnitude
5. Expected Turnover: Low/Medium/High

Maximum expression nesting depth: {max_complexity}"#,
            asset_class = asset_class,
            fields = fields,
            max_complexity = self.config.max_complexity
        )
    }

    /// Build the generation prompt for a specific hypothesis
    pub fn generation_prompt(&self, hypothesis: &str) -> String {
        format!(
            r#"Based on the following market hypothesis, generate {} novel alpha factors:

## Hypothesis
{hypothesis}

## Requirements
1. Each factor must use the function syntax exactly as specified
2. Factors should directly relate to the hypothesis
3. Include a mix of simple and complex factors
4. Ensure factors are properly nested with matching parentheses

Please generate the factors in the following JSON format:
```json
[
  {{
    "name": "factor_name",
    "expression": "rank(ts_mean(returns(close, 5), 20))",
    "rationale": "Explanation of economic intuition",
    "expected_ic": 0.03,
    "expected_turnover": "low"
  }}
]
```"#,
            self.config.num_factors,
            hypothesis = hypothesis
        )
    }

    /// Build the refinement prompt based on backtest results
    pub fn refinement_prompt(
        &self,
        factors: &[(String, String, f64, f64)], // (name, expression, ic, ic_ir)
        target_improvement: &str,
    ) -> String {
        let factor_results: Vec<String> = factors
            .iter()
            .map(|(name, expr, ic, ic_ir)| {
                format!(
                    "- {}: {} | IC: {:.4} | IC_IR: {:.4}",
                    name, expr, ic, ic_ir
                )
            })
            .collect();

        format!(
            r#"The following factors were backtested with these results:

{}

## Analysis Request
{}

## Task
Based on the results above, generate {} improved or modified factors that:
1. Address weaknesses in the original factors
2. Potentially combine successful elements from multiple factors
3. Add new variations that might improve performance
4. Maintain or reduce complexity where possible

Please provide the refined factors in the same JSON format as before."#,
            factor_results.join("\n"),
            target_improvement,
            self.config.num_factors
        )
    }

    /// Build a prompt for explaining factor performance
    pub fn explanation_prompt(&self, factor_name: &str, expression: &str, ic: f64, ic_ir: f64) -> String {
        format!(
            r#"Please analyze the following alpha factor:

**Name**: {factor_name}
**Expression**: {expression}
**Information Coefficient (IC)**: {ic:.4}
**IC Information Ratio (IC_IR)**: {ic_ir:.4}

Provide:
1. A plain-language explanation of what this factor measures
2. Economic intuition for why it might predict returns
3. Potential risks or limitations of this factor
4. Market conditions where this factor might underperform
5. Suggestions for improvement or complementary factors"#,
            factor_name = factor_name,
            expression = expression,
            ic = ic,
            ic_ir = ic_ir
        )
    }

    /// Build a prompt for generating factors from market observations
    pub fn observation_prompt(&self, observations: &[String]) -> String {
        let obs_list: Vec<String> = observations
            .iter()
            .enumerate()
            .map(|(i, obs)| format!("{}. {}", i + 1, obs))
            .collect();

        format!(
            r#"Based on the following market observations, generate alpha factors that could capture these patterns:

## Observations
{}

## Task
Design {} factors that specifically target these observed patterns. For each factor:
1. Clearly link it to one or more observations
2. Explain the expected mechanism
3. Consider potential decay or reversal of the effect

Provide factors in the standard JSON format."#,
            obs_list.join("\n"),
            self.config.num_factors
        )
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_contains_functions() {
        let builder = PromptBuilder::new();
        let prompt = builder.system_prompt();

        assert!(prompt.contains("rank(x)"));
        assert!(prompt.contains("ts_mean(x, d)"));
        assert!(prompt.contains("returns(x, d)"));
    }

    #[test]
    fn test_generation_prompt() {
        let builder = PromptBuilder::new();
        let prompt = builder.generation_prompt("momentum reversal");

        assert!(prompt.contains("momentum reversal"));
        assert!(prompt.contains("JSON format"));
    }

    #[test]
    fn test_refinement_prompt() {
        let builder = PromptBuilder::new();
        let factors = vec![(
            "momentum_5d".to_string(),
            "rank(returns(close, 5))".to_string(),
            0.03,
            0.6,
        )];

        let prompt = builder.refinement_prompt(&factors, "Improve IC stability");

        assert!(prompt.contains("momentum_5d"));
        assert!(prompt.contains("0.0300"));
        assert!(prompt.contains("Improve IC stability"));
    }
}
