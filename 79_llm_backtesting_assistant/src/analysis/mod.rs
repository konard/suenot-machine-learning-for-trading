//! LLM Analysis module
//!
//! Provides functionality for generating prompts and parsing LLM responses
//! for backtesting analysis.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::backtesting::{BacktestResults, MarketType};
use crate::error::{Error, Result};
use crate::metrics::PerformanceMetrics;

/// LLM provider trait
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a prompt to the LLM and get a response
    async fn complete(&self, prompt: &str) -> Result<String>;

    /// Get the provider name
    fn name(&self) -> &str;
}

/// OpenAI-compatible LLM client
pub struct OpenAiClient {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl OpenAiClient {
    /// Create a new OpenAI client
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "gpt-4".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// Set a custom base URL (for OpenAI-compatible APIs)
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

#[async_trait]
impl LlmProvider for OpenAiClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert quantitative analyst specializing in trading strategy analysis. Provide detailed, actionable insights based on backtesting results."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        });

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let response_json: OpenAiResponse = response.json().await?;

        response_json
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| Error::ApiError("No response from OpenAI".to_string()))
    }

    fn name(&self) -> &str {
        "OpenAI"
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
}

/// Anthropic Claude client
pub struct AnthropicClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "claude-3-opus-20240229".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }
}

#[async_trait]
impl LlmProvider for AnthropicClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        let request_body = serde_json::json!({
            "model": self.model,
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "system": "You are an expert quantitative analyst specializing in trading strategy analysis. Provide detailed, actionable insights based on backtesting results."
        });

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let response_json: AnthropicResponse = response.json().await?;

        response_json
            .content
            .into_iter()
            .next()
            .map(|c| c.text)
            .ok_or_else(|| Error::ApiError("No response from Anthropic".to_string()))
    }

    fn name(&self) -> &str {
        "Anthropic"
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: String,
}

/// Mock LLM client for testing (generates realistic mock analysis)
pub struct MockLlmClient;

impl MockLlmClient {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmProvider for MockLlmClient {
    async fn complete(&self, _prompt: &str) -> Result<String> {
        Ok(r#"
## Strategy Performance Analysis

### Overall Assessment
The backtesting results show a moderately performing strategy with room for improvement.

### Key Strengths
1. **Positive Returns**: The strategy generated positive total returns over the test period
2. **Reasonable Win Rate**: Win rate is within acceptable range for trend-following strategies
3. **Controlled Drawdowns**: Maximum drawdown remains manageable

### Areas for Improvement
1. **Risk-Adjusted Returns**: Sharpe ratio could be improved through better position sizing
2. **Trade Frequency**: Consider optimizing entry/exit timing to capture more opportunities
3. **Loss Management**: Implement tighter stop-losses to reduce average loss size

### Recommendations
1. Consider adding a volatility filter to avoid choppy market conditions
2. Implement dynamic position sizing based on market volatility
3. Add correlation analysis with benchmark indices
4. Test the strategy across different market regimes

### Risk Assessment
- **Market Risk**: Moderate exposure to directional moves
- **Liquidity Risk**: Low (assuming liquid instruments)
- **Model Risk**: Strategy relies on historical patterns continuing
"#.to_string())
    }

    fn name(&self) -> &str {
        "Mock"
    }
}

/// Prompt builder for generating analysis prompts
pub struct PromptBuilder {
    include_trades: bool,
    include_equity_curve: bool,
    focus_areas: Vec<String>,
}

impl PromptBuilder {
    /// Create a new prompt builder
    pub fn new() -> Self {
        Self {
            include_trades: true,
            include_equity_curve: false,
            focus_areas: vec![
                "risk_analysis".to_string(),
                "performance".to_string(),
                "recommendations".to_string(),
            ],
        }
    }

    /// Include individual trade details
    pub fn with_trades(mut self, include: bool) -> Self {
        self.include_trades = include;
        self
    }

    /// Include equity curve data
    pub fn with_equity_curve(mut self, include: bool) -> Self {
        self.include_equity_curve = include;
        self
    }

    /// Set focus areas for analysis
    pub fn with_focus_areas(mut self, areas: Vec<String>) -> Self {
        self.focus_areas = areas;
        self
    }

    /// Build the analysis prompt
    pub fn build(&self, results: &BacktestResults) -> String {
        let mut prompt = String::new();

        prompt.push_str("Please analyze the following backtesting results for a trading strategy:\n\n");

        // Strategy info
        prompt.push_str(&format!("## Strategy Information\n"));
        prompt.push_str(&format!("- **Strategy Name**: {}\n", results.strategy_name));
        prompt.push_str(&format!("- **Symbol**: {}\n", results.symbol));
        prompt.push_str(&format!("- **Market Type**: {}\n", results.market_type));
        prompt.push_str(&format!("- **Period**: {} to {}\n",
            results.start_date.format("%Y-%m-%d"),
            results.end_date.format("%Y-%m-%d")
        ));
        prompt.push_str(&format!("- **Initial Capital**: ${:.2}\n", results.initial_capital));
        prompt.push_str(&format!("- **Final Capital**: ${:.2}\n", results.final_capital));
        prompt.push_str("\n");

        // Parameters
        prompt.push_str(&format!("## Strategy Parameters\n"));
        prompt.push_str(&format!("```json\n{}\n```\n\n",
            serde_json::to_string_pretty(&results.parameters).unwrap_or_default()
        ));

        // Performance metrics
        prompt.push_str(&format!("## Performance Metrics\n"));
        prompt.push_str(&self.format_metrics(&results.metrics));
        prompt.push_str("\n");

        // Trade summary
        if self.include_trades && !results.trades.is_empty() {
            prompt.push_str("## Trade Summary\n");
            prompt.push_str(&format!("- Total Trades: {}\n", results.trades.len()));

            let winning = results.trades.iter().filter(|t| t.pnl > 0.0).count();
            let losing = results.trades.len() - winning;
            prompt.push_str(&format!("- Winning Trades: {}\n", winning));
            prompt.push_str(&format!("- Losing Trades: {}\n", losing));

            // Show last 5 trades
            prompt.push_str("\n### Recent Trades (last 5):\n");
            for trade in results.trades.iter().rev().take(5) {
                prompt.push_str(&format!(
                    "- {} {} @ ${:.2} -> ${:.2}, P&L: ${:.2}\n",
                    trade.entry_time.format("%Y-%m-%d"),
                    format!("{:?}", trade.side),
                    trade.entry_price,
                    trade.exit_price,
                    trade.pnl
                ));
            }
            prompt.push_str("\n");
        }

        // Analysis request
        prompt.push_str("## Analysis Request\n");
        prompt.push_str("Please provide a comprehensive analysis including:\n");
        for area in &self.focus_areas {
            match area.as_str() {
                "risk_analysis" => prompt.push_str("1. **Risk Analysis**: Evaluate the risk metrics and potential vulnerabilities\n"),
                "performance" => prompt.push_str("2. **Performance Assessment**: Analyze the returns and compare to benchmarks\n"),
                "recommendations" => prompt.push_str("3. **Recommendations**: Suggest specific improvements to the strategy\n"),
                "market_conditions" => prompt.push_str("4. **Market Conditions**: Assess how the strategy performs in different market regimes\n"),
                other => prompt.push_str(&format!("- {}\n", other)),
            }
        }

        // Market-specific considerations
        if results.market_type == MarketType::Crypto {
            prompt.push_str("\nNote: This is a cryptocurrency strategy. Please consider:\n");
            prompt.push_str("- 24/7 market operation\n");
            prompt.push_str("- Higher volatility typical of crypto markets\n");
            prompt.push_str("- Potential for flash crashes and manipulation\n");
        }

        prompt
    }

    fn format_metrics(&self, metrics: &PerformanceMetrics) -> String {
        format!(
            r#"| Metric | Value |
|--------|-------|
| Total Return | {:.2}% |
| Annualized Return | {:.2}% |
| Sharpe Ratio | {:.2} |
| Sortino Ratio | {:.2} |
| Calmar Ratio | {:.2} |
| Max Drawdown | {:.2}% |
| Volatility | {:.2}% |
| Win Rate | {:.2}% |
| Profit Factor | {:.2} |
| Avg Trade Return | ${:.2} |
| Avg Win | ${:.2} |
| Avg Loss | ${:.2} |
| Largest Win | ${:.2} |
| Largest Loss | ${:.2} |
"#,
            metrics.total_return * 100.0,
            metrics.annualized_return * 100.0,
            metrics.sharpe_ratio,
            metrics.sortino_ratio,
            metrics.calmar_ratio,
            metrics.max_drawdown * 100.0,
            metrics.volatility * 100.0,
            metrics.win_rate * 100.0,
            metrics.profit_factor,
            metrics.avg_trade_return,
            metrics.avg_win,
            metrics.avg_loss,
            metrics.largest_win,
            metrics.largest_loss,
        )
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Main backtesting assistant that orchestrates analysis
pub struct BacktestingAssistant<P: LlmProvider> {
    provider: P,
    prompt_builder: PromptBuilder,
}

impl<P: LlmProvider> BacktestingAssistant<P> {
    /// Create a new assistant with the given LLM provider
    pub fn with_provider(provider: P) -> Self {
        Self {
            provider,
            prompt_builder: PromptBuilder::new(),
        }
    }

    /// Set a custom prompt builder
    pub fn with_prompt_builder(mut self, builder: PromptBuilder) -> Self {
        self.prompt_builder = builder;
        self
    }

    /// Analyze backtest results
    pub async fn analyze(&self, results: &BacktestResults) -> Result<AnalysisResult> {
        let prompt = self.prompt_builder.build(results);
        let analysis = self.provider.complete(&prompt).await?;

        Ok(AnalysisResult {
            strategy_name: results.strategy_name.clone(),
            symbol: results.symbol.clone(),
            analysis,
            provider: self.provider.name().to_string(),
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Convenience constructor for BacktestingAssistant with OpenAI
impl BacktestingAssistant<OpenAiClient> {
    /// Create a new assistant with OpenAI
    pub fn new(api_key: String) -> Self {
        Self::with_provider(OpenAiClient::new(api_key))
    }
}

/// Result of LLM analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Strategy name that was analyzed
    pub strategy_name: String,
    /// Symbol that was analyzed
    pub symbol: String,
    /// The LLM-generated analysis
    pub analysis: String,
    /// LLM provider used
    pub provider: String,
    /// Timestamp of analysis
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl std::fmt::Display for AnalysisResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Analysis for {} ({}) ===", self.strategy_name, self.symbol)?;
        writeln!(f, "Provider: {} | Generated: {}", self.provider, self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"))?;
        writeln!(f, "{}", "=".repeat(50))?;
        writeln!(f, "{}", self.analysis)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtesting::BacktestResults;

    #[tokio::test]
    async fn test_mock_analysis() {
        let results = BacktestResults::sample();
        let assistant = BacktestingAssistant::with_provider(MockLlmClient::new());
        let analysis = assistant.analyze(&results).await.unwrap();

        assert!(!analysis.analysis.is_empty());
        assert_eq!(analysis.provider, "Mock");
    }

    #[test]
    fn test_prompt_builder() {
        let results = BacktestResults::sample();
        let prompt = PromptBuilder::new().build(&results);

        assert!(prompt.contains("Strategy Information"));
        assert!(prompt.contains("Performance Metrics"));
        assert!(prompt.contains("Sharpe Ratio"));
    }
}
