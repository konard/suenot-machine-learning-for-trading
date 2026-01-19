//! LLM module for news analysis using large language models.
//!
//! This module provides:
//! - LLM client implementations for various providers
//! - Prompt templates for news analysis
//! - Response parsing and validation
//! - Caching layer to reduce API calls

mod client;
mod parser;
mod prompts;

pub use client::{LlmClient, LlmProvider};
pub use parser::{LlmResponse, ParsedAnalysis};
pub use prompts::PromptTemplate;

use crate::news::NewsItem;
use serde::{Deserialize, Serialize};

/// Error types for LLM operations
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("API request failed: {0}")]
    ApiError(String),

    #[error("Rate limit exceeded, retry after {0} seconds")]
    RateLimitError(u64),

    #[error("Invalid API key")]
    AuthenticationError,

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Timeout after {0}ms")]
    TimeoutError(u64),

    #[error("Model not available: {0}")]
    ModelNotAvailable(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Configuration for LLM analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// API key for the LLM provider
    pub api_key: String,
    /// Provider to use (openai, anthropic, local)
    pub provider: LlmProvider,
    /// Model name/ID
    pub model: String,
    /// Maximum tokens in response
    pub max_tokens: usize,
    /// Temperature for generation (0.0 to 1.0)
    pub temperature: f64,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            provider: LlmProvider::OpenAI,
            model: "gpt-4".to_string(),
            max_tokens: 1024,
            temperature: 0.3,
            timeout_ms: 30000,
            enable_cache: true,
            cache_ttl_seconds: 300,
        }
    }
}

/// News analyzer that uses LLM for interpretation
#[derive(Debug)]
pub struct NewsAnalyzer {
    client: LlmClient,
    prompt_template: PromptTemplate,
}

impl NewsAnalyzer {
    /// Create a new news analyzer with a given LLM client
    pub fn new(client: LlmClient) -> Self {
        Self {
            client,
            prompt_template: PromptTemplate::default(),
        }
    }

    /// Create with custom prompt template
    pub fn with_template(client: LlmClient, template: PromptTemplate) -> Self {
        Self {
            client,
            prompt_template: template,
        }
    }

    /// Analyze a single news item
    pub async fn analyze(&self, news: &NewsItem) -> Result<ParsedAnalysis, LlmError> {
        let prompt = self.prompt_template.build_analysis_prompt(news);
        let response = self.client.complete(&prompt).await?;
        parser::parse_analysis_response(&response.content)
    }

    /// Analyze multiple news items in batch
    pub async fn analyze_batch(&self, news_items: &[NewsItem]) -> Vec<Result<ParsedAnalysis, LlmError>> {
        let mut results = Vec::with_capacity(news_items.len());

        for item in news_items {
            results.push(self.analyze(item).await);
        }

        results
    }

    /// Get the underlying LLM client
    pub fn client(&self) -> &LlmClient {
        &self.client
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        assert!(config.enable_cache);
        assert_eq!(config.temperature, 0.3);
        assert_eq!(config.provider, LlmProvider::OpenAI);
    }

    #[test]
    fn test_llm_error_display() {
        let err = LlmError::RateLimitError(60);
        assert!(err.to_string().contains("60"));

        let err = LlmError::TimeoutError(5000);
        assert!(err.to_string().contains("5000"));
    }
}
