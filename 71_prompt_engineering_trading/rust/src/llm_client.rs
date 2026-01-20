//! LLM Client abstraction for multiple providers.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{TradingError, TradingResult};

/// Configuration for LLM requests.
#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub temperature: f32,
    pub max_tokens: u32,
    pub model: String,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            max_tokens: 1000,
            model: "gpt-4".to_string(),
        }
    }
}

/// Trait for LLM clients.
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Complete a prompt and return the response.
    async fn complete(&self, prompt: &str, config: &LLMConfig) -> TradingResult<String>;

    /// Get the provider name.
    fn provider(&self) -> &str;
}

/// Mock LLM client for testing.
pub struct MockLLMClient {
    responses: HashMap<String, String>,
    default_response: String,
}

impl MockLLMClient {
    /// Create a new mock client with default response.
    pub fn new() -> Self {
        Self {
            responses: HashMap::new(),
            default_response: r#"{"sentiment": "NEUTRAL", "confidence": 50}"#.to_string(),
        }
    }

    /// Create a mock client with specific responses.
    pub fn with_responses(responses: HashMap<String, String>) -> Self {
        Self {
            responses,
            default_response: r#"{"sentiment": "NEUTRAL", "confidence": 50}"#.to_string(),
        }
    }

    /// Set the default response.
    pub fn set_default_response(&mut self, response: String) {
        self.default_response = response;
    }

    /// Add a response for a specific key.
    pub fn add_response(&mut self, key: &str, response: String) {
        self.responses.insert(key.to_string(), response);
    }
}

impl Default for MockLLMClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMClient for MockLLMClient {
    async fn complete(&self, prompt: &str, _config: &LLMConfig) -> TradingResult<String> {
        // Try to match prompt to a response key
        for (key, response) in &self.responses {
            if prompt.to_lowercase().contains(&key.to_lowercase()) {
                return Ok(response.clone());
            }
        }
        Ok(self.default_response.clone())
    }

    fn provider(&self) -> &str {
        "mock"
    }
}

/// OpenAI API client.
pub struct OpenAIClient {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessageContent,
}

#[derive(Deserialize)]
struct OpenAIMessageContent {
    content: String,
}

impl OpenAIClient {
    /// Create a new OpenAI client.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create with custom base URL (for proxies or compatible APIs).
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn complete(&self, prompt: &str, config: &LLMConfig) -> TradingResult<String> {
        let request = OpenAIRequest {
            model: config.model.clone(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: config.temperature,
            max_tokens: config.max_tokens,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(TradingError::LLMError(error_text));
        }

        let api_response: OpenAIResponse = response.json().await?;

        api_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| TradingError::LLMError("No response from API".to_string()))
    }

    fn provider(&self) -> &str {
        "openai"
    }
}

/// Anthropic Claude API client.
pub struct AnthropicClient {
    api_key: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

impl AnthropicClient {
    /// Create a new Anthropic client.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    async fn complete(&self, prompt: &str, config: &LLMConfig) -> TradingResult<String> {
        let request = AnthropicRequest {
            model: config.model.clone(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: config.max_tokens,
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(TradingError::LLMError(error_text));
        }

        let api_response: AnthropicResponse = response.json().await?;

        api_response
            .content
            .first()
            .map(|c| c.text.clone())
            .ok_or_else(|| TradingError::LLMError("No response from API".to_string()))
    }

    fn provider(&self) -> &str {
        "anthropic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_client() {
        let client = MockLLMClient::new();
        let config = LLMConfig::default();

        let result = client.complete("test prompt", &config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_mock_client_with_responses() {
        let mut responses = HashMap::new();
        responses.insert(
            "earnings".to_string(),
            r#"{"sentiment": "POSITIVE", "confidence": 85}"#.to_string(),
        );

        let client = MockLLMClient::with_responses(responses);
        let config = LLMConfig::default();

        let result = client
            .complete("Apple reports strong earnings", &config)
            .await
            .unwrap();
        assert!(result.contains("POSITIVE"));
    }
}
