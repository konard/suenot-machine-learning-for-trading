//! LLM client for factor generation
//!
//! Provides integration with various LLM providers (OpenAI, Anthropic, etc.)

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};

/// LLM-related errors
#[derive(Error, Debug)]
pub enum LlmError {
    #[error("API error: {0}")]
    ApiError(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Context length exceeded")]
    ContextLengthExceeded,

    #[error("Model not available: {0}")]
    ModelNotAvailable(String),
}

/// LLM client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// API key
    pub api_key: String,

    /// Model name
    #[serde(default = "default_model")]
    pub model: String,

    /// Maximum tokens in response
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for generation
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// API base URL
    #[serde(default)]
    pub base_url: Option<String>,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_model() -> String {
    "gpt-4".to_string()
}

fn default_max_tokens() -> u32 {
    2000
}

fn default_temperature() -> f32 {
    0.7
}

fn default_timeout() -> u64 {
    60
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: default_model(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            base_url: None,
            timeout_secs: default_timeout(),
        }
    }
}

/// Response from LLM
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub model: String,
    pub usage: Option<TokenUsage>,
    pub finish_reason: Option<String>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// LLM client for interacting with language models
#[derive(Clone)]
pub struct LlmClient {
    config: LlmConfig,
    client: Client,
    provider: LlmProvider,
}

#[derive(Debug, Clone, Copy)]
enum LlmProvider {
    OpenAI,
    Anthropic,
    Custom,
}

impl LlmClient {
    /// Create a new OpenAI client
    pub fn new_openai(api_key: &str) -> Result<Self, LlmError> {
        let config = LlmConfig {
            api_key: api_key.to_string(),
            base_url: Some("https://api.openai.com/v1".to_string()),
            ..Default::default()
        };

        Self::with_config(config, LlmProvider::OpenAI)
    }

    /// Create a new Anthropic client
    pub fn new_anthropic(api_key: &str) -> Result<Self, LlmError> {
        let config = LlmConfig {
            api_key: api_key.to_string(),
            model: "claude-3-opus-20240229".to_string(),
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            ..Default::default()
        };

        Self::with_config(config, LlmProvider::Anthropic)
    }

    /// Create with custom configuration
    pub fn with_config(config: LlmConfig, provider: LlmProvider) -> Result<Self, LlmError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;

        Ok(Self {
            config,
            client,
            provider,
        })
    }

    /// Send a completion request
    pub async fn complete(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        match self.provider {
            LlmProvider::OpenAI => self.complete_openai(prompt).await,
            LlmProvider::Anthropic => self.complete_anthropic(prompt).await,
            LlmProvider::Custom => self.complete_openai(prompt).await, // Default to OpenAI-compatible
        }
    }

    /// Send a chat completion request
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<LlmResponse, LlmError> {
        match self.provider {
            LlmProvider::OpenAI => self.chat_openai(messages).await,
            LlmProvider::Anthropic => self.chat_anthropic(messages).await,
            LlmProvider::Custom => self.chat_openai(messages).await,
        }
    }

    async fn complete_openai(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        self.chat_openai(&messages).await
    }

    async fn chat_openai(&self, messages: &[ChatMessage]) -> Result<LlmResponse, LlmError> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");

        let url = format!("{}/chat/completions", base_url);

        #[derive(Serialize)]
        struct OpenAIRequest {
            model: String,
            messages: Vec<ChatMessage>,
            max_tokens: u32,
            temperature: f32,
        }

        #[derive(Deserialize)]
        struct OpenAIResponse {
            choices: Vec<OpenAIChoice>,
            model: String,
            usage: Option<OpenAIUsage>,
        }

        #[derive(Deserialize)]
        struct OpenAIChoice {
            message: OpenAIMessage,
            finish_reason: Option<String>,
        }

        #[derive(Deserialize)]
        struct OpenAIMessage {
            content: String,
        }

        #[derive(Deserialize)]
        struct OpenAIUsage {
            prompt_tokens: u32,
            completion_tokens: u32,
            total_tokens: u32,
        }

        let request = OpenAIRequest {
            model: self.config.model.clone(),
            messages: messages.to_vec(),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        };

        debug!("Sending request to OpenAI: {}", url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();

            if status.as_u16() == 429 {
                return Err(LlmError::RateLimitExceeded);
            } else if status.as_u16() == 401 {
                return Err(LlmError::InvalidApiKey);
            }

            return Err(LlmError::ApiError(format!(
                "Status {}: {}",
                status, body
            )));
        }

        let openai_response: OpenAIResponse = response.json().await?;

        let choice = openai_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::InvalidResponse("No choices in response".to_string()))?;

        Ok(LlmResponse {
            content: choice.message.content,
            model: openai_response.model,
            usage: openai_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
            finish_reason: choice.finish_reason,
        })
    }

    async fn complete_anthropic(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        self.chat_anthropic(&messages).await
    }

    async fn chat_anthropic(&self, messages: &[ChatMessage]) -> Result<LlmResponse, LlmError> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");

        let url = format!("{}/messages", base_url);

        #[derive(Serialize)]
        struct AnthropicRequest {
            model: String,
            messages: Vec<ChatMessage>,
            max_tokens: u32,
        }

        #[derive(Deserialize)]
        struct AnthropicResponse {
            content: Vec<AnthropicContent>,
            model: String,
            stop_reason: Option<String>,
            usage: Option<AnthropicUsage>,
        }

        #[derive(Deserialize)]
        struct AnthropicContent {
            text: String,
        }

        #[derive(Deserialize)]
        struct AnthropicUsage {
            input_tokens: u32,
            output_tokens: u32,
        }

        let request = AnthropicRequest {
            model: self.config.model.clone(),
            messages: messages.to_vec(),
            max_tokens: self.config.max_tokens,
        };

        debug!("Sending request to Anthropic: {}", url);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();

            if status.as_u16() == 429 {
                return Err(LlmError::RateLimitExceeded);
            } else if status.as_u16() == 401 {
                return Err(LlmError::InvalidApiKey);
            }

            return Err(LlmError::ApiError(format!(
                "Status {}: {}",
                status, body
            )));
        }

        let anthropic_response: AnthropicResponse = response.json().await?;

        let content = anthropic_response
            .content
            .into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n");

        Ok(LlmResponse {
            content,
            model: anthropic_response.model,
            usage: anthropic_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            }),
            finish_reason: anthropic_response.stop_reason,
        })
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.max_tokens, 2000);
        assert!((config.temperature - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage::user("Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");

        let msg = ChatMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
    }
}
