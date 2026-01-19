//! LLM client implementations for various providers

use super::{LlmConfig, LlmError, LlmResponse};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Mutex;

/// Supported LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    /// OpenAI (GPT-4, GPT-3.5)
    OpenAI,
    /// Anthropic (Claude)
    Anthropic,
    /// Local models (Ollama, etc.)
    Local,
}

impl Default for LlmProvider {
    fn default() -> Self {
        Self::OpenAI
    }
}

/// LLM client for making API calls
#[derive(Debug)]
pub struct LlmClient {
    config: LlmConfig,
    http_client: reqwest::Client,
    cache: Mutex<LruCache<String, LlmResponse>>,
}

impl LlmClient {
    /// Create a new LLM client with configuration
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|e| LlmError::Unknown(e.to_string()))?;

        let cache_size = NonZeroUsize::new(1000).unwrap();

        Ok(Self {
            config,
            http_client,
            cache: Mutex::new(LruCache::new(cache_size)),
        })
    }

    /// Create a client for OpenAI
    pub fn new_openai(api_key: &str) -> Result<Self, LlmError> {
        let config = LlmConfig {
            api_key: api_key.to_string(),
            provider: LlmProvider::OpenAI,
            model: "gpt-4".to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a client for Anthropic (Claude)
    pub fn new_anthropic(api_key: &str) -> Result<Self, LlmError> {
        let config = LlmConfig {
            api_key: api_key.to_string(),
            provider: LlmProvider::Anthropic,
            model: "claude-3-opus-20240229".to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a client for local models
    pub fn new_local(base_url: &str) -> Result<Self, LlmError> {
        let config = LlmConfig {
            api_key: String::new(),
            provider: LlmProvider::Local,
            model: "llama2".to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Make a completion request
    pub async fn complete(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        // Check cache first
        if self.config.enable_cache {
            let cache_key = self.cache_key(prompt);
            if let Ok(mut cache) = self.cache.lock() {
                if let Some(cached) = cache.get(&cache_key) {
                    tracing::debug!("Cache hit for prompt");
                    return Ok(cached.clone());
                }
            }
        }

        // Make API call based on provider
        let response = match self.config.provider {
            LlmProvider::OpenAI => self.complete_openai(prompt).await?,
            LlmProvider::Anthropic => self.complete_anthropic(prompt).await?,
            LlmProvider::Local => self.complete_local(prompt).await?,
        };

        // Store in cache
        if self.config.enable_cache {
            let cache_key = self.cache_key(prompt);
            if let Ok(mut cache) = self.cache.lock() {
                cache.put(cache_key, response.clone());
            }
        }

        Ok(response)
    }

    /// Generate cache key for a prompt
    fn cache_key(&self, prompt: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        hasher.update(self.config.model.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Complete using OpenAI API
    async fn complete_openai(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        let request_body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial analyst specializing in cryptocurrency markets. Analyze news and extract trading-relevant information in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        });

        let response = self
            .http_client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        if response.status() == 429 {
            return Err(LlmError::RateLimitError(60));
        }

        if response.status() == 401 {
            return Err(LlmError::AuthenticationError);
        }

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!("Status {}: {}", status, text)));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| LlmError::ParseError("Missing content in response".to_string()))?
            .to_string();

        let usage = json["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize;

        Ok(LlmResponse {
            content,
            model: self.config.model.clone(),
            tokens_used: usage,
            latency_ms: 0, // Would need to track this
        })
    }

    /// Complete using Anthropic API
    async fn complete_anthropic(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        let request_body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        let response = self
            .http_client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!("Status {}: {}", status, text)));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        let content = json["content"][0]["text"]
            .as_str()
            .ok_or_else(|| LlmError::ParseError("Missing content in response".to_string()))?
            .to_string();

        Ok(LlmResponse {
            content,
            model: self.config.model.clone(),
            tokens_used: 0,
            latency_ms: 0,
        })
    }

    /// Complete using local model (Ollama-style API)
    async fn complete_local(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        let request_body = serde_json::json!({
            "model": self.config.model,
            "prompt": prompt,
            "stream": false
        });

        let response = self
            .http_client
            .post("http://localhost:11434/api/generate")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(LlmError::ModelNotAvailable(self.config.model.clone()));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        let content = json["response"]
            .as_str()
            .ok_or_else(|| LlmError::ParseError("Missing response".to_string()))?
            .to_string();

        Ok(LlmResponse {
            content,
            model: self.config.model.clone(),
            tokens_used: 0,
            latency_ms: 0,
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    /// Clear the response cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_default() {
        let provider = LlmProvider::default();
        assert_eq!(provider, LlmProvider::OpenAI);
    }

    #[test]
    fn test_client_creation() {
        let config = LlmConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        };
        let client = LlmClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_cache_key_generation() {
        let config = LlmConfig {
            api_key: "test".to_string(),
            ..Default::default()
        };
        let client = LlmClient::new(config).unwrap();

        let key1 = client.cache_key("hello");
        let key2 = client.cache_key("hello");
        let key3 = client.cache_key("world");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
