//! LLM API client for making requests to language models.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::provider::LLMProvider;

/// Message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
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

/// OpenAI-compatible request format.
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

/// OpenAI-compatible response format.
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: MessageContent,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
    content: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Anthropic request format.
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

/// Anthropic response format.
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: String,
}

/// Client for interacting with LLM APIs.
pub struct LLMClient {
    provider: LLMProvider,
    client: Client,
    temperature: f32,
    max_tokens: u32,
}

impl LLMClient {
    /// Create a new LLM client.
    pub fn new(provider: LLMProvider) -> Self {
        Self {
            provider,
            client: Client::new(),
            temperature: 0.3, // Lower temperature for more consistent risk assessments
            max_tokens: 2000,
        }
    }

    /// Set the temperature for responses.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the maximum tokens for responses.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Send a completion request to the LLM.
    pub async fn complete(&self, messages: Vec<Message>) -> Result<String> {
        match &self.provider {
            LLMProvider::OpenAI { model, api_key } => {
                self.complete_openai(model, api_key, messages).await
            }
            LLMProvider::Anthropic { model, api_key } => {
                self.complete_anthropic(model, api_key, messages).await
            }
            LLMProvider::Local { model, base_url } => {
                self.complete_local(model, base_url, messages).await
            }
            LLMProvider::AzureOpenAI {
                model,
                api_key,
                endpoint,
                deployment,
            } => {
                self.complete_azure(model, api_key, endpoint, deployment, messages)
                    .await
            }
        }
    }

    /// Complete using OpenAI API.
    async fn complete_openai(
        &self,
        model: &str,
        api_key: &str,
        messages: Vec<Message>,
    ) -> Result<String> {
        debug!("Sending request to OpenAI API");

        let request = OpenAIRequest {
            model: model.to_string(),
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            response_format: Some(ResponseFormat {
                format_type: "json_object".to_string(),
            }),
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error {}: {}", status, error_text);
        }

        let data: OpenAIResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        if let Some(usage) = &data.usage {
            info!(
                "Token usage - prompt: {}, completion: {}, total: {}",
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
            );
        }

        data.choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("No response from OpenAI"))
    }

    /// Complete using Anthropic API.
    async fn complete_anthropic(
        &self,
        model: &str,
        api_key: &str,
        messages: Vec<Message>,
    ) -> Result<String> {
        debug!("Sending request to Anthropic API");

        // Separate system message from user/assistant messages
        let (system, chat_messages): (Option<String>, Vec<Message>) = {
            let system_msg = messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone());

            let other_msgs: Vec<Message> = messages
                .into_iter()
                .filter(|m| m.role != "system")
                .collect();

            (system_msg, other_msgs)
        };

        let request = AnthropicRequest {
            model: model.to_string(),
            messages: chat_messages,
            max_tokens: self.max_tokens,
            system,
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error {}: {}", status, error_text);
        }

        let data: AnthropicResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic response")?;

        data.content
            .first()
            .map(|c| c.text.clone())
            .ok_or_else(|| anyhow::anyhow!("No response from Anthropic"))
    }

    /// Complete using local/Ollama API.
    async fn complete_local(
        &self,
        model: &str,
        base_url: &str,
        messages: Vec<Message>,
    ) -> Result<String> {
        debug!("Sending request to local LLM at {}", base_url);

        #[derive(Serialize)]
        struct OllamaRequest {
            model: String,
            messages: Vec<Message>,
            stream: bool,
        }

        #[derive(Deserialize)]
        struct OllamaResponse {
            message: MessageContent,
        }

        let request = OllamaRequest {
            model: model.to_string(),
            messages,
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/api/chat", base_url))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to local LLM")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Local LLM error {}: {}", status, error_text);
        }

        let data: OllamaResponse = response
            .json()
            .await
            .context("Failed to parse local LLM response")?;

        Ok(data.message.content)
    }

    /// Complete using Azure OpenAI API.
    async fn complete_azure(
        &self,
        model: &str,
        api_key: &str,
        endpoint: &str,
        deployment: &str,
        messages: Vec<Message>,
    ) -> Result<String> {
        debug!("Sending request to Azure OpenAI API");

        let request = OpenAIRequest {
            model: model.to_string(),
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            response_format: Some(ResponseFormat {
                format_type: "json_object".to_string(),
            }),
        };

        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version=2024-02-15-preview",
            endpoint, deployment
        );

        let response = self
            .client
            .post(&url)
            .header("api-key", api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Azure OpenAI")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Azure OpenAI API error {}: {}", status, error_text);
        }

        let data: OpenAIResponse = response
            .json()
            .await
            .context("Failed to parse Azure OpenAI response")?;

        data.choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("No response from Azure OpenAI"))
    }

    /// Get the provider name.
    pub fn provider_name(&self) -> &str {
        self.provider.provider_name()
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.provider.model_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let system = Message::system("You are a helpful assistant.");
        assert_eq!(system.role, "system");

        let user = Message::user("Hello!");
        assert_eq!(user.role, "user");

        let assistant = Message::assistant("Hi there!");
        assert_eq!(assistant.role, "assistant");
    }

    #[test]
    fn test_client_creation() {
        let provider = LLMProvider::ollama("llama2".to_string());
        let client = LLMClient::new(provider)
            .with_temperature(0.5)
            .with_max_tokens(1000);

        assert_eq!(client.model_name(), "llama2");
        assert_eq!(client.provider_name(), "Local");
    }
}
