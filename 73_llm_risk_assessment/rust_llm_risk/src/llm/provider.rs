//! LLM provider configuration.

use serde::{Deserialize, Serialize};

/// Supported LLM providers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LLMProvider {
    /// OpenAI GPT models
    OpenAI {
        model: String,
        api_key: String,
    },
    /// Anthropic Claude models
    Anthropic {
        model: String,
        api_key: String,
    },
    /// Local/self-hosted models (Ollama, vLLM, etc.)
    Local {
        model: String,
        base_url: String,
    },
    /// Azure OpenAI Service
    AzureOpenAI {
        model: String,
        api_key: String,
        endpoint: String,
        deployment: String,
    },
}

impl LLMProvider {
    /// Create an OpenAI provider with GPT-4.
    pub fn openai_gpt4(api_key: String) -> Self {
        LLMProvider::OpenAI {
            model: "gpt-4".to_string(),
            api_key,
        }
    }

    /// Create an OpenAI provider with GPT-3.5-turbo.
    pub fn openai_gpt35(api_key: String) -> Self {
        LLMProvider::OpenAI {
            model: "gpt-3.5-turbo".to_string(),
            api_key,
        }
    }

    /// Create an Anthropic provider with Claude-3.
    pub fn anthropic_claude3(api_key: String) -> Self {
        LLMProvider::Anthropic {
            model: "claude-3-sonnet-20240229".to_string(),
            api_key,
        }
    }

    /// Create a local Ollama provider.
    pub fn ollama(model: String) -> Self {
        LLMProvider::Local {
            model,
            base_url: "http://localhost:11434".to_string(),
        }
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        match self {
            LLMProvider::OpenAI { model, .. } => model,
            LLMProvider::Anthropic { model, .. } => model,
            LLMProvider::Local { model, .. } => model,
            LLMProvider::AzureOpenAI { model, .. } => model,
        }
    }

    /// Get the provider name.
    pub fn provider_name(&self) -> &str {
        match self {
            LLMProvider::OpenAI { .. } => "OpenAI",
            LLMProvider::Anthropic { .. } => "Anthropic",
            LLMProvider::Local { .. } => "Local",
            LLMProvider::AzureOpenAI { .. } => "AzureOpenAI",
        }
    }

    /// Get the API endpoint URL.
    pub fn endpoint_url(&self) -> String {
        match self {
            LLMProvider::OpenAI { .. } => {
                "https://api.openai.com/v1/chat/completions".to_string()
            }
            LLMProvider::Anthropic { .. } => {
                "https://api.anthropic.com/v1/messages".to_string()
            }
            LLMProvider::Local { base_url, .. } => {
                format!("{}/api/chat", base_url)
            }
            LLMProvider::AzureOpenAI { endpoint, deployment, .. } => {
                format!(
                    "{}/openai/deployments/{}/chat/completions?api-version=2024-02-15-preview",
                    endpoint, deployment
                )
            }
        }
    }
}

impl Default for LLMProvider {
    fn default() -> Self {
        // Default to local Ollama with llama2
        LLMProvider::ollama("llama2".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let openai = LLMProvider::openai_gpt4("test-key".to_string());
        assert_eq!(openai.model_name(), "gpt-4");
        assert_eq!(openai.provider_name(), "OpenAI");
    }

    #[test]
    fn test_endpoint_urls() {
        let openai = LLMProvider::openai_gpt4("key".to_string());
        assert!(openai.endpoint_url().contains("openai.com"));

        let anthropic = LLMProvider::anthropic_claude3("key".to_string());
        assert!(anthropic.endpoint_url().contains("anthropic.com"));

        let ollama = LLMProvider::ollama("llama2".to_string());
        assert!(ollama.endpoint_url().contains("localhost"));
    }
}
