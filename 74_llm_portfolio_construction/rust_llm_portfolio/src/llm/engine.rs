//! LLM Portfolio Engine Implementation

use crate::portfolio::{Asset, AssetScore, Confidence, MarketData, Portfolio};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LLM Portfolio Engine for analyzing assets and generating portfolios
pub struct LLMPortfolioEngine {
    api_key: Option<String>,
    base_url: String,
    model: String,
    client: Client,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct LLMScoreResponse {
    scores: Vec<LLMAssetScoreRaw>,
}

#[derive(Debug, Deserialize)]
struct LLMAssetScoreRaw {
    symbol: String,
    fundamental_score: f64,
    momentum_score: f64,
    sentiment_score: f64,
    risk_score: f64,
    reasoning: String,
}

impl LLMPortfolioEngine {
    /// Create a new LLM Portfolio Engine
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4".to_string(),
            client: Client::new(),
        }
    }

    /// Create with custom base URL (for local models or alternative providers)
    pub fn with_base_url(api_key: Option<String>, base_url: String, model: String) -> Self {
        Self {
            api_key,
            base_url,
            model,
            client: Client::new(),
        }
    }

    /// Analyze assets using mock data (for testing without API)
    pub fn analyze_assets_mock(
        &self,
        assets: &[Asset],
        _market_data: &HashMap<String, MarketData>,
        _news_headlines: &[String],
    ) -> Vec<AssetScore> {
        assets
            .iter()
            .map(|asset| {
                // Generate deterministic mock scores based on asset symbol
                let hash = asset.symbol.bytes().fold(0u64, |acc: u64, b: u8| acc.wrapping_add(b as u64));
                let base: f64 = (hash % 30) as f64 / 10.0 + 5.0; // Range 5.0-8.0

                AssetScore {
                    symbol: asset.symbol.clone(),
                    fundamental_score: (base + 0.5).min(10.0),
                    momentum_score: (base - 0.3).max(1.0),
                    sentiment_score: base,
                    risk_score: (10.0_f64 - base + 2.0).min(10.0).max(1.0),
                    overall_score: base,
                    confidence: Confidence::Medium,
                    reasoning: format!(
                        "Mock analysis for {}: Stable asset with moderate growth potential.",
                        asset.name
                    ),
                }
            })
            .collect()
    }

    /// Analyze assets using real LLM API
    pub async fn analyze_assets(
        &self,
        assets: &[Asset],
        market_data: &HashMap<String, MarketData>,
        news_headlines: &[String],
    ) -> Result<Vec<AssetScore>, Box<dyn std::error::Error>> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or("API key required for real LLM analysis")?;

        let prompt = self.build_analysis_prompt(assets, market_data, news_headlines);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: Self::system_prompt().to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt,
                },
            ],
            temperature: 0.3,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?
            .json::<ChatResponse>()
            .await?;

        let content = &response.choices[0].message.content;
        let parsed: LLMScoreResponse = serde_json::from_str(content)?;

        let scores = parsed
            .scores
            .into_iter()
            .filter_map(|llm_score: LLMAssetScoreRaw| {
                // Check if this symbol is in our assets list
                if assets.iter().any(|a| a.symbol == llm_score.symbol) {
                    let overall = (llm_score.fundamental_score
                        + llm_score.momentum_score
                        + llm_score.sentiment_score
                        + (10.0 - llm_score.risk_score))
                        / 4.0;

                    Some(AssetScore {
                        symbol: llm_score.symbol,
                        fundamental_score: llm_score.fundamental_score,
                        momentum_score: llm_score.momentum_score,
                        sentiment_score: llm_score.sentiment_score,
                        risk_score: llm_score.risk_score,
                        overall_score: overall,
                        confidence: Self::determine_confidence(overall),
                        reasoning: llm_score.reasoning,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(scores)
    }

    fn system_prompt() -> &'static str {
        r#"You are a quantitative financial analyst specializing in portfolio construction.
Your task is to analyze assets and provide scores for portfolio optimization.

For each asset, provide scores from 1-10 for:
- fundamental_score: Based on company/project fundamentals, financials, technology
- momentum_score: Based on price trends and technical indicators
- sentiment_score: Based on news, social media, and market sentiment
- risk_score: Higher = more risky (volatility, regulatory risks, etc.)

Respond ONLY with valid JSON in this exact format:
{
  "scores": [
    {
      "symbol": "SYMBOL",
      "fundamental_score": 7.5,
      "momentum_score": 6.0,
      "sentiment_score": 8.0,
      "risk_score": 4.0,
      "reasoning": "Brief explanation"
    }
  ]
}"#
    }

    fn build_analysis_prompt(
        &self,
        assets: &[Asset],
        market_data: &HashMap<String, MarketData>,
        news_headlines: &[String],
    ) -> String {
        let mut prompt = String::from("Analyze the following assets for portfolio construction:\n\n");

        prompt.push_str("## Assets:\n");
        for asset in assets {
            prompt.push_str(&format!(
                "- {} ({}): {} - Current Price: ${:.2}\n",
                asset.symbol, asset.asset_class, asset.name, asset.current_price
            ));
        }

        prompt.push_str("\n## Market Data:\n");
        for (symbol, data) in market_data {
            prompt.push_str(&format!(
                "- {}: Volatility: {:.2}%, 7d Return: {:.2}%, 30d Return: {:.2}%\n",
                symbol,
                data.volatility * 100.0,
                data.return_7d * 100.0,
                data.return_30d * 100.0
            ));
        }

        if !news_headlines.is_empty() {
            prompt.push_str("\n## Recent News Headlines:\n");
            for headline in news_headlines.iter().take(10) {
                prompt.push_str(&format!("- {}\n", headline));
            }
        }

        prompt.push_str("\nProvide your analysis as JSON.");
        prompt
    }

    fn determine_confidence(overall_score: f64) -> Confidence {
        if overall_score >= 7.0 {
            Confidence::High
        } else if overall_score >= 5.0 {
            Confidence::Medium
        } else {
            Confidence::Low
        }
    }

    /// Generate portfolio from asset scores using score-weighted allocation
    pub fn generate_portfolio(&self, scores: &[AssetScore], min_weight: f64) -> Portfolio {
        if scores.is_empty() {
            return Portfolio::new(HashMap::new());
        }

        // Filter scores above minimum threshold
        let valid_scores: Vec<_> = scores
            .iter()
            .filter(|s| s.overall_score >= 5.0)
            .collect();

        if valid_scores.is_empty() {
            return Portfolio::new(HashMap::new());
        }

        // Calculate total score
        let total_score: f64 = valid_scores.iter().map(|s| s.overall_score).sum();

        // Calculate raw weights
        let mut weights: HashMap<String, f64> = valid_scores
            .iter()
            .map(|s| (s.symbol.clone(), s.overall_score / total_score))
            .collect();

        // Apply minimum weight constraint
        let mut needs_redistribution = true;
        while needs_redistribution {
            needs_redistribution = false;
            let below_min: Vec<String> = weights
                .iter()
                .filter(|(_, &w)| w < min_weight && w > 0.0)
                .map(|(k, _): (&String, &f64)| k.clone())
                .collect();

            for symbol in below_min {
                weights.remove(&symbol);
                needs_redistribution = true;
            }

            if needs_redistribution && !weights.is_empty() {
                let total: f64 = weights.values().sum();
                for w in weights.values_mut() {
                    *w /= total;
                }
            }
        }

        let mut portfolio = Portfolio::new(weights);
        portfolio.metadata.insert("strategy".to_string(), "score_weighted".to_string());
        portfolio.metadata.insert("num_assets".to_string(), valid_scores.len().to_string());

        portfolio
    }
}

impl Default for LLMPortfolioEngine {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portfolio::AssetClass;

    #[test]
    fn test_mock_analysis() {
        let engine = LLMPortfolioEngine::new(None);

        let assets = vec![
            Asset::new("BTC", "Bitcoin", AssetClass::Crypto, 65000.0),
            Asset::new("ETH", "Ethereum", AssetClass::Crypto, 3200.0),
        ];

        let scores = engine.analyze_assets_mock(&assets, &HashMap::new(), &[]);

        assert_eq!(scores.len(), 2);
        for score in &scores {
            assert!(score.overall_score >= 1.0 && score.overall_score <= 10.0);
            assert!(score.fundamental_score >= 1.0 && score.fundamental_score <= 10.0);
        }
    }

    #[test]
    fn test_portfolio_generation() {
        let engine = LLMPortfolioEngine::new(None);

        let assets = vec![
            Asset::new("BTC", "Bitcoin", AssetClass::Crypto, 65000.0),
            Asset::new("ETH", "Ethereum", AssetClass::Crypto, 3200.0),
        ];

        let scores = engine.analyze_assets_mock(&assets, &HashMap::new(), &[]);
        let portfolio = engine.generate_portfolio(&scores, 0.05);

        let total_weight: f64 = portfolio.weights.values().sum();
        assert!((total_weight - 1.0).abs() < 0.001);
    }
}
