//! LLM adapter for intelligent execution decisions.

use crate::data::OrderBook;
use crate::execution::{ParentOrder, Side};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// LLM-related errors
#[derive(Error, Debug)]
pub enum LlmError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {0}")]
    Api(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// API endpoint URL
    pub api_url: String,
    /// API key
    pub api_key: String,
    /// Model name
    pub model: String,
    /// Maximum tokens in response
    pub max_tokens: u32,
    /// Temperature for generation
    pub temperature: f32,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_url: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key: String::new(),
            model: "gpt-4".to_string(),
            max_tokens: 512,
            temperature: 0.1,
            timeout_ms: 30000,
        }
    }
}

impl LlmConfig {
    /// Create config for OpenAI
    pub fn openai(api_key: String) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }

    /// Create config for Anthropic Claude
    pub fn anthropic(api_key: String) -> Self {
        Self {
            api_url: "https://api.anthropic.com/v1/messages".to_string(),
            api_key,
            model: "claude-3-sonnet-20240229".to_string(),
            ..Default::default()
        }
    }

    /// Create config for local LLM (e.g., Ollama)
    pub fn local(url: String, model: String) -> Self {
        Self {
            api_url: url,
            api_key: String::new(),
            model,
            ..Default::default()
        }
    }
}

/// LLM execution decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmDecision {
    /// Recommended action
    pub action: ExecutionAction,
    /// Quantity to execute (as fraction of remaining)
    pub quantity_fraction: f64,
    /// Price aggressiveness (-1.0 passive to +1.0 aggressive)
    pub aggressiveness: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Reasoning explanation
    pub reasoning: String,
    /// Recommended urgency adjustment
    pub urgency_adjustment: f64,
}

impl Default for LlmDecision {
    fn default() -> Self {
        Self {
            action: ExecutionAction::Continue,
            quantity_fraction: 0.0,
            aggressiveness: 0.0,
            confidence: 0.5,
            reasoning: "Default decision".to_string(),
            urgency_adjustment: 0.0,
        }
    }
}

/// Execution actions recommended by LLM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionAction {
    /// Execute a slice now
    Execute,
    /// Wait for better conditions
    Wait,
    /// Continue with current strategy
    Continue,
    /// Increase execution pace
    Accelerate,
    /// Decrease execution pace
    Decelerate,
    /// Pause execution
    Pause,
    /// Cancel remaining execution
    Cancel,
}

/// Market state for LLM analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    /// Current mid price
    pub mid_price: f64,
    /// Bid-ask spread in bps
    pub spread_bps: f64,
    /// Order book imbalance (-1 to +1)
    pub imbalance: f64,
    /// Bid depth (top 10 levels)
    pub bid_depth: f64,
    /// Ask depth (top 10 levels)
    pub ask_depth: f64,
    /// Recent price change (%)
    pub price_change_pct: f64,
    /// Recent volatility (%)
    pub volatility: f64,
    /// Recent volume vs average
    pub volume_ratio: f64,
    /// Funding rate (for perpetuals)
    pub funding_rate: Option<f64>,
    /// Open interest change (%)
    pub oi_change_pct: Option<f64>,
}

impl MarketState {
    /// Create from order book
    pub fn from_orderbook(orderbook: &OrderBook) -> Self {
        Self {
            mid_price: orderbook.mid_price().unwrap_or(0.0),
            spread_bps: orderbook.spread_bps().unwrap_or(0.0),
            imbalance: orderbook.imbalance(10),
            bid_depth: orderbook.bid_depth(10),
            ask_depth: orderbook.ask_depth(10),
            price_change_pct: 0.0,
            volatility: 0.0,
            volume_ratio: 1.0,
            funding_rate: None,
            oi_change_pct: None,
        }
    }
}

/// Execution context for LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Order side
    pub side: Side,
    /// Total quantity
    pub total_quantity: f64,
    /// Filled quantity
    pub filled_quantity: f64,
    /// Remaining time (seconds)
    pub remaining_time: u64,
    /// Target participation rate
    pub target_participation: f64,
    /// Actual participation rate
    pub actual_participation: f64,
    /// Current VWAP slippage (bps)
    pub vwap_slippage_bps: f64,
    /// Implementation shortfall (bps)
    pub is_bps: f64,
    /// Urgency parameter
    pub urgency: f64,
}

/// LLM adapter for execution decisions
#[derive(Debug, Clone)]
pub struct LlmAdapter {
    config: LlmConfig,
    client: reqwest::Client,
}

impl LlmAdapter {
    /// Create a new LLM adapter
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|e| LlmError::Config(e.to_string()))?;

        Ok(Self { config, client })
    }

    /// Create an adapter for OpenAI
    pub fn openai(api_key: String) -> Result<Self, LlmError> {
        Self::new(LlmConfig::openai(api_key))
    }

    /// Create an adapter for Anthropic
    pub fn anthropic(api_key: String) -> Result<Self, LlmError> {
        Self::new(LlmConfig::anthropic(api_key))
    }

    /// Create an adapter for local LLM
    pub fn local(url: String, model: String) -> Result<Self, LlmError> {
        Self::new(LlmConfig::local(url, model))
    }

    /// Build the execution prompt
    fn build_prompt(&self, market: &MarketState, context: &ExecutionContext) -> String {
        format!(
            r#"You are an expert algorithmic trading execution optimizer. Analyze the current market conditions and execution progress to recommend the next action.

## Market State
- Mid Price: {:.2}
- Spread: {:.2} bps
- Order Book Imbalance: {:.2} (positive = more bids)
- Bid Depth (10 levels): {:.4}
- Ask Depth (10 levels): {:.4}
- Recent Price Change: {:.2}%
- Volatility: {:.2}%
- Volume Ratio (vs avg): {:.2}x
{}
{}

## Execution Context
- Side: {}
- Total Quantity: {:.4}
- Filled: {:.4} ({:.1}%)
- Remaining Time: {} seconds
- Target Participation: {:.1}%
- Actual Participation: {:.1}%
- VWAP Slippage: {:.2} bps
- Implementation Shortfall: {:.2} bps
- Urgency: {:.2}

## Task
Recommend the optimal execution action. Consider:
1. Market impact vs opportunity cost tradeoff
2. Current vs target progress
3. Market conditions (spread, depth, volatility)
4. Time pressure

Respond in JSON format:
{{
    "action": "Execute|Wait|Continue|Accelerate|Decelerate|Pause|Cancel",
    "quantity_fraction": <0.0 to 1.0>,
    "aggressiveness": <-1.0 to 1.0>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>",
    "urgency_adjustment": <-0.2 to 0.2>
}}"#,
            market.mid_price,
            market.spread_bps,
            market.imbalance,
            market.bid_depth,
            market.ask_depth,
            market.price_change_pct,
            market.volatility,
            market.volume_ratio,
            market.funding_rate.map_or(String::new(), |r| format!("- Funding Rate: {:.4}%", r * 100.0)),
            market.oi_change_pct.map_or(String::new(), |r| format!("- OI Change: {:.2}%", r)),
            context.side,
            context.total_quantity,
            context.filled_quantity,
            (context.filled_quantity / context.total_quantity) * 100.0,
            context.remaining_time,
            context.target_participation * 100.0,
            context.actual_participation * 100.0,
            context.vwap_slippage_bps,
            context.is_bps,
            context.urgency,
        )
    }

    /// Parse LLM response into decision
    fn parse_response(&self, response: &str) -> Result<LlmDecision, LlmError> {
        // Try to extract JSON from the response
        let json_str = if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                &response[start..=end]
            } else {
                return Err(LlmError::Parse("No closing brace found".to_string()));
            }
        } else {
            return Err(LlmError::Parse("No JSON found in response".to_string()));
        };

        serde_json::from_str(json_str)
            .map_err(|e| LlmError::Parse(format!("Failed to parse JSON: {}", e)))
    }

    /// Get execution decision from LLM
    pub async fn get_decision(
        &self,
        market: &MarketState,
        context: &ExecutionContext,
    ) -> Result<LlmDecision, LlmError> {
        let prompt = self.build_prompt(market, context);

        // Call OpenAI-compatible API
        let request_body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert algorithmic trading execution optimizer. Respond only with valid JSON."
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
            .client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if response.status() == 429 {
            return Err(LlmError::RateLimit);
        }

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(error_text));
        }

        let response_json: serde_json::Value = response.json().await?;

        let content = response_json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| LlmError::InvalidResponse("Missing content".to_string()))?;

        self.parse_response(content)
    }

    /// Get a simple heuristic decision (fallback when LLM is unavailable)
    pub fn get_heuristic_decision(
        &self,
        market: &MarketState,
        context: &ExecutionContext,
    ) -> LlmDecision {
        let remaining_fraction = 1.0 - (context.filled_quantity / context.total_quantity);
        let time_fraction = context.remaining_time as f64 / 3600.0; // Normalize to 1 hour

        // Determine if we're behind or ahead of schedule
        let expected_fill = 1.0 - time_fraction;
        let actual_fill = context.filled_quantity / context.total_quantity;
        let progress_diff = actual_fill - expected_fill;

        // Adjust based on market conditions
        let spread_factor = if market.spread_bps < 5.0 {
            1.2 // Tight spread, be more aggressive
        } else if market.spread_bps > 20.0 {
            0.8 // Wide spread, be more passive
        } else {
            1.0
        };

        // Consider order book imbalance
        let imbalance_factor = match context.side {
            Side::Buy => {
                if market.imbalance > 0.3 {
                    0.8 // More bids, slow down buying
                } else if market.imbalance < -0.3 {
                    1.2 // More asks, speed up buying
                } else {
                    1.0
                }
            }
            Side::Sell => {
                if market.imbalance < -0.3 {
                    0.8 // More asks, slow down selling
                } else if market.imbalance > 0.3 {
                    1.2 // More bids, speed up selling
                } else {
                    1.0
                }
            }
        };

        // Calculate quantity fraction
        let base_fraction = remaining_fraction / (context.remaining_time as f64 / 60.0).max(1.0);
        let quantity_fraction = (base_fraction * spread_factor * imbalance_factor)
            .clamp(0.01, 0.25);

        // Determine action
        let (action, aggressiveness) = if progress_diff < -0.1 {
            // Behind schedule
            (ExecutionAction::Accelerate, 0.3)
        } else if progress_diff > 0.1 {
            // Ahead of schedule
            (ExecutionAction::Decelerate, -0.3)
        } else if market.spread_bps > 30.0 {
            // Very wide spread
            (ExecutionAction::Wait, -0.5)
        } else {
            (ExecutionAction::Execute, 0.0)
        };

        LlmDecision {
            action,
            quantity_fraction,
            aggressiveness,
            confidence: 0.7,
            reasoning: format!(
                "Heuristic: progress_diff={:.2}, spread={:.1}bps, imbalance={:.2}",
                progress_diff, market.spread_bps, market.imbalance
            ),
            urgency_adjustment: progress_diff.clamp(-0.1, 0.1),
        }
    }
}

/// Build execution context from parent order
pub fn build_execution_context(
    order: &ParentOrder,
    target_participation: f64,
    actual_participation: f64,
    vwap: f64,
    market_vwap: f64,
) -> ExecutionContext {
    let vwap_slippage_bps = if market_vwap > 0.0 && vwap > 0.0 {
        ((vwap - market_vwap) / market_vwap * 10000.0) * order.side.sign()
    } else {
        0.0
    };

    let is_bps = if let (Some(arrival), Some(avg)) = (order.arrival_price, order.average_price) {
        if arrival > 0.0 {
            ((avg - arrival) / arrival * 10000.0) * order.side.sign()
        } else {
            0.0
        }
    } else {
        0.0
    };

    ExecutionContext {
        side: order.side,
        total_quantity: order.total_quantity,
        filled_quantity: order.filled_quantity,
        remaining_time: order.remaining_time(),
        target_participation,
        actual_participation,
        vwap_slippage_bps,
        is_bps,
        urgency: order.urgency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_openai() {
        let config = LlmConfig::openai("test-key".to_string());
        assert!(config.api_url.contains("openai"));
        assert_eq!(config.api_key, "test-key");
    }

    #[test]
    fn test_market_state_from_orderbook() {
        use crate::data::OrderBook;

        let mut book = OrderBook::new("BTCUSDT".to_string());
        book.update_bid(49990.0, 1.0);
        book.update_ask(50010.0, 1.0);

        let state = MarketState::from_orderbook(&book);
        assert_eq!(state.mid_price, 50000.0);
        assert!((state.spread_bps - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_heuristic_decision() {
        let config = LlmConfig::default();
        let adapter = LlmAdapter::new(config).unwrap();

        let market = MarketState {
            mid_price: 50000.0,
            spread_bps: 5.0,
            imbalance: 0.0,
            bid_depth: 100.0,
            ask_depth: 100.0,
            price_change_pct: 0.0,
            volatility: 0.5,
            volume_ratio: 1.0,
            funding_rate: None,
            oi_change_pct: None,
        };

        let context = ExecutionContext {
            side: Side::Buy,
            total_quantity: 10.0,
            filled_quantity: 5.0,
            remaining_time: 1800,
            target_participation: 0.1,
            actual_participation: 0.1,
            vwap_slippage_bps: 2.0,
            is_bps: 3.0,
            urgency: 0.5,
        };

        let decision = adapter.get_heuristic_decision(&market, &context);
        assert!(decision.quantity_fraction > 0.0);
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn test_parse_response() {
        let config = LlmConfig::default();
        let adapter = LlmAdapter::new(config).unwrap();

        let response = r#"
        Here is my analysis:
        {
            "action": "Execute",
            "quantity_fraction": 0.05,
            "aggressiveness": 0.2,
            "confidence": 0.85,
            "reasoning": "Good conditions",
            "urgency_adjustment": 0.0
        }
        "#;

        let decision = adapter.parse_response(response).unwrap();
        assert_eq!(decision.action, ExecutionAction::Execute);
        assert!((decision.quantity_fraction - 0.05).abs() < 0.001);
    }
}
