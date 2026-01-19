//! Trading signal generation using LLMs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{TradingError, TradingResult};
use crate::llm_client::{LLMClient, LLMConfig};
use crate::prompts::signals as prompts;

/// Trading signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SignalDirection {
    Buy,
    Sell,
    Hold,
}

/// Trading signal with entry/exit levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub symbol: String,
    pub direction: SignalDirection,
    pub confidence: f32,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub position_size_pct: f32,
    pub timeframe: String,
    pub reasoning: String,
    pub key_factors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
}

impl TradingSignal {
    /// Calculate risk/reward ratio.
    pub fn risk_reward_ratio(&self) -> f64 {
        let risk = (self.entry_price - self.stop_loss).abs();
        let reward = (self.take_profit - self.entry_price).abs();
        if risk > 0.0 {
            reward / risk
        } else {
            0.0
        }
    }

    /// Check if signal is actionable (not HOLD).
    pub fn is_actionable(&self) -> bool {
        self.direction != SignalDirection::Hold
    }
}

/// Technical data for signal generation.
#[derive(Debug, Clone, Serialize)]
pub struct TechnicalData {
    pub symbol: String,
    pub current_price: f64,
    pub sma_20: f64,
    pub sma_50: f64,
    pub sma_200: Option<f64>,
    pub rsi: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub volume: f64,
    pub avg_volume: f64,
    pub support: f64,
    pub resistance: f64,
    pub atr: f64,
    pub trend: String,
}

/// Signal generator using LLM analysis.
pub struct SignalGenerator {
    client: Arc<dyn LLMClient>,
    config: LLMConfig,
}

impl SignalGenerator {
    /// Create a new signal generator.
    pub fn new(client: Box<dyn LLMClient>) -> Self {
        Self {
            client: Arc::from(client),
            config: LLMConfig {
                temperature: 0.3,
                max_tokens: 800,
                ..Default::default()
            },
        }
    }

    /// Generate trading signal from technical data.
    pub async fn generate_signal(&self, data: &TechnicalData) -> TradingResult<TradingSignal> {
        let prompt = self.build_technical_prompt(data);
        let response = self.client.complete(&prompt, &self.config).await?;
        self.parse_signal_response(&response, &data.symbol)
    }

    /// Generate signal from news headline.
    pub async fn generate_news_signal(
        &self,
        headline: &str,
        symbol: &str,
        current_price: f64,
    ) -> TradingResult<TradingSignal> {
        let prompt = prompts::NEWS_DRIVEN
            .replace("{headline}", headline)
            .replace("{symbol}", symbol)
            .replace("{current_price}", &format!("{:.2}", current_price));

        let response = self.client.complete(&prompt, &self.config).await?;
        self.parse_signal_response(&response, symbol)
    }

    /// Generate signal with multi-timeframe analysis.
    pub async fn generate_multi_timeframe_signal(
        &self,
        data: &HashMap<String, serde_json::Value>,
    ) -> TradingResult<TradingSignal> {
        let symbol = data
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN");

        let prompt = self.build_multi_timeframe_prompt(data);
        let response = self.client.complete(&prompt, &self.config).await?;
        self.parse_signal_response(&response, symbol)
    }

    /// Generate signal with self-consistency validation.
    pub async fn generate_with_consistency(
        &self,
        data: &TechnicalData,
        num_samples: usize,
    ) -> TradingResult<TradingSignal> {
        let prompt = self.build_technical_prompt(data);

        // Generate multiple samples
        let mut signals = Vec::new();
        for _ in 0..num_samples {
            let response = self.client.complete(&prompt, &self.config).await?;
            if let Ok(signal) = self.parse_signal_response(&response, &data.symbol) {
                signals.push(signal);
            }
        }

        if signals.is_empty() {
            return Err(TradingError::LLMError(
                "Failed to generate consistent signals".to_string(),
            ));
        }

        // Count direction occurrences
        let mut buy_count = 0;
        let mut sell_count = 0;
        let mut hold_count = 0;

        for signal in &signals {
            match signal.direction {
                SignalDirection::Buy => buy_count += 1,
                SignalDirection::Sell => sell_count += 1,
                SignalDirection::Hold => hold_count += 1,
            }
        }

        // Find majority direction
        let (majority_direction, majority_count) =
            if buy_count >= sell_count && buy_count >= hold_count {
                (SignalDirection::Buy, buy_count)
            } else if sell_count >= hold_count {
                (SignalDirection::Sell, sell_count)
            } else {
                (SignalDirection::Hold, hold_count)
            };

        // Use first matching signal as base
        let base_signal = signals
            .iter()
            .find(|s| s.direction == majority_direction)
            .unwrap();

        let agreement_ratio = majority_count as f32 / signals.len() as f32;

        Ok(TradingSignal {
            symbol: base_signal.symbol.clone(),
            direction: majority_direction,
            confidence: base_signal.confidence * agreement_ratio,
            entry_price: base_signal.entry_price,
            stop_loss: base_signal.stop_loss,
            take_profit: base_signal.take_profit,
            position_size_pct: base_signal.position_size_pct,
            timeframe: base_signal.timeframe.clone(),
            reasoning: format!(
                "[Agreement: {:.0}%] {}",
                agreement_ratio * 100.0,
                base_signal.reasoning
            ),
            key_factors: base_signal.key_factors.clone(),
            timestamp: Some(Utc::now()),
        })
    }

    /// Build prompt from technical data.
    fn build_technical_prompt(&self, data: &TechnicalData) -> String {
        prompts::BASIC_SIGNAL
            .replace("{symbol}", &data.symbol)
            .replace("{current_price}", &format!("{:.2}", data.current_price))
            .replace("{sma_20}", &format!("{:.2}", data.sma_20))
            .replace("{sma_50}", &format!("{:.2}", data.sma_50))
            .replace("{rsi}", &format!("{:.1}", data.rsi))
            .replace("{macd}", &format!("{:.2}", data.macd))
            .replace("{support}", &format!("{:.2}", data.support))
            .replace("{resistance}", &format!("{:.2}", data.resistance))
            .replace("{trend}", &data.trend)
    }

    /// Build multi-timeframe prompt.
    fn build_multi_timeframe_prompt(&self, data: &HashMap<String, serde_json::Value>) -> String {
        let get_str = |key: &str, default: &str| -> String {
            data.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or(default)
                .to_string()
        };

        let get_f64 = |key: &str, default: f64| -> f64 {
            data.get(key)
                .and_then(|v| v.as_f64())
                .unwrap_or(default)
        };

        prompts::MULTI_TIMEFRAME
            .replace("{symbol}", &get_str("symbol", "UNKNOWN"))
            .replace("{short_tf}", &get_str("short_tf", "1h"))
            .replace("{short_trend}", &get_str("short_trend", "NEUTRAL"))
            .replace("{short_rsi}", &format!("{:.1}", get_f64("short_rsi", 50.0)))
            .replace("{short_momentum}", &get_str("short_momentum", "FLAT"))
            .replace("{medium_tf}", &get_str("medium_tf", "4h"))
            .replace("{medium_trend}", &get_str("medium_trend", "NEUTRAL"))
            .replace("{medium_rsi}", &format!("{:.1}", get_f64("medium_rsi", 50.0)))
            .replace("{medium_momentum}", &get_str("medium_momentum", "FLAT"))
            .replace("{long_tf}", &get_str("long_tf", "1d"))
            .replace("{long_trend}", &get_str("long_trend", "NEUTRAL"))
            .replace("{long_rsi}", &format!("{:.1}", get_f64("long_rsi", 50.0)))
            .replace("{long_momentum}", &get_str("long_momentum", "FLAT"))
            .replace("{current_price}", &format!("{:.2}", get_f64("current_price", 0.0)))
            .replace("{support}", &format!("{:.2}", get_f64("support", 0.0)))
            .replace("{resistance}", &format!("{:.2}", get_f64("resistance", 0.0)))
    }

    /// Parse signal from LLM response.
    fn parse_signal_response(&self, response: &str, symbol: &str) -> TradingResult<TradingSignal> {
        // Find JSON in response
        let start_idx = response.find('{');
        let end_idx = response.rfind('}');

        match (start_idx, end_idx) {
            (Some(start), Some(end)) if start < end => {
                let json_str = &response[start..=end];
                let parsed: serde_json::Value = serde_json::from_str(json_str)?;

                let direction_str = parsed
                    .get("direction")
                    .and_then(|v| v.as_str())
                    .unwrap_or("HOLD");

                let direction = match direction_str.to_uppercase().as_str() {
                    "BUY" => SignalDirection::Buy,
                    "SELL" => SignalDirection::Sell,
                    _ => SignalDirection::Hold,
                };

                let confidence = parsed
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(50.0) as f32;

                let entry_price = parsed
                    .get("entry_price")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                let stop_loss = parsed
                    .get("stop_loss")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                let take_profit = parsed
                    .get("take_profit")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                let position_size_pct = parsed
                    .get("position_size_pct")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;

                let timeframe = parsed
                    .get("timeframe")
                    .and_then(|v| v.as_str())
                    .unwrap_or("1d")
                    .to_string();

                let reasoning = parsed
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let key_factors = parsed
                    .get("key_factors")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                Ok(TradingSignal {
                    symbol: symbol.to_string(),
                    direction,
                    confidence,
                    entry_price,
                    stop_loss,
                    take_profit,
                    position_size_pct,
                    timeframe,
                    reasoning,
                    key_factors,
                    timestamp: Some(Utc::now()),
                })
            }
            _ => Err(TradingError::ParseError(serde_json::Error::io(
                std::io::Error::new(std::io::ErrorKind::InvalidData, "No JSON found in response"),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::MockLLMClient;

    #[tokio::test]
    async fn test_signal_generation() {
        let mut client = MockLLMClient::new();
        client.set_default_response(
            r#"{"direction": "BUY", "confidence": 75, "entry_price": 185.0, "stop_loss": 180.0, "take_profit": 195.0, "position_size_pct": 5, "timeframe": "1d", "reasoning": "Strong momentum", "key_factors": ["bullish trend"]}"#.to_string(),
        );

        let generator = SignalGenerator::new(Box::new(client));

        let data = TechnicalData {
            symbol: "AAPL".to_string(),
            current_price: 185.0,
            sma_20: 183.0,
            sma_50: 180.0,
            sma_200: Some(175.0),
            rsi: 55.0,
            macd: 0.5,
            macd_signal: 0.3,
            volume: 45000000.0,
            avg_volume: 50000000.0,
            support: 180.0,
            resistance: 195.0,
            atr: 3.5,
            trend: "BULLISH".to_string(),
        };

        let signal = generator.generate_signal(&data).await.unwrap();

        assert_eq!(signal.direction, SignalDirection::Buy);
        assert_eq!(signal.confidence, 75.0);
        assert_eq!(signal.entry_price, 185.0);
    }

    #[test]
    fn test_risk_reward_ratio() {
        let signal = TradingSignal {
            symbol: "TEST".to_string(),
            direction: SignalDirection::Buy,
            confidence: 80.0,
            entry_price: 100.0,
            stop_loss: 95.0,
            take_profit: 115.0,
            position_size_pct: 5.0,
            timeframe: "1d".to_string(),
            reasoning: "Test".to_string(),
            key_factors: vec![],
            timestamp: None,
        };

        // Risk: 5, Reward: 15, R/R: 3.0
        assert!((signal.risk_reward_ratio() - 3.0).abs() < 0.001);
    }
}
