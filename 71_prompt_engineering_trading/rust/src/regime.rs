//! Market regime detection using LLMs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{TradingError, TradingResult};
use crate::llm_client::{LLMClient, LLMConfig};
use crate::prompts::regime as prompts;

/// Market regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MarketRegime {
    /// Strong uptrend with low volatility
    RiskOnTrending,
    /// Uptrend with high volatility
    RiskOnVolatile,
    /// Strong downtrend
    RiskOffTrending,
    /// Crash mode, extreme fear
    RiskOffPanic,
    /// Sideways consolidation
    Ranging,
    /// Regime change in progress
    Transitional,
}

impl MarketRegime {
    /// Get recommended strategies for this regime.
    pub fn recommended_strategies(&self) -> Vec<&'static str> {
        match self {
            Self::RiskOnTrending => vec!["trend_following", "momentum", "buy_dips"],
            Self::RiskOnVolatile => vec!["mean_reversion", "volatility_selling", "range_trading"],
            Self::RiskOffTrending => vec!["short_selling", "put_buying", "defensive_rotation"],
            Self::RiskOffPanic => vec!["cash", "tail_hedges", "volatility_buying"],
            Self::Ranging => vec!["mean_reversion", "premium_selling", "pairs_trading"],
            Self::Transitional => vec!["reduce_exposure", "straddles", "diversification"],
        }
    }

    /// Get strategies to avoid in this regime.
    pub fn avoid_strategies(&self) -> Vec<&'static str> {
        match self {
            Self::RiskOnTrending => vec!["short_selling", "volatility_buying"],
            Self::RiskOnVolatile => vec!["trend_following", "leveraged_positions"],
            Self::RiskOffTrending => vec!["buy_dips", "momentum_long"],
            Self::RiskOffPanic => vec!["any_long_exposure", "leverage"],
            Self::Ranging => vec!["trend_following", "breakout_trading"],
            Self::Transitional => vec!["concentrated_positions", "directional_bets"],
        }
    }

    /// Get position sizing multiplier for this regime.
    pub fn position_multiplier(&self) -> f64 {
        match self {
            Self::RiskOnTrending => 1.2,
            Self::RiskOnVolatile => 0.8,
            Self::RiskOffTrending => 0.6,
            Self::RiskOffPanic => 0.3,
            Self::Ranging => 0.7,
            Self::Transitional => 0.5,
        }
    }
}

/// Result of regime analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAnalysis {
    pub regime: MarketRegime,
    pub confidence: f32,
    pub key_drivers: Vec<String>,
    pub recommended_strategies: Vec<String>,
    pub avoid_strategies: Vec<String>,
    pub position_sizing: String,
    pub next_regime_probabilities: HashMap<String, f64>,
    pub reasoning: String,
}

/// Market data for regime detection.
#[derive(Debug, Clone, Serialize)]
pub struct MarketData {
    // Equities
    pub sp500_price: f64,
    pub sp500_change: f64,
    pub nasdaq_price: f64,
    pub nasdaq_change: f64,
    pub russell_price: f64,
    pub russell_change: f64,

    // Volatility
    pub vix: f64,
    pub vix_avg: f64,
    pub vix_term_structure: String,

    // Rates
    pub treasury_10y: f64,
    pub treasury_2y: f64,
    pub credit_spread: f64,

    // Macro
    pub dxy: f64,
    pub gold: f64,
    pub oil: f64,

    // Breadth
    pub adv_dec_ratio: f64,
    pub pct_above_200: f64,
}

/// Market regime detector using LLM analysis.
pub struct MarketRegimeDetector {
    client: Arc<dyn LLMClient>,
    config: LLMConfig,
}

impl MarketRegimeDetector {
    /// Create a new regime detector.
    pub fn new(client: Box<dyn LLMClient>) -> Self {
        Self {
            client: Arc::from(client),
            config: LLMConfig {
                temperature: 0.3,
                max_tokens: 1000,
                ..Default::default()
            },
        }
    }

    /// Detect current market regime.
    pub async fn detect_regime(&self, data: &MarketData) -> TradingResult<RegimeAnalysis> {
        let prompt = self.build_prompt(data);
        let response = self.client.complete(&prompt, &self.config).await?;
        self.parse_response(&response)
    }

    /// Detect regime with confidence validation.
    pub async fn detect_with_confidence(
        &self,
        data: &MarketData,
        num_samples: usize,
    ) -> TradingResult<RegimeAnalysis> {
        let prompt = self.build_prompt(data);

        // Generate multiple samples
        let mut results = Vec::new();
        for _ in 0..num_samples {
            let response = self.client.complete(&prompt, &self.config).await?;
            if let Ok(result) = self.parse_response(&response) {
                results.push(result);
            }
        }

        if results.is_empty() {
            return Ok(RegimeAnalysis {
                regime: MarketRegime::Transitional,
                confidence: 0.0,
                key_drivers: vec!["Unable to determine".to_string()],
                recommended_strategies: vec!["reduce_exposure".to_string()],
                avoid_strategies: vec!["concentrated_positions".to_string()],
                position_sizing: "REDUCE".to_string(),
                next_regime_probabilities: HashMap::new(),
                reasoning: "Failed to generate consistent regime analysis".to_string(),
            });
        }

        // Count regime occurrences
        let mut regime_counts: HashMap<MarketRegime, usize> = HashMap::new();
        for result in &results {
            *regime_counts.entry(result.regime).or_insert(0) += 1;
        }

        // Find majority regime
        let (majority_regime, majority_count) = regime_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(r, c)| (*r, *c))
            .unwrap();

        // Use first matching result as base
        let base_result = results
            .iter()
            .find(|r| r.regime == majority_regime)
            .unwrap();

        let agreement_ratio = majority_count as f32 / results.len() as f32;

        Ok(RegimeAnalysis {
            regime: majority_regime,
            confidence: base_result.confidence * agreement_ratio,
            key_drivers: base_result.key_drivers.clone(),
            recommended_strategies: base_result.recommended_strategies.clone(),
            avoid_strategies: base_result.avoid_strategies.clone(),
            position_sizing: base_result.position_sizing.clone(),
            next_regime_probabilities: base_result.next_regime_probabilities.clone(),
            reasoning: format!(
                "[Agreement: {:.0}%] {}",
                agreement_ratio * 100.0,
                base_result.reasoning
            ),
        })
    }

    /// Build prompt from market data.
    fn build_prompt(&self, data: &MarketData) -> String {
        prompts::COMPREHENSIVE
            .replace("{sp500_price}", &format!("{:.2}", data.sp500_price))
            .replace("{sp500_change}", &format!("{:.1}", data.sp500_change))
            .replace("{nasdaq_price}", &format!("{:.2}", data.nasdaq_price))
            .replace("{nasdaq_change}", &format!("{:.1}", data.nasdaq_change))
            .replace("{russell_price}", &format!("{:.2}", data.russell_price))
            .replace("{russell_change}", &format!("{:.1}", data.russell_change))
            .replace("{vix}", &format!("{:.1}", data.vix))
            .replace("{vix_avg}", &format!("{:.1}", data.vix_avg))
            .replace("{vix_term_structure}", &data.vix_term_structure)
            .replace("{treasury_10y}", &format!("{:.2}", data.treasury_10y))
            .replace("{treasury_2y}", &format!("{:.2}", data.treasury_2y))
            .replace("{credit_spread}", &format!("{:.0}", data.credit_spread))
            .replace("{dxy}", &format!("{:.1}", data.dxy))
            .replace("{gold}", &format!("{:.0}", data.gold))
            .replace("{oil}", &format!("{:.2}", data.oil))
            .replace("{adv_dec_ratio}", &format!("{:.2}", data.adv_dec_ratio))
            .replace("{pct_above_200}", &format!("{:.0}", data.pct_above_200))
    }

    /// Parse LLM response into RegimeAnalysis.
    fn parse_response(&self, response: &str) -> TradingResult<RegimeAnalysis> {
        // Find JSON in response
        let start_idx = response.find('{');
        let end_idx = response.rfind('}');

        match (start_idx, end_idx) {
            (Some(start), Some(end)) if start < end => {
                let json_str = &response[start..=end];
                let parsed: serde_json::Value = serde_json::from_str(json_str)?;

                let regime_str = parsed
                    .get("regime")
                    .and_then(|v| v.as_str())
                    .unwrap_or("TRANSITIONAL");

                let regime = match regime_str.to_uppercase().as_str() {
                    "RISK_ON_TRENDING" => MarketRegime::RiskOnTrending,
                    "RISK_ON_VOLATILE" => MarketRegime::RiskOnVolatile,
                    "RISK_OFF_TRENDING" => MarketRegime::RiskOffTrending,
                    "RISK_OFF_PANIC" => MarketRegime::RiskOffPanic,
                    "RANGING" => MarketRegime::Ranging,
                    _ => MarketRegime::Transitional,
                };

                let confidence = parsed
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(50.0) as f32;

                let key_drivers = parsed
                    .get("key_drivers")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let recommended_strategies = parsed
                    .get("recommended_strategies")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        regime
                            .recommended_strategies()
                            .iter()
                            .map(|s| s.to_string())
                            .collect()
                    });

                let avoid_strategies = parsed
                    .get("avoid_strategies")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        regime
                            .avoid_strategies()
                            .iter()
                            .map(|s| s.to_string())
                            .collect()
                    });

                let position_sizing = parsed
                    .get("position_sizing")
                    .and_then(|v| v.as_str())
                    .unwrap_or("MAINTAIN")
                    .to_string();

                let next_regime_probabilities = parsed
                    .get("next_regime_probabilities")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                            .collect()
                    })
                    .unwrap_or_default();

                let reasoning = parsed
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                Ok(RegimeAnalysis {
                    regime,
                    confidence,
                    key_drivers,
                    recommended_strategies,
                    avoid_strategies,
                    position_sizing,
                    next_regime_probabilities,
                    reasoning,
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
    async fn test_regime_detection() {
        let mut client = MockLLMClient::new();
        client.set_default_response(
            r#"{"regime": "RISK_ON_TRENDING", "confidence": 75, "key_drivers": ["strong momentum"], "recommended_strategies": ["trend_following"], "avoid_strategies": ["short_selling"], "position_sizing": "MAINTAIN", "next_regime_probabilities": {}, "reasoning": "Bullish market"}"#.to_string(),
        );

        let detector = MarketRegimeDetector::new(Box::new(client));

        let data = MarketData {
            sp500_price: 4800.0,
            sp500_change: 1.5,
            nasdaq_price: 15000.0,
            nasdaq_change: 2.0,
            russell_price: 2000.0,
            russell_change: 1.0,
            vix: 14.0,
            vix_avg: 16.0,
            vix_term_structure: "Contango".to_string(),
            treasury_10y: 4.2,
            treasury_2y: 4.6,
            credit_spread: 100.0,
            dxy: 104.0,
            gold: 2050.0,
            oil: 78.0,
            adv_dec_ratio: 1.8,
            pct_above_200: 65.0,
        };

        let result = detector.detect_regime(&data).await.unwrap();

        assert_eq!(result.regime, MarketRegime::RiskOnTrending);
        assert_eq!(result.confidence, 75.0);
    }

    #[test]
    fn test_regime_strategies() {
        let regime = MarketRegime::RiskOnTrending;
        assert!(regime.recommended_strategies().contains(&"trend_following"));
        assert!(regime.avoid_strategies().contains(&"short_selling"));
        assert!((regime.position_multiplier() - 1.2).abs() < 0.001);
    }
}
