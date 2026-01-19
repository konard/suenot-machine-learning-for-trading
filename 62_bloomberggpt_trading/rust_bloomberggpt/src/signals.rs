//! Trading Signal Generation Module
//!
//! Converts sentiment analysis results into actionable trading signals
//! with position sizing and confidence weighting.

use crate::error::{Error, Result};
use crate::sentiment::{Sentiment, SentimentResult};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of trading signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalType {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold current position
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl SignalType {
    /// Convert signal to position multiplier (-1.0 to 1.0)
    pub fn to_position(&self) -> f64 {
        match self {
            SignalType::StrongBuy => 1.0,
            SignalType::Buy => 0.5,
            SignalType::Hold => 0.0,
            SignalType::Sell => -0.5,
            SignalType::StrongSell => -1.0,
        }
    }

    /// Create signal type from score
    pub fn from_score(score: f64) -> Self {
        if score >= 0.6 {
            SignalType::StrongBuy
        } else if score >= 0.2 {
            SignalType::Buy
        } else if score <= -0.6 {
            SignalType::StrongSell
        } else if score <= -0.2 {
            SignalType::Sell
        } else {
            SignalType::Hold
        }
    }
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::StrongBuy => write!(f, "STRONG BUY"),
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Hold => write!(f, "HOLD"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::StrongSell => write!(f, "STRONG SELL"),
        }
    }
}

/// Source of the trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalSource {
    /// News article
    News,
    /// Earnings report
    Earnings,
    /// Social media
    Social,
    /// SEC filing
    Filing,
    /// Analyst report
    Analyst,
    /// Technical analysis
    Technical,
    /// Aggregated from multiple sources
    Aggregated,
}

impl SignalSource {
    /// Get reliability weight for source
    pub fn weight(&self) -> f64 {
        match self {
            SignalSource::Earnings => 1.0,
            SignalSource::Filing => 0.9,
            SignalSource::Analyst => 0.8,
            SignalSource::News => 0.7,
            SignalSource::Technical => 0.6,
            SignalSource::Social => 0.3,
            SignalSource::Aggregated => 1.0,
        }
    }
}

/// A trading signal with all relevant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Symbol this signal applies to
    pub symbol: String,
    /// Signal type (buy/sell/hold)
    pub signal_type: SignalType,
    /// Signal strength (-1.0 to 1.0)
    pub strength: f64,
    /// Confidence in the signal (0.0 to 1.0)
    pub confidence: f64,
    /// Source of the signal
    pub source: SignalSource,
    /// When the signal was generated
    pub timestamp: DateTime<Utc>,
    /// When the signal expires
    pub expiry: DateTime<Utc>,
    /// Reasoning for the signal
    pub reason: String,
    /// Original sentiment result (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentiment_result: Option<SentimentResult>,
}

impl TradingSignal {
    /// Check if signal is still valid
    pub fn is_valid(&self) -> bool {
        Utc::now() < self.expiry
    }

    /// Calculate position size based on signal
    pub fn position_size(&self, max_position: f64) -> f64 {
        self.signal_type.to_position() * self.strength * self.confidence * max_position
    }

    /// Apply time decay to signal strength
    pub fn decayed_strength(&self, decay_hours: f64) -> f64 {
        let hours_elapsed = (Utc::now() - self.timestamp).num_hours() as f64;
        let decay_factor = (-hours_elapsed / decay_hours).exp();
        self.strength * decay_factor
    }
}

/// Configuration for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Signal validity duration in hours
    pub signal_duration_hours: i64,
    /// Decay rate for signal strength
    pub decay_hours: f64,
    /// Source weights override
    pub source_weights: HashMap<String, f64>,
    /// Minimum sentiment score to generate signal
    pub min_sentiment_score: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            signal_duration_hours: 24,
            decay_hours: 12.0,
            source_weights: HashMap::new(),
            min_sentiment_score: 0.2,
        }
    }
}

/// Generator for trading signals from sentiment analysis
pub struct TradingSignalGenerator {
    config: SignalConfig,
}

impl TradingSignalGenerator {
    /// Create new signal generator with default config
    pub fn new() -> Self {
        Self {
            config: SignalConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate trading signal from sentiment result
    pub fn generate_signal(
        &self,
        sentiment: &SentimentResult,
        symbol: &str,
        source: SignalSource,
    ) -> Result<Option<TradingSignal>> {
        // Check minimum thresholds
        if sentiment.confidence < self.config.min_confidence {
            return Ok(None);
        }

        if sentiment.score.abs() < self.config.min_sentiment_score {
            return Ok(None);
        }

        let now = Utc::now();
        let expiry = now + Duration::hours(self.config.signal_duration_hours);

        // Apply source weight
        let source_weight = self.config
            .source_weights
            .get(&format!("{:?}", source).to_lowercase())
            .copied()
            .unwrap_or_else(|| source.weight());

        let weighted_score = sentiment.score * source_weight;
        let signal_type = SignalType::from_score(weighted_score);

        // Skip hold signals
        if signal_type == SignalType::Hold {
            return Ok(None);
        }

        let signal = TradingSignal {
            symbol: symbol.to_string(),
            signal_type,
            strength: weighted_score.abs(),
            confidence: sentiment.confidence * source_weight,
            source,
            timestamp: now,
            expiry,
            reason: format!(
                "Sentiment analysis: {} (score: {:.2}, confidence: {:.2})",
                sentiment.sentiment, sentiment.score, sentiment.confidence
            ),
            sentiment_result: Some(sentiment.clone()),
        };

        Ok(Some(signal))
    }

    /// Aggregate multiple signals for the same symbol
    pub fn aggregate_signals(&self, signals: &[TradingSignal]) -> Option<TradingSignal> {
        if signals.is_empty() {
            return None;
        }

        // Group by symbol
        let mut by_symbol: HashMap<&str, Vec<&TradingSignal>> = HashMap::new();
        for signal in signals {
            by_symbol
                .entry(&signal.symbol)
                .or_default()
                .push(signal);
        }

        // Aggregate first symbol (for simplicity)
        let (symbol, symbol_signals) = by_symbol.iter().next()?;

        // Calculate weighted average
        let mut total_weight = 0.0;
        let mut weighted_strength = 0.0;
        let mut weighted_confidence = 0.0;

        for signal in symbol_signals {
            let weight = signal.source.weight() * signal.confidence;
            total_weight += weight;
            weighted_strength += signal.strength * signal.signal_type.to_position() * weight;
            weighted_confidence += signal.confidence * weight;
        }

        if total_weight == 0.0 {
            return None;
        }

        let avg_strength = weighted_strength / total_weight;
        let avg_confidence = weighted_confidence / total_weight;
        let signal_type = SignalType::from_score(avg_strength);

        let now = Utc::now();

        Some(TradingSignal {
            symbol: symbol.to_string(),
            signal_type,
            strength: avg_strength.abs(),
            confidence: avg_confidence,
            source: SignalSource::Aggregated,
            timestamp: now,
            expiry: now + Duration::hours(self.config.signal_duration_hours),
            reason: format!(
                "Aggregated from {} signals",
                symbol_signals.len()
            ),
            sentiment_result: None,
        })
    }

    /// Get all valid (non-expired) signals
    pub fn filter_valid(&self, signals: &[TradingSignal]) -> Vec<TradingSignal> {
        signals
            .iter()
            .filter(|s| s.is_valid())
            .cloned()
            .collect()
    }

    /// Get signals sorted by strength
    pub fn rank_signals(&self, signals: &[TradingSignal]) -> Vec<TradingSignal> {
        let mut sorted: Vec<_> = signals.iter().cloned().collect();
        sorted.sort_by(|a, b| {
            let score_a = a.strength * a.confidence;
            let score_b = b.strength * b.confidence;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }
}

impl Default for TradingSignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sentiment::Sentiment;
    use std::collections::HashMap;

    fn create_test_sentiment(score: f64, confidence: f64) -> SentimentResult {
        SentimentResult {
            text: "Test text".to_string(),
            sentiment: Sentiment::from_score(score),
            score,
            confidence,
            entity_sentiments: HashMap::new(),
            source: "test".to_string(),
        }
    }

    #[test]
    fn test_signal_type_from_score() {
        assert_eq!(SignalType::from_score(0.8), SignalType::StrongBuy);
        assert_eq!(SignalType::from_score(0.4), SignalType::Buy);
        assert_eq!(SignalType::from_score(0.0), SignalType::Hold);
        assert_eq!(SignalType::from_score(-0.4), SignalType::Sell);
        assert_eq!(SignalType::from_score(-0.8), SignalType::StrongSell);
    }

    #[test]
    fn test_generate_signal_positive() {
        let generator = TradingSignalGenerator::new();
        let sentiment = create_test_sentiment(0.7, 0.8);

        let signal = generator
            .generate_signal(&sentiment, "AAPL", SignalSource::News)
            .unwrap();

        assert!(signal.is_some());
        let signal = signal.unwrap();
        assert_eq!(signal.symbol, "AAPL");
        assert!(matches!(
            signal.signal_type,
            SignalType::Buy | SignalType::StrongBuy
        ));
    }

    #[test]
    fn test_generate_signal_low_confidence() {
        let generator = TradingSignalGenerator::new();
        let sentiment = create_test_sentiment(0.7, 0.3);

        let signal = generator
            .generate_signal(&sentiment, "AAPL", SignalSource::News)
            .unwrap();

        assert!(signal.is_none());
    }

    #[test]
    fn test_signal_decay() {
        let signal = TradingSignal {
            symbol: "AAPL".to_string(),
            signal_type: SignalType::Buy,
            strength: 1.0,
            confidence: 0.9,
            source: SignalSource::News,
            timestamp: Utc::now() - Duration::hours(6),
            expiry: Utc::now() + Duration::hours(18),
            reason: "Test".to_string(),
            sentiment_result: None,
        };

        let decayed = signal.decayed_strength(12.0);
        assert!(decayed < 1.0);
        assert!(decayed > 0.0);
    }
}
