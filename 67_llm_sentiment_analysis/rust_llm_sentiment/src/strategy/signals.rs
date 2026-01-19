//! # Trading Signals
//!
//! Convert sentiment scores to actionable trading signals.

use crate::defaults;
use crate::sentiment::{SentimentLevel, SentimentResult};
use serde::{Deserialize, Serialize};

/// Trading signal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingSignal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold / no action
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl TradingSignal {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            TradingSignal::StrongBuy => "STRONG BUY",
            TradingSignal::Buy => "BUY",
            TradingSignal::Hold => "HOLD",
            TradingSignal::Sell => "SELL",
            TradingSignal::StrongSell => "STRONG SELL",
        }
    }

    /// Check if signal is bullish (buy or strong buy)
    pub fn is_bullish(&self) -> bool {
        matches!(self, TradingSignal::Buy | TradingSignal::StrongBuy)
    }

    /// Check if signal is bearish (sell or strong sell)
    pub fn is_bearish(&self) -> bool {
        matches!(self, TradingSignal::Sell | TradingSignal::StrongSell)
    }

    /// Get numeric direction (-1, 0, 1)
    pub fn direction(&self) -> i32 {
        match self {
            TradingSignal::StrongBuy | TradingSignal::Buy => 1,
            TradingSignal::Hold => 0,
            TradingSignal::Sell | TradingSignal::StrongSell => -1,
        }
    }
}

/// Signal strength/conviction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalStrength {
    /// Very weak signal
    VeryWeak,
    /// Weak signal
    Weak,
    /// Moderate signal
    Moderate,
    /// Strong signal
    Strong,
    /// Very strong signal
    VeryStrong,
}

impl SignalStrength {
    /// Convert confidence to signal strength
    pub fn from_confidence(confidence: f64) -> Self {
        if confidence < 0.4 {
            SignalStrength::VeryWeak
        } else if confidence < 0.55 {
            SignalStrength::Weak
        } else if confidence < 0.7 {
            SignalStrength::Moderate
        } else if confidence < 0.85 {
            SignalStrength::Strong
        } else {
            SignalStrength::VeryStrong
        }
    }

    /// Get position size multiplier (0.0 to 1.0)
    pub fn size_multiplier(&self) -> f64 {
        match self {
            SignalStrength::VeryWeak => 0.2,
            SignalStrength::Weak => 0.4,
            SignalStrength::Moderate => 0.6,
            SignalStrength::Strong => 0.8,
            SignalStrength::VeryStrong => 1.0,
        }
    }
}

/// Generated trading signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedSignal {
    /// Trading signal direction
    pub signal: TradingSignal,
    /// Signal strength
    pub strength: SignalStrength,
    /// Underlying sentiment score
    pub sentiment_score: f64,
    /// Confidence level
    pub confidence: f64,
    /// Suggested position size (0.0 to 1.0)
    pub position_size: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Reasoning for the signal
    pub reason: String,
}

/// Trading signal generator
pub struct SignalGenerator {
    /// Bullish threshold
    bullish_threshold: f64,
    /// Bearish threshold
    bearish_threshold: f64,
    /// Strong signal threshold (absolute value)
    strong_threshold: f64,
    /// Minimum confidence for signal
    min_confidence: f64,
    /// Maximum position size
    max_position_size: f64,
    /// Default stop loss
    stop_loss_pct: f64,
    /// Default take profit
    take_profit_pct: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalGenerator {
    /// Create a new signal generator with default settings
    pub fn new() -> Self {
        Self {
            bullish_threshold: defaults::BULLISH_THRESHOLD,
            bearish_threshold: defaults::BEARISH_THRESHOLD,
            strong_threshold: 0.6,
            min_confidence: defaults::MIN_CONFIDENCE,
            max_position_size: defaults::MAX_POSITION_SIZE,
            stop_loss_pct: defaults::STOP_LOSS_PCT,
            take_profit_pct: defaults::TAKE_PROFIT_PCT,
        }
    }

    /// Set thresholds
    pub fn with_thresholds(
        mut self,
        bullish: f64,
        bearish: f64,
        strong: f64,
    ) -> Self {
        self.bullish_threshold = bullish;
        self.bearish_threshold = bearish;
        self.strong_threshold = strong;
        self
    }

    /// Set minimum confidence
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Set risk parameters
    pub fn with_risk_params(
        mut self,
        max_position: f64,
        stop_loss: f64,
        take_profit: f64,
    ) -> Self {
        self.max_position_size = max_position;
        self.stop_loss_pct = stop_loss;
        self.take_profit_pct = take_profit;
        self
    }

    /// Generate trading signal from a single sentiment result
    pub fn generate(&self, result: &SentimentResult) -> GeneratedSignal {
        let score = result.score;
        let confidence = result.confidence;

        // Determine signal direction
        let signal = if confidence < self.min_confidence {
            TradingSignal::Hold
        } else if score > self.strong_threshold {
            TradingSignal::StrongBuy
        } else if score > self.bullish_threshold {
            TradingSignal::Buy
        } else if score < -self.strong_threshold {
            TradingSignal::StrongSell
        } else if score < self.bearish_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        };

        // Calculate position size
        let strength = SignalStrength::from_confidence(confidence);
        let base_size = score.abs() * strength.size_multiplier();
        let position_size = base_size.min(self.max_position_size);

        // Adjust stop loss and take profit based on sentiment strength
        let volatility_factor = 1.0 + (1.0 - confidence) * 0.5;
        let stop_loss = self.stop_loss_pct * volatility_factor;
        let take_profit = self.take_profit_pct * (1.0 + score.abs() * 0.5);

        // Generate reason
        let reason = self.generate_reason(&signal, score, confidence, result);

        GeneratedSignal {
            signal,
            strength,
            sentiment_score: score,
            confidence,
            position_size,
            stop_loss,
            take_profit,
            reason,
        }
    }

    /// Generate signals from multiple sentiment results
    pub fn generate_batch(&self, results: &[SentimentResult]) -> Vec<GeneratedSignal> {
        results.iter().map(|r| self.generate(r)).collect()
    }

    /// Generate aggregated signal from multiple results
    pub fn generate_aggregated(&self, results: &[SentimentResult]) -> GeneratedSignal {
        if results.is_empty() {
            return GeneratedSignal {
                signal: TradingSignal::Hold,
                strength: SignalStrength::VeryWeak,
                sentiment_score: 0.0,
                confidence: 0.0,
                position_size: 0.0,
                stop_loss: self.stop_loss_pct,
                take_profit: self.take_profit_pct,
                reason: "No sentiment data available".to_string(),
            };
        }

        // Calculate weighted average sentiment
        let total_weight: f64 = results.iter().map(|r| r.confidence).sum();

        // Guard against zero total confidence to avoid NaN
        if total_weight == 0.0 {
            return GeneratedSignal {
                signal: TradingSignal::Hold,
                strength: SignalStrength::VeryWeak,
                sentiment_score: 0.0,
                confidence: 0.0,
                position_size: 0.0,
                stop_loss: self.stop_loss_pct,
                take_profit: self.take_profit_pct,
                reason: "Aggregated confidence is zero".to_string(),
            };
        }

        let weighted_score: f64 = results
            .iter()
            .map(|r| r.score * r.confidence)
            .sum::<f64>()
            / total_weight;

        let avg_confidence = total_weight / results.len() as f64;

        // Calculate agreement (lower std = higher agreement)
        let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
        let std_dev = Self::calculate_std(&scores);
        let agreement_bonus = (0.5 - std_dev).max(0.0) * 0.2;
        let adjusted_confidence = (avg_confidence + agreement_bonus).min(0.95);

        // Create synthetic result for signal generation
        let synthetic_result = SentimentResult {
            text: String::new(),
            score: weighted_score,
            level: SentimentLevel::from_score(weighted_score),
            confidence: adjusted_confidence,
            lexicon_score: weighted_score,
            crypto_score: 0.0,
            key_words: vec![],
            crypto_terms: vec![],
            timestamp: chrono::Utc::now(),
        };

        let mut signal = self.generate(&synthetic_result);
        signal.reason = format!(
            "Aggregated from {} sources. Score: {:.2}, Agreement: {:.0}%",
            results.len(),
            weighted_score,
            (1.0 - std_dev) * 100.0
        );

        signal
    }

    /// Generate reason text for signal
    fn generate_reason(
        &self,
        signal: &TradingSignal,
        score: f64,
        confidence: f64,
        result: &SentimentResult,
    ) -> String {
        let direction = if score > 0.0 { "positive" } else if score < 0.0 { "negative" } else { "neutral" };
        let confidence_level = if confidence > 0.8 {
            "high"
        } else if confidence > 0.6 {
            "moderate"
        } else {
            "low"
        };

        let key_factors = if !result.key_words.is_empty() || !result.crypto_terms.is_empty() {
            let words: Vec<&str> = result
                .key_words
                .iter()
                .chain(result.crypto_terms.iter())
                .take(3)
                .map(|s| s.as_str())
                .collect();
            format!(" Key factors: {}", words.join(", "))
        } else {
            String::new()
        };

        format!(
            "{} signal based on {} sentiment (score: {:.2}) with {} confidence.{}",
            signal.as_str(),
            direction,
            score,
            confidence_level,
            key_factors
        )
    }

    /// Calculate standard deviation
    fn calculate_std(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}

/// Position sizing calculator
pub struct PositionSizing {
    /// Maximum position as fraction of portfolio
    max_position: f64,
    /// Kelly fraction (usually 0.25-0.5 for safety)
    kelly_fraction: f64,
}

impl Default for PositionSizing {
    fn default() -> Self {
        Self {
            max_position: 0.5,
            kelly_fraction: 0.5,
        }
    }
}

impl PositionSizing {
    /// Create new position sizing calculator
    pub fn new(max_position: f64, kelly_fraction: f64) -> Self {
        Self {
            max_position,
            kelly_fraction,
        }
    }

    /// Calculate position size based on sentiment and confidence
    pub fn calculate(&self, sentiment_score: f64, confidence: f64) -> f64 {
        // Base size from sentiment magnitude
        let base_size = sentiment_score.abs();

        // Adjust by confidence using Kelly-like formula
        let edge = confidence - 0.5; // Edge over random
        let adjusted_size = if edge > 0.0 {
            base_size * (1.0 + edge * 2.0) * self.kelly_fraction
        } else {
            base_size * 0.5 * self.kelly_fraction
        };

        // Cap at maximum position
        adjusted_size.min(self.max_position)
    }

    /// Calculate position size with volatility adjustment
    pub fn calculate_with_volatility(
        &self,
        sentiment_score: f64,
        confidence: f64,
        volatility: f64,
    ) -> f64 {
        let base = self.calculate(sentiment_score, confidence);

        // Reduce position in high volatility
        let volatility_factor = 1.0 / (1.0 + volatility);

        (base * volatility_factor).min(self.max_position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_result(score: f64, confidence: f64) -> SentimentResult {
        SentimentResult {
            text: String::new(),
            score,
            level: SentimentLevel::from_score(score),
            confidence,
            lexicon_score: score,
            crypto_score: 0.0,
            key_words: vec![],
            crypto_terms: vec![],
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_bullish_signal() {
        let generator = SignalGenerator::new();
        let result = make_result(0.5, 0.8);
        let signal = generator.generate(&result);

        assert!(signal.signal.is_bullish());
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_bearish_signal() {
        let generator = SignalGenerator::new();
        let result = make_result(-0.5, 0.8);
        let signal = generator.generate(&result);

        assert!(signal.signal.is_bearish());
    }

    #[test]
    fn test_hold_signal_low_confidence() {
        let generator = SignalGenerator::new();
        let result = make_result(0.5, 0.3); // High sentiment but low confidence
        let signal = generator.generate(&result);

        assert_eq!(signal.signal, TradingSignal::Hold);
    }

    #[test]
    fn test_hold_signal_neutral() {
        let generator = SignalGenerator::new();
        let result = make_result(0.1, 0.8); // Neutral sentiment
        let signal = generator.generate(&result);

        assert_eq!(signal.signal, TradingSignal::Hold);
    }

    #[test]
    fn test_strong_buy_signal() {
        let generator = SignalGenerator::new();
        let result = make_result(0.8, 0.9);
        let signal = generator.generate(&result);

        assert_eq!(signal.signal, TradingSignal::StrongBuy);
    }

    #[test]
    fn test_aggregated_signal() {
        let generator = SignalGenerator::new();
        let results = vec![
            make_result(0.6, 0.8),
            make_result(0.5, 0.7),
            make_result(0.4, 0.75),
        ];
        let signal = generator.generate_aggregated(&results);

        assert!(signal.signal.is_bullish());
        assert!(signal.reason.contains("Aggregated"));
    }

    #[test]
    fn test_position_sizing() {
        let sizer = PositionSizing::default();

        let strong = sizer.calculate(0.8, 0.9);
        let weak = sizer.calculate(0.3, 0.5);

        assert!(strong > weak);
        assert!(strong <= 0.5); // Max position
    }

    #[test]
    fn test_position_sizing_volatility() {
        let sizer = PositionSizing::default();

        let low_vol = sizer.calculate_with_volatility(0.5, 0.8, 0.1);
        let high_vol = sizer.calculate_with_volatility(0.5, 0.8, 0.5);

        assert!(low_vol > high_vol);
    }
}
