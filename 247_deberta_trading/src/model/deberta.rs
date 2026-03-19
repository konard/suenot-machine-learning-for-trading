//! DeBERTa-inspired sentiment analysis for financial text.
//!
//! Provides a rule-based sentiment classifier using a financial lexicon,
//! with an architecture that mirrors DeBERTa's disentangled approach
//! by separately scoring content and positional features.

use std::collections::HashSet;
use tracing::{debug, info};

/// Sentiment label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentimentLabel {
    Positive,
    Negative,
    Neutral,
}

impl std::fmt::Display for SentimentLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SentimentLabel::Positive => write!(f, "positive"),
            SentimentLabel::Negative => write!(f, "negative"),
            SentimentLabel::Neutral => write!(f, "neutral"),
        }
    }
}

/// Result from sentiment prediction.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub text: String,
    pub label: SentimentLabel,
    pub score: f64,
    pub positive_prob: f64,
    pub negative_prob: f64,
    pub neutral_prob: f64,
}

/// Trading signal generated from sentiment analysis.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub signal: f64,
    pub confidence: f64,
    pub n_positive: usize,
    pub n_negative: usize,
    pub n_neutral: usize,
}

/// DeBERTa-inspired sentiment model for financial text.
///
/// Uses a financial lexicon with content and position-aware scoring
/// to classify sentiment of financial headlines and text.
pub struct SentimentModel {
    positive_words: HashSet<String>,
    negative_words: HashSet<String>,
    /// Bigrams that flip or amplify sentiment.
    positive_bigrams: Vec<(String, String)>,
    negative_bigrams: Vec<(String, String)>,
}

impl SentimentModel {
    /// Create a new sentiment model with default financial lexicon.
    pub fn new() -> Self {
        let positive_words: HashSet<String> = [
            "beat", "beats", "surge", "surges", "surged", "rally", "rallied",
            "gain", "gains", "profit", "profits", "growth", "record", "high",
            "bullish", "upgrade", "upgraded", "outperform", "exceeded", "exceeds",
            "strong", "positive", "rises", "rose", "soar", "soars", "soared",
            "boom", "recovery", "rebounds", "optimistic", "approval", "approved",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let negative_words: HashSet<String> = [
            "miss", "misses", "drop", "drops", "dropped", "fall", "falls",
            "fell", "loss", "losses", "decline", "declined", "low", "bearish",
            "downgrade", "downgraded", "underperform", "crash", "crashed",
            "plunge", "plunged", "weak", "negative", "risk", "recession",
            "selloff", "bankruptcy", "default", "fraud", "investigation",
            "recall", "breach",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let positive_bigrams = vec![
            ("beats".into(), "expectations".into()),
            ("record".into(), "revenue".into()),
            ("strong".into(), "growth".into()),
            ("all-time".into(), "high".into()),
            ("exceeds".into(), "estimates".into()),
        ];

        let negative_bigrams = vec![
            ("misses".into(), "expectations".into()),
            ("profit".into(), "warning".into()),
            ("rate".into(), "hike".into()),
            ("security".into(), "breach".into()),
            ("regulatory".into(), "crackdown".into()),
        ];

        info!("SentimentModel initialized with financial lexicon");

        Self {
            positive_words,
            negative_words,
            positive_bigrams,
            negative_bigrams,
        }
    }

    /// Predict sentiment for a single text.
    pub fn predict(&self, text: &str) -> SentimentResult {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let word_set: HashSet<String> = words.iter().map(|w| w.to_string()).collect();

        // Content scoring
        let pos_count = word_set.iter().filter(|w| self.positive_words.contains(*w)).count();
        let neg_count = word_set.iter().filter(|w| self.negative_words.contains(*w)).count();

        // Position-aware bigram scoring (DeBERTa-inspired)
        let mut pos_bigram_score = 0.0_f64;
        let mut neg_bigram_score = 0.0_f64;

        for i in 0..words.len().saturating_sub(1) {
            let bigram = (words[i].to_string(), words[i + 1].to_string());
            for pb in &self.positive_bigrams {
                if bigram.0 == pb.0 && bigram.1 == pb.1 {
                    pos_bigram_score += 0.3;
                }
            }
            for nb in &self.negative_bigrams {
                if bigram.0 == nb.0 && bigram.1 == nb.1 {
                    neg_bigram_score += 0.3;
                }
            }
        }

        let total = pos_count + neg_count;
        let (label, positive_prob, negative_prob, neutral_prob) = if total == 0
            && pos_bigram_score == 0.0
            && neg_bigram_score == 0.0
        {
            (SentimentLabel::Neutral, 0.15, 0.15, 0.70)
        } else {
            let pos_score = pos_count as f64 + pos_bigram_score;
            let neg_score = neg_count as f64 + neg_bigram_score;
            let total_score = pos_score + neg_score;

            let pos_ratio = pos_score / total_score;
            let neg_ratio = neg_score / total_score;

            if pos_ratio > 0.6 {
                let s = 0.5 + 0.4 * pos_ratio;
                (SentimentLabel::Positive, s, (1.0 - s) * 0.4, (1.0 - s) * 0.6)
            } else if neg_ratio > 0.6 {
                let s = 0.5 + 0.4 * neg_ratio;
                (SentimentLabel::Negative, (1.0 - s) * 0.4, s, (1.0 - s) * 0.6)
            } else {
                (
                    SentimentLabel::Neutral,
                    0.25 + 0.15 * pos_ratio,
                    0.25 + 0.15 * neg_ratio,
                    0.50,
                )
            }
        };

        let score = match label {
            SentimentLabel::Positive => positive_prob,
            SentimentLabel::Negative => negative_prob,
            SentimentLabel::Neutral => neutral_prob,
        };

        debug!("Predicted {:?} ({:.4}) for: {}", label, score, text);

        SentimentResult {
            text: text.to_string(),
            label,
            score,
            positive_prob,
            negative_prob,
            neutral_prob,
        }
    }

    /// Predict sentiment for multiple texts.
    pub fn predict_batch(&self, texts: &[&str]) -> Vec<SentimentResult> {
        texts.iter().map(|t| self.predict(t)).collect()
    }

    /// Generate a trading signal from multiple texts.
    pub fn get_trading_signal(&self, texts: &[&str], threshold: f64) -> TradingSignal {
        let predictions = self.predict_batch(texts);

        let mut n_positive = 0_usize;
        let mut n_negative = 0_usize;
        let mut n_neutral = 0_usize;
        let mut signal_score = 0.0_f64;
        let mut total_weight = 0.0_f64;

        for pred in &predictions {
            match pred.label {
                SentimentLabel::Positive => {
                    n_positive += 1;
                    signal_score += pred.score;
                }
                SentimentLabel::Negative => {
                    n_negative += 1;
                    signal_score -= pred.score;
                }
                SentimentLabel::Neutral => {
                    n_neutral += 1;
                }
            }
            total_weight += pred.score;
        }

        let signal = if total_weight > 0.0 {
            signal_score / total_weight
        } else {
            0.0
        };

        let confidence = signal.abs();
        let final_signal = if confidence < threshold { 0.0 } else { signal };

        TradingSignal {
            signal: final_signal,
            confidence,
            n_positive,
            n_negative,
            n_neutral,
        }
    }
}

impl Default for SentimentModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_sentiment() {
        let model = SentimentModel::new();
        let result = model.predict("Apple revenue beats Q3 expectations");
        assert_eq!(result.label, SentimentLabel::Positive);
        assert!(result.score > 0.5);
    }

    #[test]
    fn test_negative_sentiment() {
        let model = SentimentModel::new();
        let result = model.predict("Stock crashes amid fraud investigation");
        assert_eq!(result.label, SentimentLabel::Negative);
        assert!(result.score > 0.5);
    }

    #[test]
    fn test_neutral_sentiment() {
        let model = SentimentModel::new();
        let result = model.predict("The market opened today as usual");
        assert_eq!(result.label, SentimentLabel::Neutral);
    }

    #[test]
    fn test_trading_signal() {
        let model = SentimentModel::new();
        let texts = vec![
            "Revenue beats expectations",
            "Strong growth in Q4",
            "Minor decline in margins",
        ];
        let signal = model.get_trading_signal(&texts, 0.1);
        assert!(signal.signal > 0.0); // Overall positive
        assert!(signal.n_positive >= 2);
    }
}
