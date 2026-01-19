//! Trading signal generation module
//!
//! This module generates actionable trading signals from sentiment analysis results.

use crate::sentiment::{SentimentAnalyzer, SentimentResult};
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalType {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold/no action signal
    Hold,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Buy => write!(f, "buy"),
            SignalType::Sell => write!(f, "sell"),
            SignalType::Hold => write!(f, "hold"),
        }
    }
}

/// A trading signal with associated metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Symbol/ticker
    pub symbol: String,
    /// Signal type (buy/sell/hold)
    pub signal_type: SignalType,
    /// Signal strength (-1.0 to 1.0)
    pub strength: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Reason for the signal
    pub reason: String,
    /// Timestamp when signal was generated
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(
        symbol: impl Into<String>,
        signal_type: SignalType,
        strength: f64,
        confidence: f64,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            signal_type,
            strength,
            confidence,
            reason: reason.into(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// News item with source information
#[derive(Debug, Clone)]
pub struct NewsItem {
    /// News text content
    pub text: String,
    /// Source of the news
    pub source: String,
}

impl NewsItem {
    /// Create a new news item
    pub fn new(text: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            source: source.into(),
        }
    }
}

/// Trading signal generator using FinGPT sentiment analysis
pub struct TradingSignalGenerator {
    analyzer: SentimentAnalyzer,
    source_weights: HashMap<String, f64>,
    sentiment_threshold: f64,
}

impl TradingSignalGenerator {
    /// Create a new signal generator
    pub fn new(analyzer: SentimentAnalyzer) -> Self {
        let mut source_weights = HashMap::new();
        source_weights.insert("sec_filing".to_string(), 1.0);
        source_weights.insert("reuters".to_string(), 0.9);
        source_weights.insert("bloomberg".to_string(), 0.9);
        source_weights.insert("wsj".to_string(), 0.85);
        source_weights.insert("cnbc".to_string(), 0.75);
        source_weights.insert("twitter".to_string(), 0.5);
        source_weights.insert("reddit".to_string(), 0.4);

        Self {
            analyzer,
            source_weights,
            sentiment_threshold: 0.3,
        }
    }

    /// Set sentiment threshold for generating buy/sell signals
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.sentiment_threshold = threshold;
        self
    }

    /// Generate a trading signal from multiple news items
    pub async fn generate_signal(
        &self,
        symbol: &str,
        news_items: &[NewsItem],
    ) -> Result<TradingSignal> {
        if news_items.is_empty() {
            return Ok(TradingSignal::new(
                symbol,
                SignalType::Hold,
                0.0,
                0.5,
                "No news data available",
            ));
        }

        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        let mut sentiment_reasons = Vec::new();

        for item in news_items {
            let result = self.analyzer.analyze(&item.text).await?;
            let weight = self.source_weights
                .get(&item.source.to_lowercase())
                .copied()
                .unwrap_or(0.5);

            weighted_score += result.score * weight;
            total_weight += weight;

            sentiment_reasons.push(format!(
                "{}: {} ({:.0}%)",
                item.source,
                result.sentiment,
                result.confidence * 100.0
            ));
        }

        let avg_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        };

        let (signal_type, reason) = if avg_score > self.sentiment_threshold {
            (SignalType::Buy, format!(
                "Positive sentiment detected. Sources: {}",
                sentiment_reasons.join(", ")
            ))
        } else if avg_score < -self.sentiment_threshold {
            (SignalType::Sell, format!(
                "Negative sentiment detected. Sources: {}",
                sentiment_reasons.join(", ")
            ))
        } else {
            (SignalType::Hold, format!(
                "Mixed/neutral sentiment. Sources: {}",
                sentiment_reasons.join(", ")
            ))
        };

        let confidence = (avg_score.abs() / self.sentiment_threshold).min(1.0) * 0.5 + 0.5;

        Ok(TradingSignal::new(
            symbol,
            signal_type,
            avg_score,
            confidence,
            reason,
        ))
    }

    /// Analyze sentiment for a single text and return a signal
    pub async fn analyze_single(
        &self,
        symbol: &str,
        text: &str,
        source: &str,
    ) -> Result<(SentimentResult, TradingSignal)> {
        let sentiment = self.analyzer.analyze(text).await?;
        let news_item = NewsItem::new(text, source);
        let signal = self.generate_signal(symbol, &[news_item]).await?;
        Ok((sentiment, signal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_buy_signal() {
        let analyzer = SentimentAnalyzer::new_mock();
        let generator = TradingSignalGenerator::new(analyzer);

        let news = vec![
            NewsItem::new("Company reports record profits and strong growth", "reuters"),
        ];

        let signal = generator.generate_signal("AAPL", &news).await.unwrap();
        assert_eq!(signal.signal_type, SignalType::Buy);
    }

    #[tokio::test]
    async fn test_sell_signal() {
        let analyzer = SentimentAnalyzer::new_mock();
        let generator = TradingSignalGenerator::new(analyzer);

        let news = vec![
            NewsItem::new("Stock crashes amid fraud investigation and layoffs", "bloomberg"),
        ];

        let signal = generator.generate_signal("XYZ", &news).await.unwrap();
        assert_eq!(signal.signal_type, SignalType::Sell);
    }
}
