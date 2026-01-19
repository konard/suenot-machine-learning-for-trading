//! Trading signal generation using RAG.

use crate::retriever::{DocumentRetriever, SearchResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Trading signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalDirection {
    /// Buy signal
    Long,
    /// Sell signal
    Short,
    /// No action
    Neutral,
}

impl std::fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalDirection::Long => write!(f, "LONG"),
            SignalDirection::Short => write!(f, "SHORT"),
            SignalDirection::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// A generated trading signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Stock ticker symbol
    pub ticker: String,
    /// Signal direction
    pub direction: SignalDirection,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Explanation for the signal
    pub reasoning: String,
    /// Sources used for analysis
    pub sources: Vec<String>,
    /// When the signal was generated
    pub timestamp: DateTime<Utc>,
    /// Number of documents analyzed
    pub documents_analyzed: usize,
    /// Positive signal count
    pub positive_signals: usize,
    /// Negative signal count
    pub negative_signals: usize,
}

impl TradingSignal {
    /// Create a new trading signal.
    pub fn new(
        ticker: String,
        direction: SignalDirection,
        confidence: f64,
        reasoning: String,
        sources: Vec<String>,
    ) -> Self {
        Self {
            ticker,
            direction,
            confidence,
            reasoning,
            sources,
            timestamp: Utc::now(),
            documents_analyzed: 0,
            positive_signals: 0,
            negative_signals: 0,
        }
    }

    /// Create a neutral signal with no data.
    pub fn no_data(ticker: String) -> Self {
        Self::new(
            ticker,
            SignalDirection::Neutral,
            0.0,
            "No relevant documents found for analysis.".to_string(),
            Vec::new(),
        )
    }
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TradingSignal({}: {}, confidence={:.0}%)",
            self.ticker,
            self.direction,
            self.confidence * 100.0
        )
    }
}

/// Sentiment analyzer for financial text.
pub struct SentimentAnalyzer {
    positive_keywords: HashSet<String>,
    negative_keywords: HashSet<String>,
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with default keywords.
    pub fn new() -> Self {
        let positive = [
            "beat", "exceeded", "growth", "upgrade", "bullish", "strong",
            "positive", "increase", "surge", "rally", "outperform", "buy",
            "profit", "gains", "record", "breakthrough", "optimistic",
            "accelerate", "expand", "success", "momentum", "upside",
        ];

        let negative = [
            "miss", "below", "decline", "downgrade", "bearish", "weak",
            "negative", "decrease", "drop", "fall", "underperform", "sell",
            "loss", "losses", "concern", "risk", "warning", "slowdown",
            "cut", "reduce", "disappoint", "downside", "pressure",
        ];

        Self {
            positive_keywords: positive.iter().map(|s| s.to_string()).collect(),
            negative_keywords: negative.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Analyze sentiment of text.
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let text_lower = text.to_lowercase();

        let positive_count = self
            .positive_keywords
            .iter()
            .filter(|w| text_lower.contains(w.as_str()))
            .count();

        let negative_count = self
            .negative_keywords
            .iter()
            .filter(|w| text_lower.contains(w.as_str()))
            .count();

        let total = positive_count + negative_count;

        let (sentiment, score) = if total == 0 {
            (Sentiment::Neutral, 0.5)
        } else if positive_count > negative_count {
            let s = 0.5 + (positive_count - negative_count) as f64 / (total * 2) as f64;
            (Sentiment::Positive, s.min(1.0))
        } else {
            let s = 0.5 - (negative_count - positive_count) as f64 / (total * 2) as f64;
            (Sentiment::Negative, s.max(0.0))
        };

        SentimentResult {
            sentiment,
            score,
            positive_count,
            negative_count,
        }
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Sentiment classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

/// Result of sentiment analysis.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub sentiment: Sentiment,
    pub score: f64,
    pub positive_count: usize,
    pub negative_count: usize,
}

/// RAG-based trading signal generator.
pub struct RAGSignalGenerator {
    retriever: DocumentRetriever,
    sentiment_analyzer: SentimentAnalyzer,
}

impl RAGSignalGenerator {
    /// Create a new signal generator.
    pub fn new(retriever: DocumentRetriever) -> Self {
        Self {
            retriever,
            sentiment_analyzer: SentimentAnalyzer::new(),
        }
    }

    /// Generate a trading signal for a ticker.
    pub fn generate_signal(&self, ticker: &str) -> TradingSignal {
        self.generate_signal_with_query(ticker, None, 5)
    }

    /// Generate a trading signal with custom query.
    pub fn generate_signal_with_query(
        &self,
        ticker: &str,
        query: Option<&str>,
        top_k: usize,
    ) -> TradingSignal {
        let query = query.unwrap_or_else(|| {
            // Default query
            "market sentiment outlook analysis news"
        });

        // Retrieve relevant documents
        let results = self.retriever.search(query, top_k, Some(ticker));

        if results.is_empty() {
            return TradingSignal::no_data(ticker.to_string());
        }

        // Analyze sentiment
        self.analyze_results(ticker, &results)
    }

    /// Analyze search results to generate signal.
    fn analyze_results(&self, ticker: &str, results: &[SearchResult]) -> TradingSignal {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        let mut positive_signals = 0;
        let mut negative_signals = 0;
        let mut sources: HashSet<String> = HashSet::new();

        for result in results {
            let sentiment = self.sentiment_analyzer.analyze(&result.document.text);

            // Weight by retrieval relevance
            let weight = result.score;
            total_score += sentiment.score * weight;
            total_weight += weight;

            match sentiment.sentiment {
                Sentiment::Positive => positive_signals += 1,
                Sentiment::Negative => negative_signals += 1,
                Sentiment::Neutral => {}
            }

            sources.insert(result.document.source.clone());
        }

        // Calculate weighted average
        let avg_sentiment = if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.5
        };

        // Determine direction
        let (direction, confidence) = if avg_sentiment > 0.6 {
            (SignalDirection::Long, avg_sentiment.min(0.9))
        } else if avg_sentiment < 0.4 {
            (SignalDirection::Short, (1.0 - avg_sentiment).min(0.9))
        } else {
            (SignalDirection::Neutral, 0.3 + (avg_sentiment - 0.5).abs())
        };

        let reasoning = format!(
            "Analysis of {} documents: {} positive, {} negative signals. \
             Average sentiment score: {:.2}.",
            results.len(),
            positive_signals,
            negative_signals,
            avg_sentiment
        );

        let mut signal = TradingSignal::new(
            ticker.to_string(),
            direction,
            confidence,
            reasoning,
            sources.into_iter().collect(),
        );

        signal.documents_analyzed = results.len();
        signal.positive_signals = positive_signals;
        signal.negative_signals = negative_signals;

        signal
    }

    /// Generate signals for multiple tickers.
    pub fn batch_generate(&self, tickers: &[&str]) -> Vec<TradingSignal> {
        tickers.iter().map(|t| self.generate_signal(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;

    fn create_test_retriever() -> DocumentRetriever {
        let mut retriever = DocumentRetriever::new();
        retriever.add_documents(vec![
            Document::new(
                "Tesla news: Company reported record deliveries beating expectations. \
                 Strong growth in China. Market sentiment is bullish. Analysts upgrade to buy.",
                Some("TSLA".to_string()),
                "Reuters",
                Utc::now(),
            ),
            Document::new(
                "Tesla market analysis: Company faces competition concerns. Margins under pressure. \
                 Negative sentiment from some analysts who downgrade amid slowdown fears.",
                Some("TSLA".to_string()),
                "Bloomberg",
                Utc::now(),
            ),
        ]);
        retriever
    }

    #[test]
    fn test_sentiment_analyzer() {
        let analyzer = SentimentAnalyzer::new();

        let result = analyzer.analyze("Strong growth and record profits");
        assert_eq!(result.sentiment, Sentiment::Positive);
        assert!(result.positive_count > 0);

        let result = analyzer.analyze("Decline in sales amid weak demand");
        assert_eq!(result.sentiment, Sentiment::Negative);
        assert!(result.negative_count > 0);
    }

    #[test]
    fn test_signal_generation() {
        let retriever = create_test_retriever();
        let generator = RAGSignalGenerator::new(retriever);

        let signal = generator.generate_signal("TSLA");

        assert_eq!(signal.ticker, "TSLA");
        assert!(signal.documents_analyzed > 0);
        assert!(!signal.sources.is_empty());
    }

    #[test]
    fn test_no_data_signal() {
        let retriever = DocumentRetriever::new();
        let generator = RAGSignalGenerator::new(retriever);

        let signal = generator.generate_signal("UNKNOWN");

        assert_eq!(signal.direction, SignalDirection::Neutral);
        assert_eq!(signal.confidence, 0.0);
    }
}
