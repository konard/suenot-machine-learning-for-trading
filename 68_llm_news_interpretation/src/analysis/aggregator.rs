//! Signal aggregation from multiple news sources

use super::{EventType, SentimentResult};
use crate::news::NewsSource;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Aggregated trading signal from multiple news items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedSignal {
    /// Target asset symbol
    pub symbol: String,
    /// Overall sentiment score (-1.0 to 1.0)
    pub sentiment: f64,
    /// Confidence in the signal (0.0 to 1.0)
    pub confidence: f64,
    /// Number of news items contributing to signal
    pub news_count: usize,
    /// Dominant event type
    pub dominant_event: EventType,
    /// Signal strength (absolute value of weighted sentiment)
    pub strength: f64,
    /// Suggested action
    pub action: SignalAction,
    /// Timestamp of aggregation
    pub timestamp: DateTime<Utc>,
    /// Individual signals that contributed
    pub components: Vec<SignalComponent>,
}

/// Individual signal component from a single news item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalComponent {
    /// News item ID
    pub news_id: String,
    /// Source of news
    pub source: NewsSource,
    /// Sentiment from this item
    pub sentiment: f64,
    /// Weight applied to this component
    pub weight: f64,
    /// Event type
    pub event_type: EventType,
}

/// Trading action suggested by signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalAction {
    /// Strong buy recommendation
    StrongBuy,
    /// Buy recommendation
    Buy,
    /// Hold/neutral
    Hold,
    /// Sell recommendation
    Sell,
    /// Strong sell recommendation
    StrongSell,
}

impl SignalAction {
    /// Convert weighted sentiment to action
    pub fn from_sentiment(weighted_sentiment: f64) -> Self {
        if weighted_sentiment > 0.5 {
            SignalAction::StrongBuy
        } else if weighted_sentiment > 0.15 {
            SignalAction::Buy
        } else if weighted_sentiment > -0.15 {
            SignalAction::Hold
        } else if weighted_sentiment > -0.5 {
            SignalAction::Sell
        } else {
            SignalAction::StrongSell
        }
    }

    /// Get position size multiplier
    pub fn position_multiplier(&self) -> f64 {
        match self {
            SignalAction::StrongBuy => 1.0,
            SignalAction::Buy => 0.5,
            SignalAction::Hold => 0.0,
            SignalAction::Sell => -0.5,
            SignalAction::StrongSell => -1.0,
        }
    }
}

/// Signal aggregator for combining multiple news signals
#[derive(Debug)]
pub struct SignalAggregator {
    /// Minimum confidence threshold for signals
    min_confidence: f64,
    /// Maximum age of news to consider (in seconds)
    max_news_age_secs: i64,
    /// Source reliability weights
    source_weights: HashMap<NewsSource, f64>,
    /// Event type impact weights
    event_weights: HashMap<EventType, f64>,
    /// Decay factor for older news
    time_decay_factor: f64,
}

impl Default for SignalAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalAggregator {
    /// Create a new signal aggregator
    pub fn new() -> Self {
        Self {
            min_confidence: 0.3,
            max_news_age_secs: 3600, // 1 hour
            source_weights: Self::default_source_weights(),
            event_weights: Self::default_event_weights(),
            time_decay_factor: 0.95,
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set maximum news age
    pub fn with_max_age_secs(mut self, seconds: i64) -> Self {
        self.max_news_age_secs = seconds;
        self
    }

    /// Aggregate signals for a symbol
    pub fn aggregate(&self, symbol: &str, signals: Vec<NewsSignal>) -> Option<AggregatedSignal> {
        let now = Utc::now();

        // Filter and weight signals
        let valid_signals: Vec<_> = signals
            .into_iter()
            .filter(|s| {
                let age = (now - s.timestamp).num_seconds();
                age <= self.max_news_age_secs && s.sentiment.confidence >= self.min_confidence
            })
            .collect();

        if valid_signals.is_empty() {
            return None;
        }

        // Calculate weighted sentiment
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        let mut event_counts: HashMap<EventType, usize> = HashMap::new();
        let mut components = Vec::new();

        for signal in &valid_signals {
            let age_secs = (now - signal.timestamp).num_seconds() as f64;
            let time_weight = self.time_decay_factor.powf(age_secs / 300.0); // Decay every 5 min
            let source_weight = self.source_weights.get(&signal.source).copied().unwrap_or(0.5);
            let event_weight = self.event_weights.get(&signal.event_type).copied().unwrap_or(1.0);

            let weight = time_weight * source_weight * event_weight * signal.sentiment.confidence;
            total_weight += weight;
            weighted_sum += signal.sentiment.score * weight;

            *event_counts.entry(signal.event_type).or_insert(0) += 1;

            components.push(SignalComponent {
                news_id: signal.news_id.clone(),
                source: signal.source,
                sentiment: signal.sentiment.score,
                weight,
                event_type: signal.event_type,
            });
        }

        let sentiment = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        // Find dominant event type
        let dominant_event = event_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(event, _)| event)
            .unwrap_or(EventType::Unknown);

        // Calculate overall confidence
        let news_count = valid_signals.len();
        let count_factor = (news_count as f64 / 5.0).min(1.0);
        let avg_confidence: f64 = valid_signals.iter().map(|s| s.sentiment.confidence).sum::<f64>()
            / news_count as f64;
        let confidence = (avg_confidence * count_factor).clamp(0.0, 1.0);

        let strength = (sentiment * confidence).abs();
        let action = SignalAction::from_sentiment(sentiment * confidence);

        Some(AggregatedSignal {
            symbol: symbol.to_string(),
            sentiment,
            confidence,
            news_count,
            dominant_event,
            strength,
            action,
            timestamp: now,
            components,
        })
    }

    /// Get default source reliability weights
    fn default_source_weights() -> HashMap<NewsSource, f64> {
        let mut weights = HashMap::new();
        weights.insert(NewsSource::CoinDesk, 1.0);
        weights.insert(NewsSource::CoinTelegraph, 0.9);
        weights.insert(NewsSource::Bloomberg, 1.0);
        weights.insert(NewsSource::Reuters, 1.0);
        weights.insert(NewsSource::Twitter, 0.6);
        weights.insert(NewsSource::Reddit, 0.5);
        weights.insert(NewsSource::Telegram, 0.4);
        weights.insert(NewsSource::Other, 0.3);
        weights
    }

    /// Get default event type weights
    fn default_event_weights() -> HashMap<EventType, f64> {
        let mut weights = HashMap::new();
        weights.insert(EventType::Regulatory, 1.5);
        weights.insert(EventType::Security, 1.5);
        weights.insert(EventType::Adoption, 1.3);
        weights.insert(EventType::Partnership, 1.0);
        weights.insert(EventType::Technical, 0.8);
        weights.insert(EventType::Market, 0.7);
        weights.insert(EventType::Sentiment, 0.5);
        weights.insert(EventType::Macro, 1.2);
        weights.insert(EventType::Unknown, 0.3);
        weights
    }
}

/// Input signal from a single news analysis
#[derive(Debug, Clone)]
pub struct NewsSignal {
    /// News item ID
    pub news_id: String,
    /// Source of the news
    pub source: NewsSource,
    /// Sentiment analysis result
    pub sentiment: SentimentResult,
    /// Event type classification
    pub event_type: EventType,
    /// Timestamp of the news
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signal(
        id: &str,
        source: NewsSource,
        score: f64,
        confidence: f64,
        event: EventType,
    ) -> NewsSignal {
        NewsSignal {
            news_id: id.to_string(),
            source,
            sentiment: SentimentResult::new(score, confidence),
            event_type: event,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_aggregate_single_signal() {
        let aggregator = SignalAggregator::new();
        let signals = vec![create_test_signal(
            "1",
            NewsSource::CoinDesk,
            0.7,
            0.9,
            EventType::Regulatory,
        )];

        let result = aggregator.aggregate("BTC", signals);
        assert!(result.is_some());

        let agg = result.unwrap();
        assert_eq!(agg.symbol, "BTC");
        assert!(agg.sentiment > 0.0);
        assert_eq!(agg.news_count, 1);
    }

    #[test]
    fn test_aggregate_mixed_signals() {
        let aggregator = SignalAggregator::new();
        let signals = vec![
            create_test_signal("1", NewsSource::CoinDesk, 0.8, 0.9, EventType::Adoption),
            create_test_signal("2", NewsSource::Twitter, -0.3, 0.5, EventType::Sentiment),
            create_test_signal("3", NewsSource::Bloomberg, 0.5, 0.8, EventType::Partnership),
        ];

        let result = aggregator.aggregate("ETH", signals);
        assert!(result.is_some());

        let agg = result.unwrap();
        assert_eq!(agg.news_count, 3);
        // Weighted towards positive due to higher source weights
        assert!(agg.sentiment > 0.0);
    }

    #[test]
    fn test_signal_action_from_sentiment() {
        assert_eq!(SignalAction::from_sentiment(0.7), SignalAction::StrongBuy);
        assert_eq!(SignalAction::from_sentiment(0.3), SignalAction::Buy);
        assert_eq!(SignalAction::from_sentiment(0.0), SignalAction::Hold);
        assert_eq!(SignalAction::from_sentiment(-0.3), SignalAction::Sell);
        assert_eq!(SignalAction::from_sentiment(-0.7), SignalAction::StrongSell);
    }

    #[test]
    fn test_position_multiplier() {
        assert_eq!(SignalAction::StrongBuy.position_multiplier(), 1.0);
        assert_eq!(SignalAction::Hold.position_multiplier(), 0.0);
        assert_eq!(SignalAction::StrongSell.position_multiplier(), -1.0);
    }

    #[test]
    fn test_filter_low_confidence() {
        let aggregator = SignalAggregator::new().with_min_confidence(0.5);
        let signals = vec![create_test_signal(
            "1",
            NewsSource::Twitter,
            0.5,
            0.2, // Below threshold
            EventType::Sentiment,
        )];

        let result = aggregator.aggregate("BTC", signals);
        assert!(result.is_none());
    }
}
