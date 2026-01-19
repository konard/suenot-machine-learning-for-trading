//! # Sentiment Aggregator
//!
//! Aggregates sentiment from multiple sources with time-weighted averaging.

use super::analyzer::SentimentResult;
use crate::data::NewsSource;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Aggregated sentiment result
#[derive(Debug, Clone)]
pub struct AggregatedSentiment {
    /// Weighted average sentiment score
    pub score: f64,
    /// Number of sources analyzed
    pub source_count: usize,
    /// Number of texts analyzed
    pub text_count: usize,
    /// Breakdown by source
    pub source_scores: HashMap<String, (f64, usize)>,
    /// Average confidence
    pub confidence: f64,
    /// Timestamp of aggregation
    pub timestamp: DateTime<Utc>,
    /// Score standard deviation (agreement measure)
    pub std_dev: f64,
}

/// Sentiment aggregator for combining multiple sentiment results
pub struct SentimentAggregator {
    /// Time decay half-life in hours
    decay_half_life_hours: f64,
    /// Source weights
    source_weights: HashMap<NewsSource, f64>,
}

impl Default for SentimentAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentAggregator {
    /// Create a new sentiment aggregator
    pub fn new() -> Self {
        let mut source_weights = HashMap::new();
        source_weights.insert(NewsSource::News, 1.2);
        source_weights.insert(NewsSource::Twitter, 1.0);
        source_weights.insert(NewsSource::Reddit, 0.9);
        source_weights.insert(NewsSource::Telegram, 0.7);
        source_weights.insert(NewsSource::Discord, 0.6);
        source_weights.insert(NewsSource::StockTwits, 0.5);
        source_weights.insert(NewsSource::Unknown, 0.5);

        Self {
            decay_half_life_hours: 6.0,
            source_weights,
        }
    }

    /// Set time decay half-life
    pub fn with_decay(mut self, half_life_hours: f64) -> Self {
        self.decay_half_life_hours = half_life_hours;
        self
    }

    /// Calculate time-based weight (exponential decay)
    fn time_weight(&self, timestamp: DateTime<Utc>, now: DateTime<Utc>) -> f64 {
        let age_hours = (now - timestamp).num_minutes() as f64 / 60.0;
        let lambda = 0.693 / self.decay_half_life_hours; // ln(2) / half_life
        (-lambda * age_hours).exp()
    }

    /// Get source weight
    fn source_weight(&self, source: NewsSource) -> f64 {
        *self.source_weights.get(&source).unwrap_or(&0.5)
    }

    /// Aggregate sentiment results with time weighting
    pub fn aggregate(
        &self,
        results: &[SentimentResult],
        sources: &[NewsSource],
    ) -> AggregatedSentiment {
        if results.is_empty() {
            return AggregatedSentiment {
                score: 0.0,
                source_count: 0,
                text_count: 0,
                source_scores: HashMap::new(),
                confidence: 0.0,
                timestamp: Utc::now(),
                std_dev: 0.0,
            };
        }

        let now = Utc::now();
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut confidence_sum = 0.0;
        let mut source_data: HashMap<String, (f64, f64, usize)> = HashMap::new(); // sum, weight, count
        let mut scores: Vec<f64> = Vec::new();

        for (result, source) in results.iter().zip(sources.iter()) {
            let time_w = self.time_weight(result.timestamp, now);
            let source_w = self.source_weight(*source);
            let combined_weight = time_w * source_w * result.confidence;

            weighted_sum += result.score * combined_weight;
            total_weight += combined_weight;
            confidence_sum += result.confidence;
            scores.push(result.score);

            let source_name = source.name().to_string();
            let entry = source_data.entry(source_name).or_insert((0.0, 0.0, 0));
            entry.0 += result.score * combined_weight;
            entry.1 += combined_weight;
            entry.2 += 1;
        }

        let aggregated_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        let avg_confidence = confidence_sum / results.len() as f64;

        // Calculate source breakdown
        let source_scores: HashMap<String, (f64, usize)> = source_data
            .into_iter()
            .map(|(name, (sum, weight, count))| {
                let avg = if weight > 0.0 { sum / weight } else { 0.0 };
                (name, (avg, count))
            })
            .collect();

        // Calculate standard deviation
        let std_dev = Self::calculate_std(&scores);

        AggregatedSentiment {
            score: aggregated_score,
            source_count: source_scores.len(),
            text_count: results.len(),
            source_scores,
            confidence: avg_confidence,
            timestamp: now,
            std_dev,
        }
    }

    /// Simple aggregation without time weighting
    pub fn simple_average(&self, results: &[SentimentResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        let sum: f64 = results.iter().map(|r| r.score).sum();
        sum / results.len() as f64
    }

    /// Weighted average by confidence
    pub fn confidence_weighted_average(&self, results: &[SentimentResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        let weighted_sum: f64 = results.iter().map(|r| r.score * r.confidence).sum();
        let total_confidence: f64 = results.iter().map(|r| r.confidence).sum();

        if total_confidence > 0.0 {
            weighted_sum / total_confidence
        } else {
            0.0
        }
    }

    /// Majority vote sentiment (returns the most common sentiment direction)
    pub fn majority_sentiment(&self, results: &[SentimentResult]) -> (f64, f64) {
        if results.is_empty() {
            return (0.0, 0.0);
        }

        let positive = results.iter().filter(|r| r.score > 0.2).count();
        let negative = results.iter().filter(|r| r.score < -0.2).count();
        let neutral = results.len() - positive - negative;

        let max_count = positive.max(negative).max(neutral);
        let agreement = max_count as f64 / results.len() as f64;

        let direction = if positive > negative && positive > neutral {
            1.0
        } else if negative > positive && negative > neutral {
            -1.0
        } else {
            0.0
        };

        (direction, agreement)
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

/// Source-specific aggregation
#[derive(Debug, Clone)]
pub struct SourceAggregation {
    /// Twitter sentiment
    pub twitter: Option<f64>,
    /// Reddit sentiment
    pub reddit: Option<f64>,
    /// News sentiment
    pub news: Option<f64>,
    /// Other sources sentiment
    pub other: Option<f64>,
    /// Combined sentiment
    pub combined: f64,
}

impl SourceAggregation {
    /// Create from aggregated sentiment
    pub fn from_aggregated(agg: &AggregatedSentiment) -> Self {
        Self {
            twitter: agg.source_scores.get("twitter").map(|(s, _)| *s),
            reddit: agg.source_scores.get("reddit").map(|(s, _)| *s),
            news: agg.source_scores.get("news").map(|(s, _)| *s),
            other: {
                let other_sources: Vec<f64> = agg
                    .source_scores
                    .iter()
                    .filter(|(k, _)| !["twitter", "reddit", "news"].contains(&k.as_str()))
                    .map(|(_, (s, _))| *s)
                    .collect();
                if other_sources.is_empty() {
                    None
                } else {
                    Some(other_sources.iter().sum::<f64>() / other_sources.len() as f64)
                }
            },
            combined: agg.score,
        }
    }

    /// Check if sources agree on direction
    pub fn sources_agree(&self) -> bool {
        let scores: Vec<f64> = [self.twitter, self.reddit, self.news, self.other]
            .into_iter()
            .flatten()
            .collect();

        if scores.is_empty() {
            return true;
        }

        let positive = scores.iter().filter(|&&s| s > 0.1).count();
        let negative = scores.iter().filter(|&&s| s < -0.1).count();

        // Sources agree if most are in the same direction
        positive == scores.len() || negative == scores.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_result(score: f64, confidence: f64) -> SentimentResult {
        SentimentResult {
            text: String::new(),
            score,
            level: crate::sentiment::SentimentLevel::from_score(score),
            confidence,
            lexicon_score: score,
            crypto_score: 0.0,
            key_words: vec![],
            crypto_terms: vec![],
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_simple_average() {
        let aggregator = SentimentAggregator::new();
        let results = vec![
            make_result(0.5, 0.8),
            make_result(-0.5, 0.8),
        ];

        let avg = aggregator.simple_average(&results);
        assert!((avg - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_confidence_weighted() {
        let aggregator = SentimentAggregator::new();
        let results = vec![
            make_result(0.8, 0.9),  // High confidence positive
            make_result(-0.2, 0.3), // Low confidence negative
        ];

        let avg = aggregator.confidence_weighted_average(&results);
        assert!(avg > 0.0); // Should be positive due to higher confidence
    }

    #[test]
    fn test_aggregate_with_sources() {
        let aggregator = SentimentAggregator::new();
        let results = vec![
            make_result(0.5, 0.8),
            make_result(0.3, 0.7),
            make_result(0.4, 0.9),
        ];
        let sources = vec![NewsSource::Twitter, NewsSource::Reddit, NewsSource::News];

        let agg = aggregator.aggregate(&results, &sources);
        assert!(agg.score > 0.0);
        assert_eq!(agg.text_count, 3);
        assert_eq!(agg.source_count, 3);
    }

    #[test]
    fn test_majority_sentiment() {
        let aggregator = SentimentAggregator::new();
        let results = vec![
            make_result(0.5, 0.8),
            make_result(0.4, 0.7),
            make_result(0.3, 0.6),
            make_result(-0.2, 0.5),
        ];

        let (direction, agreement) = aggregator.majority_sentiment(&results);
        assert_eq!(direction, 1.0); // Majority positive
        assert!(agreement > 0.5);
    }

    #[test]
    fn test_time_decay() {
        let aggregator = SentimentAggregator::new();
        let now = Utc::now();

        let recent = aggregator.time_weight(now, now);
        let old = aggregator.time_weight(now - Duration::hours(12), now);

        assert!(recent > old);
        assert!((recent - 1.0).abs() < 0.01);
    }
}
