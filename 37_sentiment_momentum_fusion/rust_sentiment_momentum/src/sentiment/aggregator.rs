//! Агрегатор настроений
//!
//! Объединяет sentiment-данные из разных источников
//! с учётом времени и вовлечённости.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use super::mock_data::SentimentMessage;

/// Агрегированные данные настроений
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedSentiment {
    /// Символ актива
    pub symbol: String,
    /// Временная метка
    pub timestamp: DateTime<Utc>,
    /// Взвешенное среднее настроение
    pub sentiment: f64,
    /// Количество сообщений
    pub volume: usize,
    /// Уверенность (на основе согласованности)
    pub confidence: f64,
    /// Доля бычьих сообщений
    pub bullish_ratio: f64,
    /// Стандартное отклонение настроений
    pub sentiment_std: f64,
}

/// Агрегатор настроений
pub struct SentimentAggregator {
    /// Период полураспада временного веса (в часах)
    decay_halflife_hours: f64,
    /// Минимальное количество сообщений для надёжной агрегации
    min_messages: usize,
}

impl Default for SentimentAggregator {
    fn default() -> Self {
        Self::new(24.0, 5)
    }
}

impl SentimentAggregator {
    /// Создание нового агрегатора
    ///
    /// # Аргументы
    ///
    /// * `decay_halflife_hours` - Период полураспада (24 часа = вчерашнее сообщение
    ///   весит в 2 раза меньше)
    /// * `min_messages` - Минимальное количество сообщений для надёжной оценки
    pub fn new(decay_halflife_hours: f64, min_messages: usize) -> Self {
        Self {
            decay_halflife_hours,
            min_messages,
        }
    }

    /// Агрегация настроений за период
    pub fn aggregate(
        &self,
        symbol: &str,
        reference_time: DateTime<Utc>,
        messages: &[SentimentMessage],
    ) -> AggregatedSentiment {
        if messages.is_empty() {
            return AggregatedSentiment {
                symbol: symbol.to_string(),
                timestamp: reference_time,
                sentiment: 0.0,
                volume: 0,
                confidence: 0.0,
                bullish_ratio: 0.5,
                sentiment_std: 0.0,
            };
        }

        // Рассчитываем веса
        let weights: Vec<f64> = messages
            .iter()
            .map(|msg| {
                // Временной вес
                let hours_ago = (reference_time - msg.timestamp).num_seconds() as f64 / 3600.0;
                let time_weight = 0.5_f64.powf(hours_ago / self.decay_halflife_hours);

                // Вес по вовлечённости
                let engagement = msg.engagement();
                let engagement_weight = (1.0 + engagement as f64).ln();

                time_weight * engagement_weight
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();

        if total_weight == 0.0 {
            return AggregatedSentiment {
                symbol: symbol.to_string(),
                timestamp: reference_time,
                sentiment: 0.0,
                volume: messages.len(),
                confidence: 0.0,
                bullish_ratio: 0.5,
                sentiment_std: 0.0,
            };
        }

        // Нормализуем веса
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

        // Взвешенное среднее настроение
        let weighted_sentiment: f64 = messages
            .iter()
            .zip(normalized_weights.iter())
            .map(|(msg, weight)| msg.sentiment * weight)
            .sum();

        // Расчёт стандартного отклонения
        let sentiment_variance: f64 = messages
            .iter()
            .zip(normalized_weights.iter())
            .map(|(msg, weight)| {
                let diff = msg.sentiment - weighted_sentiment;
                diff * diff * weight
            })
            .sum();
        let sentiment_std = sentiment_variance.sqrt();

        // Уверенность на основе согласованности и объёма
        let agreement_confidence = 1.0 - sentiment_std.min(1.0);
        let volume_confidence = (messages.len() as f64 / self.min_messages as f64).min(1.0);
        let confidence = agreement_confidence * 0.7 + volume_confidence * 0.3;

        // Доля бычьих сообщений
        let bullish_count = messages.iter().filter(|msg| msg.sentiment > 0.1).count();
        let bullish_ratio = bullish_count as f64 / messages.len() as f64;

        AggregatedSentiment {
            symbol: symbol.to_string(),
            timestamp: reference_time,
            sentiment: weighted_sentiment,
            volume: messages.len(),
            confidence,
            bullish_ratio,
            sentiment_std,
        }
    }

    /// Агрегация по временным окнам
    pub fn aggregate_by_windows(
        &self,
        symbol: &str,
        messages: &[SentimentMessage],
        window_hours: i64,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Vec<AggregatedSentiment> {
        let mut results = Vec::new();
        let mut current_time = start_time;
        let window_duration = Duration::hours(window_hours);

        while current_time < end_time {
            let window_end = current_time + window_duration;

            // Фильтруем сообщения для текущего окна
            let window_messages: Vec<SentimentMessage> = messages
                .iter()
                .filter(|msg| msg.timestamp >= current_time && msg.timestamp < window_end)
                .cloned()
                .collect();

            let aggregated = self.aggregate(symbol, window_end, &window_messages);
            results.push(aggregated);

            current_time = window_end;
        }

        results
    }

    /// Расчёт скользящего среднего настроения
    pub fn rolling_sentiment(
        &self,
        aggregated: &[AggregatedSentiment],
        window_size: usize,
    ) -> Vec<f64> {
        if aggregated.len() < window_size {
            return vec![0.0; aggregated.len()];
        }

        let mut result = Vec::with_capacity(aggregated.len());

        for i in 0..aggregated.len() {
            if i < window_size - 1 {
                result.push(0.0);
            } else {
                let window_sum: f64 = aggregated[i + 1 - window_size..=i]
                    .iter()
                    .map(|a| a.sentiment)
                    .sum();
                result.push(window_sum / window_size as f64);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_aggregation() {
        let aggregator = SentimentAggregator::new(24.0, 5);
        let now = Utc::now();

        let messages = vec![
            SentimentMessage {
                id: "1".to_string(),
                timestamp: now - Duration::hours(1),
                text: "Bullish!".to_string(),
                sentiment: 0.8,
                source: "twitter".to_string(),
                likes: 100,
                retweets: 50,
                replies: 10,
            },
            SentimentMessage {
                id: "2".to_string(),
                timestamp: now - Duration::hours(2),
                text: "Also bullish!".to_string(),
                sentiment: 0.6,
                source: "twitter".to_string(),
                likes: 50,
                retweets: 20,
                replies: 5,
            },
        ];

        let result = aggregator.aggregate("BTCUSDT", now, &messages);

        assert!(result.sentiment > 0.0);
        assert_eq!(result.volume, 2);
        assert!(result.bullish_ratio > 0.5);
    }

    #[test]
    fn test_empty_messages() {
        let aggregator = SentimentAggregator::default();
        let result = aggregator.aggregate("BTCUSDT", Utc::now(), &[]);

        assert_eq!(result.sentiment, 0.0);
        assert_eq!(result.volume, 0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_time_decay() {
        let aggregator = SentimentAggregator::new(1.0, 1); // 1 hour halflife
        let now = Utc::now();

        let messages = vec![
            SentimentMessage {
                id: "1".to_string(),
                timestamp: now,
                text: "Recent positive".to_string(),
                sentiment: 1.0,
                source: "twitter".to_string(),
                likes: 10,
                retweets: 0,
                replies: 0,
            },
            SentimentMessage {
                id: "2".to_string(),
                timestamp: now - Duration::hours(2),
                text: "Old negative".to_string(),
                sentiment: -1.0,
                source: "twitter".to_string(),
                likes: 10,
                retweets: 0,
                replies: 0,
            },
        ];

        let result = aggregator.aggregate("BTCUSDT", now, &messages);

        // Более свежее сообщение должно иметь больший вес
        assert!(result.sentiment > 0.0);
    }
}
