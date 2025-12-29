//! Генерация тестовых sentiment-данных
//!
//! Модуль для создания реалистичных тестовых данных
//! при отсутствии реального API социальных сетей.

use chrono::{DateTime, Duration, Utc};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Сообщение с sentiment-оценкой
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentMessage {
    /// Уникальный идентификатор
    pub id: String,
    /// Временная метка
    pub timestamp: DateTime<Utc>,
    /// Текст сообщения
    pub text: String,
    /// Оценка настроения от -1.0 до 1.0
    pub sentiment: f64,
    /// Источник (twitter, reddit, telegram)
    pub source: String,
    /// Количество лайков
    pub likes: u32,
    /// Количество ретвитов/репостов
    pub retweets: u32,
    /// Количество ответов
    pub replies: u32,
}

impl SentimentMessage {
    /// Общий engagement (вовлечённость)
    pub fn engagement(&self) -> u32 {
        self.likes + self.retweets * 2 + self.replies * 3
    }
}

/// Агрегированные sentiment-данные
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentData {
    /// Символ актива
    pub symbol: String,
    /// Временная метка
    pub timestamp: DateTime<Utc>,
    /// Средний sentiment
    pub sentiment: f64,
    /// Количество сообщений
    pub volume: u32,
    /// Уверенность
    pub confidence: f64,
    /// Доля бычьих сообщений
    pub bullish_ratio: f64,
}

/// Генератор тестовых sentiment-данных
pub struct MockSentimentGenerator {
    /// Генератор случайных чисел
    rng: StdRng,
    /// Базовый sentiment (-1.0 до 1.0)
    base_sentiment: f64,
    /// Волатильность sentiment
    volatility: f64,
    /// Среднее количество сообщений в час
    avg_messages_per_hour: u32,
}

impl Default for MockSentimentGenerator {
    fn default() -> Self {
        Self::new(42, 0.0, 0.3, 50)
    }
}

impl MockSentimentGenerator {
    /// Создание нового генератора
    ///
    /// # Аргументы
    ///
    /// * `seed` - Seed для воспроизводимости
    /// * `base_sentiment` - Начальный средний sentiment
    /// * `volatility` - Волатильность изменений
    /// * `avg_messages_per_hour` - Среднее количество сообщений в час
    pub fn new(seed: u64, base_sentiment: f64, volatility: f64, avg_messages_per_hour: u32) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            base_sentiment,
            volatility,
            avg_messages_per_hour,
        }
    }

    /// Генерация сообщений за период
    pub fn generate_messages(
        &mut self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Vec<SentimentMessage> {
        let mut messages = Vec::new();
        let hours = (end_time - start_time).num_hours() as u32;

        if hours == 0 {
            return messages;
        }

        let mut current_sentiment = self.base_sentiment;

        for hour in 0..hours {
            // Количество сообщений в этом часе
            let hour_messages =
                self.avg_messages_per_hour / 2 + self.rng.gen_range(0..self.avg_messages_per_hour);

            // Обновляем базовый sentiment (случайное блуждание)
            current_sentiment +=
                self.rng.gen_range(-self.volatility..self.volatility);
            current_sentiment = current_sentiment.clamp(-1.0, 1.0);

            for i in 0..hour_messages {
                let minute_offset = self.rng.gen_range(0..60);
                let timestamp =
                    start_time + Duration::hours(hour as i64) + Duration::minutes(minute_offset);

                // Sentiment с шумом вокруг текущего базового
                let msg_sentiment =
                    current_sentiment + self.rng.gen_range(-0.3..0.3);
                let msg_sentiment = msg_sentiment.clamp(-1.0, 1.0);

                // Генерируем engagement (логнормальное распределение)
                let base_engagement = self.rng.gen_range(1.0_f64..10.0);
                let likes = (base_engagement.powi(2) * self.rng.gen_range(1.0..5.0)) as u32;
                let retweets = (likes as f64 * self.rng.gen_range(0.1..0.5)) as u32;
                let replies = (likes as f64 * self.rng.gen_range(0.05..0.2)) as u32;

                messages.push(SentimentMessage {
                    id: format!("{}_{}_{}", symbol, hour, i),
                    timestamp,
                    text: self.generate_text(msg_sentiment),
                    sentiment: msg_sentiment,
                    source: self.random_source(),
                    likes,
                    retweets,
                    replies,
                });
            }
        }

        messages
    }

    /// Генерация агрегированных данных
    pub fn generate_aggregated(
        &mut self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        interval_hours: i64,
    ) -> Vec<SentimentData> {
        let mut data = Vec::new();
        let mut current_time = start_time;
        let mut current_sentiment = self.base_sentiment;

        while current_time < end_time {
            // Случайное блуждание sentiment
            current_sentiment +=
                self.rng.gen_range(-self.volatility..self.volatility);
            current_sentiment = current_sentiment.clamp(-1.0, 1.0);

            // Объём сообщений
            let volume = (self.avg_messages_per_hour as f64 * interval_hours as f64
                * self.rng.gen_range(0.5..1.5)) as u32;

            // Уверенность
            let confidence = self.rng.gen_range(0.5..0.95);

            // Доля бычьих
            let bullish_ratio = if current_sentiment > 0.0 {
                0.5 + current_sentiment * 0.4 + self.rng.gen_range(-0.1..0.1)
            } else {
                0.5 + current_sentiment * 0.4 + self.rng.gen_range(-0.1..0.1)
            };

            data.push(SentimentData {
                symbol: symbol.to_string(),
                timestamp: current_time,
                sentiment: current_sentiment,
                volume,
                confidence,
                bullish_ratio: bullish_ratio.clamp(0.0, 1.0),
            });

            current_time = current_time + Duration::hours(interval_hours);
        }

        data
    }

    /// Генерация sentiment с корреляцией к цене
    ///
    /// Симулирует реалистичную связь между ценой и настроениями
    pub fn generate_with_price_correlation(
        &mut self,
        symbol: &str,
        price_returns: &[f64],
        start_time: DateTime<Utc>,
        interval_hours: i64,
        correlation: f64,
    ) -> Vec<SentimentData> {
        let mut data = Vec::new();
        let mut current_time = start_time;

        for (i, &price_return) in price_returns.iter().enumerate() {
            // Sentiment = корреляция с ценой + шум
            let price_component = price_return * 10.0 * correlation; // Масштабируем
            let noise = self.rng.gen_range(-self.volatility..self.volatility);
            let sentiment = (price_component + noise).clamp(-1.0, 1.0);

            // Иногда добавляем опережающий sentiment (sentiment leads price)
            let lead_adjustment = if self.rng.gen_bool(0.3) {
                self.rng.gen_range(-0.2..0.2)
            } else {
                0.0
            };

            let final_sentiment = (sentiment + lead_adjustment).clamp(-1.0, 1.0);

            let volume = (self.avg_messages_per_hour as f64 * interval_hours as f64
                * (1.0 + price_return.abs() * 5.0) // Больше активности при волатильности
                * self.rng.gen_range(0.5..1.5)) as u32;

            data.push(SentimentData {
                symbol: symbol.to_string(),
                timestamp: current_time,
                sentiment: final_sentiment,
                volume,
                confidence: self.rng.gen_range(0.5..0.95),
                bullish_ratio: (0.5 + final_sentiment * 0.4).clamp(0.0, 1.0),
            });

            current_time = current_time + Duration::hours(interval_hours);
        }

        data
    }

    /// Генерация случайного источника
    fn random_source(&mut self) -> String {
        match self.rng.gen_range(0..3) {
            0 => "twitter".to_string(),
            1 => "reddit".to_string(),
            _ => "telegram".to_string(),
        }
    }

    /// Генерация текста на основе sentiment
    fn generate_text(&mut self, sentiment: f64) -> String {
        let positive_phrases = [
            "Bullish on this!",
            "To the moon!",
            "Great buying opportunity",
            "Very optimistic about this",
            "HODL!",
            "This is going up",
        ];

        let negative_phrases = [
            "Bearish...",
            "Looks like a dump incoming",
            "Stay away",
            "This is going down",
            "Sell now",
            "Not looking good",
        ];

        let neutral_phrases = [
            "Watching closely",
            "Interesting price action",
            "Let's see what happens",
            "Consolidating...",
            "Wait for confirmation",
        ];

        let phrases = if sentiment > 0.3 {
            &positive_phrases[..]
        } else if sentiment < -0.3 {
            &negative_phrases[..]
        } else {
            &neutral_phrases[..]
        };

        phrases[self.rng.gen_range(0..phrases.len())].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_generation() {
        let mut generator = MockSentimentGenerator::default();
        let now = Utc::now();
        let start = now - Duration::hours(24);

        let messages = generator.generate_messages("BTCUSDT", start, now);

        assert!(!messages.is_empty());
        assert!(messages.iter().all(|m| m.sentiment >= -1.0 && m.sentiment <= 1.0));
    }

    #[test]
    fn test_aggregated_generation() {
        let mut generator = MockSentimentGenerator::new(42, 0.2, 0.2, 100);
        let now = Utc::now();
        let start = now - Duration::days(7);

        let data = generator.generate_aggregated("BTCUSDT", start, now, 1);

        assert!(!data.is_empty());
        assert!(data.iter().all(|d| d.sentiment >= -1.0 && d.sentiment <= 1.0));
    }

    #[test]
    fn test_engagement() {
        let msg = SentimentMessage {
            id: "1".to_string(),
            timestamp: Utc::now(),
            text: "Test".to_string(),
            sentiment: 0.5,
            source: "twitter".to_string(),
            likes: 100,
            retweets: 50,
            replies: 10,
        };

        // likes + retweets*2 + replies*3 = 100 + 100 + 30 = 230
        assert_eq!(msg.engagement(), 230);
    }
}
