//! Модуль анализа настроений
//!
//! Содержит компоненты для анализа и агрегации sentiment-данных.

mod aggregator;
mod analyzer;
mod mock_data;

pub use aggregator::SentimentAggregator;
pub use analyzer::{SentimentAnalyzer, SentimentScore};
pub use mock_data::{MockSentimentGenerator, SentimentData, SentimentMessage};
