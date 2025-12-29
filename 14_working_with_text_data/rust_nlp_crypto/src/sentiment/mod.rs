//! Модуль анализа настроений
//!
//! Включает:
//! - Лексикон-базированный анализ
//! - Наивный Байесовский классификатор
//! - Агрегацию результатов

mod analyzer;
mod lexicon;
mod naive_bayes;

pub use analyzer::SentimentAnalyzer;
pub use lexicon::{CryptoLexicon, SentimentLexicon};
pub use naive_bayes::NaiveBayesClassifier;
