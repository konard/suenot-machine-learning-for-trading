//! Модуль обработки естественного языка (NLP)
//!
//! Включает:
//! - Токенизацию текста
//! - Предобработку (очистка, нормализация, стемминг)
//! - Векторизацию (Bag of Words, TF-IDF)

mod preprocessor;
mod tokenizer;
mod vectorizer;

pub use preprocessor::Preprocessor;
pub use tokenizer::Tokenizer;
pub use vectorizer::{BagOfWords, TfIdf, Vectorizer};
