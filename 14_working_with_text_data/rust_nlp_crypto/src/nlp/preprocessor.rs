//! Предобработка текста
//!
//! Включает:
//! - Очистку от специальных символов
//! - Удаление стоп-слов
//! - Стемминг (приведение к корню)
//! - Нормализацию

use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;

/// Английские стоп-слова
const ENGLISH_STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need",
    "dare", "ought", "used", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn",
];

/// Крипто-специфичные стоп-слова (часто встречающиеся, но не информативные)
const CRYPTO_STOP_WORDS: &[&str] = &[
    "crypto", "cryptocurrency", "token", "coin", "blockchain", "trading", "trade",
    "market", "price", "buy", "sell", "holder", "holding", "wallet", "exchange",
];

/// Предобработчик текста
#[derive(Debug, Clone)]
pub struct Preprocessor {
    /// Стоп-слова для удаления
    stop_words: HashSet<String>,
    /// Использовать стемминг
    use_stemming: bool,
    /// Удалять числа
    remove_numbers: bool,
    /// Приводить к нижнему регистру
    lowercase: bool,
    /// Минимальная длина слова
    min_word_length: usize,
    /// Стеммер
    stemmer: Option<Stemmer>,
}

impl Preprocessor {
    /// Создать новый предобработчик с настройками по умолчанию
    pub fn new() -> Self {
        let mut stop_words = HashSet::new();
        for word in ENGLISH_STOP_WORDS {
            stop_words.insert(word.to_string());
        }

        Self {
            stop_words,
            use_stemming: true,
            remove_numbers: false,
            lowercase: true,
            min_word_length: 2,
            stemmer: Some(Stemmer::create(Algorithm::English)),
        }
    }

    /// Добавить крипто-специфичные стоп-слова
    pub fn with_crypto_stopwords(mut self) -> Self {
        for word in CRYPTO_STOP_WORDS {
            self.stop_words.insert(word.to_string());
        }
        self
    }

    /// Добавить пользовательские стоп-слова
    pub fn with_custom_stopwords(mut self, words: &[&str]) -> Self {
        for word in words {
            self.stop_words.insert(word.to_string());
        }
        self
    }

    /// Включить/выключить стемминг
    pub fn with_stemming(mut self, enabled: bool) -> Self {
        self.use_stemming = enabled;
        if enabled && self.stemmer.is_none() {
            self.stemmer = Some(Stemmer::create(Algorithm::English));
        }
        self
    }

    /// Включить удаление чисел
    pub fn with_remove_numbers(mut self, remove: bool) -> Self {
        self.remove_numbers = remove;
        self
    }

    /// Установить минимальную длину слова
    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_word_length = len;
        self
    }

    /// Предобработать список токенов
    pub fn process(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .filter_map(|token| self.process_token(token))
            .collect()
    }

    /// Предобработать один токен
    pub fn process_token(&self, token: &str) -> Option<String> {
        // Приводим к нижнему регистру
        let processed = if self.lowercase {
            token.to_lowercase()
        } else {
            token.to_string()
        };

        // Проверяем длину
        if processed.len() < self.min_word_length {
            return None;
        }

        // Удаляем числа если нужно
        if self.remove_numbers && processed.chars().all(|c| c.is_numeric()) {
            return None;
        }

        // Проверяем стоп-слова
        if self.stop_words.contains(&processed) {
            return None;
        }

        // Применяем стемминг
        let result = if self.use_stemming {
            if let Some(ref stemmer) = self.stemmer {
                stemmer.stem(&processed).to_string()
            } else {
                processed
            }
        } else {
            processed
        };

        // Финальная проверка длины после стемминга
        if result.len() < self.min_word_length {
            return None;
        }

        Some(result)
    }

    /// Предобработать сырой текст
    pub fn process_text(&self, text: &str) -> Vec<String> {
        // Простая токенизация по пробелам
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| {
                s.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|s| !s.is_empty())
            .collect();

        self.process(&tokens)
    }

    /// Проверить, является ли слово стоп-словом
    pub fn is_stopword(&self, word: &str) -> bool {
        self.stop_words.contains(&word.to_lowercase())
    }

    /// Получить стем слова
    pub fn stem(&self, word: &str) -> String {
        if let Some(ref stemmer) = self.stemmer {
            stemmer.stem(&word.to_lowercase()).to_string()
        } else {
            word.to_lowercase()
        }
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Утилиты для очистки текста
pub struct TextCleaner;

impl TextCleaner {
    /// Удалить HTML теги
    pub fn remove_html_tags(text: &str) -> String {
        let re = regex::Regex::new(r"<[^>]+>").unwrap();
        re.replace_all(text, "").to_string()
    }

    /// Удалить URL
    pub fn remove_urls(text: &str) -> String {
        let re = regex::Regex::new(r"https?://\S+").unwrap();
        re.replace_all(text, "").to_string()
    }

    /// Удалить множественные пробелы
    pub fn normalize_whitespace(text: &str) -> String {
        let re = regex::Regex::new(r"\s+").unwrap();
        re.replace_all(text.trim(), " ").to_string()
    }

    /// Удалить специальные символы (оставить только буквы, цифры, пробелы)
    pub fn remove_special_chars(text: &str) -> String {
        text.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect()
    }

    /// Полная очистка текста
    pub fn clean(text: &str) -> String {
        let result = Self::remove_html_tags(text);
        let result = Self::remove_urls(&result);
        let result = Self::remove_special_chars(&result);
        Self::normalize_whitespace(&result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopword_removal() {
        let preprocessor = Preprocessor::new();
        let tokens = vec![
            "the".to_string(),
            "bitcoin".to_string(),
            "is".to_string(),
            "rising".to_string(),
        ];
        let result = preprocessor.process(&tokens);

        // "the" и "is" должны быть удалены
        assert!(!result.contains(&"the".to_string()));
        assert!(!result.contains(&"is".to_string()));
    }

    #[test]
    fn test_stemming() {
        let preprocessor = Preprocessor::new();
        let result = preprocessor.stem("running");

        assert_eq!(result, "run");
    }

    #[test]
    fn test_crypto_stopwords() {
        let preprocessor = Preprocessor::new().with_crypto_stopwords();

        assert!(preprocessor.is_stopword("crypto"));
        assert!(preprocessor.is_stopword("blockchain"));
    }

    #[test]
    fn test_text_cleaner() {
        let text = "<p>Check https://example.com   for more info!</p>";
        let cleaned = TextCleaner::clean(text);

        assert!(!cleaned.contains("<p>"));
        assert!(!cleaned.contains("https"));
        assert!(!cleaned.contains("!"));
    }

    #[test]
    fn test_min_length() {
        let preprocessor = Preprocessor::new().with_min_length(4);
        let result = preprocessor.process_token("hi");

        assert!(result.is_none());
    }
}
