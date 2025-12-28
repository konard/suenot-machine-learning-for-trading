//! Токенизатор текста
//!
//! Разбивает текст на отдельные токены (слова, числа, символы)

use crate::models::{Token, TokenType};
use regex::Regex;
use std::sync::LazyLock;

/// Регулярные выражения для различных типов токенов
static URL_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s]+").unwrap());
static MENTION_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"@\w+").unwrap());
static HASHTAG_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"#\w+").unwrap());
static CRYPTO_SYMBOL_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\$[A-Z]{2,10}").unwrap());
static NUMBER_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+\.?\d*$").unwrap());

/// Известные символы криптовалют
const CRYPTO_SYMBOLS: &[&str] = &[
    "BTC", "ETH", "USDT", "USDC", "BNB", "XRP", "ADA", "DOGE", "SOL", "DOT",
    "MATIC", "SHIB", "LTC", "TRX", "AVAX", "LINK", "ATOM", "UNI", "XLM", "ETC",
    "APT", "ARB", "OP", "SUI", "PEPE", "WLD", "SEI", "TIA", "JUP", "PYTH",
    "BITCOIN", "ETHEREUM", "SOLANA", "CARDANO", "RIPPLE", "DOGECOIN",
];

/// Токенизатор текста
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Минимальная длина слова
    min_word_length: usize,
    /// Максимальная длина слова
    max_word_length: usize,
    /// Сохранять ли регистр
    preserve_case: bool,
    /// Извлекать ли специальные токены (URL, mentions, hashtags)
    extract_special: bool,
}

impl Tokenizer {
    /// Создать новый токенизатор с настройками по умолчанию
    pub fn new() -> Self {
        Self {
            min_word_length: 2,
            max_word_length: 50,
            preserve_case: false,
            extract_special: true,
        }
    }

    /// Установить минимальную длину слова
    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_word_length = len;
        self
    }

    /// Установить максимальную длину слова
    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_word_length = len;
        self
    }

    /// Сохранять регистр
    pub fn preserve_case(mut self, preserve: bool) -> Self {
        self.preserve_case = preserve;
        self
    }

    /// Извлекать специальные токены
    pub fn extract_special(mut self, extract: bool) -> Self {
        self.extract_special = extract;
        self
    }

    /// Токенизировать текст
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position = 0;

        // Сначала извлекаем специальные токены
        let mut processed_text = text.to_string();

        if self.extract_special {
            // Извлекаем URL
            for url_match in URL_REGEX.find_iter(text) {
                tokens.push(Token {
                    original: url_match.as_str().to_string(),
                    normalized: "[URL]".to_string(),
                    position,
                    token_type: TokenType::Url,
                });
                position += 1;
            }
            processed_text = URL_REGEX.replace_all(&processed_text, " ").to_string();

            // Извлекаем mentions (@user)
            for mention in MENTION_REGEX.find_iter(text) {
                tokens.push(Token {
                    original: mention.as_str().to_string(),
                    normalized: "[MENTION]".to_string(),
                    position,
                    token_type: TokenType::Mention,
                });
                position += 1;
            }
            processed_text = MENTION_REGEX.replace_all(&processed_text, " ").to_string();

            // Извлекаем hashtags (#topic)
            for hashtag in HASHTAG_REGEX.find_iter(text) {
                let tag = hashtag.as_str();
                tokens.push(Token {
                    original: tag.to_string(),
                    normalized: tag[1..].to_lowercase(), // Без #
                    position,
                    token_type: TokenType::Hashtag,
                });
                position += 1;
            }
            processed_text = HASHTAG_REGEX.replace_all(&processed_text, " ").to_string();

            // Извлекаем символы криптовалют ($BTC)
            for symbol in CRYPTO_SYMBOL_REGEX.find_iter(text) {
                let sym = symbol.as_str();
                tokens.push(Token {
                    original: sym.to_string(),
                    normalized: sym[1..].to_uppercase(), // Без $
                    position,
                    token_type: TokenType::CryptoSymbol,
                });
                position += 1;
            }
            processed_text = CRYPTO_SYMBOL_REGEX.replace_all(&processed_text, " ").to_string();
        }

        // Разбиваем оставшийся текст на слова
        for word in processed_text.split_whitespace() {
            let cleaned = self.clean_word(word);

            if cleaned.is_empty() {
                continue;
            }

            // Проверяем длину
            if cleaned.len() < self.min_word_length || cleaned.len() > self.max_word_length {
                continue;
            }

            let token_type = self.determine_token_type(&cleaned);
            let normalized = if self.preserve_case {
                cleaned.clone()
            } else {
                cleaned.to_lowercase()
            };

            tokens.push(Token {
                original: cleaned,
                normalized,
                position,
                token_type,
            });
            position += 1;
        }

        tokens
    }

    /// Токенизировать и вернуть только нормализованные строки
    pub fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        self.tokenize(text)
            .into_iter()
            .map(|t| t.normalized)
            .collect()
    }

    /// Очистить слово от знаков препинания
    fn clean_word(&self, word: &str) -> String {
        word.chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
            .collect()
    }

    /// Определить тип токена
    fn determine_token_type(&self, word: &str) -> TokenType {
        let upper = word.to_uppercase();

        // Проверяем, является ли это символом криптовалюты
        if CRYPTO_SYMBOLS.contains(&upper.as_str()) {
            return TokenType::CryptoSymbol;
        }

        // Проверяем, является ли это числом
        if NUMBER_REGEX.is_match(word) {
            return TokenType::Number;
        }

        // Проверяем эмодзи (упрощённо)
        if word.chars().any(|c| c > '\u{1F300}') {
            return TokenType::Emoji;
        }

        TokenType::Word
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("Hello World");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].normalized, "hello");
        assert_eq!(tokens[1].normalized, "world");
    }

    #[test]
    fn test_crypto_symbol_detection() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("I love BTC and ETH");

        let crypto_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| t.token_type == TokenType::CryptoSymbol)
            .collect();

        assert_eq!(crypto_tokens.len(), 2);
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("Check @user and #crypto $BTC https://example.com");

        let has_mention = tokens.iter().any(|t| t.token_type == TokenType::Mention);
        let has_hashtag = tokens.iter().any(|t| t.token_type == TokenType::Hashtag);
        let has_url = tokens.iter().any(|t| t.token_type == TokenType::Url);

        assert!(has_mention);
        assert!(has_hashtag);
        assert!(has_url);
    }

    #[test]
    fn test_min_length_filter() {
        let tokenizer = Tokenizer::new().with_min_length(3);
        let tokens = tokenizer.tokenize("I am a test");

        // "I", "am", "a" should be filtered out
        assert!(tokens.iter().all(|t| t.normalized.len() >= 3));
    }
}
