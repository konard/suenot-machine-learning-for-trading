//! Словари настроений
//!
//! Содержит:
//! - Общий словарь настроений
//! - Крипто-специфичный словарь

use std::collections::HashMap;

/// Трейт для словаря настроений
pub trait SentimentLexicon {
    /// Получить оценку слова (-1.0 до 1.0)
    fn get_score(&self, word: &str) -> Option<f64>;

    /// Проверить наличие слова в словаре
    fn contains(&self, word: &str) -> bool;

    /// Получить все слова словаря
    fn words(&self) -> Vec<&str>;
}

/// Крипто-специфичный словарь настроений
#[derive(Debug, Clone)]
pub struct CryptoLexicon {
    /// Позитивные слова с оценками
    positive: HashMap<String, f64>,
    /// Негативные слова с оценками
    negative: HashMap<String, f64>,
    /// Модификаторы (усилители/ослабители)
    modifiers: HashMap<String, f64>,
    /// Отрицания
    negations: Vec<String>,
}

impl CryptoLexicon {
    /// Создать новый крипто-словарь с предустановленными значениями
    pub fn new() -> Self {
        let mut positive = HashMap::new();
        let mut negative = HashMap::new();
        let mut modifiers = HashMap::new();

        // Сильно позитивные слова (0.7 - 1.0)
        let strong_positive = [
            ("moon", 0.9),
            ("mooning", 0.95),
            ("bullish", 0.8),
            ("pump", 0.7),
            ("pumping", 0.8),
            ("ath", 0.85),        // all-time high
            ("breakout", 0.75),
            ("rally", 0.8),
            ("surge", 0.8),
            ("skyrocket", 0.9),
            ("lambo", 0.85),
            ("hodl", 0.6),
            ("diamond", 0.7),     // diamond hands
            ("gains", 0.75),
            ("profit", 0.7),
            ("rich", 0.7),
            ("millionaire", 0.8),
            ("winner", 0.75),
            ("winning", 0.75),
            ("success", 0.7),
            ("amazing", 0.8),
            ("excellent", 0.8),
            ("incredible", 0.85),
            ("fantastic", 0.8),
            ("great", 0.7),
            ("awesome", 0.75),
            ("love", 0.7),
            ("bullrun", 0.85),
            ("adoption", 0.65),
            ("partnership", 0.6),
            ("accumulate", 0.5),
            ("accumulating", 0.5),
            ("undervalued", 0.6),
            ("gem", 0.7),
            ("opportunity", 0.55),
        ];

        // Умеренно позитивные слова (0.3 - 0.6)
        let moderate_positive = [
            ("up", 0.4),
            ("rise", 0.5),
            ("rising", 0.5),
            ("green", 0.5),
            ("buy", 0.4),
            ("buying", 0.4),
            ("long", 0.4),
            ("support", 0.45),
            ("recover", 0.5),
            ("recovery", 0.5),
            ("bounce", 0.45),
            ("stable", 0.35),
            ("growing", 0.5),
            ("growth", 0.5),
            ("positive", 0.5),
            ("good", 0.5),
            ("nice", 0.45),
            ("cool", 0.4),
            ("interesting", 0.35),
            ("promising", 0.55),
            ("potential", 0.45),
            ("strong", 0.5),
            ("healthy", 0.5),
        ];

        // Сильно негативные слова (-0.7 до -1.0)
        let strong_negative = [
            ("crash", -0.9),
            ("crashing", -0.95),
            ("bearish", -0.8),
            ("dump", -0.75),
            ("dumping", -0.8),
            ("scam", -0.95),
            ("fraud", -0.95),
            ("rugpull", -1.0),
            ("rug", -0.9),
            ("ponzi", -0.95),
            ("collapse", -0.9),
            ("disaster", -0.9),
            ("catastrophe", -0.95),
            ("dead", -0.85),
            ("rekt", -0.85),
            ("liquidated", -0.8),
            ("liquidation", -0.8),
            ("bankrupt", -0.9),
            ("worthless", -0.85),
            ("terrible", -0.8),
            ("horrible", -0.85),
            ("awful", -0.8),
            ("worst", -0.85),
            ("hate", -0.75),
            ("fear", -0.6),
            ("panic", -0.8),
            ("fud", -0.5),
            ("hack", -0.85),
            ("hacked", -0.9),
            ("exploit", -0.85),
            ("stolen", -0.9),
            ("lost", -0.7),
            ("lose", -0.7),
            ("losing", -0.7),
        ];

        // Умеренно негативные слова (-0.3 до -0.6)
        let moderate_negative = [
            ("down", -0.4),
            ("fall", -0.5),
            ("falling", -0.5),
            ("red", -0.5),
            ("sell", -0.4),
            ("selling", -0.4),
            ("short", -0.4),
            ("resistance", -0.35),
            ("decline", -0.5),
            ("drop", -0.5),
            ("dropping", -0.55),
            ("correction", -0.4),
            ("dip", -0.35),
            ("dipping", -0.4),
            ("weak", -0.5),
            ("bad", -0.5),
            ("risk", -0.4),
            ("risky", -0.45),
            ("volatile", -0.3),
            ("uncertainty", -0.4),
            ("concern", -0.45),
            ("worried", -0.5),
            ("warning", -0.5),
            ("caution", -0.4),
            ("overvalued", -0.5),
            ("bubble", -0.6),
        ];

        // Добавляем все слова в словари
        for (word, score) in strong_positive.iter().chain(moderate_positive.iter()) {
            positive.insert(word.to_string(), *score);
        }

        for (word, score) in strong_negative.iter().chain(moderate_negative.iter()) {
            negative.insert(word.to_string(), *score);
        }

        // Модификаторы (усилители и ослабители)
        let modifier_words = [
            ("very", 1.5),
            ("really", 1.4),
            ("extremely", 1.8),
            ("incredibly", 1.7),
            ("super", 1.5),
            ("absolutely", 1.6),
            ("totally", 1.4),
            ("completely", 1.5),
            ("highly", 1.4),
            ("quite", 1.2),
            ("somewhat", 0.8),
            ("slightly", 0.7),
            ("barely", 0.6),
            ("little", 0.7),
            ("maybe", 0.8),
            ("perhaps", 0.8),
            ("possibly", 0.7),
        ];

        for (word, multiplier) in modifier_words {
            modifiers.insert(word.to_string(), multiplier);
        }

        // Отрицания
        let negations = vec![
            "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
            "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
            "cant", "can't", "couldnt", "couldn't", "wont", "won't",
            "wouldnt", "wouldn't", "shouldnt", "shouldn't", "isnt", "isn't",
            "arent", "aren't", "wasnt", "wasn't", "werent", "weren't",
            "havent", "haven't", "hasnt", "hasn't", "hadnt", "hadn't",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            positive,
            negative,
            modifiers,
            negations,
        }
    }

    /// Проверить, является ли слово отрицанием
    pub fn is_negation(&self, word: &str) -> bool {
        self.negations.contains(&word.to_lowercase())
    }

    /// Получить модификатор слова
    pub fn get_modifier(&self, word: &str) -> Option<f64> {
        self.modifiers.get(&word.to_lowercase()).copied()
    }

    /// Добавить пользовательское слово
    pub fn add_word(&mut self, word: &str, score: f64) {
        let word_lower = word.to_lowercase();
        if score >= 0.0 {
            self.positive.insert(word_lower, score);
        } else {
            self.negative.insert(word_lower, score);
        }
    }

    /// Получить статистику словаря
    pub fn stats(&self) -> LexiconStats {
        LexiconStats {
            positive_count: self.positive.len(),
            negative_count: self.negative.len(),
            modifier_count: self.modifiers.len(),
            negation_count: self.negations.len(),
        }
    }
}

impl Default for CryptoLexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentLexicon for CryptoLexicon {
    fn get_score(&self, word: &str) -> Option<f64> {
        let word_lower = word.to_lowercase();

        self.positive
            .get(&word_lower)
            .or_else(|| self.negative.get(&word_lower))
            .copied()
    }

    fn contains(&self, word: &str) -> bool {
        let word_lower = word.to_lowercase();
        self.positive.contains_key(&word_lower) || self.negative.contains_key(&word_lower)
    }

    fn words(&self) -> Vec<&str> {
        self.positive
            .keys()
            .chain(self.negative.keys())
            .map(|s| s.as_str())
            .collect()
    }
}

/// Статистика словаря
#[derive(Debug, Clone)]
pub struct LexiconStats {
    pub positive_count: usize,
    pub negative_count: usize,
    pub modifier_count: usize,
    pub negation_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_lexicon_positive() {
        let lexicon = CryptoLexicon::new();

        assert!(lexicon.get_score("moon").unwrap() > 0.5);
        assert!(lexicon.get_score("bullish").unwrap() > 0.5);
        assert!(lexicon.get_score("MOON").unwrap() > 0.5); // Case insensitive
    }

    #[test]
    fn test_crypto_lexicon_negative() {
        let lexicon = CryptoLexicon::new();

        assert!(lexicon.get_score("crash").unwrap() < -0.5);
        assert!(lexicon.get_score("scam").unwrap() < -0.5);
    }

    #[test]
    fn test_negation_detection() {
        let lexicon = CryptoLexicon::new();

        assert!(lexicon.is_negation("not"));
        assert!(lexicon.is_negation("don't"));
        assert!(!lexicon.is_negation("moon"));
    }

    #[test]
    fn test_modifier() {
        let lexicon = CryptoLexicon::new();

        assert!(lexicon.get_modifier("very").unwrap() > 1.0);
        assert!(lexicon.get_modifier("slightly").unwrap() < 1.0);
    }
}
