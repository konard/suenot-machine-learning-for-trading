//! Анализатор настроений
//!
//! Модуль для анализа настроений в текстах.
//! В реальном приложении здесь можно интегрировать
//! FinBERT или другие NLP-модели.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Оценка настроения
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SentimentScore {
    /// Оценка от -1.0 (негатив) до 1.0 (позитив)
    pub score: f64,
    /// Уверенность модели от 0.0 до 1.0
    pub confidence: f64,
    /// Вероятность позитивного настроения
    pub positive_prob: f64,
    /// Вероятность нейтрального настроения
    pub neutral_prob: f64,
    /// Вероятность негативного настроения
    pub negative_prob: f64,
}

impl SentimentScore {
    /// Создание новой оценки
    pub fn new(positive_prob: f64, neutral_prob: f64, negative_prob: f64) -> Self {
        let score = positive_prob - negative_prob;
        let confidence = positive_prob.max(neutral_prob).max(negative_prob);

        Self {
            score,
            confidence,
            positive_prob,
            neutral_prob,
            negative_prob,
        }
    }

    /// Быстрое создание оценки из одного значения
    pub fn from_score(score: f64) -> Self {
        let score = score.clamp(-1.0, 1.0);

        // Приблизительное распределение вероятностей
        let (positive_prob, neutral_prob, negative_prob) = if score > 0.0 {
            let pos = 0.5 + score * 0.4;
            let neg = 0.5 - score * 0.4;
            (pos, 0.1, neg.max(0.0))
        } else {
            let neg = 0.5 - score * 0.4;
            let pos = 0.5 + score * 0.4;
            (pos.max(0.0), 0.1, neg)
        };

        Self {
            score,
            confidence: 0.7 + score.abs() * 0.25,
            positive_prob,
            neutral_prob,
            negative_prob,
        }
    }

    /// Является ли настроение позитивным
    pub fn is_positive(&self) -> bool {
        self.score > 0.1
    }

    /// Является ли настроение негативным
    pub fn is_negative(&self) -> bool {
        self.score < -0.1
    }

    /// Является ли настроение нейтральным
    pub fn is_neutral(&self) -> bool {
        self.score.abs() <= 0.1
    }

    /// Метка настроения
    pub fn label(&self) -> &'static str {
        if self.is_positive() {
            "positive"
        } else if self.is_negative() {
            "negative"
        } else {
            "neutral"
        }
    }
}

impl Default for SentimentScore {
    fn default() -> Self {
        Self::new(0.33, 0.34, 0.33)
    }
}

/// Анализатор настроений на основе словаря
///
/// Простая реализация для демонстрации.
/// В продакшене следует использовать FinBERT или подобные модели.
pub struct SentimentAnalyzer {
    /// Позитивные слова и их веса
    positive_words: HashMap<String, f64>,
    /// Негативные слова и их веса
    negative_words: HashMap<String, f64>,
    /// Усилители (very, extremely, etc.)
    intensifiers: HashMap<String, f64>,
    /// Отрицания
    negations: Vec<String>,
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentAnalyzer {
    /// Создание нового анализатора с криптовалютным словарём
    pub fn new() -> Self {
        let mut positive_words = HashMap::new();
        let mut negative_words = HashMap::new();

        // Позитивные слова для крипто-контекста
        for (word, weight) in [
            // Английские
            ("moon", 0.9),
            ("bullish", 0.85),
            ("pump", 0.7),
            ("hodl", 0.6),
            ("buy", 0.5),
            ("long", 0.5),
            ("growth", 0.6),
            ("profit", 0.7),
            ("gains", 0.7),
            ("up", 0.4),
            ("rise", 0.5),
            ("rocket", 0.8),
            ("lambo", 0.7),
            ("ath", 0.8), // all-time high
            ("breakout", 0.7),
            ("support", 0.4),
            ("accumulate", 0.5),
            ("strong", 0.5),
            ("winning", 0.7),
            ("success", 0.7),
            // Русские
            ("рост", 0.6),
            ("покупка", 0.5),
            ("прибыль", 0.7),
            ("луна", 0.9),
            ("ракета", 0.8),
            ("бычий", 0.85),
            ("отлично", 0.7),
            ("супер", 0.6),
            ("топ", 0.5),
        ] {
            positive_words.insert(word.to_lowercase(), weight);
        }

        // Негативные слова
        for (word, weight) in [
            // Английские
            ("dump", -0.8),
            ("crash", -0.9),
            ("bearish", -0.85),
            ("sell", -0.5),
            ("short", -0.5),
            ("loss", -0.7),
            ("scam", -0.9),
            ("rug", -0.95), // rug pull
            ("rekt", -0.8),
            ("down", -0.4),
            ("fall", -0.5),
            ("fear", -0.6),
            ("panic", -0.8),
            ("drop", -0.6),
            ("correction", -0.4),
            ("resistance", -0.3),
            ("weak", -0.5),
            ("losing", -0.7),
            ("failed", -0.7),
            ("fud", -0.6), // fear, uncertainty, doubt
            // Русские
            ("падение", -0.6),
            ("продажа", -0.5),
            ("убыток", -0.7),
            ("скам", -0.9),
            ("медвежий", -0.85),
            ("ужас", -0.8),
            ("паника", -0.8),
            ("слив", -0.7),
        ] {
            negative_words.insert(word.to_lowercase(), weight);
        }

        let intensifiers = [
            ("very", 1.5),
            ("extremely", 2.0),
            ("super", 1.5),
            ("absolutely", 1.8),
            ("totally", 1.5),
            ("очень", 1.5),
            ("крайне", 2.0),
        ]
        .iter()
        .map(|(k, v)| (k.to_lowercase(), *v))
        .collect();

        let negations = vec![
            "not", "no", "never", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't",
            "won't", "не", "нет", "никогда",
        ]
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect();

        Self {
            positive_words,
            negative_words,
            intensifiers,
            negations,
        }
    }

    /// Анализ текста
    pub fn analyze(&self, text: &str) -> SentimentScore {
        let words: Vec<String> = text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        if words.is_empty() {
            return SentimentScore::default();
        }

        let mut total_score = 0.0;
        let mut word_count = 0;
        let mut negation_active = false;
        let mut intensifier = 1.0;

        for (i, word) in words.iter().enumerate() {
            // Проверяем отрицание
            if self.negations.contains(word) {
                negation_active = true;
                continue;
            }

            // Проверяем усилитель
            if let Some(&int_value) = self.intensifiers.get(word) {
                intensifier = int_value;
                continue;
            }

            // Проверяем позитивные слова
            if let Some(&weight) = self.positive_words.get(word) {
                let mut score = weight * intensifier;
                if negation_active {
                    score = -score * 0.5; // Отрицание снижает интенсивность
                }
                total_score += score;
                word_count += 1;
            }

            // Проверяем негативные слова
            if let Some(&weight) = self.negative_words.get(word) {
                let mut score = weight * intensifier;
                if negation_active {
                    score = -score * 0.5;
                }
                total_score += score;
                word_count += 1;
            }

            // Сбрасываем модификаторы после использования
            if i > 0 {
                negation_active = false;
                intensifier = 1.0;
            }
        }

        if word_count == 0 {
            return SentimentScore::default();
        }

        let avg_score = (total_score / word_count as f64).clamp(-1.0, 1.0);
        SentimentScore::from_score(avg_score)
    }

    /// Пакетный анализ нескольких текстов
    pub fn analyze_batch(&self, texts: &[&str]) -> Vec<SentimentScore> {
        texts.iter().map(|text| self.analyze(text)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_sentiment() {
        let analyzer = SentimentAnalyzer::new();

        let score = analyzer.analyze("Bitcoin to the moon! Very bullish!");
        assert!(score.is_positive());
        assert!(score.score > 0.5);
    }

    #[test]
    fn test_negative_sentiment() {
        let analyzer = SentimentAnalyzer::new();

        let score = analyzer.analyze("Market crash! Panic selling everywhere!");
        assert!(score.is_negative());
        assert!(score.score < -0.5);
    }

    #[test]
    fn test_neutral_sentiment() {
        let analyzer = SentimentAnalyzer::new();

        let score = analyzer.analyze("The price is at 50000 today");
        assert!(score.is_neutral() || score.score.abs() < 0.3);
    }

    #[test]
    fn test_negation() {
        let analyzer = SentimentAnalyzer::new();

        let positive = analyzer.analyze("This is bullish");
        let negated = analyzer.analyze("This is not bullish");

        assert!(positive.score > negated.score);
    }

    #[test]
    fn test_russian_sentiment() {
        let analyzer = SentimentAnalyzer::new();

        let score = analyzer.analyze("Крипта на луну! Рост продолжается!");
        assert!(score.is_positive());

        let score = analyzer.analyze("Ужас! Паника на рынке!");
        assert!(score.is_negative());
    }
}
