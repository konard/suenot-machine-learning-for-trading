//! Анализатор настроений
//!
//! Определяет полярность текста на основе лексикона

use crate::models::{Polarity, ScoredWord, SentimentResult};
use crate::nlp::{Preprocessor, Tokenizer};
use crate::sentiment::lexicon::{CryptoLexicon, SentimentLexicon};

/// Анализатор настроений
#[derive(Debug, Clone)]
pub struct SentimentAnalyzer {
    /// Токенизатор
    tokenizer: Tokenizer,
    /// Предобработчик
    preprocessor: Preprocessor,
    /// Лексикон настроений
    lexicon: CryptoLexicon,
    /// Окно для обнаружения отрицаний (слов до sentiment слова)
    negation_window: usize,
}

impl SentimentAnalyzer {
    /// Создать новый анализатор
    pub fn new() -> Self {
        Self {
            tokenizer: Tokenizer::new(),
            preprocessor: Preprocessor::new().with_stemming(false), // Не стемим для sentiment
            lexicon: CryptoLexicon::new(),
            negation_window: 3,
        }
    }

    /// Установить пользовательский лексикон
    pub fn with_lexicon(mut self, lexicon: CryptoLexicon) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Установить окно отрицания
    pub fn with_negation_window(mut self, window: usize) -> Self {
        self.negation_window = window;
        self
    }

    /// Анализировать текст
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let tokens = self.tokenizer.tokenize_to_strings(text);

        let mut total_score = 0.0;
        let mut word_count = 0;
        let mut key_words = Vec::new();
        let mut current_modifier = 1.0;
        let mut negation_active = false;
        let mut words_since_negation = 0;

        for (i, token) in tokens.iter().enumerate() {
            let token_lower = token.to_lowercase();

            // Проверяем отрицание
            if self.lexicon.is_negation(&token_lower) {
                negation_active = true;
                words_since_negation = 0;
                continue;
            }

            // Проверяем модификатор
            if let Some(modifier) = self.lexicon.get_modifier(&token_lower) {
                current_modifier = modifier;
                continue;
            }

            // Проверяем sentiment слово
            if let Some(base_score) = self.lexicon.get_score(&token_lower) {
                let mut score = base_score * current_modifier;

                // Применяем отрицание если оно активно и в пределах окна
                if negation_active && words_since_negation < self.negation_window {
                    score = -score * 0.8; // Инвертируем с небольшим затуханием
                }

                total_score += score;
                word_count += 1;

                key_words.push(ScoredWord {
                    word: token.clone(),
                    score,
                });

                // Сбрасываем модификатор после использования
                current_modifier = 1.0;
            }

            // Увеличиваем счётчик слов после отрицания
            if negation_active {
                words_since_negation += 1;
                if words_since_negation >= self.negation_window {
                    negation_active = false;
                }
            }
        }

        // Нормализуем оценку
        let normalized_score = if word_count > 0 {
            (total_score / word_count as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Вычисляем уверенность на основе количества найденных слов
        let confidence = self.calculate_confidence(word_count, tokens.len());

        SentimentResult {
            text: text.to_string(),
            polarity: Polarity::from_score(normalized_score),
            score: normalized_score,
            confidence,
            key_words,
        }
    }

    /// Анализировать несколько текстов
    pub fn analyze_batch(&self, texts: &[String]) -> Vec<SentimentResult> {
        texts.iter().map(|t| self.analyze(t)).collect()
    }

    /// Агрегировать результаты анализа
    pub fn aggregate(&self, results: &[SentimentResult]) -> AggregatedSentiment {
        if results.is_empty() {
            return AggregatedSentiment::default();
        }

        let total_score: f64 = results.iter().map(|r| r.score).sum();
        let avg_score = total_score / results.len() as f64;

        let total_confidence: f64 = results.iter().map(|r| r.confidence).sum();
        let avg_confidence = total_confidence / results.len() as f64;

        let positive_count = results.iter().filter(|r| r.polarity == Polarity::Positive).count();
        let negative_count = results.iter().filter(|r| r.polarity == Polarity::Negative).count();
        let neutral_count = results.iter().filter(|r| r.polarity == Polarity::Neutral).count();

        // Собираем топ ключевые слова
        let mut all_words: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for result in results {
            for word in &result.key_words {
                *all_words.entry(word.word.clone()).or_insert(0.0) += word.score;
            }
        }

        let mut top_words: Vec<_> = all_words.into_iter().collect();
        top_words.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        top_words.truncate(10);

        AggregatedSentiment {
            average_score: avg_score,
            average_confidence: avg_confidence,
            total_count: results.len(),
            positive_count,
            negative_count,
            neutral_count,
            top_keywords: top_words,
            overall_polarity: Polarity::from_score(avg_score),
        }
    }

    /// Вычислить уверенность на основе количества sentiment слов
    fn calculate_confidence(&self, sentiment_words: usize, total_words: usize) -> f64 {
        if total_words == 0 {
            return 0.0;
        }

        // Базовая уверенность от отношения sentiment слов к общему количеству
        let ratio = sentiment_words as f64 / total_words as f64;

        // Бонус за наличие нескольких sentiment слов
        let word_bonus = (sentiment_words as f64).min(5.0) / 5.0;

        // Комбинируем
        ((ratio * 0.5 + word_bonus * 0.5) * 100.0).min(100.0) / 100.0
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Агрегированные результаты анализа настроений
#[derive(Debug, Clone)]
pub struct AggregatedSentiment {
    /// Средняя оценка
    pub average_score: f64,
    /// Средняя уверенность
    pub average_confidence: f64,
    /// Общее количество текстов
    pub total_count: usize,
    /// Количество позитивных
    pub positive_count: usize,
    /// Количество негативных
    pub negative_count: usize,
    /// Количество нейтральных
    pub neutral_count: usize,
    /// Топ ключевые слова
    pub top_keywords: Vec<(String, f64)>,
    /// Общая полярность
    pub overall_polarity: Polarity,
}

impl Default for AggregatedSentiment {
    fn default() -> Self {
        Self {
            average_score: 0.0,
            average_confidence: 0.0,
            total_count: 0,
            positive_count: 0,
            negative_count: 0,
            neutral_count: 0,
            top_keywords: Vec::new(),
            overall_polarity: Polarity::Neutral,
        }
    }
}

impl std::fmt::Display for AggregatedSentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let polarity_emoji = match self.overall_polarity {
            Polarity::Positive => "📈",
            Polarity::Negative => "📉",
            Polarity::Neutral => "➡️",
        };

        write!(
            f,
            "{} Overall: {:?} (score: {:.2}, confidence: {:.0}%)\n\
             Analyzed: {} texts ({} positive, {} negative, {} neutral)",
            polarity_emoji,
            self.overall_polarity,
            self.average_score,
            self.average_confidence * 100.0,
            self.total_count,
            self.positive_count,
            self.negative_count,
            self.neutral_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_sentiment() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("Bitcoin is mooning! Very bullish!");

        assert_eq!(result.polarity, Polarity::Positive);
        assert!(result.score > 0.5);
    }

    #[test]
    fn test_negative_sentiment() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("Market is crashing, this is a scam!");

        assert_eq!(result.polarity, Polarity::Negative);
        assert!(result.score < -0.5);
    }

    #[test]
    fn test_negation() {
        let analyzer = SentimentAnalyzer::new();

        // "not bullish" should be negative
        let result = analyzer.analyze("This is not bullish at all");
        assert!(result.score < 0.0 || result.polarity != Polarity::Positive);
    }

    #[test]
    fn test_modifier() {
        let analyzer = SentimentAnalyzer::new();

        let normal = analyzer.analyze("bullish");
        let intensified = analyzer.analyze("very bullish");

        // "very bullish" should have higher score
        assert!(intensified.score > normal.score);
    }

    #[test]
    fn test_aggregate() {
        let analyzer = SentimentAnalyzer::new();
        let results = vec![
            analyzer.analyze("Very bullish on BTC!"),
            analyzer.analyze("ETH is mooning!"),
            analyzer.analyze("Market looks good"),
        ];

        let aggregated = analyzer.aggregate(&results);

        assert_eq!(aggregated.total_count, 3);
        assert_eq!(aggregated.overall_polarity, Polarity::Positive);
    }
}
