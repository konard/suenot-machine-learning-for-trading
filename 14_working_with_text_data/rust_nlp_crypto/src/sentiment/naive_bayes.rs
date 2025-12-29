//! Наивный Байесовский классификатор
//!
//! Реализация мультиномиального Naive Bayes для классификации текста

use std::collections::HashMap;

/// Наивный Байесовский классификатор
#[derive(Debug, Clone)]
pub struct NaiveBayesClassifier {
    /// Вероятности классов P(class)
    class_probs: HashMap<String, f64>,
    /// Вероятности слов P(word|class) для каждого класса
    word_probs: HashMap<String, HashMap<String, f64>>,
    /// Словарь всех слов
    vocabulary: Vec<String>,
    /// Сглаживание Лапласа
    alpha: f64,
    /// Обучена ли модель
    trained: bool,
}

impl NaiveBayesClassifier {
    /// Создать новый классификатор
    pub fn new() -> Self {
        Self {
            class_probs: HashMap::new(),
            word_probs: HashMap::new(),
            vocabulary: Vec::new(),
            alpha: 1.0, // Laplace smoothing
            trained: false,
        }
    }

    /// Установить параметр сглаживания
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Обучить классификатор
    ///
    /// # Arguments
    /// * `documents` - Список токенизированных документов
    /// * `labels` - Метки классов для каждого документа
    pub fn fit(&mut self, documents: &[Vec<String>], labels: &[String]) {
        assert_eq!(documents.len(), labels.len(), "Documents and labels must have same length");

        // Подсчёт классов
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for label in labels {
            *class_counts.entry(label.clone()).or_insert(0) += 1;
        }

        let total_docs = labels.len() as f64;

        // Вычисляем P(class)
        for (class, count) in &class_counts {
            self.class_probs.insert(class.clone(), *count as f64 / total_docs);
        }

        // Собираем словарь
        let mut vocab_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        for doc in documents {
            for word in doc {
                vocab_set.insert(word.clone());
            }
        }
        self.vocabulary = vocab_set.into_iter().collect();
        self.vocabulary.sort();

        let vocab_size = self.vocabulary.len() as f64;

        // Подсчёт слов для каждого класса
        let mut class_word_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut class_total_words: HashMap<String, usize> = HashMap::new();

        for (doc, label) in documents.iter().zip(labels.iter()) {
            let word_counts = class_word_counts.entry(label.clone()).or_insert_with(HashMap::new);
            let total = class_total_words.entry(label.clone()).or_insert(0);

            for word in doc {
                *word_counts.entry(word.clone()).or_insert(0) += 1;
                *total += 1;
            }
        }

        // Вычисляем P(word|class) с Laplace smoothing
        for (class, word_counts) in &class_word_counts {
            let total_words = *class_total_words.get(class).unwrap() as f64;
            let mut probs = HashMap::new();

            for word in &self.vocabulary {
                let count = *word_counts.get(word).unwrap_or(&0) as f64;
                // P(word|class) = (count + alpha) / (total + alpha * vocab_size)
                let prob = (count + self.alpha) / (total_words + self.alpha * vocab_size);
                probs.insert(word.clone(), prob.ln()); // Храним log для численной стабильности
            }

            self.word_probs.insert(class.clone(), probs);
        }

        self.trained = true;
    }

    /// Предсказать класс для документа
    pub fn predict(&self, document: &[String]) -> Option<String> {
        if !self.trained {
            return None;
        }

        self.predict_proba(document)
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(class, _)| class)
    }

    /// Получить вероятности всех классов для документа
    pub fn predict_proba(&self, document: &[String]) -> HashMap<String, f64> {
        let mut scores: HashMap<String, f64> = HashMap::new();

        for (class, class_prob) in &self.class_probs {
            let mut log_prob = class_prob.ln(); // log P(class)

            // Добавляем log P(word|class) для каждого слова
            if let Some(word_probs) = self.word_probs.get(class) {
                for word in document {
                    if let Some(&log_word_prob) = word_probs.get(word) {
                        log_prob += log_word_prob;
                    }
                    // Игнорируем неизвестные слова
                }
            }

            scores.insert(class.clone(), log_prob);
        }

        // Нормализуем в вероятности (softmax)
        let max_score = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = scores.values().map(|s| (s - max_score).exp()).sum();

        for (_, score) in scores.iter_mut() {
            *score = ((*score - max_score).exp()) / sum_exp;
        }

        scores
    }

    /// Предсказать для нескольких документов
    pub fn predict_batch(&self, documents: &[Vec<String>]) -> Vec<Option<String>> {
        documents.iter().map(|doc| self.predict(doc)).collect()
    }

    /// Оценить точность на тестовых данных
    pub fn score(&self, documents: &[Vec<String>], labels: &[String]) -> f64 {
        let predictions = self.predict_batch(documents);

        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(pred, label)| pred.as_ref() == Some(label))
            .count();

        correct as f64 / labels.len() as f64
    }

    /// Получить классы
    pub fn classes(&self) -> Vec<&String> {
        self.class_probs.keys().collect()
    }

    /// Получить размер словаря
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

impl Default for NaiveBayesClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Готовый классификатор для sentiment analysis
pub struct SentimentNaiveBayes {
    classifier: NaiveBayesClassifier,
}

impl SentimentNaiveBayes {
    /// Создать новый классификатор настроений
    pub fn new() -> Self {
        Self {
            classifier: NaiveBayesClassifier::new(),
        }
    }

    /// Обучить на размеченных данных
    pub fn fit(&mut self, texts: &[Vec<String>], sentiments: &[&str]) {
        let labels: Vec<String> = sentiments.iter().map(|s| s.to_string()).collect();
        self.classifier.fit(texts, &labels);
    }

    /// Предсказать sentiment
    pub fn predict(&self, document: &[String]) -> Sentiment {
        match self.classifier.predict(document).as_deref() {
            Some("positive") => Sentiment::Positive,
            Some("negative") => Sentiment::Negative,
            _ => Sentiment::Neutral,
        }
    }

    /// Получить вероятности
    pub fn predict_proba(&self, document: &[String]) -> SentimentProba {
        let probs = self.classifier.predict_proba(document);
        SentimentProba {
            positive: *probs.get("positive").unwrap_or(&0.0),
            neutral: *probs.get("neutral").unwrap_or(&0.0),
            negative: *probs.get("negative").unwrap_or(&0.0),
        }
    }
}

impl Default for SentimentNaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

/// Результат классификации sentiment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sentiment {
    Positive,
    Neutral,
    Negative,
}

/// Вероятности sentiment
#[derive(Debug, Clone)]
pub struct SentimentProba {
    pub positive: f64,
    pub neutral: f64,
    pub negative: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_bayes_basic() {
        let documents = vec![
            vec!["good".to_string(), "great".to_string(), "awesome".to_string()],
            vec!["excellent".to_string(), "amazing".to_string()],
            vec!["bad".to_string(), "terrible".to_string()],
            vec!["awful".to_string(), "horrible".to_string(), "worst".to_string()],
        ];
        let labels = vec![
            "positive".to_string(),
            "positive".to_string(),
            "negative".to_string(),
            "negative".to_string(),
        ];

        let mut classifier = NaiveBayesClassifier::new();
        classifier.fit(&documents, &labels);

        // Тест на позитивном документе
        let positive_doc = vec!["good".to_string(), "great".to_string()];
        let prediction = classifier.predict(&positive_doc);
        assert_eq!(prediction, Some("positive".to_string()));

        // Тест на негативном документе
        let negative_doc = vec!["bad".to_string(), "terrible".to_string()];
        let prediction = classifier.predict(&negative_doc);
        assert_eq!(prediction, Some("negative".to_string()));
    }

    #[test]
    fn test_naive_bayes_probabilities() {
        let documents = vec![
            vec!["moon".to_string()],
            vec!["crash".to_string()],
        ];
        let labels = vec!["positive".to_string(), "negative".to_string()];

        let mut classifier = NaiveBayesClassifier::new();
        classifier.fit(&documents, &labels);

        let probs = classifier.predict_proba(&["moon".to_string()]);

        assert!(probs.get("positive").unwrap() > probs.get("negative").unwrap());
    }

    #[test]
    fn test_accuracy_score() {
        let train_docs = vec![
            vec!["buy".to_string()],
            vec!["buy".to_string()],
            vec!["sell".to_string()],
            vec!["sell".to_string()],
        ];
        let train_labels = vec![
            "positive".to_string(),
            "positive".to_string(),
            "negative".to_string(),
            "negative".to_string(),
        ];

        let mut classifier = NaiveBayesClassifier::new();
        classifier.fit(&train_docs, &train_labels);

        let test_docs = vec![
            vec!["buy".to_string()],
            vec!["sell".to_string()],
        ];
        let test_labels = vec!["positive".to_string(), "negative".to_string()];

        let accuracy = classifier.score(&test_docs, &test_labels);
        assert_eq!(accuracy, 1.0); // Perfect accuracy on simple test
    }
}
