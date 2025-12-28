//! Векторизация текста
//!
//! Преобразование текста в числовые векторы:
//! - Bag of Words (мешок слов)
//! - TF-IDF (Term Frequency - Inverse Document Frequency)

use crate::models::DocumentTermMatrix;
use std::collections::HashMap;

/// Трейт для векторизаторов
pub trait Vectorizer {
    /// Обучить векторизатор на корпусе документов
    fn fit(&mut self, documents: &[Vec<String>]);

    /// Преобразовать документ в вектор
    fn transform(&self, document: &[String]) -> Vec<f64>;

    /// Обучить и преобразовать
    fn fit_transform(&mut self, documents: &[Vec<String>]) -> DocumentTermMatrix;

    /// Получить словарь
    fn vocabulary(&self) -> &HashMap<String, usize>;
}

/// Bag of Words векторизатор
#[derive(Debug, Clone)]
pub struct BagOfWords {
    /// Словарь: слово -> индекс
    vocabulary: HashMap<String, usize>,
    /// Обратный словарь: индекс -> слово
    terms: Vec<String>,
    /// Минимальная частота документа
    min_df: usize,
    /// Максимальная частота документа (доля)
    max_df: f64,
    /// Бинарный режим (1 если слово есть, 0 если нет)
    binary: bool,
}

impl BagOfWords {
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            terms: Vec::new(),
            min_df: 1,
            max_df: 1.0,
            binary: false,
        }
    }

    /// Установить минимальную частоту документа
    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Установить максимальную частоту документа (доля от общего числа документов)
    pub fn with_max_df(mut self, max_df: f64) -> Self {
        self.max_df = max_df;
        self
    }

    /// Бинарный режим
    pub fn with_binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Получить количество терминов
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }
}

impl Default for BagOfWords {
    fn default() -> Self {
        Self::new()
    }
}

impl Vectorizer for BagOfWords {
    fn fit(&mut self, documents: &[Vec<String>]) {
        // Подсчитываем частоту документов для каждого терма
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let unique_terms: std::collections::HashSet<_> = doc.iter().collect();
            for term in unique_terms {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Фильтруем по min_df и max_df
        let n_docs = documents.len();
        let max_count = (self.max_df * n_docs as f64).ceil() as usize;

        self.vocabulary.clear();
        self.terms.clear();

        let mut filtered_terms: Vec<_> = doc_freq
            .into_iter()
            .filter(|(_, count)| *count >= self.min_df && *count <= max_count)
            .collect();

        // Сортируем для детерминированного порядка
        filtered_terms.sort_by(|a, b| a.0.cmp(&b.0));

        for (idx, (term, _)) in filtered_terms.into_iter().enumerate() {
            self.vocabulary.insert(term.clone(), idx);
            self.terms.push(term);
        }
    }

    fn transform(&self, document: &[String]) -> Vec<f64> {
        let mut vector = vec![0.0; self.terms.len()];

        for term in document {
            if let Some(&idx) = self.vocabulary.get(term) {
                if self.binary {
                    vector[idx] = 1.0;
                } else {
                    vector[idx] += 1.0;
                }
            }
        }

        vector
    }

    fn fit_transform(&mut self, documents: &[Vec<String>]) -> DocumentTermMatrix {
        self.fit(documents);

        let matrix: Vec<Vec<f64>> = documents.iter().map(|doc| self.transform(doc)).collect();

        let doc_names: Vec<String> = (0..documents.len())
            .map(|i| format!("doc_{}", i))
            .collect();

        DocumentTermMatrix {
            documents: doc_names,
            vocabulary: self.vocabulary.clone(),
            terms: self.terms.clone(),
            matrix,
        }
    }

    fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }
}

/// TF-IDF векторизатор
#[derive(Debug, Clone)]
pub struct TfIdf {
    /// Базовый Bag of Words
    bow: BagOfWords,
    /// IDF значения для каждого терма
    idf: Vec<f64>,
    /// Нормализовать векторы
    normalize: bool,
    /// Сглаживание IDF (добавить 1 к знаменателю)
    smooth_idf: bool,
}

impl TfIdf {
    pub fn new() -> Self {
        Self {
            bow: BagOfWords::new(),
            idf: Vec::new(),
            normalize: true,
            smooth_idf: true,
        }
    }

    /// Установить минимальную частоту документа
    pub fn with_min_df(mut self, min_df: usize) -> Self {
        self.bow = self.bow.with_min_df(min_df);
        self
    }

    /// Установить максимальную частоту документа
    pub fn with_max_df(mut self, max_df: f64) -> Self {
        self.bow = self.bow.with_max_df(max_df);
        self
    }

    /// Нормализовать векторы (L2 нормализация)
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Сглаживание IDF
    pub fn with_smooth_idf(mut self, smooth: bool) -> Self {
        self.smooth_idf = smooth;
        self
    }

    /// Рассчитать IDF
    fn calculate_idf(&mut self, documents: &[Vec<String>]) {
        let n_docs = documents.len() as f64;
        let n_terms = self.bow.n_terms();

        self.idf = vec![0.0; n_terms];

        // Подсчёт документной частоты
        let mut doc_freq = vec![0usize; n_terms];

        for doc in documents {
            let unique_terms: std::collections::HashSet<_> = doc.iter().collect();
            for term in unique_terms {
                if let Some(&idx) = self.bow.vocabulary.get(term) {
                    doc_freq[idx] += 1;
                }
            }
        }

        // Расчёт IDF
        for (idx, df) in doc_freq.into_iter().enumerate() {
            let df_smooth = if self.smooth_idf {
                df as f64 + 1.0
            } else {
                df as f64.max(1.0)
            };

            let n_smooth = if self.smooth_idf {
                n_docs + 1.0
            } else {
                n_docs
            };

            self.idf[idx] = (n_smooth / df_smooth).ln() + 1.0;
        }
    }

    /// L2 нормализация вектора
    fn l2_normalize(vector: &mut [f64]) {
        let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Получить IDF значения
    pub fn get_idf(&self) -> &[f64] {
        &self.idf
    }

    /// Получить топ-N терминов по IDF
    pub fn top_terms(&self, n: usize) -> Vec<(String, f64)> {
        let mut term_idf: Vec<_> = self
            .bow
            .terms
            .iter()
            .zip(self.idf.iter())
            .map(|(term, idf)| (term.clone(), *idf))
            .collect();

        term_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        term_idf.truncate(n);
        term_idf
    }
}

impl Default for TfIdf {
    fn default() -> Self {
        Self::new()
    }
}

impl Vectorizer for TfIdf {
    fn fit(&mut self, documents: &[Vec<String>]) {
        self.bow.fit(documents);
        self.calculate_idf(documents);
    }

    fn transform(&self, document: &[String]) -> Vec<f64> {
        let mut tf_vector = self.bow.transform(document);

        // Умножаем TF на IDF
        for (idx, tf) in tf_vector.iter_mut().enumerate() {
            *tf *= self.idf[idx];
        }

        // Нормализация
        if self.normalize {
            Self::l2_normalize(&mut tf_vector);
        }

        tf_vector
    }

    fn fit_transform(&mut self, documents: &[Vec<String>]) -> DocumentTermMatrix {
        self.fit(documents);

        let matrix: Vec<Vec<f64>> = documents.iter().map(|doc| self.transform(doc)).collect();

        let doc_names: Vec<String> = (0..documents.len())
            .map(|i| format!("doc_{}", i))
            .collect();

        DocumentTermMatrix {
            documents: doc_names,
            vocabulary: self.bow.vocabulary.clone(),
            terms: self.bow.terms.clone(),
            matrix,
        }
    }

    fn vocabulary(&self) -> &HashMap<String, usize> {
        self.bow.vocabulary()
    }
}

/// Вычисление косинусного сходства между векторами
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bag_of_words() {
        let documents = vec![
            vec!["bitcoin".to_string(), "price".to_string(), "up".to_string()],
            vec![
                "ethereum".to_string(),
                "price".to_string(),
                "down".to_string(),
            ],
        ];

        let mut bow = BagOfWords::new();
        let dtm = bow.fit_transform(&documents);

        assert_eq!(dtm.n_documents(), 2);
        assert!(dtm.n_terms() >= 4); // bitcoin, price, up, ethereum, down
    }

    #[test]
    fn test_tfidf() {
        let documents = vec![
            vec!["btc".to_string(), "moon".to_string()],
            vec!["btc".to_string(), "crash".to_string()],
            vec!["eth".to_string(), "moon".to_string()],
        ];

        let mut tfidf = TfIdf::new();
        let dtm = tfidf.fit_transform(&documents);

        assert_eq!(dtm.n_documents(), 3);

        // "moon" и "btc" встречаются в 2 документах -> более низкий IDF
        // "crash" и "eth" встречаются в 1 документе -> более высокий IDF
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 1.0];

        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001); // Идентичные векторы

        let c = vec![0.0, 1.0, 0.0];
        let sim2 = cosine_similarity(&a, &c);
        assert!((sim2 - 0.0).abs() < 0.001); // Ортогональные векторы
    }
}
