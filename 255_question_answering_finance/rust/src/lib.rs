use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Document Types ───────────────────────────────────────────────

/// A financial document with metadata.
#[derive(Debug, Clone)]
pub struct FinancialDocument {
    pub id: usize,
    pub title: String,
    pub content: String,
    pub source: String,
    pub doc_type: DocumentType,
    pub date: String,
}

/// Types of financial documents.
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentType {
    EarningsReport,
    SECFiling,
    AnalystNote,
    NewsArticle,
    MarketData,
    Whitepaper,
}

// ─── Document Store ───────────────────────────────────────────────

/// Stores and retrieves financial documents using keyword-based TF-IDF search.
#[derive(Debug)]
pub struct DocumentStore {
    documents: Vec<FinancialDocument>,
    next_id: usize,
    // Inverted index: term -> list of (doc_id, term_frequency)
    index: HashMap<String, Vec<(usize, f64)>>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            next_id: 0,
            index: HashMap::new(),
        }
    }

    /// Add a document to the store and update the index.
    pub fn add_document(
        &mut self,
        title: &str,
        content: &str,
        source: &str,
        doc_type: DocumentType,
        date: &str,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let doc = FinancialDocument {
            id,
            title: title.to_string(),
            content: content.to_string(),
            source: source.to_string(),
            doc_type,
            date: date.to_string(),
        };

        // Build inverted index
        let tokens = tokenize(&doc.content);
        let total_tokens = tokens.len() as f64;
        let mut term_counts: HashMap<String, f64> = HashMap::new();
        for token in &tokens {
            *term_counts.entry(token.clone()).or_insert(0.0) += 1.0;
        }
        for (term, count) in &term_counts {
            let tf = count / total_tokens;
            self.index
                .entry(term.clone())
                .or_default()
                .push((id, tf));
        }

        // Also index the title
        let title_tokens = tokenize(&doc.title);
        let title_total = title_tokens.len() as f64;
        let mut title_counts: HashMap<String, f64> = HashMap::new();
        for token in &title_tokens {
            *title_counts.entry(token.clone()).or_insert(0.0) += 1.0;
        }
        for (term, count) in &title_counts {
            let tf = count / title_total;
            // Boost title terms
            self.index
                .entry(term.clone())
                .or_default()
                .push((id, tf * 2.0));
        }

        self.documents.push(doc);
        id
    }

    /// Retrieve a document by ID.
    pub fn get_document(&self, id: usize) -> Option<&FinancialDocument> {
        self.documents.iter().find(|d| d.id == id)
    }

    /// Search documents by query using TF-IDF scoring.
    /// Returns (doc_id, relevance_score) pairs sorted by score descending.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f64)> {
        let query_tokens = tokenize(query);
        let num_docs = self.documents.len() as f64;
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings) = self.index.get(token) {
                // IDF = log(N / df)
                let df = postings.len() as f64;
                let idf = (num_docs / df).ln().max(0.0);

                for &(doc_id, tf) in postings {
                    *scores.entry(doc_id).or_insert(0.0) += tf * idf;
                }
            }
        }

        let mut results: Vec<(usize, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Number of documents in the store.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

impl Default for DocumentStore {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Question Classification ──────────────────────────────────────

/// Types of financial questions.
#[derive(Debug, Clone, PartialEq)]
pub enum QuestionType {
    Factoid,      // "What was the revenue?"
    Numerical,    // "What is the growth rate?"
    Boolean,      // "Did revenue increase?"
    Comparative,  // "Which had higher growth?"
    Temporal,     // "When did they first report profit?"
}

/// Classifies questions by type using keyword heuristics.
#[derive(Debug)]
pub struct QuestionClassifier;

impl QuestionClassifier {
    pub fn new() -> Self {
        Self
    }

    /// Classify a question into one of the supported types.
    pub fn classify(&self, question: &str) -> QuestionType {
        let q = question.to_lowercase();

        // Boolean patterns
        if q.starts_with("did ")
            || q.starts_with("does ")
            || q.starts_with("is ")
            || q.starts_with("are ")
            || q.starts_with("was ")
            || q.starts_with("were ")
            || q.starts_with("has ")
            || q.starts_with("have ")
            || q.starts_with("can ")
            || q.starts_with("will ")
        {
            return QuestionType::Boolean;
        }

        // Temporal patterns
        if q.starts_with("when ")
            || q.contains("what date")
            || q.contains("what year")
            || q.contains("what quarter")
            || q.contains("since when")
        {
            return QuestionType::Temporal;
        }

        // Comparative patterns
        if q.contains("which ")
            && (q.contains("higher")
                || q.contains("lower")
                || q.contains("more")
                || q.contains("less")
                || q.contains("better")
                || q.contains("worse")
                || q.contains("largest")
                || q.contains("smallest"))
        {
            return QuestionType::Comparative;
        }
        if q.contains("compare") || q.contains("versus") || q.contains(" vs ") {
            return QuestionType::Comparative;
        }

        // Numerical patterns
        if q.contains("growth")
            || q.contains("percentage")
            || q.contains("ratio")
            || q.contains("margin")
            || q.contains("how much")
            || q.contains("how many")
            || q.contains("calculate")
            || q.contains("compute")
            || q.contains("change")
            || q.contains("increase")
            || q.contains("decrease")
            || q.contains("difference")
            || q.contains("average")
            || q.contains("total")
            || q.contains("sum of")
        {
            return QuestionType::Numerical;
        }

        // Default to factoid
        QuestionType::Factoid
    }
}

impl Default for QuestionClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Extractive QA ────────────────────────────────────────────────

/// Answer from the extractive QA module.
#[derive(Debug, Clone)]
pub struct QAAnswer {
    pub text: String,
    pub confidence: f64,
    pub source_doc_id: Option<usize>,
    pub question_type: QuestionType,
}

/// Extracts answer spans from context using keyword matching and proximity.
#[derive(Debug)]
pub struct ExtractiveQA;

impl ExtractiveQA {
    pub fn new() -> Self {
        Self
    }

    /// Extract an answer from a context passage given a question.
    pub fn extract(&self, question: &str, context: &str) -> QAAnswer {
        let question_tokens = tokenize(question);
        let sentences = split_sentences(context);

        if sentences.is_empty() {
            return QAAnswer {
                text: String::new(),
                confidence: 0.0,
                source_doc_id: None,
                question_type: QuestionType::Factoid,
            };
        }

        // Score each sentence by keyword overlap with the question
        let mut best_score = 0.0_f64;
        let mut best_sentence = &sentences[0];

        for sentence in &sentences {
            let sent_tokens: Vec<String> = tokenize(sentence);
            let overlap: f64 = question_tokens
                .iter()
                .filter(|qt| sent_tokens.contains(qt))
                .count() as f64;
            // Normalize by question length
            let score = if !question_tokens.is_empty() {
                overlap / question_tokens.len() as f64
            } else {
                0.0
            };

            if score > best_score {
                best_score = score;
                best_sentence = sentence;
            }
        }

        // Extract a focused answer span from the best sentence
        let answer_span = extract_answer_span(question, best_sentence);
        let confidence = (best_score * 0.8).min(0.99);

        QAAnswer {
            text: answer_span,
            confidence,
            source_doc_id: None,
            question_type: QuestionType::Factoid,
        }
    }
}

impl Default for ExtractiveQA {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Numerical Reasoner ──────────────────────────────────────────

/// Supported numerical operations.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericalOp {
    GrowthRate,
    Ratio,
    Difference,
    Sum,
    Average,
    Percentage,
}

/// Handles questions requiring numerical computation.
#[derive(Debug)]
pub struct NumericalReasoner;

impl NumericalReasoner {
    pub fn new() -> Self {
        Self
    }

    /// Parse financial numbers from text.
    /// Handles currency symbols, percentages, and magnitude suffixes.
    pub fn parse_numbers(&self, text: &str) -> Vec<f64> {
        let mut numbers = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            // Skip currency symbols
            if chars[i] == '$' || chars[i] == '€' || chars[i] == '£' {
                i += 1;
                continue;
            }

            // Try to parse a number
            if chars[i].is_ascii_digit() || (chars[i] == '-' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
                let start = i;
                if chars[i] == '-' {
                    i += 1;
                }
                while i < len && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == ',') {
                    // Skip commas used as thousand separators
                    if chars[i] == ','
                        && i + 1 < len
                        && chars[i + 1].is_ascii_digit()
                        && i > start
                        && chars[i - 1].is_ascii_digit()
                    {
                        i += 1;
                        continue;
                    }
                    i += 1;
                }

                let num_str: String = chars[start..i]
                    .iter()
                    .filter(|c| **c != ',')
                    .collect();

                if let Ok(mut num) = num_str.parse::<f64>() {
                    // Check for magnitude suffix
                    let rest: String = chars[i..].iter().collect();
                    let rest_lower = rest.to_lowercase();
                    if rest_lower.starts_with(" billion") || rest_lower.starts_with("b ") {
                        num *= 1_000_000_000.0;
                        i += if rest_lower.starts_with(" billion") { 8 } else { 2 };
                    } else if rest_lower.starts_with(" million") || rest_lower.starts_with("m ") {
                        num *= 1_000_000.0;
                        i += if rest_lower.starts_with(" million") { 8 } else { 2 };
                    } else if rest_lower.starts_with(" trillion") {
                        num *= 1_000_000_000_000.0;
                        i += 9;
                    }
                    numbers.push(num);
                }
                continue;
            }
            i += 1;
        }
        numbers
    }

    /// Detect the numerical operation required by the question.
    pub fn detect_operation(&self, question: &str) -> NumericalOp {
        let q = question.to_lowercase();

        if q.contains("growth") || q.contains("grew") || q.contains("increase rate") {
            NumericalOp::GrowthRate
        } else if q.contains("ratio") || q.contains("divided by") || q.contains("per") {
            NumericalOp::Ratio
        } else if q.contains("difference") || q.contains("change") || q.contains("how much more") {
            NumericalOp::Difference
        } else if q.contains("total") || q.contains("sum") || q.contains("combined") {
            NumericalOp::Sum
        } else if q.contains("average") || q.contains("mean") {
            NumericalOp::Average
        } else if q.contains("percentage") || q.contains("percent") || q.contains("margin") {
            NumericalOp::Percentage
        } else {
            NumericalOp::Difference
        }
    }

    /// Execute a numerical operation on extracted values.
    pub fn compute(&self, op: &NumericalOp, values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }

        match op {
            NumericalOp::GrowthRate => {
                if values.len() >= 2 && values[0] != 0.0 {
                    Some((values[1] - values[0]) / values[0].abs() * 100.0)
                } else {
                    None
                }
            }
            NumericalOp::Ratio => {
                if values.len() >= 2 && values[1] != 0.0 {
                    Some(values[0] / values[1])
                } else {
                    None
                }
            }
            NumericalOp::Difference => {
                if values.len() >= 2 {
                    Some(values[1] - values[0])
                } else {
                    None
                }
            }
            NumericalOp::Sum => {
                Some(values.iter().sum())
            }
            NumericalOp::Average => {
                if !values.is_empty() {
                    Some(values.iter().sum::<f64>() / values.len() as f64)
                } else {
                    None
                }
            }
            NumericalOp::Percentage => {
                if values.len() >= 2 && values[1] != 0.0 {
                    Some(values[0] / values[1] * 100.0)
                } else if values.len() == 1 {
                    Some(values[0])
                } else {
                    None
                }
            }
        }
    }

    /// Answer a numerical question given a context.
    pub fn answer(&self, question: &str, context: &str) -> QAAnswer {
        let numbers = self.parse_numbers(context);
        let op = self.detect_operation(question);
        let result = self.compute(&op, &numbers);

        match result {
            Some(value) => {
                let text = match &op {
                    NumericalOp::GrowthRate => format!("{:.2}%", value),
                    NumericalOp::Percentage => format!("{:.2}%", value),
                    NumericalOp::Ratio => format!("{:.4}", value),
                    _ => {
                        if value.abs() >= 1_000_000_000.0 {
                            format!("${:.2} billion", value / 1_000_000_000.0)
                        } else if value.abs() >= 1_000_000.0 {
                            format!("${:.2} million", value / 1_000_000.0)
                        } else {
                            format!("{:.2}", value)
                        }
                    }
                };
                let confidence = if numbers.len() >= 2 { 0.75 } else { 0.4 };
                QAAnswer {
                    text,
                    confidence,
                    source_doc_id: None,
                    question_type: QuestionType::Numerical,
                }
            }
            None => QAAnswer {
                text: "Unable to compute answer from available data.".to_string(),
                confidence: 0.0,
                source_doc_id: None,
                question_type: QuestionType::Numerical,
            },
        }
    }
}

impl Default for NumericalReasoner {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Financial QA Pipeline ────────────────────────────────────────

/// Full QA pipeline combining retrieval, classification, extraction,
/// and numerical reasoning.
#[derive(Debug)]
pub struct FinancialQAPipeline {
    pub store: DocumentStore,
    pub classifier: QuestionClassifier,
    pub extractor: ExtractiveQA,
    pub reasoner: NumericalReasoner,
}

impl FinancialQAPipeline {
    pub fn new() -> Self {
        Self {
            store: DocumentStore::new(),
            classifier: QuestionClassifier::new(),
            extractor: ExtractiveQA::new(),
            reasoner: NumericalReasoner::new(),
        }
    }

    /// Add a document to the pipeline's store.
    pub fn add_document(
        &mut self,
        title: &str,
        content: &str,
        source: &str,
        doc_type: DocumentType,
        date: &str,
    ) -> usize {
        self.store.add_document(title, content, source, doc_type, date)
    }

    /// Answer a question using the full pipeline.
    pub fn ask(&self, question: &str) -> Vec<QAAnswer> {
        let question_type = self.classifier.classify(question);

        // Retrieve relevant documents
        let results = self.store.search(question, 3);
        if results.is_empty() {
            return vec![QAAnswer {
                text: "No relevant documents found.".to_string(),
                confidence: 0.0,
                source_doc_id: None,
                question_type,
            }];
        }

        let mut answers = Vec::new();

        for (doc_id, relevance) in &results {
            if let Some(doc) = self.store.get_document(*doc_id) {
                let answer = match &question_type {
                    QuestionType::Numerical => {
                        let mut ans = self.reasoner.answer(question, &doc.content);
                        ans.source_doc_id = Some(*doc_id);
                        ans.confidence *= relevance.min(1.0);
                        ans
                    }
                    QuestionType::Boolean => {
                        let mut ans = self.extractor.extract(question, &doc.content);
                        ans.question_type = QuestionType::Boolean;
                        ans.source_doc_id = Some(*doc_id);
                        // Convert to yes/no
                        let q_lower = question.to_lowercase();
                        let positive_keywords = ["increase", "grew", "rise", "gain", "profit", "positive"];
                        let has_positive = positive_keywords.iter().any(|k| doc.content.to_lowercase().contains(k));
                        let is_negative_q = q_lower.contains("decrease") || q_lower.contains("decline") || q_lower.contains("loss");
                        let bool_answer = if is_negative_q { !has_positive } else { has_positive };
                        ans.text = if bool_answer { "Yes".to_string() } else { "No".to_string() };
                        ans.confidence = (ans.confidence * relevance.min(1.0)).max(0.3);
                        ans
                    }
                    _ => {
                        let mut ans = self.extractor.extract(question, &doc.content);
                        ans.question_type = question_type.clone();
                        ans.source_doc_id = Some(*doc_id);
                        ans.confidence *= relevance.min(1.0);
                        ans
                    }
                };
                answers.push(answer);
            }
        }

        // Sort by confidence descending
        answers.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        answers
    }
}

impl Default for FinancialQAPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Bybit Client ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct OrderbookResult {
    pub b: Vec<Vec<String>>, // bids: [price, size]
    pub a: Vec<Vec<String>>, // asks: [price, size]
}

/// A parsed kline bar.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Async client for Bybit V5 API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data.
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        let resp: BybitResponse<KlineResult> = self.client.get(&url).send().await?.json().await?;

        let mut klines = Vec::new();
        for item in &resp.result.list {
            if item.len() >= 6 {
                klines.push(Kline {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                });
            }
        }
        klines.reverse(); // Bybit returns newest first
        Ok(klines)
    }

    /// Fetch order book snapshot.
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> anyhow::Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );
        let resp: BybitResponse<OrderbookResult> =
            self.client.get(&url).send().await?.json().await?;

        let bids: Vec<(f64, f64)> = resp
            .result
            .b
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = resp
            .result
            .a
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok((bids, asks))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Confidence Calibration ───────────────────────────────────────

/// Measures Expected Calibration Error for QA confidence scores.
pub fn expected_calibration_error(predictions: &[(f64, bool)], num_bins: usize) -> f64 {
    if predictions.is_empty() || num_bins == 0 {
        return 0.0;
    }

    let mut ece = 0.0;
    let n = predictions.len() as f64;

    for bin_idx in 0..num_bins {
        let lower = bin_idx as f64 / num_bins as f64;
        let upper = (bin_idx + 1) as f64 / num_bins as f64;

        let bin_preds: Vec<&(f64, bool)> = predictions
            .iter()
            .filter(|(conf, _)| *conf >= lower && *conf < upper)
            .collect();

        if bin_preds.is_empty() {
            continue;
        }

        let bin_size = bin_preds.len() as f64;
        let avg_confidence: f64 = bin_preds.iter().map(|(c, _)| c).sum::<f64>() / bin_size;
        let accuracy: f64 = bin_preds.iter().filter(|(_, correct)| *correct).count() as f64 / bin_size;

        ece += (bin_size / n) * (accuracy - avg_confidence).abs();
    }

    ece
}

// ─── Utility Functions ────────────────────────────────────────────

/// Tokenize text into lowercase words, removing punctuation.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .filter(|s| !STOP_WORDS.contains(s))
        .map(String::from)
        .collect()
}

/// Split text into sentences.
pub fn split_sentences(text: &str) -> Vec<String> {
    text.split(|c: char| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extract an answer span from a sentence based on question keywords.
fn extract_answer_span(question: &str, sentence: &str) -> String {
    let q_tokens = tokenize(question);

    // Look for numbers or named entities near question keywords
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.is_empty() {
        return sentence.to_string();
    }

    // Find the position of highest keyword concentration
    let window_size = 8.min(words.len());
    let mut best_start = 0;
    let mut best_score = 0.0_f64;

    for start in 0..=words.len().saturating_sub(window_size) {
        let window: Vec<String> = words[start..start + window_size.min(words.len() - start)]
            .iter()
            .map(|w| w.to_lowercase().replace(|c: char| !c.is_alphanumeric(), ""))
            .collect();

        let score: f64 = window
            .iter()
            .filter(|w| {
                // Boost numbers and financial terms
                w.parse::<f64>().is_ok()
                    || w.starts_with('$')
                    || w.ends_with('%')
                    || q_tokens.contains(w)
            })
            .count() as f64;

        if score > best_score {
            best_score = score;
            best_start = start;
        }
    }

    let end = (best_start + window_size).min(words.len());
    words[best_start..end].join(" ")
}

/// Common English stop words to filter from tokenization.
const STOP_WORDS: &[&str] = &[
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
    "in", "with", "to", "for", "of", "by", "from", "as", "it", "that",
    "this", "was", "are", "be", "has", "had", "have", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "not", "no", "so", "if", "than", "too", "very", "can", "just",
    "its", "his", "her", "their", "our", "my", "your",
];

// ─── Synthetic Data Generation ────────────────────────────────────

/// Generate synthetic financial documents for testing.
pub fn generate_sample_documents() -> Vec<(String, String, String, DocumentType, String)> {
    vec![
        (
            "Apple Q3 2024 Earnings Report".to_string(),
            "Apple Inc. reported revenue of $94.8 billion for the third quarter of fiscal year 2024, \
             representing a 5% increase year over year. Net income was $23.6 billion, or $1.53 per \
             diluted share. The company's Services segment generated $24.2 billion in revenue, \
             reaching an all-time high. iPhone revenue was $46.3 billion. The gross margin was \
             46.3%, compared to 44.5% in the year-ago quarter. Apple returned over $32 billion \
             to shareholders through dividends and share repurchases."
                .to_string(),
            "SEC Filing".to_string(),
            DocumentType::EarningsReport,
            "2024-08-01".to_string(),
        ),
        (
            "Bitcoin Market Analysis Q1 2024".to_string(),
            "Bitcoin reached a new all-time high of $73,750 in March 2024, driven by institutional \
             demand following the approval of spot Bitcoin ETFs. Trading volume on Bybit averaged \
             $5.2 billion daily. The total crypto market capitalization exceeded $2.7 trillion. \
             Bitcoin dominance rose to 52.4%, up from 48.1% at the start of the year. The average \
             transaction fee was $8.50, while the hash rate reached 580 EH/s."
                .to_string(),
            "Market Report".to_string(),
            DocumentType::MarketData,
            "2024-04-01".to_string(),
        ),
        (
            "Ethereum DeFi Protocol Analysis".to_string(),
            "Total value locked in Ethereum DeFi protocols reached $48.5 billion. Uniswap \
             processed $1.2 trillion in cumulative trading volume. Aave held $12.3 billion in \
             deposits with an average lending rate of 3.8%. MakerDAO's DAI stablecoin had a \
             market cap of $5.4 billion with a collateralization ratio of 150%. Gas fees averaged \
             25 gwei during peak hours and 8 gwei during off-peak times."
                .to_string(),
            "DeFi Report".to_string(),
            DocumentType::Whitepaper,
            "2024-03-15".to_string(),
        ),
        (
            "S&P 500 Performance Summary 2024".to_string(),
            "The S&P 500 index returned 15.3% in the first half of 2024, with technology stocks \
             leading gains. The index reached 5,500 points. The average price-to-earnings ratio \
             was 21.5. Dividend yield for the index was 1.3%. The top performer was NVIDIA with \
             a 150% gain, while the worst performer declined 35%. Total market capitalization of \
             S&P 500 companies reached $47 trillion. Earnings per share growth was 8.2% year over year."
                .to_string(),
            "Market Analysis".to_string(),
            DocumentType::NewsArticle,
            "2024-07-01".to_string(),
        ),
        (
            "Federal Reserve Policy Impact on Markets".to_string(),
            "The Federal Reserve maintained the federal funds rate at 5.25-5.50% through Q2 2024. \
             The yield on 10-year Treasury bonds was 4.3%. Core PCE inflation declined to 2.6% \
             from 2.9% in the prior quarter. The unemployment rate was 3.7%. Market participants \
             priced in 2 rate cuts for the second half of 2024. The dollar index (DXY) traded at \
             104.5, down from 106.0 at the start of the year."
                .to_string(),
            "Economic Report".to_string(),
            DocumentType::AnalystNote,
            "2024-06-15".to_string(),
        ),
    ]
}

/// Generate synthetic QA evaluation data for calibration testing.
pub fn generate_qa_evaluation_data(n: usize) -> Vec<(f64, bool)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let confidence: f64 = rng.gen_range(0.1..1.0);
        // Simulate roughly calibrated predictions
        let correct = rng.gen::<f64>() < confidence * 0.9 + 0.05;
        data.push((confidence, correct));
    }
    data
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_store_add_and_retrieve() {
        let mut store = DocumentStore::new();
        let id = store.add_document(
            "Test Report",
            "Revenue was $100 million in 2024.",
            "SEC",
            DocumentType::EarningsReport,
            "2024-01-01",
        );
        assert_eq!(id, 0);
        assert_eq!(store.len(), 1);

        let doc = store.get_document(id).unwrap();
        assert_eq!(doc.title, "Test Report");
        assert_eq!(doc.doc_type, DocumentType::EarningsReport);
    }

    #[test]
    fn test_document_store_search() {
        let mut store = DocumentStore::new();
        store.add_document(
            "Apple Earnings",
            "Apple reported revenue of $94.8 billion for Q3 2024.",
            "SEC",
            DocumentType::EarningsReport,
            "2024-08-01",
        );
        store.add_document(
            "Bitcoin Analysis",
            "Bitcoin reached a new all-time high of $73,750.",
            "Market",
            DocumentType::MarketData,
            "2024-04-01",
        );

        let results = store.search("Apple revenue", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Apple doc should rank first
    }

    #[test]
    fn test_question_classifier() {
        let clf = QuestionClassifier::new();

        assert_eq!(clf.classify("What was Apple's revenue?"), QuestionType::Factoid);
        assert_eq!(clf.classify("What is the growth rate of revenue?"), QuestionType::Numerical);
        assert_eq!(clf.classify("Did revenue increase year over year?"), QuestionType::Boolean);
        assert_eq!(clf.classify("Which segment had higher growth?"), QuestionType::Comparative);
        assert_eq!(clf.classify("When did they first report profit?"), QuestionType::Temporal);
    }

    #[test]
    fn test_extractive_qa() {
        let qa = ExtractiveQA::new();
        let context = "Apple reported revenue of $94.8 billion for Q3 2024. Net income was $23.6 billion.";
        let answer = qa.extract("What was Apple's revenue?", context);
        assert!(!answer.text.is_empty());
        assert!(answer.confidence > 0.0);
    }

    #[test]
    fn test_numerical_reasoner_parse() {
        let reasoner = NumericalReasoner::new();

        let nums = reasoner.parse_numbers("Revenue was $94.8 billion and expenses were $71.2 billion.");
        assert!(nums.len() >= 2);
        assert!((nums[0] - 94.8e9).abs() < 1e6);
        assert!((nums[1] - 71.2e9).abs() < 1e6);
    }

    #[test]
    fn test_numerical_reasoner_parse_simple() {
        let reasoner = NumericalReasoner::new();

        let nums = reasoner.parse_numbers("The price is 100 and the target is 120.");
        assert!(nums.len() >= 2);
        assert!((nums[0] - 100.0).abs() < 1e-9);
        assert!((nums[1] - 120.0).abs() < 1e-9);
    }

    #[test]
    fn test_numerical_reasoner_operations() {
        let reasoner = NumericalReasoner::new();

        // Growth rate: (120 - 100) / 100 * 100 = 20%
        let result = reasoner.compute(&NumericalOp::GrowthRate, &[100.0, 120.0]);
        assert!((result.unwrap() - 20.0).abs() < 1e-9);

        // Ratio: 100 / 50 = 2.0
        let result = reasoner.compute(&NumericalOp::Ratio, &[100.0, 50.0]);
        assert!((result.unwrap() - 2.0).abs() < 1e-9);

        // Difference: 120 - 100 = 20
        let result = reasoner.compute(&NumericalOp::Difference, &[100.0, 120.0]);
        assert!((result.unwrap() - 20.0).abs() < 1e-9);

        // Sum: 100 + 200 + 300 = 600
        let result = reasoner.compute(&NumericalOp::Sum, &[100.0, 200.0, 300.0]);
        assert!((result.unwrap() - 600.0).abs() < 1e-9);

        // Average: (100 + 200) / 2 = 150
        let result = reasoner.compute(&NumericalOp::Average, &[100.0, 200.0]);
        assert!((result.unwrap() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_numerical_reasoner_detect_operation() {
        let reasoner = NumericalReasoner::new();

        assert_eq!(reasoner.detect_operation("What is the revenue growth?"), NumericalOp::GrowthRate);
        assert_eq!(reasoner.detect_operation("What is the P/E ratio?"), NumericalOp::Ratio);
        assert_eq!(reasoner.detect_operation("What is the total revenue?"), NumericalOp::Sum);
        assert_eq!(reasoner.detect_operation("What is the average price?"), NumericalOp::Average);
    }

    #[test]
    fn test_numerical_reasoner_answer() {
        let reasoner = NumericalReasoner::new();
        let context = "Revenue was 100 million in 2023 and 120 million in 2024.";
        let answer = reasoner.answer("What is the revenue growth?", context);
        assert!(answer.confidence > 0.0);
        assert!(answer.text.contains('%'));
    }

    #[test]
    fn test_pipeline_full() {
        let mut pipeline = FinancialQAPipeline::new();

        // Add sample documents
        let samples = generate_sample_documents();
        for (title, content, source, doc_type, date) in &samples {
            pipeline.add_document(title, content, source, doc_type.clone(), date);
        }

        // Ask a factoid question
        let answers = pipeline.ask("What was Apple's revenue in Q3 2024?");
        assert!(!answers.is_empty());
        assert!(answers[0].confidence > 0.0);

        // Ask a numerical question
        let answers = pipeline.ask("What is the growth rate of Bitcoin dominance?");
        assert!(!answers.is_empty());

        // Ask a boolean question
        let answers = pipeline.ask("Did Apple's revenue increase?");
        assert!(!answers.is_empty());
        assert!(answers[0].text == "Yes" || answers[0].text == "No");
    }

    #[test]
    fn test_pipeline_empty_store() {
        let pipeline = FinancialQAPipeline::new();
        let answers = pipeline.ask("What is the revenue?");
        assert!(!answers.is_empty());
        assert_eq!(answers[0].confidence, 0.0);
    }

    #[test]
    fn test_calibration_error() {
        // Perfect calibration: all predictions at 0.8 confidence with 80% accuracy
        let mut perfect = Vec::new();
        for _ in 0..80 {
            perfect.push((0.8, true));
        }
        for _ in 0..20 {
            perfect.push((0.8, false));
        }
        let ece = expected_calibration_error(&perfect, 10);
        assert!(ece < 0.05, "ECE for well-calibrated data should be low: {}", ece);

        // Terrible calibration: high confidence but all wrong
        let terrible: Vec<(f64, bool)> = (0..100).map(|_| (0.95, false)).collect();
        let ece = expected_calibration_error(&terrible, 10);
        assert!(ece > 0.5, "ECE for terribly calibrated data should be high: {}", ece);
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Apple's revenue was $94.8 billion!");
        assert!(tokens.contains(&"apple".to_string()));
        assert!(tokens.contains(&"revenue".to_string()));
        assert!(tokens.contains(&"billion".to_string()));
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Revenue was high. Expenses were low. Net income grew.");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_synthetic_data() {
        let docs = generate_sample_documents();
        assert_eq!(docs.len(), 5);

        let eval_data = generate_qa_evaluation_data(100);
        assert_eq!(eval_data.len(), 100);
        for (conf, _) in &eval_data {
            assert!(*conf >= 0.0 && *conf <= 1.0);
        }
    }

    #[test]
    fn test_numerical_edge_cases() {
        let reasoner = NumericalReasoner::new();

        // Empty values
        assert!(reasoner.compute(&NumericalOp::GrowthRate, &[]).is_none());

        // Division by zero
        assert!(reasoner.compute(&NumericalOp::GrowthRate, &[0.0, 100.0]).is_none());
        assert!(reasoner.compute(&NumericalOp::Ratio, &[100.0, 0.0]).is_none());

        // Single value
        assert!(reasoner.compute(&NumericalOp::Sum, &[42.0]).is_some());
        assert!((reasoner.compute(&NumericalOp::Sum, &[42.0]).unwrap() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_document_store_empty_search() {
        let store = DocumentStore::new();
        let results = store.search("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_question_types_comprehensive() {
        let clf = QuestionClassifier::new();

        // Boolean variations
        assert_eq!(clf.classify("Is the company profitable?"), QuestionType::Boolean);
        assert_eq!(clf.classify("Has revenue increased?"), QuestionType::Boolean);
        assert_eq!(clf.classify("Are expenses declining?"), QuestionType::Boolean);
        assert_eq!(clf.classify("Was there a loss?"), QuestionType::Boolean);

        // Numerical variations
        assert_eq!(clf.classify("How much revenue did they generate?"), QuestionType::Numerical);
        assert_eq!(clf.classify("Calculate the margin."), QuestionType::Numerical);
        assert_eq!(clf.classify("What is the percentage change?"), QuestionType::Numerical);

        // Comparative
        assert_eq!(clf.classify("Which company had higher revenue?"), QuestionType::Comparative);
        assert_eq!(clf.classify("Compare Apple and Google revenue."), QuestionType::Comparative);
    }
}
