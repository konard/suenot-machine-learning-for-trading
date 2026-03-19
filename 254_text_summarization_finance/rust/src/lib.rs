use rand::Rng;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

// ─── Text Utilities ───────────────────────────────────────────────

/// Simple whitespace tokenizer with lowercasing and punctuation removal.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Split text into sentences using period, exclamation mark, and question mark.
pub fn sentence_split(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() && trimmed.split_whitespace().count() >= 2 {
                sentences.push(trimmed);
            }
            current = String::new();
        }
    }

    // Handle remaining text without terminal punctuation
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && trimmed.split_whitespace().count() >= 3 {
        sentences.push(trimmed);
    }

    sentences
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ─── TF-IDF Vectorizer ───────────────────────────────────────────

/// Computes TF-IDF scores for words across a corpus of documents.
#[derive(Debug)]
pub struct TfIdfVectorizer {
    /// Maps each word to its index in the vocabulary.
    vocab: HashMap<String, usize>,
    /// IDF value for each word in the vocabulary.
    idf: Vec<f64>,
    /// Number of documents in the corpus.
    num_docs: usize,
}

impl TfIdfVectorizer {
    /// Build a TF-IDF vectorizer from a corpus of documents.
    /// Each document is a string of text.
    pub fn fit(documents: &[&str]) -> Self {
        let num_docs = documents.len();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut vocab_set: HashSet<String> = HashSet::new();

        // Count document frequency for each word
        for doc in documents {
            let tokens = tokenize(doc);
            let unique_tokens: HashSet<String> = tokens.into_iter().collect();
            for token in &unique_tokens {
                *doc_freq.entry(token.clone()).or_insert(0) += 1;
                vocab_set.insert(token.clone());
            }
        }

        // Build sorted vocabulary for deterministic ordering
        let mut vocab_list: Vec<String> = vocab_set.into_iter().collect();
        vocab_list.sort();

        let mut vocab = HashMap::new();
        let mut idf = Vec::with_capacity(vocab_list.len());

        for (idx, word) in vocab_list.iter().enumerate() {
            vocab.insert(word.clone(), idx);
            let df = *doc_freq.get(word).unwrap_or(&0);
            // IDF = log(N / (1 + df))
            let idf_val = (num_docs as f64 / (1.0 + df as f64)).ln();
            idf.push(idf_val);
        }

        Self {
            vocab,
            idf,
            num_docs,
        }
    }

    /// Compute the TF-IDF vector for a given text.
    pub fn transform(&self, text: &str) -> Vec<f64> {
        let tokens = tokenize(text);
        let total = tokens.len() as f64;
        if total == 0.0 {
            return vec![0.0; self.vocab.len()];
        }

        // Count term frequencies
        let mut tf_counts: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *tf_counts.entry(token.clone()).or_insert(0) += 1;
        }

        let mut tfidf = vec![0.0; self.vocab.len()];
        for (word, &count) in &tf_counts {
            if let Some(&idx) = self.vocab.get(word) {
                let tf = count as f64 / total;
                tfidf[idx] = tf * self.idf[idx];
            }
        }
        tfidf
    }

    /// Compute the average TF-IDF score for a sentence (scalar).
    pub fn sentence_score(&self, sentence: &str) -> f64 {
        let tokens = tokenize(sentence);
        if tokens.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut count = 0;
        for token in &tokens {
            if let Some(&idx) = self.vocab.get(token) {
                total_score += self.idf[idx]; // Use IDF as proxy when no doc context
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        total_score / count as f64
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the number of documents used for fitting.
    pub fn num_documents(&self) -> usize {
        self.num_docs
    }
}

// ─── Sentence Scorer ─────────────────────────────────────────────

/// Financial keywords that indicate important content.
const FINANCIAL_KEYWORDS: &[&str] = &[
    "revenue", "profit", "loss", "growth", "earnings", "income",
    "margin", "ebitda", "eps", "guidance", "forecast", "dividend",
    "debt", "cash", "flow", "operating", "net", "gross",
    "quarter", "annual", "fiscal", "billion", "million",
    "increased", "decreased", "declined", "improved", "exceeded",
    "acquisition", "merger", "restructuring", "buyback", "share",
];

/// Scores sentences for extractive summarization using multiple features.
#[derive(Debug)]
pub struct SentenceScorer {
    /// Weight for position score.
    pub w_position: f64,
    /// Weight for TF-IDF score.
    pub w_tfidf: f64,
    /// Weight for length score.
    pub w_length: f64,
    /// Weight for financial keyword density.
    pub w_financial: f64,
    /// Weight for numerical density.
    pub w_numerical: f64,
    /// Maximum ideal sentence length in words.
    pub max_length: usize,
}

impl SentenceScorer {
    pub fn new() -> Self {
        Self {
            w_position: 0.25,
            w_tfidf: 0.25,
            w_length: 0.10,
            w_financial: 0.25,
            w_numerical: 0.15,
            max_length: 40,
        }
    }

    /// Compute position score: earlier sentences score higher.
    pub fn position_score(&self, sentence_index: usize, total_sentences: usize) -> f64 {
        if total_sentences == 0 {
            return 0.0;
        }
        1.0 - (sentence_index as f64 / total_sentences as f64)
    }

    /// Compute length score: penalize too-short and too-long sentences.
    pub fn length_score(&self, sentence: &str) -> f64 {
        let word_count = sentence.split_whitespace().count();
        if word_count < 3 {
            return 0.1; // Too short
        }
        let capped = word_count.min(self.max_length) as f64;
        capped / self.max_length as f64
    }

    /// Compute financial keyword density.
    pub fn financial_keyword_density(&self, sentence: &str) -> f64 {
        let tokens = tokenize(sentence);
        if tokens.is_empty() {
            return 0.0;
        }
        let keyword_count = tokens
            .iter()
            .filter(|t| FINANCIAL_KEYWORDS.contains(&t.as_str()))
            .count();
        keyword_count as f64 / tokens.len() as f64
    }

    /// Compute numerical density (fraction of tokens that contain digits).
    pub fn numerical_density(&self, sentence: &str) -> f64 {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        let num_count = words
            .iter()
            .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
            .count();
        num_count as f64 / words.len() as f64
    }

    /// Compute the overall score for a sentence.
    pub fn score(
        &self,
        sentence: &str,
        sentence_index: usize,
        total_sentences: usize,
        tfidf_score: f64,
    ) -> f64 {
        let pos = self.position_score(sentence_index, total_sentences);
        let len = self.length_score(sentence);
        let fin = self.financial_keyword_density(sentence);
        let num = self.numerical_density(sentence);

        self.w_position * pos
            + self.w_tfidf * tfidf_score
            + self.w_length * len
            + self.w_financial * fin
            + self.w_numerical * num
    }

    /// Score all sentences in a document, returning (sentence_index, score) pairs.
    pub fn score_sentences(
        &self,
        sentences: &[String],
        vectorizer: &TfIdfVectorizer,
    ) -> Vec<(usize, f64)> {
        let total = sentences.len();
        sentences
            .iter()
            .enumerate()
            .map(|(i, sent)| {
                let tfidf = vectorizer.sentence_score(sent);
                let score = self.score(sent, i, total, tfidf);
                (i, score)
            })
            .collect()
    }
}

impl Default for SentenceScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Text Summarizer ─────────────────────────────────────────────

/// Extractive text summarizer that selects top-k sentences.
#[derive(Debug)]
pub struct TextSummarizer {
    scorer: SentenceScorer,
    /// Number of sentences to select for the summary.
    num_sentences: usize,
}

impl TextSummarizer {
    pub fn new(num_sentences: usize) -> Self {
        Self {
            scorer: SentenceScorer::new(),
            num_sentences,
        }
    }

    /// Create with custom scorer.
    pub fn with_scorer(num_sentences: usize, scorer: SentenceScorer) -> Self {
        Self {
            scorer,
            num_sentences,
        }
    }

    /// Summarize a document by selecting the top-k sentences.
    /// Returns the summary as a string with sentences in their original order.
    pub fn summarize(&self, document: &str, vectorizer: &TfIdfVectorizer) -> String {
        let sentences = sentence_split(document);
        if sentences.is_empty() {
            return String::new();
        }

        let k = self.num_sentences.min(sentences.len());
        let mut scored = self.scorer.score_sentences(&sentences, vectorizer);

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k indices
        let mut selected_indices: Vec<usize> = scored.iter().take(k).map(|(i, _)| *i).collect();

        // Sort by original position to maintain document order
        selected_indices.sort();

        // Build summary
        let summary_sentences: Vec<&str> = selected_indices
            .iter()
            .map(|&i| sentences[i].as_str())
            .collect();

        summary_sentences.join(" ")
    }

    /// Summarize and return individual sentences with scores.
    pub fn summarize_with_scores(
        &self,
        document: &str,
        vectorizer: &TfIdfVectorizer,
    ) -> Vec<(String, f64)> {
        let sentences = sentence_split(document);
        if sentences.is_empty() {
            return Vec::new();
        }

        let k = self.num_sentences.min(sentences.len());
        let mut scored = self.scorer.score_sentences(&sentences, vectorizer);

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .iter()
            .take(k)
            .map(|(i, score)| (sentences[*i].clone(), *score))
            .collect()
    }

    pub fn num_sentences(&self) -> usize {
        self.num_sentences
    }
}

// ─── Sentiment Scorer ─────────────────────────────────────────────

/// Positive financial words.
const POSITIVE_WORDS: &[&str] = &[
    "growth", "increased", "improved", "strong", "positive", "exceeded",
    "record", "robust", "optimistic", "favorable", "upgrade", "outperform",
    "beat", "surpassed", "expansion", "higher", "gains", "profit",
    "recovery", "momentum", "innovation", "breakthrough", "achievement",
    "opportunity", "confident", "accelerated", "thriving", "outstanding",
];

/// Negative financial words.
const NEGATIVE_WORDS: &[&str] = &[
    "loss", "declined", "decreased", "weak", "negative", "missed",
    "downturn", "risk", "concern", "unfavorable", "downgrade", "underperform",
    "below", "deteriorated", "contraction", "lower", "losses", "deficit",
    "recession", "slowdown", "challenge", "headwind", "uncertainty",
    "warning", "disappointing", "volatile", "impaired", "restructuring",
];

/// Simple lexicon-based sentiment scorer for financial text.
#[derive(Debug)]
pub struct SentimentScorer {
    positive_words: HashSet<String>,
    negative_words: HashSet<String>,
}

impl SentimentScorer {
    pub fn new() -> Self {
        Self {
            positive_words: POSITIVE_WORDS.iter().map(|w| w.to_string()).collect(),
            negative_words: NEGATIVE_WORDS.iter().map(|w| w.to_string()).collect(),
        }
    }

    /// Compute sentiment score for a text. Returns value in [-1.0, 1.0].
    /// Positive values indicate positive sentiment, negative values indicate negative.
    pub fn score(&self, text: &str) -> f64 {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return 0.0;
        }

        let mut pos_count = 0;
        let mut neg_count = 0;

        for token in &tokens {
            if self.positive_words.contains(token) {
                pos_count += 1;
            }
            if self.negative_words.contains(token) {
                neg_count += 1;
            }
        }

        let total = pos_count + neg_count;
        if total == 0 {
            return 0.0;
        }

        (pos_count as f64 - neg_count as f64) / total as f64
    }

    /// Classify sentiment as Positive, Negative, or Neutral.
    pub fn classify(&self, text: &str) -> &'static str {
        let score = self.score(text);
        if score > 0.1 {
            "Positive"
        } else if score < -0.1 {
            "Negative"
        } else {
            "Neutral"
        }
    }
}

impl Default for SentimentScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Summary Trader ──────────────────────────────────────────────

/// Trading signal derived from summarization sentiment.
#[derive(Debug, Clone, PartialEq)]
pub enum TradingSignal {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingSignal::Buy => write!(f, "BUY"),
            TradingSignal::Sell => write!(f, "SELL"),
            TradingSignal::Hold => write!(f, "HOLD"),
        }
    }
}

/// Generates trading signals from summary sentiment scores.
#[derive(Debug)]
pub struct SummaryTrader {
    /// Sentiment threshold for buy signals.
    pub buy_threshold: f64,
    /// Sentiment threshold for sell signals (negative).
    pub sell_threshold: f64,
}

impl SummaryTrader {
    pub fn new(buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            buy_threshold,
            sell_threshold,
        }
    }

    /// Generate a trading signal from a sentiment score.
    pub fn signal(&self, sentiment: f64) -> TradingSignal {
        if sentiment >= self.buy_threshold {
            TradingSignal::Buy
        } else if sentiment <= self.sell_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }

    /// Process multiple documents: summarize, score sentiment, generate signals.
    pub fn process_documents(
        &self,
        documents: &[&str],
        summarizer: &TextSummarizer,
        vectorizer: &TfIdfVectorizer,
        sentiment_scorer: &SentimentScorer,
    ) -> Vec<(String, f64, TradingSignal)> {
        documents
            .iter()
            .map(|doc| {
                let summary = summarizer.summarize(doc, vectorizer);
                let sentiment = sentiment_scorer.score(&summary);
                let signal = self.signal(sentiment);
                (summary, sentiment, signal)
            })
            .collect()
    }
}

impl Default for SummaryTrader {
    fn default() -> Self {
        Self::new(0.15, -0.15)
    }
}

// ─── Bybit Client ──────────────────────────────────────────────────

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
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ─────────────────────────────────────

/// Generate synthetic financial documents for testing.
pub fn generate_synthetic_documents() -> Vec<String> {
    vec![
        "The company reported strong revenue growth of 15% year over year, driven by robust demand in cloud services. Operating income increased to $2.3 billion, exceeding analyst expectations. Management raised full-year guidance, citing momentum in enterprise contracts. The gross margin improved by 200 basis points to 68%. Cash flow from operations reached a record $1.8 billion. The board approved a $5 billion share buyback program.".to_string(),
        "Quarterly earnings declined 8% compared to the prior year due to unfavorable market conditions. Revenue missed consensus estimates by $150 million. The company announced restructuring charges of $400 million related to workforce reduction. Operating margin contracted to 12% from 18% in the previous quarter. Management expressed concern about weakening consumer demand and rising input costs. Guidance for next quarter was lowered significantly.".to_string(),
        "Net income was flat compared to last year at $1.2 billion. Revenue grew modestly at 3%, in line with expectations. The company completed its acquisition of TechCorp for $2.1 billion. Research and development spending increased 10% as the company invested in AI capabilities. Debt levels remained stable with a leverage ratio of 2.5x. Management maintained its annual dividend of $3.50 per share.".to_string(),
        "The firm achieved record quarterly revenue of $45 billion, representing 22% growth. Earnings per share surged to $4.50, beating estimates by 30%. Strong performance was driven by breakthrough innovation in semiconductor technology. International markets contributed 60% of total revenue. The company announced expansion plans with $10 billion in capital expenditure. Free cash flow generation accelerated to $8 billion.".to_string(),
        "The company reported a net loss of $500 million for the quarter amid challenging macroeconomic headwinds. Revenue decreased 12% year over year as customer churn accelerated. Impaired goodwill charges totaled $1.2 billion following a strategic review. The CEO issued a warning about continued deterioration in key markets. Volatile commodity prices added uncertainty to the outlook. The company suspended its dividend program to preserve cash.".to_string(),
    ]
}

/// Generate labeled training data for sentiment classification.
/// Each sample has features [tfidf_avg, financial_density, numerical_density, position_avg, length_avg]
/// and a binary label (1.0 = positive sentiment, 0.0 = negative sentiment).
pub fn generate_training_data(n: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let tfidf_avg: f64 = rng.gen_range(0.0..1.0);
        let financial_density: f64 = rng.gen_range(0.0..0.5);
        let numerical_density: f64 = rng.gen_range(0.0..0.4);
        let position_avg: f64 = rng.gen_range(0.0..1.0);
        let length_avg: f64 = rng.gen_range(0.0..1.0);

        // Signal: higher TF-IDF and financial density correlate with positive sentiment
        let signal = 0.3 * tfidf_avg + 0.3 * financial_density + 0.1 * numerical_density
            + 0.2 * position_avg
            - 0.1 * length_avg;
        let prob = 1.0 / (1.0 + (-signal * 4.0).exp());
        let label = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };

        data.push((
            vec![tfidf_avg, financial_density, numerical_density, position_avg, length_avg],
            label,
        ));
    }
    data
}

/// Generate synthetic kline data for testing when Bybit is unavailable.
pub fn generate_synthetic_klines(n: usize, start_price: f64) -> Vec<Kline> {
    let mut rng = rand::thread_rng();
    let mut price = start_price;
    let mut klines = Vec::with_capacity(n);

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02) * price;
        let open = price;
        let close = price + change;
        let high = open.max(close) + rng.gen_range(0.0..0.01) * price;
        let low = open.min(close) - rng.gen_range(0.0..0.01) * price;
        let volume = rng.gen_range(10.0..1000.0);

        klines.push(Kline {
            timestamp: 1700000000000 + (i as u64 * 60000),
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }
    klines
}

// ─── ROUGE Metrics ─────────────────────────────────────────────────

/// Compute ROUGE-1 (unigram) recall between generated and reference summaries.
pub fn rouge_1(generated: &str, reference: &str) -> f64 {
    let gen_tokens: Vec<String> = tokenize(generated);
    let ref_tokens: Vec<String> = tokenize(reference);

    if ref_tokens.is_empty() {
        return 0.0;
    }

    let gen_set: HashMap<String, usize> = {
        let mut m = HashMap::new();
        for t in &gen_tokens {
            *m.entry(t.clone()).or_insert(0) += 1;
        }
        m
    };

    let ref_set: HashMap<String, usize> = {
        let mut m = HashMap::new();
        for t in &ref_tokens {
            *m.entry(t.clone()).or_insert(0) += 1;
        }
        m
    };

    let overlap: usize = ref_set
        .iter()
        .map(|(word, &ref_count)| {
            let gen_count = gen_set.get(word).copied().unwrap_or(0);
            gen_count.min(ref_count)
        })
        .sum();

    overlap as f64 / ref_tokens.len() as f64
}

/// Compute ROUGE-2 (bigram) recall between generated and reference summaries.
pub fn rouge_2(generated: &str, reference: &str) -> f64 {
    let gen_tokens = tokenize(generated);
    let ref_tokens = tokenize(reference);

    if ref_tokens.len() < 2 {
        return 0.0;
    }

    let bigrams = |tokens: &[String]| -> HashMap<(String, String), usize> {
        let mut m = HashMap::new();
        for pair in tokens.windows(2) {
            *m.entry((pair[0].clone(), pair[1].clone())).or_insert(0) += 1;
        }
        m
    };

    let gen_bigrams = bigrams(&gen_tokens);
    let ref_bigrams = bigrams(&ref_tokens);

    let overlap: usize = ref_bigrams
        .iter()
        .map(|(bigram, &ref_count)| {
            let gen_count = gen_bigrams.get(bigram).copied().unwrap_or(0);
            gen_count.min(ref_count)
        })
        .sum();

    let total_ref_bigrams: usize = ref_bigrams.values().sum();
    if total_ref_bigrams == 0 {
        return 0.0;
    }

    overlap as f64 / total_ref_bigrams as f64
}

/// Compute ROUGE-L (longest common subsequence) F1 score.
pub fn rouge_l(generated: &str, reference: &str) -> f64 {
    let gen_tokens = tokenize(generated);
    let ref_tokens = tokenize(reference);

    if gen_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    // Compute LCS length using dynamic programming
    let m = gen_tokens.len();
    let n = ref_tokens.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if gen_tokens[i - 1] == ref_tokens[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let lcs_len = dp[m][n] as f64;
    let precision = lcs_len / m as f64;
    let recall = lcs_len / n as f64;

    if precision + recall == 0.0 {
        return 0.0;
    }

    let beta = 1.0; // F1 score
    (1.0 + beta * beta) * precision * recall / (beta * beta * precision + recall)
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World! This is a Test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_sentence_split() {
        let text = "Revenue increased by 15%. The company is growing. This is good news.";
        let sentences = sentence_split(text);
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("Revenue"));
        assert!(sentences[1].contains("growing"));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_tfidf_vectorizer() {
        let docs = vec![
            "revenue growth was strong this quarter",
            "the company reported a net loss",
            "earnings exceeded expectations with record revenue",
        ];
        let vectorizer = TfIdfVectorizer::fit(&docs);
        assert!(vectorizer.vocab_size() > 0);
        assert_eq!(vectorizer.num_documents(), 3);

        let vec1 = vectorizer.transform(docs[0]);
        assert_eq!(vec1.len(), vectorizer.vocab_size());
        // At least some non-zero values
        assert!(vec1.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_tfidf_sentence_score() {
        let docs = vec![
            "revenue growth exceeded expectations",
            "the weather is nice today",
            "revenue increased significantly this quarter",
        ];
        let vectorizer = TfIdfVectorizer::fit(&docs);
        let score1 = vectorizer.sentence_score("revenue growth exceeded");
        let score2 = vectorizer.sentence_score("completely unknown words xyz");
        // Known words should score higher than unknown
        assert!(score1 > score2);
    }

    #[test]
    fn test_sentence_scorer_position() {
        let scorer = SentenceScorer::new();
        let first = scorer.position_score(0, 10);
        let last = scorer.position_score(9, 10);
        assert!(first > last);
        assert!((first - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sentence_scorer_financial_keywords() {
        let scorer = SentenceScorer::new();
        let financial = scorer.financial_keyword_density("Revenue growth and profit increased significantly.");
        let generic = scorer.financial_keyword_density("The weather was sunny and pleasant today.");
        assert!(financial > generic);
    }

    #[test]
    fn test_sentence_scorer_numerical_density() {
        let scorer = SentenceScorer::new();
        let with_numbers = scorer.numerical_density("Revenue was $2.3 billion, up 15% from Q3 2023.");
        let without_numbers = scorer.numerical_density("The company is doing well overall.");
        assert!(with_numbers > without_numbers);
    }

    #[test]
    fn test_text_summarizer() {
        let docs = generate_synthetic_documents();
        let doc_refs: Vec<&str> = docs.iter().map(|d| d.as_str()).collect();
        let vectorizer = TfIdfVectorizer::fit(&doc_refs);
        let summarizer = TextSummarizer::new(2);

        let summary = summarizer.summarize(&docs[0], &vectorizer);
        assert!(!summary.is_empty());
        // Summary should be shorter than original
        assert!(summary.len() < docs[0].len());
    }

    #[test]
    fn test_sentiment_scorer_positive() {
        let scorer = SentimentScorer::new();
        let sentiment = scorer.score("Strong growth and improved earnings exceeded expectations.");
        assert!(sentiment > 0.0, "Expected positive sentiment, got {}", sentiment);
        assert_eq!(scorer.classify("Strong growth and improved earnings exceeded expectations."), "Positive");
    }

    #[test]
    fn test_sentiment_scorer_negative() {
        let scorer = SentimentScorer::new();
        let sentiment = scorer.score("Significant loss and declining revenue with negative outlook.");
        assert!(sentiment < 0.0, "Expected negative sentiment, got {}", sentiment);
        assert_eq!(scorer.classify("Significant loss and declining revenue with negative outlook."), "Negative");
    }

    #[test]
    fn test_sentiment_scorer_neutral() {
        let scorer = SentimentScorer::new();
        let sentiment = scorer.score("The meeting was held at the headquarters building.");
        assert!((sentiment).abs() <= 0.1, "Expected neutral sentiment, got {}", sentiment);
    }

    #[test]
    fn test_summary_trader_signals() {
        let trader = SummaryTrader::default();
        assert_eq!(trader.signal(0.5), TradingSignal::Buy);
        assert_eq!(trader.signal(-0.5), TradingSignal::Sell);
        assert_eq!(trader.signal(0.0), TradingSignal::Hold);
    }

    #[test]
    fn test_summary_trader_process_documents() {
        let docs = generate_synthetic_documents();
        let doc_refs: Vec<&str> = docs.iter().map(|d| d.as_str()).collect();
        let vectorizer = TfIdfVectorizer::fit(&doc_refs);
        let summarizer = TextSummarizer::new(2);
        let sentiment_scorer = SentimentScorer::new();
        let trader = SummaryTrader::default();

        let results = trader.process_documents(&doc_refs, &summarizer, &vectorizer, &sentiment_scorer);
        assert_eq!(results.len(), 5);
        for (summary, sentiment, signal) in &results {
            assert!(!summary.is_empty());
            assert!(*sentiment >= -1.0 && *sentiment <= 1.0);
            assert!(
                *signal == TradingSignal::Buy
                    || *signal == TradingSignal::Sell
                    || *signal == TradingSignal::Hold
            );
        }
    }

    #[test]
    fn test_rouge_1_identical() {
        let text = "revenue growth was strong";
        let score = rouge_1(text, text);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_1_partial() {
        let gen = "revenue growth was strong";
        let reference = "revenue growth exceeded expectations this quarter";
        let score = rouge_1(gen, reference);
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_rouge_2() {
        let gen = "revenue growth was strong";
        let reference = "revenue growth was strong this quarter";
        let score = rouge_2(gen, reference);
        assert!(score > 0.0);
    }

    #[test]
    fn test_rouge_l() {
        let gen = "the company reported strong revenue growth";
        let reference = "the company reported strong revenue growth this quarter";
        let score = rouge_l(gen, reference);
        assert!(score > 0.0);
    }

    #[test]
    fn test_synthetic_documents() {
        let docs = generate_synthetic_documents();
        assert_eq!(docs.len(), 5);
        for doc in &docs {
            assert!(!doc.is_empty());
            let sentences = sentence_split(doc);
            assert!(sentences.len() >= 3);
        }
    }

    #[test]
    fn test_generate_training_data() {
        let data = generate_training_data(100);
        assert_eq!(data.len(), 100);
        for (features, label) in &data {
            assert_eq!(features.len(), 5);
            assert!(*label == 0.0 || *label == 1.0);
        }
    }

    #[test]
    fn test_generate_synthetic_klines() {
        let klines = generate_synthetic_klines(50, 50000.0);
        assert_eq!(klines.len(), 50);
        for kline in &klines {
            assert!(kline.high >= kline.low);
            assert!(kline.volume > 0.0);
        }
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-9);
    }
}
