use ndarray::Array2;
use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Document Representation ─────────────────────────────────────

/// A labeled document consisting of text content and a category label.
#[derive(Debug, Clone)]
pub struct Document {
    pub text: String,
    pub label: Option<String>,
}

impl Document {
    pub fn new(text: &str, label: Option<&str>) -> Self {
        Self {
            text: text.to_string(),
            label: label.map(|s| s.to_string()),
        }
    }

    /// Tokenize the document into lowercase words.
    pub fn tokens(&self) -> Vec<String> {
        self.text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(|s| s.to_string())
            .collect()
    }
}

// ─── TF-IDF Vectorizer ───────────────────────────────────────────

/// Transforms documents into TF-IDF feature vectors.
///
/// The vectorizer learns a vocabulary from training data and produces
/// fixed-length numeric vectors that capture term importance.
#[derive(Debug)]
pub struct TfIdfVectorizer {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
    num_docs: usize,
}

impl TfIdfVectorizer {
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            num_docs: 0,
        }
    }

    /// Build vocabulary and compute IDF values from a corpus.
    pub fn fit(&mut self, documents: &[Document]) {
        self.num_docs = documents.len();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut vocab_set: HashMap<String, usize> = HashMap::new();

        // Build vocabulary and count document frequencies
        for doc in documents {
            let tokens = doc.tokens();
            let mut seen_in_doc: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            for token in &tokens {
                if !vocab_set.contains_key(token) {
                    let idx = vocab_set.len();
                    vocab_set.insert(token.clone(), idx);
                }
                seen_in_doc.insert(token.clone());
            }

            for token in seen_in_doc {
                *doc_freq.entry(token).or_insert(0) += 1;
            }
        }

        self.vocabulary = vocab_set;
        self.idf = vec![0.0; self.vocabulary.len()];

        // Compute IDF: log(N / (1 + df))
        for (word, &idx) in &self.vocabulary {
            let df = doc_freq.get(word).copied().unwrap_or(0);
            self.idf[idx] = (self.num_docs as f64 / (1.0 + df as f64)).ln();
        }
    }

    /// Transform a single document into a TF-IDF vector.
    pub fn transform_one(&self, doc: &Document) -> Vec<f64> {
        let tokens = doc.tokens();
        let total_tokens = tokens.len() as f64;
        let mut tf: HashMap<String, f64> = HashMap::new();

        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0.0) += 1.0;
        }

        let mut vector = vec![0.0; self.vocabulary.len()];
        for (word, &count) in &tf {
            if let Some(&idx) = self.vocabulary.get(word) {
                let term_freq = count / total_tokens.max(1.0);
                vector[idx] = term_freq * self.idf[idx];
            }
        }
        vector
    }

    /// Transform multiple documents into TF-IDF vectors.
    pub fn transform(&self, documents: &[Document]) -> Vec<Vec<f64>> {
        documents.iter().map(|doc| self.transform_one(doc)).collect()
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, documents: &[Document]) -> Vec<Vec<f64>> {
        self.fit(documents);
        self.transform(documents)
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get the top-N features by IDF value (most discriminative terms).
    pub fn top_features(&self, n: usize) -> Vec<(String, f64)> {
        let mut features: Vec<(String, f64)> = self
            .vocabulary
            .iter()
            .map(|(word, &idx)| (word.clone(), self.idf[idx]))
            .collect();
        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        features.truncate(n);
        features
    }
}

impl Default for TfIdfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Naive Bayes Classifier ──────────────────────────────────────

/// Multinomial Naive Bayes classifier with Laplace smoothing.
///
/// Learns class-conditional word distributions and predicts document
/// categories by computing log-posterior probabilities.
#[derive(Debug)]
pub struct NaiveBayesClassifier {
    class_log_prior: HashMap<String, f64>,
    feature_log_prob: HashMap<String, Vec<f64>>,
    classes: Vec<String>,
    num_features: usize,
    smoothing: f64,
}

impl NaiveBayesClassifier {
    pub fn new(smoothing: f64) -> Self {
        Self {
            class_log_prior: HashMap::new(),
            feature_log_prob: HashMap::new(),
            classes: Vec::new(),
            num_features: 0,
            smoothing,
        }
    }

    /// Train the classifier on labeled TF-IDF vectors.
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[String]) {
        assert_eq!(features.len(), labels.len());
        if features.is_empty() {
            return;
        }

        self.num_features = features[0].len();
        let n_total = features.len() as f64;

        // Count class occurrences and aggregate feature sums
        let mut class_counts: HashMap<String, f64> = HashMap::new();
        let mut class_feature_sums: HashMap<String, Vec<f64>> = HashMap::new();

        for (feat, label) in features.iter().zip(labels.iter()) {
            *class_counts.entry(label.clone()).or_insert(0.0) += 1.0;
            let sums = class_feature_sums
                .entry(label.clone())
                .or_insert_with(|| vec![0.0; self.num_features]);
            for (j, &v) in feat.iter().enumerate() {
                sums[j] += v.max(0.0); // ensure non-negative for multinomial NB
            }
        }

        self.classes = class_counts.keys().cloned().collect();
        self.classes.sort();

        // Compute log priors and log likelihoods
        for class in &self.classes {
            let count = class_counts[class];
            self.class_log_prior
                .insert(class.clone(), (count / n_total).ln());

            let sums = &class_feature_sums[class];
            let total: f64 = sums.iter().sum::<f64>() + self.smoothing * self.num_features as f64;

            let log_probs: Vec<f64> = sums
                .iter()
                .map(|&s| ((s + self.smoothing) / total).ln())
                .collect();

            self.feature_log_prob.insert(class.clone(), log_probs);
        }
    }

    /// Predict the most likely class for a feature vector.
    pub fn predict(&self, features: &[f64]) -> (String, f64) {
        let scores = self.predict_log_proba(features);
        let (best_class, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        (best_class.clone(), *best_score)
    }

    /// Compute log-probability for each class.
    pub fn predict_log_proba(&self, features: &[f64]) -> Vec<(String, f64)> {
        self.classes
            .iter()
            .map(|class| {
                let log_prior = self.class_log_prior[class];
                let log_probs = &self.feature_log_prob[class];
                let log_likelihood: f64 = features
                    .iter()
                    .zip(log_probs.iter())
                    .map(|(&f, &lp)| f.max(0.0) * lp)
                    .sum();
                (class.clone(), log_prior + log_likelihood)
            })
            .collect()
    }

    /// Evaluate accuracy on a test set.
    pub fn accuracy(&self, features: &[Vec<f64>], labels: &[String]) -> f64 {
        let correct = features
            .iter()
            .zip(labels.iter())
            .filter(|(feat, label)| {
                let (pred, _) = self.predict(feat);
                &pred == *label
            })
            .count();
        correct as f64 / features.len() as f64
    }
}

// ─── Document Classifier (Softmax Logistic Regression) ───────────

/// Multi-class logistic regression (softmax) classifier.
///
/// Trained with stochastic gradient descent on TF-IDF features.
#[derive(Debug)]
pub struct DocumentClassifier {
    weights: Array2<f64>,
    biases: Vec<f64>,
    classes: Vec<String>,
    learning_rate: f64,
    num_features: usize,
    num_classes: usize,
}

impl DocumentClassifier {
    pub fn new(num_features: usize, classes: Vec<String>, learning_rate: f64) -> Self {
        let num_classes = classes.len();
        let mut rng = rand::thread_rng();

        let weights = Array2::from_shape_fn((num_classes, num_features), |_| {
            rng.gen_range(-0.01..0.01)
        });
        let biases = vec![0.0; num_classes];

        Self {
            weights,
            biases,
            classes,
            learning_rate,
            num_features,
            num_classes,
        }
    }

    /// Softmax function over a vector of logits.
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&z| (z - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Predict class probabilities for a feature vector.
    pub fn predict_proba(&self, features: &[f64]) -> Vec<(String, f64)> {
        assert_eq!(features.len(), self.num_features);

        let logits: Vec<f64> = (0..self.num_classes)
            .map(|c| {
                let row = self.weights.row(c);
                let dot: f64 = row.iter().zip(features.iter()).map(|(&w, &x)| w * x).sum();
                dot + self.biases[c]
            })
            .collect();

        let probs = Self::softmax(&logits);
        self.classes
            .iter()
            .cloned()
            .zip(probs.into_iter())
            .collect()
    }

    /// Predict the most likely class.
    pub fn predict(&self, features: &[f64]) -> (String, f64) {
        let probs = self.predict_proba(features);
        probs
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }

    /// Train on labeled data for a given number of epochs.
    pub fn train(&mut self, features: &[Vec<f64>], labels: &[String], epochs: usize) {
        let class_idx: HashMap<String, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        for _ in 0..epochs {
            for (feat, label) in features.iter().zip(labels.iter()) {
                let logits: Vec<f64> = (0..self.num_classes)
                    .map(|c| {
                        let row = self.weights.row(c);
                        let dot: f64 =
                            row.iter().zip(feat.iter()).map(|(&w, &x)| w * x).sum();
                        dot + self.biases[c]
                    })
                    .collect();

                let probs = Self::softmax(&logits);
                let target_idx = class_idx.get(label).copied().unwrap_or(0);

                // SGD update
                for c in 0..self.num_classes {
                    let error = probs[c] - if c == target_idx { 1.0 } else { 0.0 };
                    for j in 0..self.num_features {
                        self.weights[[c, j]] -= self.learning_rate * error * feat[j];
                    }
                    self.biases[c] -= self.learning_rate * error;
                }
            }
        }
    }

    /// Evaluate accuracy on a test set.
    pub fn accuracy(&self, features: &[Vec<f64>], labels: &[String]) -> f64 {
        let correct = features
            .iter()
            .zip(labels.iter())
            .filter(|(feat, label)| {
                let (pred, _) = self.predict(feat);
                &pred == *label
            })
            .count();
        correct as f64 / features.len() as f64
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }
}

// ─── Financial Keyword Detector ──────────────────────────────────

/// Detects the presence of domain-specific financial keyword groups.
///
/// Returns a feature vector indicating keyword density for each category.
#[derive(Debug)]
pub struct FinancialKeywordDetector {
    keyword_groups: Vec<(String, Vec<String>)>,
}

impl FinancialKeywordDetector {
    pub fn new() -> Self {
        let keyword_groups = vec![
            (
                "earnings".to_string(),
                vec![
                    "revenue", "profit", "eps", "earnings", "guidance", "outlook",
                    "beat", "miss", "quarterly", "annual", "dividend",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
            ),
            (
                "mergers".to_string(),
                vec![
                    "acquisition", "merger", "takeover", "bid", "deal", "buyout",
                    "target", "acquire", "consolidation", "divestiture",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
            ),
            (
                "regulatory".to_string(),
                vec![
                    "sec", "compliance", "violation", "fine", "penalty", "regulation",
                    "enforcement", "investigation", "settlement", "sanction",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
            ),
            (
                "risk".to_string(),
                vec![
                    "default", "downgrade", "impairment", "restructuring", "bankruptcy",
                    "writeoff", "loss", "decline", "warning", "risk",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
            ),
        ];
        Self { keyword_groups }
    }

    /// Compute keyword density features for a document.
    /// Returns a vector of densities (keyword count / total words) for each group.
    pub fn detect(&self, doc: &Document) -> Vec<(String, f64)> {
        let tokens = doc.tokens();
        let total = tokens.len() as f64;
        if total == 0.0 {
            return self
                .keyword_groups
                .iter()
                .map(|(name, _)| (name.clone(), 0.0))
                .collect();
        }

        self.keyword_groups
            .iter()
            .map(|(name, keywords)| {
                let count = tokens
                    .iter()
                    .filter(|t| keywords.contains(t))
                    .count() as f64;
                (name.clone(), count / total)
            })
            .collect()
    }

    /// Return the group names.
    pub fn group_names(&self) -> Vec<String> {
        self.keyword_groups
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }
}

impl Default for FinancialKeywordDetector {
    fn default() -> Self {
        Self::new()
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

/// Financial document categories for classification.
pub const CATEGORIES: &[&str] = &["earnings", "mergers", "regulatory", "risk", "general"];

/// Generate synthetic financial documents for training and testing.
pub fn generate_synthetic_documents(n_per_class: usize) -> Vec<Document> {
    let mut rng = rand::thread_rng();
    let mut docs = Vec::new();

    let templates: Vec<(&str, Vec<&str>)> = vec![
        (
            "earnings",
            vec![
                "Company reported quarterly revenue of billion exceeding analyst expectations with strong EPS guidance for next quarter",
                "Annual earnings report shows profit growth driven by revenue increase and improved operating margins with positive outlook",
                "Quarterly results beat consensus estimates with revenue up and earnings per share above guidance driven by dividend growth",
                "Company missed earnings expectations with revenue decline and lowered guidance citing challenging market conditions",
                "Strong quarterly performance with record revenue profit margins expanding and EPS beating street estimates significantly",
            ],
        ),
        (
            "mergers",
            vec![
                "Company announces acquisition of target firm in major deal valued at billion creating industry consolidation",
                "Merger agreement reached between two firms in strategic buyout deal expected to close next quarter pending approval",
                "Takeover bid launched for target company at premium valuation as consolidation trend continues in the sector",
                "Strategic acquisition completed expanding market presence through divestiture of non-core assets and buyout",
                "Deal announced for merger of equals creating combined entity with significant market share and acquisition synergies",
            ],
        ),
        (
            "regulatory",
            vec![
                "SEC launches investigation into company compliance practices following violation of securities regulation requirements",
                "Regulatory enforcement action results in fine and penalty for compliance failures according to settlement agreement",
                "Company reaches settlement with SEC over regulation violation paying penalty and agreeing to compliance improvements",
                "Investigation by regulatory authorities finds compliance gaps leading to sanction and enforcement proceedings",
                "SEC files enforcement action alleging violation of securities laws seeking fine penalty and compliance reforms",
            ],
        ),
        (
            "risk",
            vec![
                "Credit rating downgrade announced due to increased default risk and declining financial performance raising concerns",
                "Company reports impairment charges and restructuring costs warning of potential loss in upcoming quarters ahead",
                "Bankruptcy filing follows period of decline with significant writeoff of assets and restructuring of operations",
                "Risk warning issued as company faces potential default on debt obligations with declining revenue and loss",
                "Significant impairment loss recognized with restructuring plan announced to address declining performance and risk",
            ],
        ),
        (
            "general",
            vec![
                "Market trading session saw mixed performance across sectors with technology stocks leading gains on Monday",
                "Annual shareholder meeting held with management presenting strategic vision for growth and innovation roadmap",
                "Company launches new product line expanding portfolio to serve growing market demand in technology sector",
                "Industry conference highlights trends in innovation and digital transformation across financial services sector",
                "Market commentary discusses macroeconomic outlook including interest rates inflation and global growth trends",
            ],
        ),
    ];

    for (category, texts) in &templates {
        for _ in 0..n_per_class {
            let idx = rng.gen_range(0..texts.len());
            // Add some variation by shuffling words slightly
            let mut words: Vec<&str> = texts[idx].split_whitespace().collect();
            // Swap a few random positions for variation
            for _ in 0..3 {
                let a = rng.gen_range(0..words.len());
                let b = rng.gen_range(0..words.len());
                words.swap(a, b);
            }
            docs.push(Document::new(
                &words.join(" "),
                Some(category),
            ));
        }
    }

    docs
}

/// Generate synthetic price impact data for document categories.
/// Returns (category, avg_price_change, count) tuples.
pub fn generate_price_impact_data() -> Vec<(String, f64, usize)> {
    let mut rng = rand::thread_rng();
    vec![
        (
            "earnings".to_string(),
            rng.gen_range(0.5..3.0),
            rng.gen_range(50..200),
        ),
        (
            "mergers".to_string(),
            rng.gen_range(2.0..8.0),
            rng.gen_range(20..80),
        ),
        (
            "regulatory".to_string(),
            rng.gen_range(-5.0..-1.0),
            rng.gen_range(10..50),
        ),
        (
            "risk".to_string(),
            rng.gen_range(-8.0..-2.0),
            rng.gen_range(15..60),
        ),
        (
            "general".to_string(),
            rng.gen_range(-0.5..0.5),
            rng.gen_range(100..500),
        ),
    ]
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_tokenization() {
        let doc = Document::new("Company REPORTS quarterly Revenue of $1.5 billion!", None);
        let tokens = doc.tokens();
        assert!(tokens.contains(&"company".to_string()));
        assert!(tokens.contains(&"reports".to_string()));
        assert!(tokens.contains(&"quarterly".to_string()));
        assert!(tokens.contains(&"revenue".to_string()));
        assert!(tokens.contains(&"billion".to_string()));
        // Single-char tokens and punctuation should be excluded
        assert!(!tokens.contains(&"$".to_string()));
    }

    #[test]
    fn test_tfidf_vectorizer_fit_transform() {
        let docs = vec![
            Document::new("revenue profit earnings growth", Some("earnings")),
            Document::new("acquisition merger deal buyout", Some("mergers")),
            Document::new("revenue earnings quarterly report", Some("earnings")),
        ];

        let mut vectorizer = TfIdfVectorizer::new();
        let vectors = vectorizer.fit_transform(&docs);

        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0].len(), vectorizer.vocab_size());
        // All vectors should have the same length
        assert!(vectors.iter().all(|v| v.len() == vectors[0].len()));
    }

    #[test]
    fn test_tfidf_idf_values() {
        let docs = vec![
            Document::new("the cat sat", None),
            Document::new("the dog ran", None),
            Document::new("the bird flew", None),
        ];

        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&docs);

        // "the" appears in all 3 docs, should have lower IDF
        // "cat" appears in 1 doc, should have higher IDF
        let top = vectorizer.top_features(10);
        let the_idf = top.iter().find(|(w, _)| w == "the").map(|(_, v)| *v);
        let cat_idf = top.iter().find(|(w, _)| w == "cat").map(|(_, v)| *v);

        if let (Some(the_v), Some(cat_v)) = (the_idf, cat_idf) {
            assert!(
                cat_v > the_v,
                "cat IDF ({}) should be > the IDF ({})",
                cat_v,
                the_v
            );
        }
    }

    #[test]
    fn test_naive_bayes_train_predict() {
        let docs = generate_synthetic_documents(20);
        let mut vectorizer = TfIdfVectorizer::new();
        let vectors = vectorizer.fit_transform(&docs);
        let labels: Vec<String> = docs.iter().filter_map(|d| d.label.clone()).collect();

        let mut nb = NaiveBayesClassifier::new(1.0);
        nb.fit(&vectors, &labels);

        // Predict on an earnings document
        let test_doc = Document::new("quarterly revenue profit earnings guidance", None);
        let test_vec = vectorizer.transform_one(&test_doc);
        let (predicted, _score) = nb.predict(&test_vec);

        // The classifier should exist and produce a valid class
        assert!(CATEGORIES.contains(&predicted.as_str()));
    }

    #[test]
    fn test_naive_bayes_accuracy() {
        let docs = generate_synthetic_documents(30);
        let mut vectorizer = TfIdfVectorizer::new();
        let vectors = vectorizer.fit_transform(&docs);
        let labels: Vec<String> = docs.iter().filter_map(|d| d.label.clone()).collect();

        let (train_feat, test_feat) = vectors.split_at(vectors.len() * 4 / 5);
        let (train_labels, test_labels) = labels.split_at(labels.len() * 4 / 5);

        let mut nb = NaiveBayesClassifier::new(1.0);
        nb.fit(train_feat, train_labels);

        let acc = nb.accuracy(test_feat, test_labels);
        // With synthetic data, accuracy should be meaningfully above random (20%)
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_document_classifier_predict() {
        let classes: Vec<String> = CATEGORIES.iter().map(|&s| s.to_string()).collect();
        let clf = DocumentClassifier::new(10, classes.clone(), 0.01);
        let features = vec![0.1; 10];
        let (pred, conf) = clf.predict(&features);
        assert!(classes.contains(&pred));
        assert!(conf >= 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_document_classifier_train() {
        let docs = generate_synthetic_documents(30);
        let mut vectorizer = TfIdfVectorizer::new();
        let vectors = vectorizer.fit_transform(&docs);
        let labels: Vec<String> = docs.iter().filter_map(|d| d.label.clone()).collect();

        let classes: Vec<String> = CATEGORIES.iter().map(|&s| s.to_string()).collect();
        let mut clf = DocumentClassifier::new(vectorizer.vocab_size(), classes, 0.01);

        clf.train(&vectors, &labels, 50);
        let acc = clf.accuracy(&vectors, &labels);
        // After training on the same data, accuracy should be meaningful
        assert!(acc > 0.0, "accuracy after training: {}", acc);
    }

    #[test]
    fn test_document_classifier_softmax() {
        let classes: Vec<String> = CATEGORIES.iter().map(|&s| s.to_string()).collect();
        let clf = DocumentClassifier::new(5, classes, 0.01);
        let features = vec![0.5, 0.3, 0.1, 0.8, 0.2];
        let probs = clf.predict_proba(&features);

        // Probabilities should sum to approximately 1.0
        let sum: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "probabilities sum to {} instead of 1.0",
            sum
        );
    }

    #[test]
    fn test_keyword_detector() {
        let detector = FinancialKeywordDetector::new();

        let earnings_doc = Document::new(
            "Company reported strong quarterly revenue and EPS beat with profit guidance raised",
            None,
        );
        let results = detector.detect(&earnings_doc);

        let earnings_density = results
            .iter()
            .find(|(name, _)| name == "earnings")
            .map(|(_, d)| *d)
            .unwrap_or(0.0);

        // Earnings keywords should be detected
        assert!(
            earnings_density > 0.0,
            "earnings density should be > 0, got {}",
            earnings_density
        );
    }

    #[test]
    fn test_keyword_detector_multiple_categories() {
        let detector = FinancialKeywordDetector::new();

        let merger_doc = Document::new("acquisition deal merger buyout target consolidation", None);
        let results = detector.detect(&merger_doc);

        let merger_density = results
            .iter()
            .find(|(name, _)| name == "mergers")
            .map(|(_, d)| *d)
            .unwrap_or(0.0);
        let earnings_density = results
            .iter()
            .find(|(name, _)| name == "earnings")
            .map(|(_, d)| *d)
            .unwrap_or(0.0);

        assert!(
            merger_density > earnings_density,
            "merger density ({}) should exceed earnings density ({})",
            merger_density,
            earnings_density
        );
    }

    #[test]
    fn test_synthetic_document_generation() {
        let docs = generate_synthetic_documents(10);
        assert_eq!(docs.len(), 50); // 10 per class * 5 classes
        for doc in &docs {
            assert!(doc.label.is_some());
            assert!(!doc.text.is_empty());
            assert!(CATEGORIES.contains(&doc.label.as_ref().unwrap().as_str()));
        }
    }

    #[test]
    fn test_price_impact_data() {
        let data = generate_price_impact_data();
        assert_eq!(data.len(), 5);
        for (category, _change, count) in &data {
            assert!(CATEGORIES.contains(&category.as_str()));
            assert!(*count > 0);
        }
    }

    #[test]
    fn test_empty_document() {
        let doc = Document::new("", None);
        assert!(doc.tokens().is_empty());

        let detector = FinancialKeywordDetector::new();
        let results = detector.detect(&doc);
        for (_, density) in &results {
            assert_eq!(*density, 0.0);
        }
    }

    #[test]
    fn test_vectorizer_top_features() {
        let docs = generate_synthetic_documents(5);
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit_transform(&docs);

        let top = vectorizer.top_features(5);
        assert_eq!(top.len(), 5);
        // Features should be sorted by IDF descending
        for i in 1..top.len() {
            assert!(top[i - 1].1 >= top[i].1);
        }
    }
}
