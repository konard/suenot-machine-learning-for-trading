use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;

// ─── Entity Types ─────────────────────────────────────────────────

/// Financial entity categories for NER.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Org,        // Organization / company name
    Ticker,     // Stock ticker or crypto trading pair
    Money,      // Monetary value
    Percent,    // Percentage value
    Date,       // Temporal expression
    Event,      // Financial event
    Instrument, // Financial instrument
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityType::Org => write!(f, "ORG"),
            EntityType::Ticker => write!(f, "TICKER"),
            EntityType::Money => write!(f, "MONEY"),
            EntityType::Percent => write!(f, "PERCENT"),
            EntityType::Date => write!(f, "DATE"),
            EntityType::Event => write!(f, "EVENT"),
            EntityType::Instrument => write!(f, "INSTRUMENT"),
        }
    }
}

// ─── BIO Tags ─────────────────────────────────────────────────────

/// BIO tagging scheme for sequence labeling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BioTag {
    Begin(EntityType),
    Inside(EntityType),
    Outside,
}

impl std::fmt::Display for BioTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BioTag::Begin(t) => write!(f, "B-{}", t),
            BioTag::Inside(t) => write!(f, "I-{}", t),
            BioTag::Outside => write!(f, "O"),
        }
    }
}

// ─── Token ────────────────────────────────────────────────────────

/// A token with its text content and extracted features.
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub lowercase: String,
    pub is_capitalized: bool,
    pub is_all_upper: bool,
    pub is_digit: bool,
    pub has_dollar: bool,
    pub has_percent: bool,
    pub is_ticker_format: bool,
    pub length: usize,
}

impl Token {
    pub fn new(text: &str) -> Self {
        let lowercase = text.to_lowercase();
        let is_capitalized = text.chars().next().map_or(false, |c| c.is_uppercase());
        let is_all_upper = text.len() > 0 && text.chars().all(|c| c.is_uppercase() || !c.is_alphabetic());
        let is_digit = text.chars().any(|c| c.is_ascii_digit());
        let has_dollar = text.contains('$');
        let has_percent = text.contains('%');
        let alpha_chars: String = text.chars().filter(|c| c.is_alphabetic()).collect();
        let is_ticker_format = !alpha_chars.is_empty()
            && alpha_chars.len() <= 6
            && alpha_chars.chars().all(|c| c.is_uppercase())
            && text.chars().all(|c| c.is_uppercase() || !c.is_alphabetic());

        Self {
            text: text.to_string(),
            lowercase,
            is_capitalized,
            is_all_upper,
            is_digit,
            has_dollar,
            has_percent,
            is_ticker_format,
            length: text.len(),
        }
    }

    /// Convert token features to a numeric vector for the classifier.
    /// Features: [is_capitalized, is_all_upper, is_digit, has_dollar,
    ///            has_percent, is_ticker_format, length_normalized,
    ///            in_org_gazetteer, in_ticker_gazetteer, in_event_gazetteer]
    pub fn to_feature_vec(
        &self,
        in_org: bool,
        in_ticker: bool,
        in_event: bool,
    ) -> Vec<f64> {
        vec![
            self.is_capitalized as u8 as f64,
            self.is_all_upper as u8 as f64,
            self.is_digit as u8 as f64,
            self.has_dollar as u8 as f64,
            self.has_percent as u8 as f64,
            self.is_ticker_format as u8 as f64,
            (self.length as f64) / 20.0, // normalize length
            in_org as u8 as f64,
            in_ticker as u8 as f64,
            in_event as u8 as f64,
        ]
    }
}

// ─── Recognized Entity ────────────────────────────────────────────

/// A recognized entity span with type and confidence.
#[derive(Debug, Clone)]
pub struct RecognizedEntity {
    pub text: String,
    pub entity_type: EntityType,
    pub start_token: usize,
    pub end_token: usize,
    pub confidence: f64,
}

impl std::fmt::Display for RecognizedEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}: \"{}\" ({:.0}%)]",
            self.entity_type,
            self.text,
            self.confidence * 100.0
        )
    }
}

// ─── Financial NER ────────────────────────────────────────────────

/// Rule-based and gazetteer-enhanced NER for financial text.
pub struct FinancialNER {
    org_gazetteer: Vec<String>,
    ticker_gazetteer: Vec<String>,
    event_gazetteer: Vec<String>,
    instrument_gazetteer: Vec<String>,
}

impl FinancialNER {
    pub fn new() -> Self {
        Self {
            org_gazetteer: vec![
                "Goldman Sachs".into(), "JPMorgan".into(), "Morgan Stanley".into(),
                "BlackRock".into(), "Vanguard".into(), "Citadel".into(),
                "Apple".into(), "Google".into(), "Microsoft".into(),
                "Tesla".into(), "Amazon".into(), "Meta".into(),
                "Nvidia".into(), "Coinbase".into(), "Binance".into(),
                "Bybit".into(), "NYSE".into(), "NASDAQ".into(),
                "SEC".into(), "CFTC".into(), "Federal Reserve".into(),
                "Bank of America".into(), "Wells Fargo".into(),
                "Berkshire Hathaway".into(), "UBS".into(),
                "Deutsche Bank".into(), "HSBC".into(),
            ],
            ticker_gazetteer: vec![
                "AAPL".into(), "GOOGL".into(), "MSFT".into(), "TSLA".into(),
                "AMZN".into(), "META".into(), "NVDA".into(), "COIN".into(),
                "BTC".into(), "ETH".into(), "SOL".into(), "DOGE".into(),
                "BTCUSDT".into(), "ETHUSDT".into(), "SOLUSDT".into(),
                "BTCUSD".into(), "ETHUSD".into(), "SPY".into(), "QQQ".into(),
                "XRP".into(), "ADA".into(), "DOT".into(), "LINK".into(),
            ],
            event_gazetteer: vec![
                "IPO".into(), "merger".into(), "acquisition".into(),
                "stock split".into(), "dividend".into(), "earnings".into(),
                "bankruptcy".into(), "listing".into(), "delisting".into(),
                "buyback".into(), "spinoff".into(), "restructuring".into(),
                "default".into(), "downgrade".into(), "upgrade".into(),
            ],
            instrument_gazetteer: vec![
                "Treasury".into(), "futures".into(), "options".into(),
                "puts".into(), "calls".into(), "bonds".into(),
                "swap".into(), "CDO".into(), "CDS".into(),
                "ETF".into(), "mutual fund".into(),
            ],
        }
    }

    /// Add ticker symbols to the gazetteer (e.g., from Bybit API).
    pub fn add_tickers(&mut self, tickers: &[String]) {
        for t in tickers {
            if !self.ticker_gazetteer.contains(t) {
                self.ticker_gazetteer.push(t.clone());
            }
        }
    }

    /// Add organization names to the gazetteer.
    pub fn add_orgs(&mut self, orgs: &[String]) {
        for o in orgs {
            if !self.org_gazetteer.contains(o) {
                self.org_gazetteer.push(o.clone());
            }
        }
    }

    /// Tokenize input text into Token structs.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        text.split_whitespace()
            .map(|w| {
                // Strip trailing punctuation for cleaner matching
                let clean = w.trim_matches(|c: char| c == ',' || c == '.' || c == ';' || c == ':' || c == '!' || c == '?');
                Token::new(if clean.is_empty() { w } else { clean })
            })
            .collect()
    }

    /// Check if a token matches the org gazetteer (single-word match).
    pub fn is_org(&self, token: &Token) -> bool {
        self.org_gazetteer.iter().any(|org| {
            let parts: Vec<&str> = org.split_whitespace().collect();
            if parts.len() == 1 {
                parts[0].eq_ignore_ascii_case(&token.text)
            } else {
                false
            }
        })
    }

    /// Check if a token is part of a multi-word org name (returns position).
    pub fn multi_word_org_match(&self, tokens: &[Token], start: usize) -> Option<usize> {
        for org in &self.org_gazetteer {
            let parts: Vec<&str> = org.split_whitespace().collect();
            if parts.len() > 1 && start + parts.len() <= tokens.len() {
                let matches = parts.iter().enumerate().all(|(i, part)| {
                    part.eq_ignore_ascii_case(&tokens[start + i].text)
                });
                if matches {
                    return Some(parts.len());
                }
            }
        }
        None
    }

    /// Check if a token matches the ticker gazetteer.
    pub fn is_ticker(&self, token: &Token) -> bool {
        self.ticker_gazetteer.iter().any(|t| t.eq_ignore_ascii_case(&token.text))
    }

    /// Check if a token matches the event gazetteer.
    pub fn is_event(&self, token: &Token) -> bool {
        self.event_gazetteer.iter().any(|e| e.eq_ignore_ascii_case(&token.text))
    }

    /// Check if a token matches the instrument gazetteer.
    pub fn is_instrument(&self, token: &Token) -> bool {
        self.instrument_gazetteer.iter().any(|i| i.eq_ignore_ascii_case(&token.text))
    }

    /// Check if a token looks like a monetary value.
    fn is_money_token(token: &Token) -> bool {
        token.has_dollar
            || (token.is_digit && token.text.contains('.') && !token.has_percent)
            || token.text.starts_with('$')
            || token.text.starts_with('€')
            || token.text.starts_with('£')
            || token.text.starts_with('¥')
    }

    /// Check if a token looks like a money suffix (e.g., "billion", "million").
    fn is_money_suffix(token: &Token) -> bool {
        matches!(
            token.lowercase.as_str(),
            "billion" | "million" | "thousand" | "trillion"
                | "bn" | "mn" | "tn" | "k" | "m" | "b"
        )
    }

    /// Check if a token looks like a percentage.
    fn is_percent_token(token: &Token) -> bool {
        token.has_percent
            || (token.is_digit
                && token.text.ends_with('%'))
    }

    /// Check if a token looks like a date/temporal expression.
    fn is_date_token(token: &Token) -> bool {
        // Quarter references
        if token.text.len() == 2 && token.text.starts_with('Q') && token.text.chars().nth(1).map_or(false, |c| c.is_ascii_digit()) {
            return true;
        }
        // Year references (4-digit number starting with 19 or 20)
        if token.text.len() == 4 && token.text.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(year) = token.text.parse::<u32>() {
                if (1900..=2100).contains(&year) {
                    return true;
                }
            }
        }
        // Month names
        matches!(
            token.lowercase.as_str(),
            "january" | "february" | "march" | "april" | "may" | "june"
                | "july" | "august" | "september" | "october" | "november" | "december"
                | "jan" | "feb" | "mar" | "apr" | "jun" | "jul" | "aug" | "sep"
                | "oct" | "nov" | "dec"
        )
    }

    /// Predict BIO tags for a sequence of tokens.
    pub fn predict(&self, tokens: &[Token]) -> Vec<BioTag> {
        let mut tags = vec![BioTag::Outside; tokens.len()];
        let mut i = 0;

        while i < tokens.len() {
            // Check for multi-word org matches first
            if let Some(len) = self.multi_word_org_match(tokens, i) {
                tags[i] = BioTag::Begin(EntityType::Org);
                for j in 1..len {
                    tags[i + j] = BioTag::Inside(EntityType::Org);
                }
                i += len;
                continue;
            }

            let token = &tokens[i];

            // Gazetteer-based ticker check (exact match, high confidence)
            if self.is_ticker(token) {
                tags[i] = BioTag::Begin(EntityType::Ticker);
                i += 1;
                continue;
            }

            // Money check
            if Self::is_money_token(token) {
                tags[i] = BioTag::Begin(EntityType::Money);
                // Check for money suffix
                if i + 1 < tokens.len() && Self::is_money_suffix(&tokens[i + 1]) {
                    tags[i + 1] = BioTag::Inside(EntityType::Money);
                    i += 2;
                    continue;
                }
                i += 1;
                continue;
            }

            // Percent check
            if Self::is_percent_token(token) {
                tags[i] = BioTag::Begin(EntityType::Percent);
                i += 1;
                continue;
            }

            // Date check
            if Self::is_date_token(token) {
                tags[i] = BioTag::Begin(EntityType::Date);
                // Check for adjacent date parts (e.g., "Q3 2024")
                if i + 1 < tokens.len() && Self::is_date_token(&tokens[i + 1]) {
                    tags[i + 1] = BioTag::Inside(EntityType::Date);
                    i += 2;
                    continue;
                }
                i += 1;
                continue;
            }

            // Event check
            if self.is_event(token) {
                tags[i] = BioTag::Begin(EntityType::Event);
                i += 1;
                continue;
            }

            // Instrument check
            if self.is_instrument(token) {
                tags[i] = BioTag::Begin(EntityType::Instrument);
                i += 1;
                continue;
            }

            // Single-word org check
            if self.is_org(token) {
                tags[i] = BioTag::Begin(EntityType::Org);
                i += 1;
                continue;
            }

            // Heuristic ticker format check (all-uppercase, 3-5 letters, not a common word)
            if token.is_ticker_format && token.length >= 3 && token.length <= 5 {
                let common_words = ["THE", "AND", "FOR", "NOT", "BUT", "ALL", "HER", "HIS",
                    "WAS", "ARE", "HAS", "HAD", "CAN", "DID", "GET", "LET", "MAY", "NEW",
                    "NOW", "OLD", "SEE", "WAY", "WHO", "OUR", "OUT", "DAY", "USE"];
                if !common_words.contains(&token.text.as_str()) {
                    tags[i] = BioTag::Begin(EntityType::Ticker);
                    i += 1;
                    continue;
                }
            }

            // Unknown capitalized word after financial keywords may be org
            if token.is_capitalized && i > 0 {
                let prev = &tokens[i - 1].lowercase;
                if matches!(prev.as_str(), "acquired" | "bought" | "sold" | "invested" | "company" | "firm" | "corporation" | "exchange") {
                    tags[i] = BioTag::Begin(EntityType::Org);
                    // Check if next token is also capitalized (multi-word name)
                    let mut j = i + 1;
                    while j < tokens.len() && tokens[j].is_capitalized && !self.is_ticker(&tokens[j]) {
                        tags[j] = BioTag::Inside(EntityType::Org);
                        j += 1;
                    }
                    i = j;
                    continue;
                }
            }

            i += 1;
        }

        tags
    }

    /// Extract recognized entities from text.
    pub fn extract_entities(&self, text: &str) -> Vec<RecognizedEntity> {
        let tokens = self.tokenize(text);
        let tags = self.predict(&tokens);
        let mut entities = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            if let BioTag::Begin(entity_type) = &tags[i] {
                let start = i;
                let mut end = i + 1;

                // Collect continuation tokens
                while end < tokens.len() {
                    if let BioTag::Inside(inner_type) = &tags[end] {
                        if inner_type == entity_type {
                            end += 1;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                let entity_text: String = tokens[start..end]
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                // Confidence based on matching method
                let confidence = if self.is_in_any_gazetteer(&tokens[start]) {
                    0.95
                } else if tokens[start].is_ticker_format {
                    0.85
                } else {
                    0.70
                };

                entities.push(RecognizedEntity {
                    text: entity_text,
                    entity_type: *entity_type,
                    start_token: start,
                    end_token: end,
                    confidence,
                });

                i = end;
            } else {
                i += 1;
            }
        }

        entities
    }

    fn is_in_any_gazetteer(&self, token: &Token) -> bool {
        self.is_org(token) || self.is_ticker(token) || self.is_event(token) || self.is_instrument(token)
    }
}

impl Default for FinancialNER {
    fn default() -> Self {
        Self::new()
    }
}

// ─── NER Classifier (Multi-class Logistic Regression) ─────────────

/// Multi-class logistic regression for token classification.
///
/// Classifies tokens into entity types based on feature vectors.
/// Labels: 0=O, 1=B-ORG, 2=B-TICKER, 3=B-MONEY, 4=B-PERCENT,
///         5=B-DATE, 6=B-EVENT, 7=B-INSTRUMENT
#[derive(Debug)]
pub struct NERClassifier {
    weights: Vec<Array1<f64>>,
    biases: Vec<f64>,
    learning_rate: f64,
    num_features: usize,
    num_classes: usize,
}

impl NERClassifier {
    pub fn new(num_features: usize, num_classes: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_classes)
            .map(|_| {
                Array1::from_vec(
                    (0..num_features)
                        .map(|_| rng.gen_range(-0.1..0.1))
                        .collect(),
                )
            })
            .collect();
        let biases = vec![0.0; num_classes];

        Self {
            weights,
            biases,
            learning_rate,
            num_features,
            num_classes,
        }
    }

    /// Softmax over raw scores.
    fn softmax(scores: &[f64]) -> Vec<f64> {
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        exp_scores.iter().map(|e| e / sum).collect()
    }

    /// Predict class probabilities for a feature vector.
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        let scores: Vec<f64> = (0..self.num_classes)
            .map(|c| self.weights[c].dot(&x) + self.biases[c])
            .collect();
        Self::softmax(&scores)
    }

    /// Predict the most likely class and its confidence.
    pub fn predict(&self, features: &[f64]) -> (usize, f64) {
        let probs = self.predict_proba(features);
        let (best_class, &best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        (best_class, best_prob)
    }

    /// Train on labeled data for a given number of epochs.
    /// Each sample is (features, class_index).
    pub fn train(&mut self, data: &[(Vec<f64>, usize)], epochs: usize) {
        for _ in 0..epochs {
            for (features, label) in data {
                let x = Array1::from_vec(features.clone());
                let probs = self.predict_proba(features);

                for c in 0..self.num_classes {
                    let target = if c == *label { 1.0 } else { 0.0 };
                    let error = probs[c] - target;

                    for j in 0..self.num_features {
                        self.weights[c][j] -= self.learning_rate * error * x[j];
                    }
                    self.biases[c] -= self.learning_rate * error;
                }
            }
        }
    }

    /// Evaluate accuracy on a test set.
    pub fn accuracy(&self, data: &[(Vec<f64>, usize)]) -> f64 {
        let correct = data
            .iter()
            .filter(|(features, label)| {
                let (pred, _) = self.predict(features);
                pred == *label
            })
            .count();
        correct as f64 / data.len() as f64
    }

    /// Map class index to label name.
    pub fn class_name(class: usize) -> &'static str {
        match class {
            0 => "O",
            1 => "B-ORG",
            2 => "B-TICKER",
            3 => "B-MONEY",
            4 => "B-PERCENT",
            5 => "B-DATE",
            6 => "B-EVENT",
            7 => "B-INSTRUMENT",
            _ => "UNKNOWN",
        }
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

#[derive(Debug, Deserialize)]
pub struct TickerResult {
    pub list: Vec<TickerInfo>,
}

#[derive(Debug, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
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

    /// Fetch ticker information for all spot symbols.
    pub async fn get_tickers(&self) -> anyhow::Result<Vec<TickerInfo>> {
        let url = format!(
            "{}/v5/market/tickers?category=spot",
            self.base_url
        );
        let resp: BybitResponse<TickerResult> = self.client.get(&url).send().await?.json().await?;
        Ok(resp.result.list)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ─────────────────────────────────────

/// Generate synthetic NER training data.
///
/// Each sample has features [is_capitalized, is_all_upper, is_digit, has_dollar,
/// has_percent, is_ticker_format, length_norm, in_org, in_ticker, in_event]
/// and a class label (0=O, 1=B-ORG, 2=B-TICKER, 3=B-MONEY, 4=B-PERCENT,
/// 5=B-DATE, 6=B-EVENT, 7=B-INSTRUMENT).
pub fn generate_training_data(n: usize) -> Vec<(Vec<f64>, usize)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let class = rng.gen_range(0..8);
        let features = match class {
            0 => {
                // O (outside) - typical word
                vec![
                    rng.gen_range(0.0..0.3),  // rarely capitalized
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    rng.gen_range(0.2..0.6),  // medium length
                    0.0, 0.0, 0.0,
                ]
            }
            1 => {
                // B-ORG
                vec![
                    1.0,  // capitalized
                    rng.gen_range(0.0..0.3),
                    0.0, 0.0, 0.0, 0.0,
                    rng.gen_range(0.3..0.8),
                    if rng.gen_bool(0.7) { 1.0 } else { 0.0 },  // often in gazetteer
                    0.0, 0.0,
                ]
            }
            2 => {
                // B-TICKER
                vec![
                    1.0, 1.0, // all upper
                    0.0, 0.0, 0.0,
                    1.0, // ticker format
                    rng.gen_range(0.1..0.3),  // short
                    0.0,
                    if rng.gen_bool(0.8) { 1.0 } else { 0.0 },
                    0.0,
                ]
            }
            3 => {
                // B-MONEY
                vec![
                    0.0, 0.0,
                    1.0, // has digit
                    if rng.gen_bool(0.8) { 1.0 } else { 0.0 }, // usually has dollar
                    0.0, 0.0,
                    rng.gen_range(0.2..0.5),
                    0.0, 0.0, 0.0,
                ]
            }
            4 => {
                // B-PERCENT
                vec![
                    0.0, 0.0,
                    1.0,  // has digit
                    0.0,
                    1.0,  // has percent
                    0.0,
                    rng.gen_range(0.1..0.3),
                    0.0, 0.0, 0.0,
                ]
            }
            5 => {
                // B-DATE
                vec![
                    rng.gen_range(0.3..0.7),
                    rng.gen_range(0.0..0.3),
                    if rng.gen_bool(0.5) { 1.0 } else { 0.0 },
                    0.0, 0.0, 0.0,
                    rng.gen_range(0.1..0.4),
                    0.0, 0.0, 0.0,
                ]
            }
            6 => {
                // B-EVENT
                vec![
                    rng.gen_range(0.3..0.7),
                    rng.gen_range(0.2..0.5),
                    0.0, 0.0, 0.0, 0.0,
                    rng.gen_range(0.2..0.6),
                    0.0, 0.0,
                    if rng.gen_bool(0.7) { 1.0 } else { 0.0 },
                ]
            }
            7 => {
                // B-INSTRUMENT
                vec![
                    rng.gen_range(0.5..1.0),
                    rng.gen_range(0.0..0.3),
                    0.0, 0.0, 0.0, 0.0,
                    rng.gen_range(0.3..0.7),
                    0.0, 0.0, 0.0,
                ]
            }
            _ => unreachable!(),
        };

        // Add some noise
        let noisy_features: Vec<f64> = features
            .iter()
            .map(|f| (f + rng.gen_range(-0.05_f64..0.05_f64)).clamp(0.0, 1.0))
            .collect();

        data.push((noisy_features, class));
    }
    data
}

/// Generate sample financial sentences for NER demonstration.
pub fn sample_financial_texts() -> Vec<&'static str> {
    vec![
        "Goldman Sachs reported $2.5 billion revenue in Q3 2024",
        "AAPL shares rose 5.2% after Apple announced a stock split",
        "The Federal Reserve raised interest rates by 25 basis points",
        "BTCUSDT traded at $67,500 with volume of 1,234 BTC on Bybit",
        "Tesla acquired a startup for $500 million in January 2024",
        "JPMorgan upgraded NVDA to overweight with a $150 price target",
        "SEC investigation into Coinbase caused COIN to drop 12%",
        "Morgan Stanley launched a new ETF tracking S&P 500 futures",
        "ETHUSDT surged 8.3% following the Ethereum merger upgrade",
        "Bank of America reported earnings of $0.85 per share in Q2 2024",
    ]
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_features() {
        let token = Token::new("AAPL");
        assert!(token.is_all_upper);
        assert!(token.is_ticker_format);
        assert!(!token.has_dollar);
        assert!(!token.is_digit);

        let money = Token::new("$2.5B");
        assert!(money.has_dollar);
        assert!(money.is_digit);

        let pct = Token::new("12%");
        assert!(pct.has_percent);
        assert!(pct.is_digit);
    }

    #[test]
    fn test_bio_tag_display() {
        assert_eq!(format!("{}", BioTag::Begin(EntityType::Org)), "B-ORG");
        assert_eq!(format!("{}", BioTag::Inside(EntityType::Money)), "I-MONEY");
        assert_eq!(format!("{}", BioTag::Outside), "O");
    }

    #[test]
    fn test_ner_single_org() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("Apple reported earnings");
        let tags = ner.predict(&tokens);
        assert_eq!(tags[0], BioTag::Begin(EntityType::Org));
        assert_eq!(tags[1], BioTag::Outside);
    }

    #[test]
    fn test_ner_multi_word_org() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("Goldman Sachs reported revenue");
        let tags = ner.predict(&tokens);
        assert_eq!(tags[0], BioTag::Begin(EntityType::Org));
        assert_eq!(tags[1], BioTag::Inside(EntityType::Org));
        assert_eq!(tags[2], BioTag::Outside);
    }

    #[test]
    fn test_ner_ticker() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("AAPL shares rose today");
        let tags = ner.predict(&tokens);
        assert_eq!(tags[0], BioTag::Begin(EntityType::Ticker));
        assert_eq!(tags[1], BioTag::Outside);
    }

    #[test]
    fn test_ner_crypto_ticker() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("BTCUSDT surged on Bybit");
        let tags = ner.predict(&tokens);
        assert_eq!(tags[0], BioTag::Begin(EntityType::Ticker));
        assert_eq!(tags[3], BioTag::Begin(EntityType::Org));
    }

    #[test]
    fn test_ner_money() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("Revenue was $2.5 billion");
        let tags = ner.predict(&tokens);
        // "$2.5" should be B-MONEY, "billion" should be I-MONEY
        assert_eq!(tags[2], BioTag::Begin(EntityType::Money));
        assert_eq!(tags[3], BioTag::Inside(EntityType::Money));
    }

    #[test]
    fn test_ner_percent() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("Stock rose 5.2% today");
        let tags = ner.predict(&tokens);
        assert_eq!(tags[2], BioTag::Begin(EntityType::Percent));
    }

    #[test]
    fn test_ner_date() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("In Q3 2024 results");
        let tags = ner.predict(&tokens);
        assert_eq!(tags[1], BioTag::Begin(EntityType::Date));
        assert_eq!(tags[2], BioTag::Inside(EntityType::Date));
    }

    #[test]
    fn test_ner_event() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("The IPO was successful");
        let tags = ner.predict(&tokens);
        assert_eq!(tags[1], BioTag::Begin(EntityType::Event));
    }

    #[test]
    fn test_extract_entities() {
        let ner = FinancialNER::new();
        let entities = ner.extract_entities("Goldman Sachs reported $2.5 billion revenue in Q3 2024");
        assert!(entities.len() >= 3); // At least ORG, MONEY, DATE
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Org));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Money));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
    }

    #[test]
    fn test_add_tickers() {
        let mut ner = FinancialNER::new();
        ner.add_tickers(&["PEPEUSDT".to_string(), "WIFUSDT".to_string()]);
        let tokens = ner.tokenize("PEPEUSDT pumped today");
        assert!(ner.is_ticker(&tokens[0]));
    }

    #[test]
    fn test_classifier_predict() {
        let clf = NERClassifier::new(10, 8, 0.01);
        let features = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 1.0, 0.0];
        let (class, confidence) = clf.predict(&features);
        assert!(class < 8);
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_classifier_train() {
        let data = generate_training_data(500);
        let (train, test) = data.split_at(400);

        let mut clf = NERClassifier::new(10, 8, 0.01);
        let acc_before = clf.accuracy(test);

        clf.train(&train.to_vec(), 50);
        let acc_after = clf.accuracy(test);

        // After training, accuracy should improve
        assert!(acc_after > 0.0);
        assert!(acc_after >= acc_before * 0.9, "accuracy should not degrade significantly");
    }

    #[test]
    fn test_training_data_generation() {
        let data = generate_training_data(100);
        assert_eq!(data.len(), 100);
        for (features, label) in &data {
            assert_eq!(features.len(), 10);
            assert!(*label < 8);
        }
    }

    #[test]
    fn test_sample_texts() {
        let texts = sample_financial_texts();
        assert!(!texts.is_empty());
        let ner = FinancialNER::new();
        for text in texts {
            let entities = ner.extract_entities(text);
            // Each sample text should have at least one entity
            assert!(!entities.is_empty(), "No entities found in: {}", text);
        }
    }

    #[test]
    fn test_entity_display() {
        let entity = RecognizedEntity {
            text: "Goldman Sachs".to_string(),
            entity_type: EntityType::Org,
            start_token: 0,
            end_token: 2,
            confidence: 0.95,
        };
        let display = format!("{}", entity);
        assert!(display.contains("ORG"));
        assert!(display.contains("Goldman Sachs"));
        assert!(display.contains("95%"));
    }

    #[test]
    fn test_tokenize() {
        let ner = FinancialNER::new();
        let tokens = ner.tokenize("Hello, world! Price is $100.");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[4].text, "$100");
    }
}
