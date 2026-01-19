//! # Text Preprocessing
//!
//! Text cleaning and preprocessing utilities for financial text analysis.

use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use std::collections::HashSet;

/// Text preprocessor for financial text
pub struct TextPreprocessor {
    /// Regex for URL removal
    url_regex: Regex,
    /// Regex for mention removal
    mention_regex: Regex,
    /// Regex for hashtag extraction
    hashtag_regex: Regex,
    /// Regex for ticker extraction (e.g., $BTC, $ETH)
    ticker_regex: Regex,
    /// Regex for multiple whitespace
    whitespace_regex: Regex,
    /// Stop words to remove
    stop_words: HashSet<String>,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextPreprocessor {
    /// Create a new text preprocessor
    pub fn new() -> Self {
        let stop_words: HashSet<String> = vec![
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "although",
            "this", "that", "these", "those", "i", "me", "my", "myself", "we",
            "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "they", "them", "their", "theirs",
            "themselves", "what", "which", "who", "whom", "am",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            url_regex: Regex::new(r"https?://\S+|www\.\S+").unwrap(),
            mention_regex: Regex::new(r"@\w+").unwrap(),
            hashtag_regex: Regex::new(r"#(\w+)").unwrap(),
            ticker_regex: Regex::new(r"\$([A-Z]{2,5})").unwrap(),
            whitespace_regex: Regex::new(r"\s+").unwrap(),
            stop_words,
        }
    }

    /// Preprocess text for sentiment analysis
    ///
    /// Steps:
    /// 1. Unicode normalization
    /// 2. Remove URLs
    /// 3. Remove mentions
    /// 4. Convert to lowercase
    /// 5. Normalize whitespace
    pub fn preprocess(&self, text: &str) -> String {
        // Unicode normalization (NFC)
        let normalized: String = text.nfc().collect();

        // Remove URLs
        let no_urls = self.url_regex.replace_all(&normalized, "");

        // Remove mentions
        let no_mentions = self.mention_regex.replace_all(&no_urls, "");

        // Convert to lowercase
        let lowercase = no_mentions.to_lowercase();

        // Normalize whitespace
        let clean = self.whitespace_regex.replace_all(&lowercase, " ");

        clean.trim().to_string()
    }

    /// Extract stock/crypto tickers from text (e.g., $BTC, $ETH)
    pub fn extract_tickers(&self, text: &str) -> Vec<String> {
        self.ticker_regex
            .captures_iter(text)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    /// Extract hashtags from text
    pub fn extract_hashtags(&self, text: &str) -> Vec<String> {
        self.hashtag_regex
            .captures_iter(text)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_lowercase()))
            .collect()
    }

    /// Remove stop words from text
    pub fn remove_stop_words(&self, text: &str) -> String {
        text.split_whitespace()
            .filter(|word| !self.stop_words.contains(&word.to_lowercase()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Tokenize text into words
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    /// Prepare text for LLM input (minimal preprocessing)
    pub fn prepare_for_llm(&self, text: &str) -> String {
        // Unicode normalization
        let normalized: String = text.nfc().collect();

        // Normalize whitespace
        let clean = self.whitespace_regex.replace_all(&normalized, " ");

        clean.trim().to_string()
    }
}

/// Financial-specific text cleaner
pub struct FinancialTextCleaner {
    /// Preprocessor for basic text cleaning
    preprocessor: TextPreprocessor,
    /// Financial abbreviation expansions
    abbreviations: Vec<(Regex, &'static str)>,
}

impl Default for FinancialTextCleaner {
    fn default() -> Self {
        Self::new()
    }
}

impl FinancialTextCleaner {
    /// Create a new financial text cleaner
    pub fn new() -> Self {
        let abbreviations = vec![
            (Regex::new(r"\bQ[1-4]\b").unwrap(), "quarter"),
            (Regex::new(r"\bYoY\b").unwrap(), "year over year"),
            (Regex::new(r"\bQoQ\b").unwrap(), "quarter over quarter"),
            (Regex::new(r"\bMoM\b").unwrap(), "month over month"),
            (Regex::new(r"\bEPS\b").unwrap(), "earnings per share"),
            (Regex::new(r"\bP/E\b").unwrap(), "price to earnings"),
            (Regex::new(r"\bATH\b").unwrap(), "all time high"),
            (Regex::new(r"\bATL\b").unwrap(), "all time low"),
            (Regex::new(r"\bFOMO\b").unwrap(), "fear of missing out"),
            (Regex::new(r"\bFUD\b").unwrap(), "fear uncertainty doubt"),
            (Regex::new(r"\bHODL\b").unwrap(), "hold"),
            (Regex::new(r"\bDCA\b").unwrap(), "dollar cost averaging"),
        ];

        Self {
            preprocessor: TextPreprocessor::new(),
            abbreviations,
        }
    }

    /// Clean financial text with abbreviation expansion
    pub fn clean(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Expand abbreviations
        for (regex, expansion) in &self.abbreviations {
            result = regex.replace_all(&result, *expansion).to_string();
        }

        // Apply basic preprocessing
        self.preprocessor.preprocess(&result)
    }

    /// Extract financial entities (tickers, amounts, percentages)
    pub fn extract_entities(&self, text: &str) -> FinancialEntities {
        let tickers = self.preprocessor.extract_tickers(text);

        let percentage_regex = Regex::new(r"(\d+(?:\.\d+)?)\s*%").unwrap();
        let percentages: Vec<f64> = percentage_regex
            .captures_iter(text)
            .filter_map(|cap| cap.get(1).and_then(|m| m.as_str().parse().ok()))
            .collect();

        let amount_regex = Regex::new(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([BMKbmk])?").unwrap();
        let amounts: Vec<String> = amount_regex
            .captures_iter(text)
            .map(|cap| cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default())
            .collect();

        FinancialEntities {
            tickers,
            percentages,
            amounts,
        }
    }
}

/// Extracted financial entities from text
#[derive(Debug, Clone)]
pub struct FinancialEntities {
    /// Stock/crypto tickers
    pub tickers: Vec<String>,
    /// Percentage values
    pub percentages: Vec<f64>,
    /// Dollar amounts
    pub amounts: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_basic() {
        let preprocessor = TextPreprocessor::new();
        let text = "  Hello   World  ";
        assert_eq!(preprocessor.preprocess(text), "hello world");
    }

    #[test]
    fn test_remove_urls() {
        let preprocessor = TextPreprocessor::new();
        let text = "Check this out https://example.com great news!";
        assert_eq!(preprocessor.preprocess(text), "check this out great news!");
    }

    #[test]
    fn test_extract_tickers() {
        let preprocessor = TextPreprocessor::new();
        let text = "$BTC is going up, $ETH follows! But $DOGE?";
        let tickers = preprocessor.extract_tickers(text);
        assert_eq!(tickers, vec!["BTC", "ETH", "DOGE"]);
    }

    #[test]
    fn test_extract_hashtags() {
        let preprocessor = TextPreprocessor::new();
        let text = "#Bitcoin is trending! #crypto #moon";
        let hashtags = preprocessor.extract_hashtags(text);
        assert_eq!(hashtags, vec!["bitcoin", "crypto", "moon"]);
    }

    #[test]
    fn test_financial_cleaner() {
        let cleaner = FinancialTextCleaner::new();
        let text = "BTC hit ATH! HODL your coins, don't FOMO!";
        let clean = cleaner.clean(text);
        assert!(clean.contains("all time high"));
        assert!(clean.contains("hold"));
    }

    #[test]
    fn test_extract_entities() {
        let cleaner = FinancialTextCleaner::new();
        let text = "$BTC surges 10.5%! $ETH follows with 5%";
        let entities = cleaner.extract_entities(text);
        assert_eq!(entities.tickers, vec!["BTC", "ETH"]);
        assert!(entities.percentages.contains(&10.5));
        assert!(entities.percentages.contains(&5.0));
    }
}
