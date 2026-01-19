//! # Sentiment Analysis Example
//!
//! Demonstrates basic sentiment analysis using the financial lexicon.

use llm_sentiment::{SentimentAnalyzer, TextPreprocessor};

fn main() {
    println!("=== LLM Sentiment Analysis Demo ===\n");

    // Initialize analyzer
    let analyzer = SentimentAnalyzer::new();
    let preprocessor = TextPreprocessor::new();

    // Example financial texts
    let texts = vec![
        "Bitcoin surges 10% as institutional adoption accelerates!",
        "Market crash fears grow as inflation data disappoints investors.",
        "The company reported earnings in line with expectations.",
        "Stock rally continues with bullish momentum breaking resistance.",
        "Warning: Potential rug pull detected in this DeFi protocol.",
        "Tesla misses delivery targets amid supply chain issues.",
        "Apple's services revenue hits record high despite iPhone slump.",
        "Crypto market showing extreme fear after regulatory concerns.",
    ];

    println!("Analyzing {} texts...\n", texts.len());
    println!("{}", "=".repeat(70));

    for text in texts {
        let result = analyzer.analyze(text);

        println!("\nText: \"{}\"", text);
        println!("{}", "-".repeat(60));
        println!("  Sentiment Score: {:.3}", result.score);
        println!("  Sentiment Level: {}", result.level.as_str());
        println!("  Confidence: {:.1}%", result.confidence * 100.0);
        println!("  Lexicon Score: {:.3}", result.lexicon_score);
        println!("  Crypto Score: {:.3}", result.crypto_score);

        if !result.key_words.is_empty() {
            println!("  Key Words: {}", result.key_words.join(", "));
        }
        if !result.crypto_terms.is_empty() {
            println!("  Crypto Terms: {}", result.crypto_terms.join(", "));
        }
    }

    println!("\n{}", "=".repeat(70));

    // Demonstrate preprocessing
    println!("\n=== Text Preprocessing Demo ===\n");

    let raw_text = "Check out $BTC at https://example.com - it's MOONING! 🚀 @trader123";
    println!("Raw text: \"{}\"", raw_text);

    let preprocessed = preprocessor.preprocess(raw_text);
    println!("Preprocessed: \"{}\"", preprocessed);

    let tickers = preprocessor.extract_tickers(raw_text);
    println!("Extracted tickers: {:?}", tickers);

    let hashtags = preprocessor.extract_hashtags("#Bitcoin is #bullish today!");
    println!("Extracted hashtags: {:?}", hashtags);

    println!("\n=== Analysis Complete ===");
}
