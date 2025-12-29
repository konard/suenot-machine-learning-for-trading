//! Example: Sentiment analysis for crypto texts
//!
//! This example demonstrates how to use word embeddings
//! for sentiment analysis of cryptocurrency trading texts.
//!
//! Run with: cargo run --example analyze_sentiment

use crypto_embeddings::{Word2Vec, Tokenizer, SimilarityAnalyzer};
use crypto_embeddings::analysis::SentimentAnalyzer;
use anyhow::Result;

fn main() -> Result<()> {
    env_logger::init();

    println!("=== Crypto Sentiment Analysis Example ===\n");

    // Train a model on sample data first
    println!("1. Preparing training data...");
    let training_texts = vec![
        // Bullish texts
        "BTC showing strong bullish momentum breaking resistance",
        "ETH pumping hard bulls in control moon soon",
        "Massive accumulation detected whales buying Bitcoin",
        "Breakout confirmed uptrend intact higher highs",
        "Bull market back crypto adoption accelerating",
        "New ATH incoming momentum building strongly",
        "Institutional buying increasing bullish signal",
        "Support holding strong bounce expected rally",
        // Bearish texts
        "BTC dumping hard breaking support levels",
        "ETH crash imminent bears taking control",
        "Whales selling massive distribution detected",
        "Breakdown confirmed downtrend bear market",
        "Liquidations cascading panic selling",
        "Resistance rejection failed rally weak",
        "FUD spreading fear uncertainty doubt",
        "Scam project rug pull warning",
        // Neutral texts
        "BTC consolidating sideways range bound",
        "ETH testing key levels watching closely",
        "Market awaiting catalyst direction unclear",
        "Volume low waiting for breakout",
    ];

    let tokenizer = Tokenizer::new();
    let sentences: Vec<Vec<String>> = training_texts
        .iter()
        .map(|t| tokenizer.tokenize(t))
        .collect();

    println!("   Training texts: {}", training_texts.len());

    // Build and train model
    println!("\n2. Training embeddings model...");
    let mut model = Word2Vec::new(50, 3, 1);
    model.build_vocab(&sentences);
    model.train(&sentences, 10)?;
    println!("   Vocabulary size: {}", model.vocab_size());

    // Create sentiment analyzer
    let sentiment_analyzer = SentimentAnalyzer::new(model.clone());

    // Test texts
    let test_texts = vec![
        ("Bullish news", "BTC breaking out massive pump incoming bulls winning"),
        ("Bearish news", "Market crashing liquidations everywhere panic selling"),
        ("Neutral news", "BTC trading sideways waiting for direction"),
        ("Mixed news", "Some bullish signals but resistance holding"),
        ("Strong bullish", "To the moon! ATH incoming bull run confirmed"),
        ("Strong bearish", "It's over dump it all scam market dead"),
    ];

    println!("\n3. Analyzing sentiment...\n");
    println!("{:>15} | {:>8} | {:>10} | Details", "Type", "Score", "Label");
    println!("{}", "-".repeat(60));

    for (text_type, text) in &test_texts {
        let result = sentiment_analyzer.analyze(text);

        let label_str = match result.label {
            crypto_embeddings::analysis::SentimentLabel::Positive => "Positive",
            crypto_embeddings::analysis::SentimentLabel::Negative => "Negative",
            crypto_embeddings::analysis::SentimentLabel::Neutral => "Neutral",
        };

        println!(
            "{:>15} | {:>+8.2} | {:>10} | +{:.0} -{:.0}",
            text_type,
            result.score,
            label_str,
            result.positive_score,
            result.negative_score
        );

        if !result.positive_words.is_empty() || !result.negative_words.is_empty() {
            if !result.positive_words.is_empty() {
                println!("{:>15}   Positive words: {:?}", "", result.positive_words);
            }
            if !result.negative_words.is_empty() {
                println!("{:>15}   Negative words: {:?}", "", result.negative_words);
            }
        }
    }

    // Document similarity analysis
    println!("\n4. Document similarity analysis...\n");

    let similarity_analyzer = SimilarityAnalyzer::new(model);

    let reference = "BTC showing bullish momentum breaking resistance";
    let candidates = vec![
        "Bitcoin bulls pushing price higher breakout",
        "ETH consolidating near support level",
        "Market dumping bears in control",
        "Crypto adoption accelerating institutional buying",
    ];

    println!("Reference: \"{}\"\n", reference);
    println!("Similarities:");

    for candidate in &candidates {
        match similarity_analyzer.text_similarity(reference, candidate) {
            Ok(sim) => {
                let bar_len = ((sim + 1.0) / 2.0 * 20.0) as usize;
                let bar: String = "#".repeat(bar_len) + &"-".repeat(20 - bar_len);
                println!("  [{bar}] {:.3}: \"{}\"", sim, candidate);
            }
            Err(_) => println!("  Could not compute similarity for: {}", candidate),
        }
    }

    // Simulated real-time analysis
    println!("\n5. Simulated real-time sentiment tracking...\n");

    let timeline = vec![
        ("09:00", "Market opens BTC stable around 45000"),
        ("09:15", "Breaking BTC pumping breaking 46000 bulls winning"),
        ("09:30", "Massive buying pressure volume spike detected"),
        ("09:45", "ATH watch BTC approaching resistance at 47000"),
        ("10:00", "Rejection at resistance some profit taking"),
        ("10:15", "Flash crash BTC dumping hard to 44000"),
        ("10:30", "Panic selling liquidations cascade"),
        ("10:45", "Support holding at 43500 bounce starting"),
        ("11:00", "Recovery underway V shape forming"),
    ];

    println!("{:>8} | {:>6} | {:>20}", "Time", "Score", "Sentiment");
    println!("{}", "-".repeat(40));

    let sentiment_analyzer = SentimentAnalyzer::new(Word2Vec::new(10, 2, 1)); // Use lexicon-based only

    for (time, text) in &timeline {
        let result = sentiment_analyzer.analyze(text);
        let score = result.score;

        let sentiment_bar = if score > 0.3 {
            format!("{:>20}", "#".repeat((score * 10.0) as usize))
        } else if score < -0.3 {
            format!("{:<20}", "#".repeat((-score * 10.0) as usize))
        } else {
            format!("{:^20}", "---")
        };

        println!("{:>8} | {:>+6.2} | {}", time, score, sentiment_bar);
    }

    println!("\n=== Analysis Complete ===");

    Ok(())
}
