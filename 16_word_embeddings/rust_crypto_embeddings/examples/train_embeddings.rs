//! Example: Training word embeddings on crypto texts
//!
//! This example demonstrates how to train Word2Vec embeddings
//! on cryptocurrency-related text data.
//!
//! Run with: cargo run --example train_embeddings

use crypto_embeddings::{Word2Vec, Tokenizer};
use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    env_logger::init();

    println!("=== Word2Vec Training Example ===\n");

    // Sample crypto trading texts (in production, load from file)
    let texts = vec![
        "BTC showing strong bullish momentum after breaking key resistance at 45000",
        "ETH consolidating near support levels awaiting next catalyst",
        "SOL experiencing massive pump following DeFi announcement",
        "Bitcoin bulls pushing price higher amid institutional buying",
        "Ethereum gas fees dropping as layer 2 adoption increases",
        "Solana network faces congestion during NFT mint event",
        "BTC forming bullish flag pattern on 4H timeframe",
        "ETH breaking out of descending triangle with volume",
        "Market sentiment turning bullish as BTC reclaims 50k",
        "Whale accumulation detected on chain for Bitcoin",
        "Altcoins rallying as Bitcoin dominance drops",
        "DeFi TVL reaching new highs on Ethereum and Solana",
        "BTC RSI showing oversold conditions on daily chart",
        "ETH MACD crossover signals potential trend reversal",
        "SOL forming higher highs and higher lows uptrend intact",
        "Bitcoin mining difficulty adjustment incoming",
        "Ethereum staking yields attractive for long term holders",
        "Crypto market cap surpassing 2 trillion dollars",
        "BTC halving cycle analysis suggests bull run continuation",
        "Institutional investors increasing Bitcoin allocation",
        "ETH merge upgrade completed successfully network transition",
        "SOL ecosystem growing with new DeFi protocols launching",
        "Bitcoin as digital gold narrative strengthening",
        "Ethereum scaling solutions reducing transaction costs",
        "Market volatility increasing as options expiry approaches",
        "BTC testing critical support zone buyers stepping in",
        "ETH showing strength relative to BTC ETH BTC pair rising",
        "Crypto adoption growing in emerging markets",
        "Bitcoin ATH approaching as momentum builds",
        "DeFi yields compressing as market matures",
    ];

    // 1. Tokenize the corpus
    println!("1. Tokenizing corpus...");
    let tokenizer = Tokenizer::new();
    let sentences: Vec<Vec<String>> = texts
        .iter()
        .map(|text| tokenizer.tokenize(text))
        .collect();

    println!("   Sentences: {}", sentences.len());
    println!("   Sample tokens: {:?}", &sentences[0]);
    println!();

    // 2. Detect and apply phrases
    println!("2. Detecting phrases...");
    let phrases = tokenizer.detect_phrases(&sentences, 2);
    println!("   Found {} phrases", phrases.len());
    for phrase in phrases.iter().take(10) {
        println!("   - {}", phrase);
    }

    let sentences = tokenizer.apply_phrases(&sentences, &phrases);
    println!();

    // 3. Build vocabulary and train model
    println!("3. Training Word2Vec model...");
    let mut model = Word2Vec::new(50, 3, 2);
    model.build_vocab(&sentences);
    println!("   Vocabulary size: {}", model.vocab_size());

    model.train(&sentences, 10)?;
    println!("   Training complete!");
    println!();

    // 4. Explore embeddings
    println!("4. Exploring word relationships...\n");

    // Most similar words
    let test_words = ["btc", "bullish", "support", "defi"];
    for word in test_words {
        if model.contains(word) {
            println!("   Words similar to '{}':", word);
            match model.most_similar(word, 5) {
                Ok(similar) => {
                    for (w, score) in similar {
                        println!("     - {}: {:.4}", w, score);
                    }
                }
                Err(e) => println!("     Error: {}", e),
            }
            println!();
        } else {
            println!("   '{}' not in vocabulary\n", word);
        }
    }

    // 5. Word analogies
    println!("5. Testing analogies...\n");

    // BTC is to bullish as ETH is to ?
    println!("   btc + bullish - eth = ?");
    match model.analogy(
        &["btc".to_string(), "bullish".to_string()],
        &["eth".to_string()],
        5,
    ) {
        Ok(results) => {
            for (w, score) in results {
                println!("     - {}: {:.4}", w, score);
            }
        }
        Err(e) => println!("     Error: {}", e),
    }
    println!();

    // 6. Save model
    let model_path = PathBuf::from("crypto_embeddings.vec");
    println!("6. Saving model to {:?}...", model_path);
    model.save(&model_path)?;
    println!("   Model saved successfully!");
    println!();

    // 7. Load and verify
    println!("7. Reloading model...");
    let loaded_model = Word2Vec::load(&model_path)?;
    println!("   Loaded vocabulary size: {}", loaded_model.vocab_size());

    // Verify vectors match
    if model.contains("btc") && loaded_model.contains("btc") {
        let orig_vec = model.get_vector("btc")?;
        let loaded_vec = loaded_model.get_vector("btc")?;
        let diff: f32 = orig_vec.iter()
            .zip(loaded_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        println!("   Vector difference (should be 0): {:.6}", diff);
    }

    // Clean up
    std::fs::remove_file(model_path)?;
    println!("\n=== Done ===");

    Ok(())
}
