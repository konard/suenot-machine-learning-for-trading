//! Example: ELECTRA-based sentiment trading
//!
//! This example demonstrates using the ELECTRA sentiment model
//! to analyze financial news and generate trading signals.

use electra_finance::nlp::tokenizer::SimpleTokenizer;
use electra_finance::models::network::{SentimentNetwork, SentimentResult};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== ELECTRA Finance: Sentiment Trading ===\n");

    // Create tokenizer
    let tokenizer = SimpleTokenizer::new(64);
    println!("Tokenizer vocabulary size: {}", tokenizer.vocab_size());

    // Create sentiment model
    let model = SentimentNetwork::new(
        tokenizer.vocab_size(),
        64,   // embedding_dim
        128,  // hidden_dim
        3,    // num_classes: negative, neutral, positive
    );

    // Sample financial news headlines
    let news_headlines = vec![
        "Bitcoin surged past resistance level on heavy volume".to_string(),
        "Stock market rally drove prices up after strong earnings report".to_string(),
        "Crypto market showed bearish sentiment amid selling pressure".to_string(),
        "Trading volume remained flat with mixed market sentiment".to_string(),
        "Revenue growth exceeded analyst forecast pushing shares higher".to_string(),
        "Market decline led to record losses for portfolio managers".to_string(),
        "Bullish momentum pushed the asset to a record high price".to_string(),
        "Investors showed risk sentiment with heavy sell volume".to_string(),
    ];

    println!("Analyzing {} news headlines...\n", news_headlines.len());

    // Analyze each headline
    let mut all_results: Vec<SentimentResult> = Vec::new();

    for (i, headline) in news_headlines.iter().enumerate() {
        let (token_ids, attention_mask) = tokenizer.tokenize(headline);
        let result = model.predict(&token_ids, &attention_mask);

        let label_str = match result.label {
            0 => "NEGATIVE",
            1 => "NEUTRAL",
            2 => "POSITIVE",
            _ => "UNKNOWN",
        };

        println!("{}. \"{}\"", i + 1, headline);
        println!("   Sentiment: {} (conf: {:.3})", label_str, result.confidence);
        println!("   Neg: {:.3} | Neu: {:.3} | Pos: {:.3} | Signal: {:+.3}",
            result.negative, result.neutral, result.positive, result.signal());
        println!();

        all_results.push(result);
    }

    // Aggregate signal
    let (signal, confidence) = model.aggregate_signal(&all_results, 0.3);
    println!("=== Aggregated Trading Signal ===");
    println!("Signal: {:+.4}", signal);
    println!("Confidence: {:.4}", confidence);

    let action = if signal > 0.1 {
        "BUY"
    } else if signal < -0.1 {
        "SELL"
    } else {
        "HOLD"
    };
    println!("Recommended Action: {}", action);

    // Demonstrate batch processing
    println!("\n=== Batch Processing Demo ===");
    let batch_texts = vec![
        "Strong earnings beat consensus estimate".to_string(),
        "Revenue declined sharply on weak demand".to_string(),
        "Market traded sideways with low volume".to_string(),
    ];

    let (batch_ids, batch_masks) = tokenizer.tokenize_batch(&batch_texts);
    let batch_results = model.predict_batch(&batch_ids, &batch_masks);

    for (text, result) in batch_texts.iter().zip(batch_results.iter()) {
        println!("  \"{}\" -> Signal: {:+.3}", text, result.signal());
    }

    println!("\n=== Sentiment analysis complete ===");

    Ok(())
}
