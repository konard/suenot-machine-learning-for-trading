//! Sentiment Analysis Demo
//!
//! Demonstrates financial sentiment analysis using the BloombergGPT Trading library.
//!
//! Run with: cargo run --bin sentiment_demo

use bloomberggpt_trading::{SentimentAnalyzer, Sentiment};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=".repeat(60));
    println!("BloombergGPT Trading - Sentiment Analysis Demo");
    println!("=".repeat(60));

    // Create analyzer (using mock for demo - no API key needed)
    let analyzer = SentimentAnalyzer::new_mock();

    println!("\nAnalyzer: {}", analyzer.name());
    println!();

    // Sample financial texts to analyze
    let texts = vec![
        "Apple reports record quarterly earnings, beating analyst expectations by 15%",
        "Tesla stock plunges 8% after Elon Musk sells $4 billion in shares",
        "Federal Reserve maintains interest rates unchanged amid inflation concerns",
        "NVIDIA announces breakthrough AI chip, data center revenue up 171%",
        "Amazon faces regulatory scrutiny as FTC launches antitrust investigation",
        "Microsoft cloud revenue exceeds expectations, Azure growth at 29%",
        "Bitcoin crashes below $30,000 as regulatory fears intensify",
        "Goldman Sachs downgrades tech sector to neutral citing high valuations",
    ];

    println!("SENTIMENT ANALYSIS RESULTS");
    println!("{}", "-".repeat(60));
    println!();

    for text in &texts {
        let result = analyzer.analyze(text).await?;

        let sentiment_emoji = match result.sentiment {
            Sentiment::VeryPositive => "[++]",
            Sentiment::Positive => "[+ ]",
            Sentiment::Neutral => "[  ]",
            Sentiment::Negative => "[ -]",
            Sentiment::VeryNegative => "[--]",
        };

        println!("{} {}", sentiment_emoji, result.sentiment);
        println!("    Score: {:.2}, Confidence: {:.0}%", result.score, result.confidence * 100.0);
        println!("    Text: \"{}\"", truncate_text(text, 50));

        if !result.entity_sentiments.is_empty() {
            println!("    Entities:");
            for (entity, sentiment) in &result.entity_sentiments {
                println!(
                    "      - {} ({:?}): {} ({:.2})",
                    entity, sentiment.entity_type, sentiment.sentiment, sentiment.score
                );
            }
        }
        println!();
    }

    // Batch analysis example
    println!("{}", "=".repeat(60));
    println!("BATCH ANALYSIS");
    println!("{}", "-".repeat(60));

    let batch_texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
    let batch_results = analyzer.analyze_batch(&batch_texts).await?;

    // Calculate statistics
    let positive_count = batch_results
        .iter()
        .filter(|r| r.score > 0.0)
        .count();
    let negative_count = batch_results
        .iter()
        .filter(|r| r.score < 0.0)
        .count();
    let neutral_count = batch_results
        .iter()
        .filter(|r| r.score == 0.0)
        .count();

    let avg_score: f64 = batch_results.iter().map(|r| r.score).sum::<f64>()
        / batch_results.len() as f64;
    let avg_confidence: f64 = batch_results.iter().map(|r| r.confidence).sum::<f64>()
        / batch_results.len() as f64;

    println!("\nBatch Statistics:");
    println!("  Total texts analyzed: {}", batch_results.len());
    println!("  Positive: {}", positive_count);
    println!("  Negative: {}", negative_count);
    println!("  Neutral: {}", neutral_count);
    println!("  Average score: {:.3}", avg_score);
    println!("  Average confidence: {:.1}%", avg_confidence * 100.0);

    // Overall market sentiment
    let market_sentiment = if avg_score > 0.2 {
        "BULLISH"
    } else if avg_score < -0.2 {
        "BEARISH"
    } else {
        "NEUTRAL"
    };

    println!("\n  Overall Market Sentiment: {}", market_sentiment);

    println!("\n{}", "=".repeat(60));
    println!("Demo complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}

fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}
