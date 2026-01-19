//! FinGPT Sentiment Analysis Demo
//!
//! This example demonstrates financial sentiment analysis using the FinGPT-inspired
//! approach with mock data.

use fingpt_trading::{SentimentAnalyzer, Sentiment};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(70));
    println!("FinGPT Financial Sentiment Analysis Demo");
    println!("{}", "=".repeat(70));

    // Initialize the sentiment analyzer in mock mode
    let analyzer = SentimentAnalyzer::new_mock();

    // Test cases covering various financial scenarios
    let test_texts = vec![
        // Positive news
        ("Apple reported record quarterly revenue of $123.9 billion, beating analyst expectations by 5%.", "Positive"),
        ("NVIDIA stock surged to new all-time highs amid strong AI chip demand.", "Positive"),
        ("Microsoft cloud revenue growth exceeded expectations, driving stock higher.", "Positive"),
        ("Amazon announces 20% increase in Prime membership, shares rally.", "Positive"),

        // Negative news
        ("Tesla shares plunged 12% after disappointing delivery numbers.", "Negative"),
        ("Bitcoin dropped below $40,000 as regulatory concerns mount.", "Negative"),
        ("Meta faces antitrust lawsuit, stock falls 8% in after-hours trading.", "Negative"),
        ("Bank announces major layoffs amid declining profits.", "Negative"),

        // Neutral news
        ("The Federal Reserve kept interest rates unchanged at the latest meeting.", "Neutral"),
        ("Apple is expected to announce new products at next month's event.", "Neutral"),
        ("Trading volume remains steady as markets await earnings reports.", "Neutral"),
    ];

    // Analyze each text
    for (i, (text, expected)) in test_texts.iter().enumerate() {
        let result = analyzer.analyze(text).await?;

        let truncated = if text.len() > 65 {
            format!("{}...", &text[..65])
        } else {
            text.to_string()
        };

        println!("\n[{}] Text: {}", i + 1, truncated);
        println!("    Expected: {}", expected);
        println!("    Detected: {}", result.sentiment.to_string().to_uppercase());
        println!("    Confidence: {:.1}%", result.confidence * 100.0);
        println!("    Score: {:+.2}", result.score);

        let status = if (result.sentiment == Sentiment::Positive && *expected == "Positive")
            || (result.sentiment == Sentiment::Negative && *expected == "Negative")
            || (result.sentiment == Sentiment::Neutral && *expected == "Neutral")
        {
            "OK"
        } else {
            "MISMATCH"
        };
        println!("    Status: {}", status);
    }

    // Demonstrate aspect-based sentiment analysis
    println!("\n{}", "=".repeat(70));
    println!("Aspect-Based Sentiment Analysis");
    println!("{}", "=".repeat(70));

    let multi_entity_text = r#"
    In today's market session, Apple reported strong iPhone sales growth,
    beating expectations. However, Microsoft warned of cloud slowdown,
    causing concern among investors. Meanwhile, Tesla announced
    record deliveries but faces margin pressure from price cuts.
    "#;

    let entities = vec!["Apple", "Microsoft", "Tesla"];
    let aspect_results = analyzer.analyze_aspects(multi_entity_text, &entities).await?;

    println!("\nText: {}...", multi_entity_text.trim().chars().take(100).collect::<String>());
    println!("\nSentiment by Entity:");

    for entity in &entities {
        if let Some(result) = aspect_results.get(*entity) {
            println!(
                "  {}: {} (confidence: {:.1}%, score: {:+.2})",
                entity,
                result.sentiment.to_string().to_uppercase(),
                result.confidence * 100.0,
                result.score
            );
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("Demo completed successfully!");
    println!("{}", "=".repeat(70));

    Ok(())
}
