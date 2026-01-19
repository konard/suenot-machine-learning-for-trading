//! Trading Signal Generation Demo
//!
//! Demonstrates how to convert sentiment analysis into trading signals.
//!
//! Run with: cargo run --bin signal_demo

use bloomberggpt_trading::{
    SentimentAnalyzer,
    TradingSignalGenerator,
    signals::{SignalSource, SignalConfig},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=".repeat(60));
    println!("BloombergGPT Trading - Signal Generation Demo");
    println!("=".repeat(60));

    // Create analyzer and signal generator
    let analyzer = SentimentAnalyzer::new_mock();

    let config = SignalConfig {
        min_confidence: 0.5,
        signal_duration_hours: 24,
        decay_hours: 12.0,
        source_weights: HashMap::new(),
        min_sentiment_score: 0.15,
    };
    let generator = TradingSignalGenerator::with_config(config);

    // Sample news items with associated symbols and sources
    let news_items = vec![
        ("AAPL", "Apple reports record iPhone sales, revenue up 8% year-over-year", SignalSource::Earnings),
        ("AAPL", "Analysts upgrade Apple to buy after strong services growth", SignalSource::Analyst),
        ("TSLA", "Tesla faces production challenges at Gigafactory Berlin", SignalSource::News),
        ("TSLA", "Elon Musk hints at new affordable EV model", SignalSource::Social),
        ("MSFT", "Microsoft Azure wins major government cloud contract", SignalSource::News),
        ("NVDA", "NVIDIA AI chips see unprecedented demand from data centers", SignalSource::Earnings),
        ("AMZN", "Amazon Prime membership growth slows in Q3", SignalSource::Earnings),
        ("GOOGL", "Google faces antitrust lawsuit over search monopoly", SignalSource::Filing),
    ];

    println!("\nProcessing news items...\n");

    let mut all_signals = Vec::new();
    let mut signals_by_symbol: HashMap<String, Vec<_>> = HashMap::new();

    for (symbol, text, source) in &news_items {
        println!("Processing: {} - {}", symbol, truncate_text(text, 40));

        // Analyze sentiment
        let sentiment = analyzer.analyze(text).await?;

        println!(
            "  Sentiment: {} (score: {:.2}, confidence: {:.0}%)",
            sentiment.sentiment, sentiment.score, sentiment.confidence * 100.0
        );

        // Generate signal
        if let Some(signal) = generator.generate_signal(&sentiment, symbol, *source)? {
            println!(
                "  Signal: {} (strength: {:.2}, source: {:?})",
                signal.signal_type, signal.strength, signal.source
            );

            signals_by_symbol
                .entry(symbol.to_string())
                .or_default()
                .push(signal.clone());

            all_signals.push(signal);
        } else {
            println!("  Signal: None (below threshold)");
        }
        println!();
    }

    // Display signal summary
    println!("{}", "=".repeat(60));
    println!("SIGNAL SUMMARY");
    println!("{}", "-".repeat(60));

    println!("\nSignals by Symbol:");
    for (symbol, signals) in &signals_by_symbol {
        println!("\n  {}:", symbol);
        for signal in signals {
            println!(
                "    - {} | Strength: {:.2} | Confidence: {:.0}% | Source: {:?}",
                signal.signal_type,
                signal.strength,
                signal.confidence * 100.0,
                signal.source
            );
        }

        // Aggregate signals for this symbol
        if signals.len() > 1 {
            if let Some(agg) = generator.aggregate_signals(signals) {
                println!(
                    "    => Aggregated: {} | Strength: {:.2} | Confidence: {:.0}%",
                    agg.signal_type,
                    agg.strength,
                    agg.confidence * 100.0
                );
            }
        }
    }

    // Rank all signals
    println!("\n{}", "-".repeat(60));
    println!("TOP SIGNALS (Ranked by Strength x Confidence):");
    println!();

    let ranked = generator.rank_signals(&all_signals);
    for (i, signal) in ranked.iter().take(5).enumerate() {
        let score = signal.strength * signal.confidence;
        println!(
            "  {}. {} {} | Score: {:.3} | {}",
            i + 1,
            signal.symbol,
            signal.signal_type,
            score,
            signal.reason
        );
    }

    // Position sizing example
    println!("\n{}", "-".repeat(60));
    println!("POSITION SIZING (Max Position: $10,000):");
    println!();

    let max_position = 10000.0;
    for signal in ranked.iter().take(5) {
        let position = signal.position_size(max_position);
        let direction = if position >= 0.0 { "LONG" } else { "SHORT" };
        println!(
            "  {}: {} ${:.2} ({})",
            signal.symbol,
            direction,
            position.abs(),
            signal.signal_type
        );
    }

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
