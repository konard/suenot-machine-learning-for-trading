//! FinGPT Trading Signal Generation Demo
//!
//! This example demonstrates how to generate trading signals from
//! financial news using FinGPT sentiment analysis.

use fingpt_trading::{SentimentAnalyzer, TradingSignalGenerator, SignalType};
use fingpt_trading::signals::NewsItem;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(70));
    println!("FinGPT Trading Signal Generation Demo");
    println!("{}", "=".repeat(70));

    // Initialize components
    let analyzer = SentimentAnalyzer::new_mock();
    let generator = TradingSignalGenerator::new(analyzer);

    // Simulated news feed for different assets
    let news_feeds: Vec<(&str, Vec<NewsItem>)> = vec![
        (
            "AAPL",
            vec![
                NewsItem::new(
                    "Apple reports record iPhone 15 sales in Q4, beating analyst expectations by 12%.",
                    "reuters",
                ),
                NewsItem::new(
                    "Apple announces major expansion of AI features in upcoming iOS update.",
                    "bloomberg",
                ),
                NewsItem::new(
                    "Concerns raised about Apple's China market share declining.",
                    "twitter",
                ),
            ],
        ),
        (
            "TSLA",
            vec![
                NewsItem::new(
                    "Tesla Cybertruck deliveries begin, overwhelming demand reported.",
                    "reuters",
                ),
                NewsItem::new(
                    "Elon Musk sells $3 billion worth of Tesla shares.",
                    "sec_filing",
                ),
                NewsItem::new(
                    "Tesla faces increased competition from Chinese EV makers.",
                    "bloomberg",
                ),
            ],
        ),
        (
            "BTCUSDT",
            vec![
                NewsItem::new(
                    "Bitcoin ETF approval expected imminently, institutional interest surges.",
                    "bloomberg",
                ),
                NewsItem::new(
                    "Major bank announces Bitcoin custody services for institutional clients.",
                    "reuters",
                ),
                NewsItem::new(
                    "Regulatory concerns in Europe could impact crypto trading.",
                    "twitter",
                ),
            ],
        ),
    ];

    // Process each asset
    let mut all_signals = Vec::new();

    for (symbol, news_items) in &news_feeds {
        println!("\n{}", "=".repeat(70));
        println!("Processing: {}", symbol);
        println!("{}", "=".repeat(70));

        println!("\nNews Analysis:");
        for (i, item) in news_items.iter().enumerate() {
            let analyzer_ref = SentimentAnalyzer::new_mock();
            let result = analyzer_ref.analyze(&item.text).await?;

            let truncated = if item.text.len() > 60 {
                format!("{}...", &item.text[..60])
            } else {
                item.text.clone()
            };

            println!("\n  [{}] Source: {}", i + 1, item.source.to_uppercase());
            println!("      Text: {}", truncated);
            println!(
                "      Sentiment: {} (score: {:+.2}, confidence: {:.1}%)",
                result.sentiment.to_string().to_uppercase(),
                result.score,
                result.confidence * 100.0
            );
        }

        // Generate aggregated signal
        let signal = generator.generate_signal(symbol, news_items).await?;
        all_signals.push(signal.clone());

        println!("\n  AGGREGATED SIGNAL:");
        println!("    Symbol: {}", signal.symbol);
        println!("    Action: {}", signal.signal_type.to_string().to_uppercase());
        println!("    Strength: {:.2}", signal.strength);
        println!("    Confidence: {:.1}%", signal.confidence * 100.0);
    }

    // Summary of all signals
    println!("\n{}", "=".repeat(70));
    println!("TRADING SIGNALS SUMMARY");
    println!("{}", "=".repeat(70));

    // Sort by strength (absolute value)
    all_signals.sort_by(|a, b| {
        b.strength.abs().partial_cmp(&a.strength.abs()).unwrap()
    });

    println!(
        "\n{:<10} {:<8} {:<10} {:<12} {}",
        "Symbol", "Action", "Strength", "Confidence", "Recommendation"
    );
    println!("{}", "-".repeat(70));

    for signal in &all_signals {
        let recommendation = match signal.signal_type {
            SignalType::Buy if signal.strength > 0.5 => "Strong buy opportunity",
            SignalType::Buy => "Consider buying",
            SignalType::Sell if signal.strength < -0.5 => "Strong sell signal",
            SignalType::Sell => "Consider selling",
            SignalType::Hold => "Monitor closely",
        };

        println!(
            "{:<10} {:<8} {:+.2}      {:.0}%          {}",
            signal.symbol,
            signal.signal_type.to_string().to_uppercase(),
            signal.strength,
            signal.confidence * 100.0,
            recommendation
        );
    }

    // Demonstrate position sizing
    println!("\n{}", "=".repeat(70));
    println!("POSITION SIZING EXAMPLE");
    println!("{}", "=".repeat(70));

    let portfolio_value = 100_000.0_f64;
    let max_position_pct = 0.10_f64;

    println!("\nPortfolio Value: ${:.2}", portfolio_value);
    println!("Max Position Size: {:.0}%", max_position_pct * 100.0);
    println!("\nSuggested Allocations:");

    for signal in &all_signals {
        if signal.signal_type == SignalType::Hold {
            continue;
        }

        // Scale position by signal strength and confidence
        let position_pct = max_position_pct * signal.strength.abs() * signal.confidence;
        let position_value = portfolio_value * position_pct;

        let action_str = match signal.signal_type {
            SignalType::Buy => "BUY",
            SignalType::Sell => "SELL",
            SignalType::Hold => "HOLD",
        };

        println!(
            "  {}: {} ${:.2} ({:.1}% of portfolio)",
            signal.symbol, action_str, position_value, position_pct * 100.0
        );
    }

    println!("\n{}", "=".repeat(70));
    println!("Demo completed successfully!");
    println!("{}", "=".repeat(70));

    Ok(())
}
