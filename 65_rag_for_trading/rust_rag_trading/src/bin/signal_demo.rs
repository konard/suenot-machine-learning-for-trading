//! Trading signal generation demonstration.
//!
//! This example shows how to:
//! - Generate trading signals from RAG analysis
//! - Analyze sentiment from financial documents
//! - Batch process multiple tickers

use chrono::Utc;
use rag_trading::{Document, DocumentRetriever, RAGSignalGenerator, SignalDirection};

fn main() {
    println!("=== RAG Trading: Signal Generation Demo ===\n");

    // Create and populate document retriever
    let mut retriever = DocumentRetriever::new();

    // Add diverse documents for different tickers
    let documents = vec![
        // TSLA - Mixed sentiment
        Document::new(
            "Tesla reported record deliveries beating expectations. Strong growth \
             in China market. Analysts upgrade to buy rating.",
            Some("TSLA".to_string()),
            "Reuters",
            Utc::now(),
        ),
        Document::new(
            "Tesla faces competition concerns in EV market. Margins under pressure \
             from price cuts. Some analysts downgrade amid slowdown fears.",
            Some("TSLA".to_string()),
            "Bloomberg",
            Utc::now(),
        ),
        // AAPL - Positive sentiment
        Document::new(
            "Apple beats earnings expectations with record iPhone sales. Services \
             revenue shows strong growth. Bullish outlook for next quarter.",
            Some("AAPL".to_string()),
            "CNBC",
            Utc::now(),
        ),
        Document::new(
            "Apple announces breakthrough AI features. Strong momentum in \
             wearables segment. Analysts optimistic about growth prospects.",
            Some("AAPL".to_string()),
            "TechCrunch",
            Utc::now(),
        ),
        // NVDA - Very positive
        Document::new(
            "NVIDIA posts record revenue surge. AI chip demand exceeds all \
             expectations. Company raises guidance significantly.",
            Some("NVDA".to_string()),
            "Wall Street Journal",
            Utc::now(),
        ),
        Document::new(
            "NVIDIA dominates AI infrastructure market. Strong momentum and \
             accelerating growth. Multiple upgrades from analysts.",
            Some("NVDA".to_string()),
            "Barron's",
            Utc::now(),
        ),
        // META - Negative sentiment
        Document::new(
            "Meta faces advertising slowdown concerns. Revenue decline in key \
             markets. Losses mounting in Reality Labs division.",
            Some("META".to_string()),
            "Financial Times",
            Utc::now(),
        ),
        Document::new(
            "Meta's metaverse bet disappoints investors. Weak user growth \
             and concerns about competition. Analysts cut price targets.",
            Some("META".to_string()),
            "The Verge",
            Utc::now(),
        ),
    ];

    retriever.add_documents(documents);
    println!("Indexed {} documents\n", retriever.document_count());

    // Create signal generator
    let generator = RAGSignalGenerator::new(retriever);

    // Generate signals for each ticker
    let tickers = ["TSLA", "AAPL", "NVDA", "META", "UNKNOWN"];

    println!("--- Individual Signal Generation ---\n");

    for ticker in &tickers {
        let signal = generator.generate_signal(ticker);

        let direction_emoji = match signal.direction {
            SignalDirection::Long => "📈",
            SignalDirection::Short => "📉",
            SignalDirection::Neutral => "➡️",
        };

        println!("{} {} Signal:", direction_emoji, ticker);
        println!("  Direction: {}", signal.direction);
        println!("  Confidence: {:.1}%", signal.confidence * 100.0);
        println!("  Documents: {}", signal.documents_analyzed);
        println!(
            "  Sentiment: {} positive, {} negative",
            signal.positive_signals, signal.negative_signals
        );
        println!("  Sources: {:?}", signal.sources);
        println!("  Reasoning: {}", signal.reasoning);
        println!();
    }

    // Batch signal generation
    println!("--- Batch Signal Generation ---\n");

    let batch_tickers: Vec<&str> = vec!["TSLA", "AAPL", "NVDA", "META"];
    let signals = generator.batch_generate(&batch_tickers);

    println!("| Ticker | Direction | Confidence | Docs |");
    println!("|--------|-----------|------------|------|");

    for signal in &signals {
        println!(
            "| {} | {:^9} | {:>9.1}% | {:>4} |",
            signal.ticker,
            format!("{}", signal.direction),
            signal.confidence * 100.0,
            signal.documents_analyzed
        );
    }

    println!("\n--- Signal Summary ---\n");

    let long_signals: Vec<_> = signals
        .iter()
        .filter(|s| s.direction == SignalDirection::Long)
        .collect();
    let short_signals: Vec<_> = signals
        .iter()
        .filter(|s| s.direction == SignalDirection::Short)
        .collect();

    println!(
        "Long signals: {} ({:?})",
        long_signals.len(),
        long_signals.iter().map(|s| &s.ticker).collect::<Vec<_>>()
    );
    println!(
        "Short signals: {} ({:?})",
        short_signals.len(),
        short_signals.iter().map(|s| &s.ticker).collect::<Vec<_>>()
    );

    // Find highest confidence signals
    if let Some(best_long) = signals
        .iter()
        .filter(|s| s.direction == SignalDirection::Long)
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    {
        println!(
            "\nStrongest LONG: {} ({:.1}% confidence)",
            best_long.ticker,
            best_long.confidence * 100.0
        );
    }

    if let Some(best_short) = signals
        .iter()
        .filter(|s| s.direction == SignalDirection::Short)
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    {
        println!(
            "Strongest SHORT: {} ({:.1}% confidence)",
            best_short.ticker,
            best_short.confidence * 100.0
        );
    }

    println!("\n=== Demo Complete ===");
}
