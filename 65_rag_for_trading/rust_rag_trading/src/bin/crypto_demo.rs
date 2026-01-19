//! Cryptocurrency data demo using Bybit API.
//!
//! This example shows how to:
//! - Fetch cryptocurrency data from Bybit
//! - Generate trading signals for crypto assets
//! - Analyze crypto market sentiment

use chrono::Utc;
use rag_trading::{
    data::DataLoader,
    Document, DocumentRetriever, RAGSignalGenerator, SignalDirection,
};

#[tokio::main]
async fn main() {
    println!("=== RAG Trading: Cryptocurrency Demo ===\n");

    // Create data loader
    let loader = DataLoader::new();

    // Try to fetch real data from Bybit (may fail if API is unavailable)
    println!("--- Fetching Cryptocurrency Data ---\n");

    // Use mock data for demonstration (reliable without network)
    let btc_data = loader.mock_data("BTCUSDT", 30, 45000.0);
    let eth_data = loader.mock_data("ETHUSDT", 30, 2500.0);
    let sol_data = loader.mock_data("SOLUSDT", 30, 100.0);

    println!("BTCUSDT: {} bars, latest close: ${:.2}", btc_data.len(), btc_data.close.last().unwrap_or(&0.0));
    println!("ETHUSDT: {} bars, latest close: ${:.2}", eth_data.len(), eth_data.close.last().unwrap_or(&0.0));
    println!("SOLUSDT: {} bars, latest close: ${:.2}\n", sol_data.len(), sol_data.close.last().unwrap_or(&0.0));

    // Calculate returns
    println!("--- Price Statistics ---\n");

    for (name, data) in [("BTC", &btc_data), ("ETH", &eth_data), ("SOL", &sol_data)] {
        let returns = data.returns();
        if !returns.is_empty() {
            let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let max_return = returns.iter().cloned().fold(f64::MIN, f64::max);
            let min_return = returns.iter().cloned().fold(f64::MAX, f64::min);

            println!("{} Statistics:", name);
            println!("  Avg Daily Return: {:.2}%", avg_return * 100.0);
            println!("  Max Return: {:.2}%", max_return * 100.0);
            println!("  Min Return: {:.2}%", min_return * 100.0);
            println!();
        }
    }

    // Create document retriever with crypto news
    let mut retriever = DocumentRetriever::new();

    let crypto_docs = vec![
        // BTC - Bullish news
        Document::new(
            "Bitcoin ETF sees record inflows as institutional adoption accelerates. \
             Major banks announce Bitcoin custody services. Bullish sentiment grows \
             as price approaches all-time high.",
            Some("BTC".to_string()),
            "CoinDesk",
            Utc::now(),
        ),
        Document::new(
            "Bitcoin hash rate reaches new all-time high. Network security stronger \
             than ever. Mining profitability improves amid price rally.",
            Some("BTC".to_string()),
            "Bitcoin Magazine",
            Utc::now(),
        ),
        // ETH - Mixed news
        Document::new(
            "Ethereum staking rewards attract institutional investors. Layer 2 \
             adoption continues strong growth. Gas fees remain competitive.",
            Some("ETH".to_string()),
            "The Block",
            Utc::now(),
        ),
        Document::new(
            "Ethereum faces competition from alternative Layer 1 chains. \
             Some DeFi protocols migrate to other networks. Concerns about \
             network congestion during peak times.",
            Some("ETH".to_string()),
            "Decrypt",
            Utc::now(),
        ),
        // SOL - Recovery news
        Document::new(
            "Solana ecosystem shows strong recovery with new project launches. \
             Transaction volume surges as network stability improves.",
            Some("SOL".to_string()),
            "Solana Foundation",
            Utc::now(),
        ),
        Document::new(
            "Solana DeFi TVL grows despite market uncertainty. NFT marketplace \
             sees increased activity. Developer community expands.",
            Some("SOL".to_string()),
            "CryptoSlate",
            Utc::now(),
        ),
        // Market-wide sentiment
        Document::new(
            "Crypto market sentiment turns positive as regulatory clarity improves. \
             Institutional adoption expected to accelerate in 2024.",
            None,
            "Bloomberg Crypto",
            Utc::now(),
        ),
    ];

    retriever.add_documents(crypto_docs);
    println!("--- RAG Analysis ---\n");
    println!("Indexed {} crypto-related documents\n", retriever.document_count());

    // Create signal generator
    let generator = RAGSignalGenerator::new(retriever);

    // Generate signals for crypto assets
    let symbols = ["BTC", "ETH", "SOL"];

    println!("| Symbol | Direction | Confidence | Positive | Negative |");
    println!("|--------|-----------|------------|----------|----------|");

    for symbol in &symbols {
        let signal = generator.generate_signal(symbol);

        let direction_icon = match signal.direction {
            SignalDirection::Long => "📈",
            SignalDirection::Short => "📉",
            SignalDirection::Neutral => "➡️",
        };

        println!(
            "| {} {} | {:^9} | {:>9.1}% | {:>8} | {:>8} |",
            direction_icon,
            symbol,
            format!("{}", signal.direction),
            signal.confidence * 100.0,
            signal.positive_signals,
            signal.negative_signals
        );
    }

    println!();

    // Detailed analysis for each crypto
    println!("--- Detailed Signal Analysis ---\n");

    for symbol in &symbols {
        let signal = generator.generate_signal(symbol);

        println!("{}:", symbol);
        println!("  Signal: {} ({:.1}% confidence)", signal.direction, signal.confidence * 100.0);
        println!("  Analysis: {}", signal.reasoning);
        println!("  Sources: {:?}", signal.sources);
        println!();
    }

    // Try to fetch real data (optional, may timeout)
    println!("--- Attempting Live Data Fetch ---\n");

    match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        loader.fetch_crypto("BTCUSDT", "D", 5)
    ).await {
        Ok(Ok(data)) => {
            println!("Successfully fetched live BTCUSDT data:");
            for i in 0..data.len().min(5) {
                println!(
                    "  {} - O: {:.2} H: {:.2} L: {:.2} C: {:.2}",
                    data.timestamps[i].format("%Y-%m-%d"),
                    data.open[i],
                    data.high[i],
                    data.low[i],
                    data.close[i]
                );
            }
        }
        Ok(Err(e)) => {
            println!("Could not fetch live data: {}", e);
            println!("(This is expected if Bybit API is unavailable)");
        }
        Err(_) => {
            println!("Live data fetch timed out");
            println!("(Using mock data for demonstration)");
        }
    }

    println!("\n=== Demo Complete ===");
}
