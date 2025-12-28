//! Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to:
//! - Connect to Bybit API
//! - Fetch announcements and market data
//! - Save data for topic modeling

use anyhow::Result;
use chrono::Utc;
use std::path::PathBuf;
use topic_modeling::api::bybit::{BybitClient, MarketContext, MarketDocument};
use topic_modeling::utils::io::{ensure_directory, DocumentDataset};

fn main() -> Result<()> {
    env_logger::init();

    println!("=== Bybit Data Fetcher ===\n");

    // Create Bybit client (no authentication needed for public endpoints)
    let client = BybitClient::new();

    // Fetch announcements
    println!("Fetching announcements...");
    let announcements = match client.get_announcements("en-US", 50) {
        Ok(ann) => {
            println!("  Fetched {} announcements", ann.len());
            ann
        }
        Err(e) => {
            println!("  Warning: Could not fetch announcements: {}", e);
            vec![]
        }
    };

    // Fetch current market context
    println!("\nFetching market data...");
    let market_context = match client.get_tickers("spot", Some("BTCUSDT")) {
        Ok(tickers) => {
            if let Some(btc) = tickers.first() {
                let price: f64 = btc.last_price.parse().unwrap_or(0.0);
                let change: f64 = btc.price_24h_pcnt.parse().unwrap_or(0.0);
                let volume: f64 = btc.volume_24h.parse().unwrap_or(0.0);

                println!("  BTC/USDT: ${:.2}", price);
                println!("  24h Change: {:.2}%", change * 100.0);
                println!("  24h Volume: {:.2} BTC", volume);

                Some(MarketContext {
                    btc_price: price,
                    btc_change_24h: change * 100.0,
                    total_volume: volume,
                })
            } else {
                None
            }
        }
        Err(e) => {
            println!("  Warning: Could not fetch tickers: {}", e);
            None
        }
    };

    // Convert announcements to documents
    let documents: Vec<MarketDocument> = announcements
        .iter()
        .map(|ann| {
            let mut doc = MarketDocument::from_announcement(ann);
            if let Some(ref ctx) = market_context {
                doc = doc.with_market_context(ctx.clone());
            }
            doc
        })
        .collect();

    println!("\nCreated {} documents from announcements", documents.len());

    // Print sample documents
    if !documents.is_empty() {
        println!("\n=== Sample Documents ===\n");
        for (i, doc) in documents.iter().take(3).enumerate() {
            println!("Document {}:", i + 1);
            println!("  Title: {}", doc.title);
            println!("  Type: {}", doc.doc_type);
            if !doc.symbols.is_empty() {
                println!("  Symbols: {:?}", doc.symbols);
            }
            println!(
                "  Content preview: {}...",
                &doc.content.chars().take(100).collect::<String>()
            );
            println!();
        }
    }

    // Fetch some trading pairs info
    println!("=== Available Trading Pairs ===\n");
    match client.get_symbols("spot") {
        Ok(symbols) => {
            let usdt_pairs: Vec<_> = symbols
                .iter()
                .filter(|s| s.quote_coin == "USDT" && s.status == "Trading")
                .take(20)
                .collect();

            println!("Top USDT pairs:");
            for symbol in usdt_pairs {
                println!("  {} ({}/{})", symbol.symbol, symbol.base_coin, symbol.quote_coin);
            }
        }
        Err(e) => println!("Could not fetch symbols: {}", e),
    }

    // Save dataset
    if !documents.is_empty() {
        let data_dir = PathBuf::from("data");
        ensure_directory(&data_dir)?;

        let dataset = DocumentDataset::new("bybit_announcements", "Bybit Exchange API", documents);

        let json_path = data_dir.join("bybit_announcements.json");
        dataset.save_json(&json_path)?;
        println!("\n✓ Saved dataset to {:?}", json_path);
    }

    // Demonstrate sample data creation for testing
    println!("\n=== Creating Sample Data for Testing ===\n");

    let sample_documents = create_sample_documents();
    let sample_dataset = DocumentDataset::new(
        "sample_crypto_news",
        "Generated sample data for testing",
        sample_documents,
    );

    let data_dir = PathBuf::from("data");
    ensure_directory(&data_dir)?;

    let sample_path = data_dir.join("sample_crypto_news.json");
    sample_dataset.save_json(&sample_path)?;
    println!("✓ Saved sample dataset to {:?}", sample_path);

    println!("\n=== Done ===");
    Ok(())
}

/// Create sample documents for testing topic modeling
fn create_sample_documents() -> Vec<MarketDocument> {
    let sample_texts = vec![
        // Topic 1: Bitcoin trading
        ("Bitcoin Reaches New All-Time High", "BTC surpassed $70,000 as institutional investors continue to accumulate. Trading volume on major exchanges has increased significantly.", vec!["BTC"]),
        ("Bitcoin Mining Difficulty Adjustment", "The Bitcoin network has adjusted its mining difficulty upward by 5%. Hash rate continues to grow as miners expand operations.", vec!["BTC"]),
        ("BTC ETF Sees Record Inflows", "Bitcoin ETF products recorded over $1 billion in daily inflows. This marks a significant milestone for cryptocurrency adoption.", vec!["BTC"]),

        // Topic 2: Ethereum and DeFi
        ("Ethereum Layer 2 Solutions Growing", "ETH Layer 2 networks like Arbitrum and Optimism are seeing increased adoption. Transaction costs have decreased significantly.", vec!["ETH"]),
        ("DeFi TVL Reaches New Heights", "Total Value Locked in DeFi protocols has exceeded $100 billion. Ethereum remains the dominant platform for decentralized finance.", vec!["ETH"]),
        ("Ethereum Staking Rewards Update", "ETH staking rewards have stabilized around 4% APY. Validator count continues to grow post-Shanghai upgrade.", vec!["ETH"]),

        // Topic 3: NFTs and Gaming
        ("NFT Market Shows Signs of Recovery", "NFT trading volumes have increased 30% this month. Blue-chip collections are leading the recovery.", vec![]),
        ("Blockchain Gaming Adoption Accelerates", "Play-to-earn games are attracting millions of new users. Major gaming studios are entering the Web3 space.", vec![]),
        ("New NFT Marketplace Launch", "A new NFT marketplace has launched with zero-fee trading. Artists and collectors are migrating to the platform.", vec![]),

        // Topic 4: Regulation and Compliance
        ("Crypto Regulation Framework Proposed", "Lawmakers have introduced new cryptocurrency regulation bills. Industry groups are providing feedback on proposed rules.", vec![]),
        ("Exchange Compliance Update", "Major exchanges are implementing enhanced KYC procedures. Regulatory compliance costs have increased industry-wide.", vec![]),
        ("Stablecoin Oversight Guidelines Released", "Regulators have published new guidelines for stablecoin issuers. Reserve requirements and audit standards are included.", vec!["USDT"]),

        // Topic 5: Technology and Development
        ("Zero-Knowledge Proof Advancement", "New ZK-proof technology enables faster and cheaper verification. Multiple blockchain projects are integrating the technology.", vec![]),
        ("Cross-Chain Bridge Security Improvements", "Bridge protocols have implemented new security measures. Multi-signature and time-lock mechanisms are now standard.", vec![]),
        ("Smart Contract Audit Standards Updated", "Industry groups have released updated smart contract security standards. Formal verification is now recommended for high-value contracts.", vec![]),

        // Mixed topics
        ("BTC and ETH Correlation Analysis", "Bitcoin and Ethereum price correlation has reached new highs. Market analysts discuss portfolio implications.", vec!["BTC", "ETH"]),
        ("Institutional Adoption Trends", "More hedge funds are allocating to cryptocurrency. Bitcoin and Ethereum remain the preferred assets.", vec!["BTC", "ETH"]),
        ("Market Volatility Analysis", "Crypto market volatility has decreased compared to previous cycles. Mature market structure is cited as a factor.", vec!["BTC", "ETH"]),
    ];

    sample_texts
        .into_iter()
        .enumerate()
        .map(|(i, (title, content, symbols))| MarketDocument {
            id: format!("sample_{}", i),
            title: title.to_string(),
            content: content.to_string(),
            timestamp: Utc::now().timestamp_millis() as u64 - (i as u64 * 3600000),
            symbols: symbols.into_iter().map(|s| s.to_string()).collect(),
            doc_type: "news".to_string(),
            market_context: None,
        })
        .collect()
}
