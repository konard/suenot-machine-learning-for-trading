//! Cryptocurrency Portfolio Construction Example
//!
//! This example demonstrates how to build an LLM-based portfolio
//! using cryptocurrency data from Bybit exchange.

use llm_portfolio_construction::{
    data::bybit::{BybitClient, OHLCV},
    llm::engine::LLMPortfolioEngine,
    portfolio::{Asset, AssetClass, MarketData},
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLM Crypto Portfolio Construction ===\n");

    // Initialize Bybit client
    let client = BybitClient::new();

    // Define crypto assets to analyze
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"];

    println!("Fetching market data for: {:?}\n", symbols);

    // Fetch OHLCV data for each symbol
    let mut market_data: HashMap<String, MarketData> = HashMap::new();
    let mut assets: Vec<Asset> = Vec::new();

    for symbol in &symbols {
        match client.fetch_klines(symbol, "D", 30) {
            Ok(data) => {
                if data.len() >= 2 {
                    // Calculate returns
                    let prices: Vec<f64> = data.iter().map(|d| d.close).collect();

                    // Calculate 7-day and 30-day returns
                    let return_7d = if prices.len() >= 7 {
                        (prices.last().unwrap() - prices[prices.len() - 7]) / prices[prices.len() - 7]
                    } else {
                        0.0
                    };

                    let return_30d = (prices.last().unwrap() - prices[0]) / prices[0];

                    // Calculate volatility
                    let volatility = BybitClient::calculate_volatility(&data, true);

                    let current_price = *prices.last().unwrap();
                    let volume_24h = data.last().map(|d| d.volume).unwrap_or(0.0);

                    market_data.insert(
                        symbol.to_string(),
                        MarketData {
                            symbol: symbol.to_string(),
                            return_7d,
                            return_30d,
                            volatility,
                            volume_24h,
                        },
                    );

                    assets.push(Asset::new(
                        symbol,
                        &symbol.replace("USDT", ""),
                        AssetClass::Crypto,
                        current_price,
                    ));

                    println!("  {} - Price: ${:.2}, Volatility: {:.2}%",
                             symbol, current_price, volatility * 100.0);
                }
            }
            Err(e) => {
                eprintln!("  {} - Error: {}", symbol, e);
            }
        }
    }

    println!();

    // Sample news headlines (in real usage, fetch from news API)
    let news_headlines = vec![
        "Bitcoin ETF sees record inflows".to_string(),
        "Ethereum upgrade successful, gas fees reduced".to_string(),
        "Solana network processes 65,000 TPS in stress test".to_string(),
        "Cardano partners with African governments for identity solution".to_string(),
        "Polkadot parachain auctions attract major projects".to_string(),
    ];

    // Initialize LLM Portfolio Engine (mock mode for demo)
    let engine = LLMPortfolioEngine::new(None);

    println!("Analyzing assets with LLM (mock mode)...\n");

    // Analyze assets
    let scores = engine.analyze_assets_mock(&assets, &market_data, &news_headlines);

    // Display scores
    println!("Asset Scores:");
    println!("{:-<70}", "");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "Symbol", "Fundamental", "Momentum", "Sentiment", "Risk", "Overall"
    );
    println!("{:-<70}", "");

    for score in &scores {
        println!(
            "{:<10} {:>12.1} {:>12.1} {:>12.1} {:>10.1} {:>10.2}",
            score.symbol,
            score.fundamental_score,
            score.momentum_score,
            score.sentiment_score,
            score.risk_score,
            score.overall_score
        );
    }
    println!();

    // Generate portfolio
    let portfolio = engine.generate_portfolio(&scores, 0.05);

    println!("Generated Portfolio:");
    println!("{:-<50}", "");

    let mut weights: Vec<_> = portfolio.weights.iter().collect();
    weights.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (symbol, weight) in weights {
        let bar_len = (weight * 40.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("{:<10} {:>6.1}%  {}", symbol, weight * 100.0, bar);
    }

    println!();
    println!("Portfolio created at: {}", portfolio.timestamp);
    if let Some(strategy) = portfolio.metadata.get("strategy") {
        println!("Strategy: {}", strategy);
    }

    Ok(())
}
