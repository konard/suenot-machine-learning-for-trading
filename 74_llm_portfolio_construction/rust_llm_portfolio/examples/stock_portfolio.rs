//! Stock Market Portfolio Construction Example
//!
//! This example demonstrates how to build an LLM-based portfolio
//! using stock market data from Yahoo Finance.

use llm_portfolio_construction::{
    data::stock::{StockClient, StockPortfolioDataFetcher},
    llm::engine::LLMPortfolioEngine,
    portfolio::{Asset, AssetClass, MarketData},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLM Stock Portfolio Construction ===\n");

    // Initialize stock client
    let client = StockClient::new();
    let fetcher = StockPortfolioDataFetcher::new(client);

    // Define stock symbols to analyze
    let symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"];

    println!("Fetching market data for: {:?}\n", symbols);

    // Fetch portfolio data
    let symbol_refs: Vec<&str> = symbols.iter().map(|s| *s).collect();
    let portfolio_data = fetcher.fetch_portfolio_data(&symbol_refs, "1y").await?;

    // Calculate volatilities
    let volatilities = portfolio_data.volatilities();

    // Build market data for LLM analysis
    let mut market_data: HashMap<String, MarketData> = HashMap::new();
    let mut assets: Vec<Asset> = Vec::new();

    for symbol in &symbols {
        if let (Some(prices), Some(returns)) = (
            portfolio_data.prices.get(*symbol),
            portfolio_data.returns.get(*symbol),
        ) {
            let volatility = *volatilities.get(*symbol).unwrap_or(&0.0);
            let current_price = *prices.last().unwrap_or(&0.0);

            // Calculate 7-day and 30-day returns
            let return_7d: f64 = if returns.len() >= 7 {
                returns.iter().rev().take(7).sum()
            } else {
                returns.iter().sum()
            };

            let return_30d: f64 = if returns.len() >= 30 {
                returns.iter().rev().take(30).sum()
            } else {
                returns.iter().sum()
            };

            let volume_24h = portfolio_data.ohlcv.get(*symbol)
                .and_then(|data| data.last().map(|d| d.volume as f64))
                .unwrap_or(0.0);

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
                &get_company_name(symbol),
                AssetClass::Equity,
                current_price,
            ));

            println!("  {} - Price: ${:.2}, Volatility: {:.2}%",
                     symbol, current_price, volatility * 100.0);
        }
    }
    println!();

    // Sample news headlines (in real usage, fetch from news API)
    let news_headlines = vec![
        "Apple announces new AI features for iPhone".to_string(),
        "Microsoft Azure revenue grows 29% year-over-year".to_string(),
        "Google DeepMind achieves breakthrough in protein folding".to_string(),
        "Amazon AWS expands to new regions".to_string(),
        "NVIDIA H100 GPUs in high demand for AI training".to_string(),
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

    // Display correlation matrix
    println!("\nCorrelation Matrix:");
    let correlations = portfolio_data.correlation_matrix();

    print!("{:<8}", "");
    for symbol in &symbols {
        print!("{:>8}", symbol);
    }
    println!();

    for sym1 in &symbols {
        print!("{:<8}", sym1);
        for sym2 in &symbols {
            let corr = correlations
                .get(&(sym1.to_string(), sym2.to_string()))
                .unwrap_or(&0.0);
            print!("{:>8.2}", corr);
        }
        println!();
    }

    Ok(())
}

fn get_company_name(symbol: &str) -> String {
    match symbol {
        "AAPL" => "Apple Inc".to_string(),
        "MSFT" => "Microsoft Corporation".to_string(),
        "GOOGL" => "Alphabet Inc".to_string(),
        "AMZN" => "Amazon.com Inc".to_string(),
        "NVDA" => "NVIDIA Corporation".to_string(),
        _ => symbol.to_string(),
    }
}
