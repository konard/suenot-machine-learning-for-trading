//! Basic news analysis example
//!
//! This example demonstrates how to:
//! - Create an LLM client
//! - Analyze a news item
//! - Extract sentiment and event type
//!
//! Run with: cargo run --example basic_news_analysis

use llm_news_interpretation::{
    LlmClient, LlmConfig, NewsAnalyzer, NewsItem, NewsSource,
    SentimentAnalyzer, EventClassifier,
};
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== LLM News Interpretation - Basic Example ===\n");

    // Example news items to analyze
    let news_items = vec![
        NewsItem {
            id: "1".to_string(),
            title: "SEC Approves First Spot Bitcoin ETF".to_string(),
            content: "The U.S. Securities and Exchange Commission has approved the first spot Bitcoin ETF, marking a historic moment for the cryptocurrency industry. This decision is expected to bring significant institutional investment into the Bitcoin market.".to_string(),
            source: NewsSource::Bloomberg,
            url: Some("https://example.com/btc-etf".to_string()),
            published_at: Utc::now(),
            symbols: vec!["BTC".to_string()],
            relevance_score: 0.95,
        },
        NewsItem {
            id: "2".to_string(),
            title: "Major DeFi Protocol Hacked, $50M Stolen".to_string(),
            content: "A prominent DeFi protocol has suffered a major security breach, with hackers exploiting a smart contract vulnerability to drain approximately $50 million in user funds. The team has paused all operations while investigating.".to_string(),
            source: NewsSource::CoinDesk,
            url: Some("https://example.com/defi-hack".to_string()),
            published_at: Utc::now(),
            symbols: vec!["ETH".to_string()],
            relevance_score: 0.9,
        },
        NewsItem {
            id: "3".to_string(),
            title: "Ethereum Completes Major Network Upgrade".to_string(),
            content: "The Ethereum network has successfully completed its latest protocol upgrade, improving transaction throughput and reducing gas fees. Developers report the upgrade went smoothly with no issues.".to_string(),
            source: NewsSource::CoinTelegraph,
            url: Some("https://example.com/eth-upgrade".to_string()),
            published_at: Utc::now(),
            symbols: vec!["ETH".to_string()],
            relevance_score: 0.85,
        },
    ];

    // Demonstrate rule-based analysis (no API key needed)
    println!("--- Rule-Based Analysis (No API Key Required) ---\n");

    let sentiment_analyzer = SentimentAnalyzer::new();
    let event_classifier = EventClassifier::new();

    for news in &news_items {
        println!("📰 Title: {}", news.title);
        println!("   Source: {:?}", news.source);

        // Analyze sentiment
        let sentiment = sentiment_analyzer.analyze_rules(&format!("{} {}", news.title, news.content));
        println!("   Sentiment: {:.2} ({})", sentiment.score, sentiment.label);
        println!("   Confidence: {:.2}", sentiment.confidence);

        // Classify event
        let event = event_classifier.classify(&format!("{} {}", news.title, news.content));
        println!("   Event Type: {:?}", event.event_type);
        println!("   Market Impact: {:?}", event.impact);
        println!("   Time Horizon: {:?}", event.time_horizon);

        // Trading suggestion
        if sentiment.is_bullish() {
            println!("   💹 Suggestion: BULLISH - Consider buying {}", news.symbols.join(", "));
        } else if sentiment.is_bearish() {
            println!("   📉 Suggestion: BEARISH - Consider selling/avoiding {}", news.symbols.join(", "));
        } else {
            println!("   ➡️ Suggestion: NEUTRAL - Hold position");
        }

        println!();
    }

    // Demonstrate LLM-based analysis (requires API key)
    println!("--- LLM-Based Analysis (Requires API Key) ---\n");

    // Check for API key
    let api_key = std::env::var("OPENAI_API_KEY").ok();

    if let Some(key) = api_key {
        println!("✓ OpenAI API key found, performing LLM analysis...\n");

        let config = LlmConfig {
            api_key: key,
            model: "gpt-4".to_string(),
            temperature: 0.3,
            max_tokens: 500,
            ..Default::default()
        };

        let client = LlmClient::new(config)?;
        let analyzer = NewsAnalyzer::new(client);

        // Analyze first news item with LLM
        let news = &news_items[0];
        println!("Analyzing: {}", news.title);

        match analyzer.analyze_news(news).await {
            Ok(analysis) => {
                println!("LLM Analysis Result:");
                println!("  Sentiment: {:.2}", analysis.sentiment.score);
                println!("  Event Type: {:?}", analysis.event_type);
                println!("  Confidence: {:.2}", analysis.confidence);
                println!("  Affected Assets: {:?}", analysis.affected_assets);
            }
            Err(e) => {
                println!("Error during LLM analysis: {}", e);
            }
        }
    } else {
        println!("⚠ No OPENAI_API_KEY found in environment.");
        println!("  To use LLM-based analysis, set the OPENAI_API_KEY environment variable:");
        println!("  export OPENAI_API_KEY='your-api-key'");
        println!();
        println!("  Rule-based analysis (shown above) works without an API key.");
    }

    println!("\n=== Example Complete ===");

    Ok(())
}
