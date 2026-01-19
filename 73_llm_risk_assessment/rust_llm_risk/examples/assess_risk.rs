//! Example: Assess risk using LLM.
//!
//! This example demonstrates how to:
//! 1. Create an LLM client
//! 2. Build a risk assessor
//! 3. Analyze text for risk factors
//!
//! Run with: cargo run --example assess_risk
//!
//! Note: Requires an LLM API key or local Ollama instance.

use anyhow::Result;
use clap::Parser;
use llm_risk_assessment::llm::{LLMClient, LLMProvider};
use llm_risk_assessment::risk::RiskAssessor;
use tracing::info;
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(name = "assess_risk")]
#[command(about = "Assess financial risk from text using LLM")]
struct Args {
    /// Symbol to analyze
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// LLM provider (openai, anthropic, local)
    #[arg(short, long, default_value = "local")]
    provider: String,

    /// Model name
    #[arg(short, long, default_value = "llama2")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Starting risk assessment example");
    info!("Using provider: {}, model: {}", args.provider, args.model);

    // Create LLM provider based on args
    let provider = match args.provider.as_str() {
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY")
                .expect("OPENAI_API_KEY environment variable required");
            LLMProvider::openai_gpt4(api_key)
        }
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .expect("ANTHROPIC_API_KEY environment variable required");
            LLMProvider::anthropic_claude3(api_key)
        }
        _ => LLMProvider::ollama(args.model.clone()),
    };

    // Create LLM client and risk assessor
    let llm_client = LLMClient::new(provider);
    let assessor = RiskAssessor::new(llm_client);

    // Sample news text for analysis
    let sample_texts = vec![
        (
            "Bullish news",
            r#"Bitcoin ETF sees record inflows as institutional adoption accelerates.
            Major banks announce plans to offer cryptocurrency custody services.
            Market sentiment remains strongly positive with decreasing volatility.
            Trading volumes hit all-time highs across major exchanges."#
        ),
        (
            "Bearish news",
            r#"Cryptocurrency exchange faces regulatory scrutiny following audit concerns.
            Several key executives have resigned amid investigations.
            Trading volumes have dropped significantly as investors move to safer assets.
            Analysts warn of potential liquidity issues in the market."#
        ),
        (
            "Mixed signals",
            r#"Bitcoin price consolidates around key support level.
            Regulatory clarity improves in some jurisdictions while others impose restrictions.
            Institutional interest remains steady despite market uncertainty.
            Technical indicators show mixed signals with RSI near neutral levels."#
        ),
    ];

    println!("\n=== LLM Risk Assessment Demo ===\n");
    println!("Analyzing {} different market scenarios...\n", sample_texts.len());

    for (scenario_name, text) in sample_texts {
        println!("Scenario: {}", scenario_name);
        println!("{}", "-".repeat(50));

        match assessor.assess_with_context(text, &args.symbol, None).await {
            Ok(risk_score) => {
                println!("{}", risk_score);
                println!("Position Multiplier: {:.2}", risk_score.position_multiplier());

                // Analyze dimensions
                let (highest_dim, highest_score) = risk_score.highest_risk_dimension();
                let (lowest_dim, lowest_score) = risk_score.lowest_risk_dimension();

                println!("\nRisk Analysis:");
                println!("  Highest risk: {} ({:.1}/10)", highest_dim, highest_score);
                println!("  Lowest risk:  {} ({:.1}/10)", lowest_dim, lowest_score);

                if risk_score.has_high_risk_dimension(7.0) {
                    println!("  WARNING: High risk dimension detected!");
                }
            }
            Err(e) => {
                println!("Error assessing risk: {}", e);
                println!("Make sure your LLM provider is configured correctly.");
                println!("For local mode, ensure Ollama is running.");
            }
        }

        println!();
    }

    // Example of aggregating multiple assessments
    println!("=== Multiple Source Aggregation ===\n");

    let multiple_sources = vec![
        "Bitcoin showing strong momentum with increasing institutional buying.",
        "Market volatility remains elevated due to macroeconomic uncertainty.",
        "Technical analysis suggests bullish continuation pattern forming.",
    ];

    let source_refs: Vec<&str> = multiple_sources.iter().map(|s| s.as_str()).collect();

    match assessor.assess_multiple(&source_refs).await {
        Ok(aggregated_score) => {
            println!("Aggregated Risk Score from {} sources:", multiple_sources.len());
            println!("{}", aggregated_score);
        }
        Err(e) => {
            println!("Error in aggregated assessment: {}", e);
        }
    }

    println!("Risk assessment example complete!");

    Ok(())
}
