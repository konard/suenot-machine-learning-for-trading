//! Demo: Basic Multi-Agent Trading System
//!
//! This demo shows how to create and use multiple agents to analyze a stock.

use multi_agent_trading::{
    agents::{Agent, BearAgent, BullAgent, RiskManagerAgent, TechnicalAgent, TraderAgent},
    data::create_mock_data,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(60));
    println!("Multi-Agent LLM Trading System - Agent Demo");
    println!("{}", "=".repeat(60));

    // Step 1: Create market data
    println!("\n1. Loading market data...");
    let data = create_mock_data("AAPL", 252, 180.0);
    println!(
        "   Symbol: {}, {} candles loaded",
        data.symbol,
        data.len()
    );
    println!("   Latest price: ${:.2}", data.latest_close().unwrap_or(0.0));

    // Step 2: Create agents
    println!("\n2. Creating agent team...");
    let tech = TechnicalAgent::new("Technical-Analyst");
    let bull = BullAgent::new("Bull-Researcher");
    let bear = BearAgent::new("Bear-Researcher");
    let risk = RiskManagerAgent::new("Risk-Manager", 0.05, 0.15);
    let trader = TraderAgent::new("Head-Trader");

    println!("   Created 4 analysis agents + 1 trader agent");

    // Step 3: Run analyses
    println!("\n3. Running agent analyses...");
    println!("{}", "-".repeat(60));

    let mut analyses = Vec::new();

    // Technical analysis
    let tech_analysis = tech.analyze("AAPL", &data, None).await?;
    println!("\n   {} ({}):", tech.name(), tech.agent_type());
    println!("   Signal: {}", tech_analysis.signal);
    println!("   Confidence: {:.0}%", tech_analysis.confidence * 100.0);
    println!("   Reasoning: {}", tech_analysis.reasoning);
    analyses.push(tech_analysis);

    // Bull analysis
    let bull_analysis = bull.analyze("AAPL", &data, None).await?;
    println!("\n   {} ({}):", bull.name(), bull.agent_type());
    println!("   Signal: {}", bull_analysis.signal);
    println!("   Confidence: {:.0}%", bull_analysis.confidence * 100.0);
    analyses.push(bull_analysis);

    // Bear analysis
    let bear_analysis = bear.analyze("AAPL", &data, None).await?;
    println!("\n   {} ({}):", bear.name(), bear.agent_type());
    println!("   Signal: {}", bear_analysis.signal);
    println!("   Confidence: {:.0}%", bear_analysis.confidence * 100.0);
    analyses.push(bear_analysis);

    // Risk analysis
    let risk_analysis = risk.analyze("AAPL", &data, None).await?;
    println!("\n   {} ({}):", risk.name(), risk.agent_type());
    println!("   Signal: {}", risk_analysis.signal);
    println!("   Metrics: volatility={:.0}%, max_dd={:.0}%",
        risk_analysis.metrics.get("volatility").unwrap_or(&0.0) * 100.0,
        risk_analysis.metrics.get("max_drawdown").unwrap_or(&0.0) * 100.0
    );
    analyses.push(risk_analysis);

    // Step 4: Trader aggregation
    println!("\n{}", "-".repeat(60));
    println!("\n4. Trader aggregating analyses...");

    let final_decision = trader.aggregate("AAPL", &analyses);

    println!("\n   FINAL DECISION:");
    println!("   Signal: {}", final_decision.signal);
    println!("   Confidence: {:.0}%", final_decision.confidence * 100.0);
    println!("   Reasoning: {}", final_decision.reasoning);

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Symbol: AAPL");
    println!("Current Price: ${:.2}", data.latest_close().unwrap_or(0.0));
    println!("Recommendation: {}", final_decision.signal);
    println!("Overall Confidence: {:.0}%", final_decision.confidence * 100.0);
    println!(
        "Bullish Agents: {}",
        final_decision.metrics.get("bullish_count").unwrap_or(&0.0)
    );
    println!(
        "Bearish Agents: {}",
        final_decision.metrics.get("bearish_count").unwrap_or(&0.0)
    );
    println!("{}", "=".repeat(60));

    Ok(())
}
