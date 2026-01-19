//! Demo: Bull vs Bear Debate
//!
//! This demo shows how to use adversarial debate between agents.

use multi_agent_trading::{
    agents::{BearAgent, BullAgent},
    communication::Debate,
    data::create_mock_data,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(60));
    println!("Multi-Agent LLM Trading - Bull vs Bear Debate");
    println!("{}", "=".repeat(60));

    // Create market data
    println!("\n1. Setting up market scenario...");
    let data = create_mock_data("DEMO", 252, 100.0);

    let start_price = data.candles[0].close;
    let end_price = data.latest_close().unwrap_or(0.0);
    let total_return = (end_price / start_price - 1.0) * 100.0;

    println!("   Symbol: DEMO");
    println!("   Start: ${:.2}", start_price);
    println!("   Current: ${:.2}", end_price);
    println!("   Total Return: {:.1}%", total_return);

    // Create debaters
    println!("\n2. Introducing the debaters...");
    let bull = BullAgent::new("Oliver Optimist");
    let bear = BearAgent::new("Patty Pessimist");

    println!("   Bull Agent: {}", bull.name());
    println!("   Bear Agent: {}", bear.name());

    // Conduct debate
    println!("\n3. Conducting debate (3 rounds)...");
    println!("{}", "=".repeat(60));

    let mut debate = Debate::new(bull, bear, 3);
    let result = debate.conduct("DEMO", &data).await?;

    // Display rounds
    for round in &result.rounds {
        println!("\n--- ROUND {} ---", round.round);

        println!("\nBULL ({}):", round.bull_argument.agent_name);
        println!("   {}", round.bull_argument.reasoning);
        println!("   Confidence: {:.0}%", round.bull_argument.confidence * 100.0);

        println!("\nBEAR ({}):", round.bear_argument.agent_name);
        println!("   {}", round.bear_argument.reasoning);
        println!("   Confidence: {:.0}%", round.bear_argument.confidence * 100.0);
    }

    // Results
    println!("\n{}", "=".repeat(60));
    println!("DEBATE RESULTS");
    println!("{}", "=".repeat(60));

    println!("\nScores:");
    println!("   Bull avg confidence: {:.0}%", result.bull_avg_confidence * 100.0);
    println!("   Bear avg confidence: {:.0}%", result.bear_avg_confidence * 100.0);

    println!("\nWinner: {}", result.winner.to_uppercase());
    println!("Final Signal: {}", result.final_signal);
    println!("Final Confidence: {:.0}%", result.final_confidence * 100.0);
    println!("Conclusion: {}", result.conclusion);

    // Trading implications
    println!("\n{}", "=".repeat(60));
    println!("TRADING IMPLICATIONS");
    println!("{}", "=".repeat(60));

    match result.winner.as_str() {
        "bull" => {
            println!("\n   The debate favored the BULL case. Suggested actions:");
            println!("   - Consider opening a long position");
            println!("   - Use the bear's concerns as risk factors to monitor");
            println!("   - Set stop-loss based on support levels");
        }
        "bear" => {
            println!("\n   The debate favored the BEAR case. Suggested actions:");
            println!("   - Avoid new long positions or consider shorting");
            println!("   - Take profits on existing positions");
            println!("   - Watch for the bull's positive catalysts");
        }
        _ => {
            println!("\n   The debate was balanced. Suggested actions:");
            println!("   - Stay on the sidelines for now");
            println!("   - Wait for clearer signals");
            println!("   - Monitor both bull and bear factors");
        }
    }

    Ok(())
}
