//! Basic Analysis Example
//!
//! This example demonstrates how to use the LLM Backtesting Assistant
//! to analyze sample backtest results using a mock LLM provider.
//!
//! Run with: `cargo run --example basic_analysis`

use llm_backtesting_assistant::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== LLM Backtesting Assistant - Basic Analysis Example ===\n");

    // Create sample backtest results
    println!("1. Creating sample backtest results...");
    let results = BacktestResults::sample();

    println!("   Strategy: {}", results.strategy_name);
    println!("   Symbol: {}", results.symbol);
    println!("   Period: {} to {}",
        results.start_date.format("%Y-%m-%d"),
        results.end_date.format("%Y-%m-%d")
    );
    println!("   Initial Capital: ${:.2}", results.initial_capital);
    println!("   Final Capital: ${:.2}", results.final_capital);
    println!();

    // Display performance metrics
    println!("2. Performance Metrics:");
    println!("   Total Return: {:.2}%", results.metrics.total_return * 100.0);
    println!("   Sharpe Ratio: {:.2}", results.metrics.sharpe_ratio);
    println!("   Sortino Ratio: {:.2}", results.metrics.sortino_ratio);
    println!("   Max Drawdown: {:.2}%", results.metrics.max_drawdown * 100.0);
    println!("   Win Rate: {:.2}%", results.metrics.win_rate * 100.0);
    println!("   Profit Factor: {:.2}", results.metrics.profit_factor);
    println!("   Total Trades: {}", results.metrics.total_trades);
    println!();

    // Create assistant with mock LLM provider
    println!("3. Creating assistant with Mock LLM provider...");
    let assistant = BacktestingAssistant::with_provider(MockLlmClient::new());

    // Run analysis
    println!("4. Running LLM analysis...\n");
    let analysis = assistant.analyze(&results).await?;

    // Display analysis result
    println!("{}", analysis);

    // Generate report in different formats
    println!("\n5. Generating reports...");

    let report = Report::new(results).with_analysis(analysis);

    // Text report
    let text_report = report.generate(ReportFormat::Text)?;
    println!("\n--- Text Report Preview (first 500 chars) ---");
    println!("{}", &text_report[..text_report.len().min(500)]);
    println!("...\n");

    // Markdown report
    let md_report = report.generate(ReportFormat::Markdown)?;
    println!("--- Markdown Report Preview (first 500 chars) ---");
    println!("{}", &md_report[..md_report.len().min(500)]);
    println!("...\n");

    println!("=== Example Complete ===");
    println!("\nTo use with a real LLM provider, set your API key:");
    println!("  - OpenAI: OPENAI_API_KEY environment variable");
    println!("  - Anthropic: ANTHROPIC_API_KEY environment variable");

    Ok(())
}
