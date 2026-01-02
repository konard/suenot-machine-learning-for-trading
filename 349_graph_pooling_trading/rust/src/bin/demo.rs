//! Demo: Graph Pooling for Cryptocurrency Trading
//!
//! This example demonstrates:
//! 1. Fetching market data from Bybit
//! 2. Building a market graph
//! 3. Applying hierarchical graph pooling
//! 4. Generating trading signals

use graph_pooling_trading::prelude::*;
use graph_pooling_trading::{DEFAULT_SYMBOLS, StrategyConfig};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    println!("=================================================");
    println!("  Graph Pooling for Cryptocurrency Trading");
    println!("  Using Bybit Market Data");
    println!("=================================================\n");

    // Initialize Bybit client
    info!("Initializing Bybit client...");
    let client = BybitClient::new(None, None);

    // Check connection
    let server_time = client.get_server_time().await?;
    info!("Connected to Bybit. Server time: {}", server_time);

    // Select symbols to analyze
    let symbols: Vec<&str> = DEFAULT_SYMBOLS.iter().take(15).copied().collect();
    println!("\nAnalyzing {} cryptocurrencies:", symbols.len());
    for (i, symbol) in symbols.iter().enumerate() {
        println!("  {}. {}", i + 1, symbol);
    }

    // Fetch kline data
    println!("\nFetching 1-hour klines (100 candles each)...");
    let klines = client.get_klines_batch(&symbols, "60", 100).await?;
    info!("Fetched data for {} symbols", klines.len());

    // Build market graph
    println!("\nBuilding market correlation graph...");
    let graph = MarketGraph::from_klines(&klines)?;
    println!("{}", graph.summary());

    // Find asset clusters
    println!("\nDiscovering correlated asset clusters...");
    let clusters = graph.find_clusters(0.5);
    for (i, cluster) in clusters.iter().enumerate() {
        println!("  Cluster {}: {:?}", i + 1, cluster);
    }

    // Apply hierarchical pooling
    println!("\nApplying hierarchical graph pooling...");
    let config = StrategyConfig {
        n_clusters_l1: 5,
        n_clusters_l2: 2,
        ..Default::default()
    };
    let mut strategy = GraphPoolingStrategy::with_config(config);
    let signals = strategy.generate_signals(&graph)?;

    // Display signals
    println!("\nTrading Signals:");
    println!("{:-<60}", "");
    println!(
        "{:<12} {:>10} {:>10} {:>8} {:>12}",
        "Symbol", "Position", "Confidence", "Cluster", "RelMomentum"
    );
    println!("{:-<60}", "");

    for signal in &signals {
        let position_str = if signal.position > 0.0 {
            format!("+{:.4}", signal.position)
        } else {
            format!("{:.4}", signal.position)
        };

        println!(
            "{:<12} {:>10} {:>10.2}% {:>8} {:>+12.4}",
            signal.symbol,
            position_str,
            signal.confidence * 100.0,
            signal.cluster,
            signal.relative_momentum
        );
    }
    println!("{:-<60}", "");

    // Display regime
    println!("\nCurrent Market Regime: {}", strategy.current_regime());

    // Top long and short recommendations
    let mut sorted_signals = signals.clone();
    sorted_signals.sort_by(|a, b| b.position.partial_cmp(&a.position).unwrap());

    println!("\nTop Long Recommendations:");
    for signal in sorted_signals.iter().take(3) {
        if signal.position > 0.0 {
            println!(
                "  {} (position: {:.4}, confidence: {:.1}%)",
                signal.symbol,
                signal.position,
                signal.confidence * 100.0
            );
        }
    }

    println!("\nTop Short Recommendations:");
    for signal in sorted_signals.iter().rev().take(3) {
        if signal.position < 0.0 {
            println!(
                "  {} (position: {:.4}, confidence: {:.1}%)",
                signal.symbol,
                signal.position,
                signal.confidence * 100.0
            );
        }
    }

    println!("\n=================================================");
    println!("  Demo completed successfully!");
    println!("=================================================");

    Ok(())
}
