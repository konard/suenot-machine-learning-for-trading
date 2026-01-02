//! Example: Generate live trading signals
//!
//! This example demonstrates how to generate real-time trading signals
//! using data from Bybit and the Graph Transformer model.

use graph_transformer_trading::prelude::*;
use graph_transformer_trading::graph::builder::create_test_graph;
use graph_transformer_trading::transformer::model::get_predictions;
use graph_transformer_trading::strategy::portfolio::PortfolioBuilder;
use ndarray::Array2;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Live Trading Signal Generator ===\n");

    // Note: In production, you would fetch real data from Bybit
    // For this example, we'll use simulated data

    println!("1. Fetching market data...");

    // Try to fetch real data, fall back to simulation
    let (graph, features) = match fetch_real_data().await {
        Ok((g, f)) => {
            println!("   Using LIVE data from Bybit");
            (g, f)
        }
        Err(e) => {
            println!("   Could not fetch live data: {}", e);
            println!("   Using SIMULATED data for demonstration");
            create_simulated_data()
        }
    };

    println!("   Graph: {} nodes, {} edges", graph.num_nodes(), graph.num_edges());
    println!("   Features: {:?}", features.shape());

    // Create model
    println!("\n2. Loading Graph Transformer model...");
    let config = GraphTransformerConfig::minimal();
    let model = GraphTransformer::new(config);
    println!("   Model loaded successfully");

    // Generate predictions
    println!("\n3. Generating predictions...");
    let predictions = get_predictions(&model, &features, &graph);

    // Display predictions
    println!("\n4. Current Market Analysis:");
    println!("{:=<80}", "");
    println!(
        "{:<12} {:>8} {:>8} {:>8} {:>10} {:>12} {:>10}",
        "Asset", "P(Up)", "P(Down)", "Ret%", "Direction", "Confidence", "Signal"
    );
    println!("{:-<80}", "");

    for pred in &predictions {
        let direction = pred.predicted_direction();
        let conf = pred.confidence();

        let signal = if conf > 0.6 {
            match direction {
                "UP" => "🟢 LONG",
                "DOWN" => "🔴 SHORT",
                _ => "⚪ HOLD",
            }
        } else {
            "⚪ HOLD"
        };

        println!(
            "{:<12} {:>7.1}% {:>7.1}% {:>7.2}% {:>10} {:>11.1}% {:>10}",
            pred.symbol,
            pred.prob_up * 100.0,
            pred.prob_down * 100.0,
            pred.predicted_return * 100.0,
            direction,
            conf * 100.0,
            signal
        );
    }

    // Generate actionable signals
    println!("\n5. Actionable Trading Signals:");
    println!("{:=<60}", "");

    let signal_gen = SignalGenerator::new(0.55);
    let signals = signal_gen.from_predictions(&predictions);

    if signals.is_empty() {
        println!("   No strong signals at this time.");
        println!("   Market conditions are uncertain - stay cautious.");
    } else {
        for signal in &signals {
            let emoji = match signal.signal_type {
                SignalType::Long => "🟢",
                SignalType::Short => "🔴",
                SignalType::Hold => "⚪",
            };

            println!(
                "   {} {:?} {} | Confidence: {:.1}% | Expected: {:+.2}%",
                emoji,
                signal.signal_type,
                signal.symbol,
                signal.confidence * 100.0,
                signal.expected_return * 100.0
            );
        }
    }

    // Build portfolio
    println!("\n6. Suggested Portfolio Allocation:");
    println!("{:-<60}", "");

    let portfolio_builder = PortfolioBuilder::new();
    let portfolio = portfolio_builder.build(&signals);

    if portfolio.positions.is_empty() {
        println!("   100% CASH - No favorable positions identified");
    } else {
        for pos in &portfolio.positions {
            let direction = if pos.direction > 0 { "LONG " } else { "SHORT" };
            println!(
                "   {} {} - Allocation: {:.1}%",
                direction,
                pos.symbol,
                pos.weight * 100.0
            );
        }
        println!("   CASH - Allocation: {:.1}%", portfolio.cash_weight * 100.0);
    }

    // Risk metrics
    println!("\n7. Risk Summary:");
    println!("{:-<60}", "");
    println!("   Net Exposure: {:.1}%", portfolio.net_exposure() * 100.0);
    println!("   Long Exposure: {:.1}%", portfolio.long_exposure() * 100.0);
    println!("   Short Exposure: {:.1}%", portfolio.short_exposure() * 100.0);

    // Disclaimer
    println!("\n{:=<80}", "");
    println!("DISCLAIMER: This is for educational purposes only.");
    println!("Do NOT trade real money based on these signals.");
    println!("Cryptocurrency trading involves substantial risk of loss.");
    println!("{:=<80}", "");

    Ok(())
}

async fn fetch_real_data() -> Result<(MarketGraph, Array2<f64>), Box<dyn Error>> {
    let client = BybitClient::new();
    let symbols = DEFAULT_SYMBOLS;

    // Fetch tickers
    let tickers = client.get_tickers_filtered(symbols).await?;

    if tickers.is_empty() {
        return Err("No tickers fetched".into());
    }

    // Build graph from tickers
    let builder = MarketGraphBuilder::new()
        .include_sector_edges(true);
    let graph = builder.build_from_tickers(&tickers);

    // Create features from tickers
    let feature_engine = FeatureEngine::new(8);
    let features = feature_engine.compute_from_tickers(&tickers);

    Ok((graph, features))
}

fn create_simulated_data() -> (MarketGraph, Array2<f64>) {
    let graph = create_test_graph();
    let num_nodes = graph.num_nodes();

    let features = Array2::from_shape_fn((num_nodes, 8), |_| {
        rand::random::<f64>() * 2.0 - 1.0
    });

    (graph, features)
}
