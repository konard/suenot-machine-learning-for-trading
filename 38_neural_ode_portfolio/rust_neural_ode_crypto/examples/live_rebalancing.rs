//! # Live Rebalancing Example
//!
//! Demonstrates continuous portfolio rebalancing using Neural ODE.
//!
//! Run with:
//! ```bash
//! cargo run --example live_rebalancing
//! ```

use anyhow::Result;
use neural_ode_crypto::data::{BybitClient, CandleData, Timeframe, TechnicalIndicators, Features};
use neural_ode_crypto::model::NeuralODEPortfolio;
use neural_ode_crypto::strategy::ContinuousRebalancer;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== Live Rebalancing Simulation ===");

    // Portfolio configuration
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let initial_value = 100_000.0;
    let rebalance_threshold = 0.02; // 2%

    info!("Portfolio: {:?}", symbols);
    info!("Initial Value: ${:.2}", initial_value);
    info!("Rebalance Threshold: {:.1}%", rebalance_threshold * 100.0);

    // Fetch current market data
    info!("\nFetching market data from Bybit...");
    let client = BybitClient::new();

    let mut all_data = Vec::new();
    for symbol in &symbols {
        match client.get_klines(symbol, "60", 200).await {
            Ok(candles) => {
                info!("  {}: {} candles", symbol, candles.len());
                all_data.push(CandleData::new(
                    symbol.to_string(),
                    Timeframe::Hour1,
                    candles,
                ));
            }
            Err(e) => {
                warn!("  {}: Error: {}", symbol, e);
                return Err(e);
            }
        }
    }

    // Calculate features for each asset
    let indicators = TechnicalIndicators::default();
    let features = calculate_portfolio_features(&all_data, &indicators);

    info!("\nFeatures calculated: {} per asset", features.n_features);

    // Create Neural ODE model
    info!("\nInitializing Neural ODE model...");
    let model = NeuralODEPortfolio::new(
        symbols.len(),
        features.n_features,
        16, // hidden dimension
    );
    info!("Model parameters: {}", model.num_params());

    // Create rebalancer
    let rebalancer = ContinuousRebalancer::new(model, rebalance_threshold)
        .with_asset_names(symbols.iter().map(|s| s.to_string()).collect())
        .with_min_trade_size(100.0)
        .with_transaction_cost(0.001);

    // Current portfolio state (equal weight)
    let n_assets = symbols.len();
    let mut current_weights = vec![1.0 / n_assets as f64; n_assets];
    let mut portfolio_value = initial_value;

    info!("\n=== Portfolio Analysis ===");
    info!("Current weights: {:?}",
        current_weights.iter()
            .zip(symbols.iter())
            .map(|(w, s)| format!("{}: {:.1}%", s, w * 100.0))
            .collect::<Vec<_>>()
    );

    // Check if rebalancing is needed
    let decision = rebalancer.check_rebalance(&current_weights, &features);

    info!("\nRebalance Decision:");
    info!("  Should rebalance: {}", decision.should_rebalance);
    info!("  Max deviation: {:.2}%", decision.max_deviation * 100.0);
    info!("  Target weights: {:?}",
        decision.target_weights.iter()
            .zip(symbols.iter())
            .map(|(w, s)| format!("{}: {:.1}%", s, w * 100.0))
            .collect::<Vec<_>>()
    );

    // Execute rebalancing if needed
    if decision.should_rebalance {
        info!("\n=== Executing Rebalance ===");

        let result = rebalancer.execute_rebalance(
            &current_weights,
            &features,
            portfolio_value,
        );

        if result.executed {
            info!("Trades:");
            for trade in &result.trades {
                let action = if trade.dollar_amount > 0.0 { "Buy" } else { "Sell" };
                info!("  {} {} ${:.2} ({:+.2}%)",
                    action,
                    trade.asset_name,
                    trade.dollar_amount.abs(),
                    trade.weight_change * 100.0
                );
            }

            info!("\nTransaction cost: ${:.2}", result.transaction_cost);
            info!("Total turnover: {:.2}%", result.turnover * 100.0);

            // Update state
            portfolio_value -= result.transaction_cost;
            current_weights = result.new_weights;

            info!("\nNew portfolio:");
            info!("  Value: ${:.2}", portfolio_value);
            info!("  Weights: {:?}",
                current_weights.iter()
                    .zip(symbols.iter())
                    .map(|(w, s)| format!("{}: {:.1}%", s, w * 100.0))
                    .collect::<Vec<_>>()
            );
        }
    } else {
        info!("\nNo rebalancing needed.");
    }

    // Simulate trajectory
    info!("\n=== Projected Trajectory (next 24 hours) ===");
    let trajectory = rebalancer.model().solve_trajectory(
        &current_weights,
        &features,
        (0.0, 1.0),  // 1 unit = 24 hours
        7,
    );

    for (t, weights) in &trajectory {
        let hour = (t * 24.0) as i32;
        let weights_str: Vec<String> = weights.iter()
            .zip(symbols.iter())
            .map(|(w, s)| format!("{}:{:.1}%", s.replace("USDT", ""), w * 100.0))
            .collect();
        info!("  +{}h: [{}]", hour, weights_str.join(" "));
    }

    info!("\nDone!");
    Ok(())
}

fn calculate_portfolio_features(
    data: &[CandleData],
    indicators: &TechnicalIndicators,
) -> Features {
    let n_assets = data.len();

    let mut all_features = Vec::new();
    let mut feature_names = Vec::new();

    for asset_data in data {
        let asset_features = indicators.calculate_all(asset_data);

        if feature_names.is_empty() {
            feature_names = asset_features.names.clone();
        }

        all_features.push(asset_features.data[0].clone());
    }

    Features {
        n_assets,
        n_features: feature_names.len(),
        data: all_features,
        names: feature_names,
    }
}
