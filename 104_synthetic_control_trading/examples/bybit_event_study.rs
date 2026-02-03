//! Bybit Event Study Example
//!
//! Demonstrates using SCM with cryptocurrency data from Bybit.

use synthetic_control_trading::{BybitClient, SyntheticControlModel};
use synthetic_control_trading::data::bybit::extract_returns;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Synthetic Control Method - Bybit Event Study");
    println!("=============================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch data for treated asset (BTC)
    println!("Fetching BTC/USDT data...");
    let btc_klines = client.fetch_klines("BTCUSDT", "D", 100).await?;
    let btc_returns = extract_returns(&btc_klines);

    // Fetch data for donor assets
    println!("Fetching donor data...");
    let eth_klines = client.fetch_klines("ETHUSDT", "D", 100).await?;
    let bnb_klines = client.fetch_klines("BNBUSDT", "D", 100).await?;
    let sol_klines = client.fetch_klines("SOLUSDT", "D", 100).await?;
    let xrp_klines = client.fetch_klines("XRPUSDT", "D", 100).await?;

    let eth_returns = extract_returns(&eth_klines);
    let bnb_returns = extract_returns(&bnb_klines);
    let sol_returns = extract_returns(&sol_klines);
    let xrp_returns = extract_returns(&xrp_klines);

    // Split into pre and post treatment periods
    // Assume event happened at day 60
    let event_day = 60;

    let btc_pre = &btc_returns[..event_day];
    let btc_post = &btc_returns[event_day..];

    let donors_pre = vec![
        eth_returns[..event_day].to_vec(),
        bnb_returns[..event_day].to_vec(),
        sol_returns[..event_day].to_vec(),
        xrp_returns[..event_day].to_vec(),
    ];

    let donors_post = vec![
        eth_returns[event_day..].to_vec(),
        bnb_returns[event_day..].to_vec(),
        sol_returns[event_day..].to_vec(),
        xrp_returns[event_day..].to_vec(),
    ];

    println!("\nData summary:");
    println!("  Pre-treatment periods: {}", btc_pre.len());
    println!("  Post-treatment periods: {}", btc_post.len());
    println!("  Number of donors: {}", donors_pre.len());

    // Fit SCM model
    println!("\nFitting Synthetic Control Model...");
    let mut model = SyntheticControlModel::new();
    model.fit_with_labels(
        btc_pre,
        &donors_pre,
        vec![
            "ETH".to_string(),
            "BNB".to_string(),
            "SOL".to_string(),
            "XRP".to_string(),
        ],
    )?;

    println!(
        "Pre-treatment RMSE: {:.6}",
        model.pre_treatment_rmse().unwrap()
    );

    // Print weights
    println!("\nSynthetic BTC composition:");
    if let Some(weights_map) = model.weights_map() {
        for (label, weight) in weights_map {
            if weight > 0.01 {
                println!("  {}: {:.2}%", label, weight * 100.0);
            }
        }
    }

    // Estimate treatment effects
    let effects = model.estimate_effects(btc_post, &donors_post)?;

    println!("\nTreatment Effect Analysis:");
    println!("  Post-treatment periods: {}", effects.effects.len());
    println!("  Cumulative effect: {:.4}", effects.cumulative);
    println!("  CAR: {:.2}%", effects.car * 100.0);

    // Print daily effects
    println!("\n  Daily effects (first 10 days):");
    for (i, effect) in effects.effects.iter().take(10).enumerate() {
        println!("    Day {}: {:.4}", i + 1, effect);
    }

    // Run placebo tests
    println!("\nRunning placebo tests for inference...");
    let placebo_results = model.run_placebo_tests(btc_post, &donors_post)?;

    println!(
        "  P-value: {:.4} (rank {}/{})",
        placebo_results.p_value, placebo_results.rank, placebo_results.n_placebos + 1
    );

    // Interpretation
    println!("\nInterpretation:");
    if effects.car > 0.0 {
        println!("  BTC outperformed its synthetic control by {:.2}%", effects.car * 100.0);
    } else {
        println!("  BTC underperformed its synthetic control by {:.2}%", effects.car.abs() * 100.0);
    }

    if placebo_results.p_value < 0.1 {
        println!("  This effect is statistically significant at the 10% level.");
    } else {
        println!("  This effect is NOT statistically significant (p > 0.10).");
    }

    println!("\nEvent study completed!");

    Ok(())
}
