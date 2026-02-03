//! Bybit Live Data Example
//!
//! Demonstrates how to fetch live cryptocurrency data from Bybit
//! and use PCMCI causal discovery for trading analysis and prediction.

use pcmci_causal_discovery::{
    BacktestConfig, Backtester, BybitClient, CausalTradingStrategy, IndependenceTest, PCMCIConfig,
};
use pcmci_causal_discovery::data::standardize;
use pcmci_causal_discovery::model::PCMCICausalDiscovery;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("=== PCMCI Causal Discovery with Bybit Data ===\n");

    // Initialize Bybit client
    let client = BybitClient::new();

    // Fetch recent klines for BTCUSDT
    println!("Fetching BTCUSDT data from Bybit...");
    let symbol = "BTCUSDT";
    let interval = "15"; // 15-minute candles
    let limit = 500;

    let data = match client.get_klines(symbol, interval, limit).await {
        Ok(data) => data,
        Err(e) => {
            println!("Error fetching data: {}", e);
            println!("\nNote: This example requires internet access to the Bybit API.");
            println!("If you're running offline, use the simple_backtest example instead.");
            return Ok(());
        }
    };

    println!("Fetched {} bars of {} data\n", data.len(), symbol);

    // Show some recent data
    println!("=== Recent Bars ===\n");
    for bar in data.bars.iter().rev().take(5) {
        let dt = chrono::DateTime::from_timestamp_millis(bar.timestamp)
            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        println!(
            "{}: O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
            dt, bar.open, bar.high, bar.low, bar.close, bar.volume
        );
    }

    // --- Step 1: Run causal discovery on the data ---
    println!("\n=== Causal Discovery Analysis ===\n");

    let causal_vars = data.to_causal_variables();
    let var_names = ["returns", "volume_change", "volatility", "spread"];

    println!(
        "Extracted {} observations of {} causal variables",
        causal_vars.nrows(),
        causal_vars.ncols()
    );
    println!("Variables: {:?}\n", var_names);

    // Standardize for analysis
    let standardized = standardize(&causal_vars);

    // Run PCMCI
    let pcmci_config = PCMCIConfig {
        max_lag: 5,
        significance_level: 0.05,
        test_type: IndependenceTest::ParCorr,
        max_conds_dim: Some(4),
    };

    let mut pcmci = PCMCICausalDiscovery::new(pcmci_config.clone());
    let graph = pcmci.fit(&standardized);

    // Print discovered causal links
    println!("Discovered Causal Links (p < 0.05):\n");
    let significant = graph.get_significant_links(0.05);
    if significant.is_empty() {
        println!("  No significant causal links found.");
        println!("  This may indicate that the market is efficient at this timescale,");
        println!("  or that more data / different lags are needed.\n");
    } else {
        for link in &significant {
            let source_name = var_names.get(link.source).unwrap_or(&"Unknown");
            let target_name = var_names.get(link.target).unwrap_or(&"Unknown");
            println!(
                "  {} (t-{}) -> {}: strength={:.4}, p-value={:.4}",
                source_name, link.lag, target_name, link.strength, link.p_value
            );
        }
        println!();
    }

    // Focus on what causes returns
    println!("=== Causes of Returns ===\n");
    let return_parents = graph.get_parents(0, 0.05);
    if return_parents.is_empty() {
        println!("  No significant causes of returns found.");
    } else {
        for link in &return_parents {
            let source_name = var_names.get(link.source).unwrap_or(&"Unknown");
            let direction = if link.strength > 0.0 { "positive" } else { "negative" };
            println!(
                "  {} (lag {}) has {} effect on returns (strength={:.4})",
                source_name, link.lag, direction, link.strength
            );
        }
    }

    // --- Step 2: Run backtest ---
    println!("\n=== Running Causal Trading Backtest ===\n");

    let backtest_config = BacktestConfig {
        initial_capital: 10_000.0,
        position_size: 0.05,
        stop_loss: Some(0.015),
        take_profit: Some(0.03),
        fee_rate: 0.0004,
        slippage: 0.001,
        lookback_window: 100,
        reestimate_freq: 50,
        min_link_strength: 0.1,
        significance_level: 0.1,
    };

    let strategy_pcmci_config = PCMCIConfig {
        max_lag: 3,
        significance_level: 0.1,
        test_type: IndependenceTest::ParCorr,
        max_conds_dim: Some(3),
    };

    let mut strategy = CausalTradingStrategy::new(strategy_pcmci_config, backtest_config.clone());

    let backtester = Backtester::new(backtest_config);
    let result = backtester.run(&mut strategy, &data);

    // Print backtest results
    println!("=== Backtest Results for {} ===\n", symbol);
    println!("Total Return:          {:.2}%", result.total_return * 100.0);
    println!("Annual Return:         {:.2}%", result.annual_return * 100.0);
    println!("Sharpe Ratio:          {:.2}", result.sharpe_ratio);
    println!("Max Drawdown:          {:.2}%", result.max_drawdown * 100.0);
    println!("Win Rate:              {:.2}%", result.win_rate * 100.0);
    println!("Profit Factor:         {:.2}", result.profit_factor);
    println!("Total Trades:          {}", result.total_trades);
    println!("Graph Updates:         {}", result.n_graph_updates);
    println!("Avg Significant Links: {:.1}", result.avg_significant_links);

    // Print final causal graph
    println!("\n=== Final Causal Graph ===\n");
    strategy.print_graph_summary();

    // Generate current prediction
    println!("\n=== Current Market Analysis ===\n");
    if causal_vars.nrows() >= 20 {
        let recent = causal_vars
            .slice(ndarray::s![(causal_vars.nrows() - 20).., ..])
            .to_owned();

        let signal = strategy.generate_signal(&recent);

        let current_price = data.bars.last().unwrap().close;
        println!("Current {} price: ${:.2}", symbol, current_price);
        println!("Signal:           {:?}", signal.signal);
        println!("Strength:         {:.4}", signal.strength);
        println!("Causal links:     {}", signal.n_links);

        if let Some(ref primary) = signal.primary_cause {
            let source_name = var_names.get(primary.source).unwrap_or(&"Unknown");
            println!(
                "Primary driver:   {} (lag {}, strength={:.4})",
                source_name, primary.lag, primary.strength
            );
        }

        match signal.signal {
            pcmci_causal_discovery::strategy::Signal::Long => {
                println!("\nCausal analysis suggests: BULLISH");
                println!("Identified causal factors predict upward movement.");
            }
            pcmci_causal_discovery::strategy::Signal::Short => {
                println!("\nCausal analysis suggests: BEARISH");
                println!("Identified causal factors predict downward movement.");
            }
            pcmci_causal_discovery::strategy::Signal::Hold => {
                println!("\nCausal analysis suggests: NEUTRAL");
                println!("No strong causal signal detected - recommend holding.");
            }
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
