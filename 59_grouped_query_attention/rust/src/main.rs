//! GQA Trading Demo
//!
//! This binary demonstrates the Grouped Query Attention trading model.

use clap::Parser;
use gqa_trading::{
    data::{generate_synthetic_data, load_bybit_data},
    model::GQATrader,
    predict::analyze_prediction,
    strategy::{backtest_strategy, compare_strategies, BacktestConfig},
};

/// GQA Trading Model Demo
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Symbol to analyze (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Time interval (e.g., 1h, 4h, 1d)
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "500")]
    limit: usize,

    /// Use synthetic data instead of real data
    #[arg(long)]
    synthetic: bool,

    /// Run backtest
    #[arg(long)]
    backtest: bool,

    /// Run strategy comparison
    #[arg(long)]
    compare: bool,
}

fn main() -> anyhow::Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Grouped Query Attention (GQA) Trading Model Demo         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Load or generate data
    let data = if args.synthetic {
        println!("\n📊 Generating synthetic data...");
        generate_synthetic_data(args.limit, 50000.0, 0.02)
    } else {
        println!("\n📊 Loading {} data from Bybit...", args.symbol);
        match load_bybit_data(&args.symbol, &args.interval, args.limit) {
            Ok(data) => data,
            Err(e) => {
                log::warn!("Failed to fetch data: {}. Using synthetic data.", e);
                generate_synthetic_data(args.limit, 50000.0, 0.02)
            }
        }
    };

    println!("   Loaded {} candles", data.len());
    println!("   Latest close: ${:.2}", data.latest_close());

    // Create model
    println!("\n🧠 Creating GQA Trading Model...");
    let model = GQATrader::new(
        5,   // input_dim (OHLCV)
        64,  // d_model
        8,   // num_heads
        2,   // num_kv_heads (GQA grouping)
        4,   // num_layers
    );

    println!("   Model parameters: ~{}", model.param_count());

    // Show memory savings
    let stats = gqa_trading::model::GroupedQueryAttention::new(64, 8, 2).memory_stats(100);
    println!("\n💾 Memory Efficiency:");
    println!("   GQA KV cache: {} KB", stats.gqa_kv_cache_bytes / 1024);
    println!("   MHA KV cache: {} KB", stats.mha_kv_cache_bytes / 1024);
    println!("   Memory savings: {:.0}%", stats.memory_savings * 100.0);

    // Make a prediction on latest data
    println!("\n🔮 Making prediction on latest sequence...");
    let seq_len: usize = 60;
    if data.len() >= seq_len {
        let start_idx = data.len() - seq_len;
        let sequence = data.data.slice(ndarray::s![start_idx.., ..]).to_owned();
        let analysis = analyze_prediction(&model, &sequence);

        println!("   Prediction: {}", analysis.prediction_label);
        println!("   Confidence: {:.1}%", analysis.confidence * 100.0);
        println!("   Signal: {}", analysis.signal);
        println!("   Probabilities:");
        println!("     DOWN:    {:.1}%", analysis.probabilities.down * 100.0);
        println!("     NEUTRAL: {:.1}%", analysis.probabilities.neutral * 100.0);
        println!("     UP:      {:.1}%", analysis.probabilities.up * 100.0);
        println!("   Recommended: {}", analysis.recommended_action);
    }

    // Run backtest if requested
    if args.backtest {
        println!("\n📈 Running Backtest...");
        let config = BacktestConfig {
            seq_len: 60,
            initial_capital: 10000.0,
            confidence_threshold: 0.3,
            transaction_cost: 0.001,
            ..Default::default()
        };

        let result = backtest_strategy(&model, &data.data, config);
        result.print_summary();

        println!("\n   Trade History:");
        for (i, trade) in result.trades.iter().take(5).enumerate() {
            println!(
                "   {}. {} at ${:.2} -> ${:.2} ({:+.2}%)",
                i + 1,
                trade.direction,
                trade.entry_price,
                trade.exit_price.unwrap_or(0.0),
                trade.pnl_percent.unwrap_or(0.0) * 100.0
            );
        }
        if result.trades.len() > 5 {
            println!("   ... and {} more trades", result.trades.len() - 5);
        }
    }

    // Compare strategies if requested
    if args.compare {
        println!("\n⚖️  Comparing Strategies...");
        let configs = vec![
            (
                "Conservative",
                BacktestConfig {
                    confidence_threshold: 0.5,
                    stop_loss: Some(0.01),
                    ..Default::default()
                },
            ),
            (
                "Moderate",
                BacktestConfig {
                    confidence_threshold: 0.3,
                    stop_loss: Some(0.02),
                    ..Default::default()
                },
            ),
            (
                "Aggressive",
                BacktestConfig {
                    confidence_threshold: 0.1,
                    stop_loss: Some(0.03),
                    ..Default::default()
                },
            ),
        ];

        compare_strategies(&model, &data.data, &configs);
    }

    println!("\n✅ Demo completed successfully!");

    Ok(())
}
