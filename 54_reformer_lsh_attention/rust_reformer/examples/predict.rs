//! Example: Make predictions with a Reformer model
//!
//! This example demonstrates how to:
//! 1. Load recent market data
//! 2. Prepare features for prediction
//! 3. Run inference with a Reformer model
//! 4. Interpret the results

use clap::Parser;
use reformer::{BybitClient, DataLoader, ReformerConfig, ReformerModel, AttentionType};

/// Make predictions with Reformer
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Sequence length for model input
    #[arg(long, default_value = "168")]
    seq_len: usize,

    /// Show attention bucket assignments
    #[arg(long)]
    show_buckets: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Reformer Prediction Example ===");
    println!("Symbol: {}", args.symbol);
    println!();

    // Fetch recent data
    println!("Fetching recent market data...");
    let client = BybitClient::new();

    // Need extra data for feature computation
    let klines = client.get_klines(&args.symbol, "60", args.seq_len + 100).await?;
    println!("Fetched {} klines", klines.len());

    // Get current price
    let current_price = klines.last().map(|k| k.close).unwrap_or(0.0);
    let current_time = klines.last().map(|k| k.timestamp).unwrap_or(0);

    let current_date = chrono::DateTime::from_timestamp_millis(current_time as i64)
        .map(|d| d.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_default();

    println!("\nCurrent Price: ${:.2}", current_price);
    println!("Timestamp: {}", current_date);

    // Prepare features
    println!("\nPreparing features...");
    let loader = DataLoader::new();
    let features = loader.prepare_inference(&klines, args.seq_len)?;

    println!("Feature shape: {:?}", features.dim());

    // Create model
    let config = ReformerConfig {
        seq_len: args.seq_len,
        n_features: features.ncols(),
        d_model: 64,
        n_heads: 4,
        d_ff: 256,
        n_layers: 4,
        n_hash_rounds: 4,
        n_buckets: 16,
        prediction_horizon: 24,
        attention_type: AttentionType::LSH,
        ..Default::default()
    };

    println!("\nCreating Reformer model...");
    println!("  Model dimension: {}", config.d_model);
    println!("  Attention heads: {}", config.n_heads);
    println!("  LSH hash rounds: {}", config.n_hash_rounds);
    println!("  LSH buckets: {}", config.n_buckets);

    let model = ReformerModel::new(config);

    // Make prediction
    println!("\nMaking prediction...");
    let (prediction, buckets) = model.predict_with_attention(&features);

    // Display results
    println!("\n=== Prediction Results ===");
    println!("Predicted returns for next 24 hours:");
    println!();

    let mut cum_return = 0.0;
    for (i, &pred) in prediction.iter().enumerate() {
        cum_return += pred;
        let predicted_price = current_price * (1.0 + cum_return);

        let signal = if pred > 0.01 {
            "BULLISH"
        } else if pred < -0.01 {
            "BEARISH"
        } else {
            "NEUTRAL"
        };

        if i < 6 || i >= prediction.len() - 2 {
            println!(
                "Hour {:2}: {:+.4}% (Cumulative: {:+.4}%) -> ${:.2} [{}]",
                i + 1,
                pred * 100.0,
                cum_return * 100.0,
                predicted_price,
                signal
            );
        } else if i == 6 {
            println!("  ...");
        }
    }

    // Summary
    println!("\n=== Summary ===");
    let final_return = cum_return;
    let final_price = current_price * (1.0 + final_return);

    println!("24h Predicted Return: {:+.2}%", final_return * 100.0);
    println!("24h Predicted Price: ${:.2}", final_price);
    println!("Price Change: ${:+.2}", final_price - current_price);

    // Trading signal
    let overall_signal = if final_return > 0.02 {
        "STRONG BUY"
    } else if final_return > 0.005 {
        "BUY"
    } else if final_return < -0.02 {
        "STRONG SELL"
    } else if final_return < -0.005 {
        "SELL"
    } else {
        "HOLD"
    };

    println!("\nOverall Signal: {}", overall_signal);

    // Show bucket assignments if requested
    if args.show_buckets && !buckets.is_empty() {
        println!("\n=== LSH Bucket Assignments ===");
        println!("(Shows which positions were hashed to which buckets)");
        println!();

        // Count bucket distribution
        let mut bucket_counts = std::collections::HashMap::new();
        for &b in &buckets {
            *bucket_counts.entry(b).or_insert(0) += 1;
        }

        let mut sorted_buckets: Vec<_> = bucket_counts.iter().collect();
        sorted_buckets.sort_by_key(|(_, &count)| std::cmp::Reverse(count));

        println!("Bucket distribution (top 5):");
        for (bucket, count) in sorted_buckets.iter().take(5) {
            let pct = **count as f64 / buckets.len() as f64 * 100.0;
            println!("  Bucket {}: {} positions ({:.1}%)", bucket, count, pct);
        }
    }

    println!("\n=== Disclaimer ===");
    println!("This is a demonstration with randomly initialized weights.");
    println!("Do not use for actual trading decisions.");
    println!("Real predictions require proper model training.");

    Ok(())
}
