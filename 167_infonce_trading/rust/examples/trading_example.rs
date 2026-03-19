//! # InfoNCE Trading Example
//!
//! Fetches BTCUSDT candles from Bybit, builds contrastive pairs,
//! trains a simple encoder with InfoNCE loss, and displays the
//! learned embeddings.
//!
//! Run:
//! ```bash
//! cargo run --example trading_example
//! ```

use infonce_trading::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== InfoNCE Trading Example ===\n");

    // ------------------------------------------------------------------
    // 1. Fetch market data from Bybit
    // ------------------------------------------------------------------
    println!("[1/4] Fetching BTCUSDT 5-minute candles from Bybit...");
    let candles = fetch_bybit_klines("BTCUSDT", "5", 200).await?;
    println!("      Received {} candles", candles.len());

    if candles.len() < 30 {
        anyhow::bail!("Not enough candles received (need at least 30, got {})", candles.len());
    }

    // Print a summary of the data
    let first = &candles[0];
    let last = &candles[candles.len() - 1];
    println!(
        "      Time range: {} -> {} (timestamps)",
        first.timestamp, last.timestamp
    );
    println!(
        "      Price range: {:.2} -> {:.2}",
        first.open, last.close
    );
    println!();

    // ------------------------------------------------------------------
    // 2. Create contrastive pairs
    // ------------------------------------------------------------------
    println!("[2/4] Generating contrastive pairs...");
    let config = InfoNCEConfig {
        temperature: 0.1,
        embedding_dim: 32,
        window_size: 5,
        positive_range: 3,
        num_negatives: 8,
        learning_rate: 0.005,
        epochs: 30,
    };

    let samples = generate_contrastive_pairs(&candles, &config);
    println!("      Created {} contrastive samples", samples.len());
    println!(
        "      Window size: {} candles ({} features each)",
        config.window_size,
        config.window_size * FEATURES_PER_CANDLE
    );
    println!("      Negatives per sample: {}", config.num_negatives);
    println!();

    if samples.is_empty() {
        anyhow::bail!("No contrastive samples could be generated. Try adjusting the config.");
    }

    // ------------------------------------------------------------------
    // 3. Train the encoder
    // ------------------------------------------------------------------
    println!("[3/4] Training encoder with InfoNCE loss...");
    println!("      Temperature: {}", config.temperature);
    println!("      Learning rate: {}", config.learning_rate);
    println!("      Epochs: {}", config.epochs);
    println!();

    let input_dim = config.window_size * FEATURES_PER_CANDLE;
    let mut encoder = Encoder::new(input_dim, 128, 64, config.embedding_dim);

    let losses = encoder.train(&samples, &config);

    println!();
    let first_loss = losses.first().unwrap_or(&0.0);
    let last_loss = losses.last().unwrap_or(&0.0);
    println!(
        "      Training complete: loss {:.4} -> {:.4} ({:.1}% reduction)",
        first_loss,
        last_loss,
        (1.0 - last_loss / first_loss) * 100.0
    );
    println!();

    // ------------------------------------------------------------------
    // 4. Show learned representations
    // ------------------------------------------------------------------
    println!("[4/4] Learned representations for selected windows:\n");

    // Encode several windows and show their pairwise similarity
    let window_indices: Vec<usize> = (0..5)
        .map(|i| i * (candles.len() - config.window_size) / 5)
        .collect();

    let embeddings: Vec<(usize, ndarray::Array1<f64>)> = window_indices
        .iter()
        .map(|&idx| {
            let feat = window_features(&candles[idx..idx + config.window_size]);
            let emb = encoder.forward(&feat);
            (idx, emb)
        })
        .collect();

    // Print embeddings (first 8 dims)
    for (idx, emb) in &embeddings {
        let candle = &candles[*idx];
        let ret = (candles[idx + config.window_size - 1].close - candle.open) / candle.open * 100.0;
        print!("  Window @{:>3} (ret={:>+6.2}%): [", idx, ret);
        for (i, val) in emb.iter().enumerate().take(8) {
            if i > 0 {
                print!(", ");
            }
            print!("{:>+.3}", val);
        }
        println!(", ...]");
    }
    println!();

    // Pairwise similarity matrix
    println!("  Pairwise cosine similarity:");
    print!("          ");
    for (idx, _) in &embeddings {
        print!("  W@{:<3}", idx);
    }
    println!();
    for (i, (idx_i, emb_i)) in embeddings.iter().enumerate() {
        print!("  W@{:<3}  ", idx_i);
        for (j, (_, emb_j)) in embeddings.iter().enumerate() {
            let sim = cosine_similarity(emb_i, emb_j);
            if i == j {
                print!("  1.000");
            } else {
                print!("  {:>+.3}", sim);
            }
        }
        println!();
    }
    println!();

    // Identify most and least similar pairs
    let mut best = (0usize, 0usize, f64::NEG_INFINITY);
    let mut worst = (0usize, 0usize, f64::INFINITY);
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            if sim > best.2 {
                best = (embeddings[i].0, embeddings[j].0, sim);
            }
            if sim < worst.2 {
                worst = (embeddings[i].0, embeddings[j].0, sim);
            }
        }
    }
    println!(
        "  Most similar:  W@{} and W@{} (sim = {:.4})",
        best.0, best.1, best.2
    );
    println!(
        "  Least similar: W@{} and W@{} (sim = {:.4})",
        worst.0, worst.1, worst.2
    );

    println!("\n=== Done ===");
    Ok(())
}
