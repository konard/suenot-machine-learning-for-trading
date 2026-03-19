use vq_vae_trading::*;

fn main() -> anyhow::Result<()> {
    println!("=== VQ-VAE Trading Example ===\n");

    // --- Fetch data from Bybit ---
    println!("Fetching BTCUSDT daily candles from Bybit...");
    let candles = match fetch_bybit_ohlcv("BTCUSDT", "D", 200) {
        Ok(c) => {
            println!("Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("Failed to fetch from Bybit: {}. Using synthetic data.", e);
            generate_synthetic_candles(200)
        }
    };

    if candles.len() < 10 {
        println!("Not enough candles, using synthetic data.");
        let candles = generate_synthetic_candles(200);
        return run_vqvae_pipeline(&candles);
    }

    run_vqvae_pipeline(&candles)
}

fn run_vqvae_pipeline(candles: &[OhlcvCandle]) -> anyhow::Result<()> {
    // --- Normalize OHLCV data ---
    println!("\nNormalizing OHLCV data...");
    let normalized = normalize_ohlcv(candles);
    println!("Normalized {} data points", normalized.len());

    // --- Create sliding windows ---
    let window_size = 5;
    let mut windows = create_windows(&normalized, window_size);
    println!(
        "Created {} windows of size {} (feature dim: {})",
        windows.len(),
        window_size,
        windows.first().map_or(0, |w| w.len())
    );

    if windows.is_empty() {
        println!("Not enough data to create windows.");
        return Ok(());
    }

    // --- Z-score normalize ---
    zscore_normalize(&mut windows);
    println!("Applied z-score normalization");

    // --- Configure VQ-VAE ---
    let input_dim = windows[0].len(); // window_size * 5 features
    let hidden_dim = 32;
    let embedding_dim = 16;
    let num_embeddings = 32; // 32 codebook entries (market patterns)
    let beta = 0.25;
    let learning_rate = 0.001;

    let mut vqvae = VQVAE::new(
        input_dim,
        hidden_dim,
        embedding_dim,
        num_embeddings,
        beta,
        learning_rate,
    );

    // --- Train VQ-VAE ---
    println!("\nTraining VQ-VAE...");
    println!(
        "  Input dim: {}, Hidden: {}, Embedding: {}, Codebook size: {}",
        input_dim, hidden_dim, embedding_dim, num_embeddings
    );

    let num_epochs = 20;
    for epoch in 0..num_epochs {
        let loss = vqvae.train_epoch_stochastic(&windows, windows.len());
        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            println!("  Epoch {}/{}: loss = {:.4}", epoch + 1, num_epochs, loss);
        }
    }

    // --- Encode data to tokens ---
    println!("\nEncoding market data to codebook tokens...");
    let tokens = vqvae.encode_to_tokens(&windows);

    // Print first 50 tokens
    let display_len = tokens.len().min(50);
    print!("  Token sequence (first {}): [", display_len);
    for (i, &t) in tokens.iter().take(display_len).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", t);
    }
    println!("]");

    // --- Codebook usage analysis ---
    println!("\nCodebook Usage Analysis:");
    let usage = codebook_usage(&tokens, num_embeddings);
    let active_codes = usage.iter().filter(|&&c| c > 0).count();
    println!(
        "  Active codes: {} / {} ({:.1}%)",
        active_codes,
        num_embeddings,
        100.0 * active_codes as f64 / num_embeddings as f64
    );

    let perplexity = codebook_perplexity(&tokens, num_embeddings);
    println!("  Perplexity: {:.2} (ideal: {:.2})", perplexity, num_embeddings as f64);

    // Print top-5 most used codes
    let mut usage_sorted: Vec<(usize, usize)> = usage.iter().copied().enumerate().collect();
    usage_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("  Top-5 most used codes:");
    for (code, count) in usage_sorted.iter().take(5) {
        println!(
            "    Code {}: {} times ({:.1}%)",
            code,
            count,
            100.0 * *count as f64 / tokens.len() as f64
        );
    }

    // --- Anomaly detection ---
    println!("\nAnomaly Detection (95th percentile):");
    let anomalies = detect_anomalies(&vqvae, &windows, 95.0);
    println!("  Found {} anomalous windows", anomalies.len());
    for (idx, score) in anomalies.iter().take(5) {
        let candle_idx = idx + window_size; // offset due to windowing + normalization
        let timestamp = if candle_idx < candles.len() {
            candles[candle_idx].timestamp
        } else {
            0
        };
        println!(
            "    Window {}: anomaly score = {:.4} (timestamp: {})",
            idx, score, timestamp
        );
    }

    // --- Map specific days to codebook entries ---
    println!("\nDay-to-Codebook Mapping (last 10 windows):");
    let start = tokens.len().saturating_sub(10);
    for i in start..tokens.len() {
        let candle_idx = i + window_size;
        let timestamp = if candle_idx < candles.len() {
            candles[candle_idx].timestamp
        } else {
            0
        };
        let score = vqvae.anomaly_score(&windows[i]);
        println!(
            "  Window {}: Code {} (distance: {:.4}, timestamp: {})",
            i, tokens[i], score, timestamp
        );
    }

    // --- Bigram transition statistics ---
    println!("\nBigram Transition Statistics:");
    if tokens.len() > 1 {
        let mut transitions: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for pair in tokens.windows(2) {
            *transitions.entry((pair[0], pair[1])).or_insert(0) += 1;
        }
        let mut transitions_vec: Vec<_> = transitions.into_iter().collect();
        transitions_vec.sort_by(|a, b| b.1.cmp(&a.1));

        println!("  Top-5 most common transitions:");
        for ((from, to), count) in transitions_vec.iter().take(5) {
            println!(
                "    Code {} -> Code {}: {} times ({:.1}%)",
                from,
                to,
                count,
                100.0 * *count as f64 / (tokens.len() - 1) as f64
            );
        }
    }

    println!("\n=== VQ-VAE Trading Example Complete ===");
    Ok(())
}

/// Generate synthetic OHLCV candles for testing when API is unavailable.
fn generate_synthetic_candles(n: usize) -> Vec<OhlcvCandle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64; // Starting BTC-like price

    for i in 0..n {
        let daily_return = rng.gen_range(-0.05..0.05);
        let volatility = rng.gen_range(0.005..0.03);

        let open = price * (1.0 + rng.gen_range(-0.005..0.005));
        let close = price * (1.0 + daily_return);
        let high = open.max(close) * (1.0 + volatility);
        let low = open.min(close) * (1.0 - volatility);
        let volume = rng.gen_range(1000.0..10000.0) * (1.0 + daily_return.abs() * 10.0);

        candles.push(OhlcvCandle {
            timestamp: 1700000000000 + (i as u64) * 86400000,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}
