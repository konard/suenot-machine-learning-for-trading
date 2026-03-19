//! Trading example: Train a Quantum GAN on Bybit BTCUSDT data
//! and generate synthetic price paths.

use quantum_gan_finance::*;

fn main() -> anyhow::Result<()> {
    println!("=== Quantum GAN for Finance - Trading Example ===\n");

    // ── Step 1: Fetch real data from Bybit ──
    println!("Fetching BTCUSDT hourly klines from Bybit...");
    let prices = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(p) => {
            println!("Fetched {} price points", p.len());
            p
        }
        Err(e) => {
            println!("Could not fetch from Bybit: {}. Using synthetic data.", e);
            generate_synthetic_prices(200)
        }
    };

    // ── Step 2: Compute log returns ──
    let returns = compute_log_returns(&prices);
    println!("Computed {} log returns", returns.len());

    println!("\nReal data statistics:");
    println!("  Mean:     {:.6}", mean(&returns));
    println!("  Std Dev:  {:.6}", std_dev(&returns));
    println!("  Skewness: {:.6}", skewness(&returns));
    println!("  Kurtosis: {:.6}", kurtosis(&returns));

    // ── Step 3: Normalize returns for GAN training ──
    let (normalized_returns, min_val, max_val) = normalize(&returns);
    println!(
        "\nNormalized returns to [0,1] (original range: [{:.6}, {:.6}])",
        min_val, max_val
    );

    // ── Step 4: Train Quantum GAN ──
    println!("\nTraining Quantum GAN...");
    println!("  Qubits: 3, Layers: 2");
    println!("  Generator LR: 0.1, Discriminator LR: 0.01");
    println!("  Epochs: 50, Batch size: 32\n");

    let mut trainer = QGANTrainer::new(3, 2, 0.1, 0.01);
    let losses = trainer.train(&normalized_returns, 50, 32);

    println!("\nTraining complete!");
    println!(
        "Final generator loss: {:.4}",
        losses.last().unwrap_or(&0.0)
    );

    // ── Step 5: Generate synthetic returns ──
    let n_synthetic = returns.len();
    let synthetic_normalized = trainer.generate(n_synthetic);
    let synthetic_returns = denormalize(&synthetic_normalized, min_val, max_val);

    println!("\nSynthetic data statistics:");
    println!("  Mean:     {:.6}", mean(&synthetic_returns));
    println!("  Std Dev:  {:.6}", std_dev(&synthetic_returns));
    println!("  Skewness: {:.6}", skewness(&synthetic_returns));
    println!("  Kurtosis: {:.6}", kurtosis(&synthetic_returns));

    // ── Step 6: Compare real vs synthetic ──
    println!("\n=== Statistical Comparison ===");
    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Metric", "Real", "Synthetic", "Diff"
    );
    println!("{}", "-".repeat(50));

    let metrics = [
        ("Mean", mean(&returns), mean(&synthetic_returns)),
        ("Std Dev", std_dev(&returns), std_dev(&synthetic_returns)),
        ("Skewness", skewness(&returns), skewness(&synthetic_returns)),
        ("Kurtosis", kurtosis(&returns), kurtosis(&synthetic_returns)),
    ];

    for (name, real_val, synth_val) in &metrics {
        println!(
            "{:<12} {:>12.6} {:>12.6} {:>12.6}",
            name,
            real_val,
            synth_val,
            (real_val - synth_val).abs()
        );
    }

    // ── Step 7: Generate synthetic price path ──
    println!("\n=== Synthetic Price Path (first 20 steps) ===");
    let start_price = prices.last().copied().unwrap_or(50000.0);
    let synthetic_path = generate_price_path(start_price, &synthetic_returns);

    for (i, price) in synthetic_path.iter().take(20).enumerate() {
        println!("  Step {:>3}: ${:.2}", i, price);
    }

    println!("\nDone!");
    Ok(())
}

/// Generate a price path from returns starting at initial_price.
fn generate_price_path(initial_price: f64, returns: &[f64]) -> Vec<f64> {
    let mut path = Vec::with_capacity(returns.len() + 1);
    path.push(initial_price);
    for r in returns {
        let prev = *path.last().unwrap();
        path.push(prev * r.exp());
    }
    path
}

/// Generate synthetic prices for fallback when API is unavailable.
fn generate_synthetic_prices(n: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0_f64;
    prices.push(price);
    for _ in 1..n {
        let ret = rng.gen::<f64>() * 0.04 - 0.02; // uniform in [-0.02, 0.02]
        price *= (1.0 + ret);
        prices.push(price);
    }
    prices
}
