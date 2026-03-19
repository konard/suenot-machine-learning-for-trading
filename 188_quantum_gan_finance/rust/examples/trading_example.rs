use quantum_gan_finance::{
    compare_distributions, compute_log_returns, fetch_bybit_klines, mean, skewness, std_dev,
    variance, QuantumGAN,
};

fn main() {
    println!("=== Quantum GAN Finance: Trading Example ===\n");

    // --- Step 1: Fetch real BTCUSDT data from Bybit ---
    println!("Step 1: Fetching BTCUSDT hourly data from Bybit...");
    let closes = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} close prices.", c.len());
            c
        }
        Err(e) => {
            eprintln!("  Failed to fetch data from Bybit: {}", e);
            eprintln!("  Using synthetic fallback data for demonstration...");
            generate_fallback_prices(200)
        }
    };

    // --- Step 2: Compute log returns ---
    println!("\nStep 2: Computing log returns...");
    let returns = compute_log_returns(&closes);
    println!("  Computed {} log returns.", returns.len());

    if returns.is_empty() {
        eprintln!("No valid returns. Exiting.");
        return;
    }

    // Print real data statistics
    println!("\n  Real data statistics:");
    println!("    Mean:     {:.6}", mean(&returns));
    println!("    Std Dev:  {:.6}", std_dev(&returns));
    println!("    Variance: {:.6}", variance(&returns));
    println!("    Skewness: {:.6}", skewness(&returns));

    // --- Step 3: Set up Quantum GAN ---
    let r_min = returns.iter().cloned().fold(f64::INFINITY, f64::min) * 1.5;
    let r_max = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.5;

    let num_qubits = 4;
    let num_layers = 3;
    let epochs = 200;

    println!(
        "\nStep 3: Training Quantum GAN ({} qubits, {} layers)...",
        num_qubits, num_layers
    );
    println!("  Return range: [{:.6}, {:.6}]", r_min, r_max);
    println!("  Number of bins: {}", 1 << num_qubits);
    println!("  Training epochs: {}\n", epochs);

    let mut qgan = QuantumGAN::new(num_qubits, num_layers, r_min, r_max);
    let losses = qgan.train(&returns, epochs, 32);

    // --- Step 4: Generate synthetic returns ---
    let num_synthetic = returns.len();
    println!(
        "\nStep 4: Generating {} synthetic returns...",
        num_synthetic
    );
    let synthetic = qgan.generate_returns(num_synthetic);

    // --- Step 5: Compare distributions ---
    println!("\nStep 5: Comparing real vs. generated distributions...");
    compare_distributions(&returns, &synthetic);

    // --- Step 6: Training convergence summary ---
    println!("\n=== Training Convergence ===");
    if let Some((first_d, first_g)) = losses.first() {
        println!("  Initial  - D_loss: {:.4}, G_loss: {:.4}", first_d, first_g);
    }
    if let Some((last_d, last_g)) = losses.last() {
        println!("  Final    - D_loss: {:.4}, G_loss: {:.4}", last_d, last_g);
    }

    // --- Step 7: Example generated samples ---
    println!("\n=== Sample Generated Returns ===");
    let display_count = 10.min(synthetic.len());
    for (i, &r) in synthetic.iter().take(display_count).enumerate() {
        println!("  Sample {:2}: {:.6} ({:.2}%)", i + 1, r, r * 100.0);
    }

    println!("\n=== Quantum GAN Finance Example Complete ===");
}

/// Generate fallback price data when Bybit API is unavailable.
/// Produces a random walk with realistic BTC-like volatility.
fn generate_fallback_prices(n: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0_f64;
    prices.push(price);

    for _ in 1..n {
        // BTC hourly returns: roughly mean=0, std=0.005
        let ret: f64 = rng.gen::<f64>() * 0.01 - 0.005;
        // Add occasional larger moves for realistic tails
        let tail_event: f64 = if rng.gen::<f64>() < 0.05 {
            (rng.gen::<f64>() - 0.5) * 0.04
        } else {
            0.0
        };
        price *= (ret + tail_event).exp();
        prices.push(price);
    }

    prices
}
