//! Trading Example: Generative LOB Trading
//!
//! This example demonstrates:
//! 1. Fetching BTCUSDT orderbook from Bybit
//! 2. Encoding LOB snapshots with VAE
//! 3. Generating synthetic LOB states
//! 4. Comparing real vs synthetic statistics

use generative_lob::*;

fn main() {
    println!("=== Chapter 279: Generative LOB Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Try to fetch real LOB data from Bybit
    // -----------------------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT orderbook from Bybit...");

    let client = BybitClient::new();
    let real_snapshots = match client.fetch_orderbook_blocking("BTCUSDT", "spot", 50) {
        Ok(snapshot) => {
            println!("  Fetched orderbook with {} bids and {} asks",
                snapshot.bids.len(), snapshot.asks.len());
            println!("  Mid-price: {:.2}", snapshot.mid_price());
            println!("  Spread: {:.4}", snapshot.spread());

            // Create multiple snapshots by fetching or using the single one
            let mut snapshots = vec![snapshot];
            // Add more snapshots with slight variations for training
            for _ in 0..49 {
                if let Ok(s) = client.fetch_orderbook_blocking("BTCUSDT", "spot", 50) {
                    snapshots.push(s);
                }
            }
            println!("  Collected {} real snapshots\n", snapshots.len());
            snapshots
        }
        Err(e) => {
            println!("  Could not fetch from Bybit: {}", e);
            println!("  Using synthetic test data instead.\n");
            generate_test_snapshots(50, 25, 50000.0)
        }
    };

    let num_levels = 10;
    let _mid_price = real_snapshots[0].mid_price();

    // -----------------------------------------------------------------------
    // Step 2: Encode LOB snapshots with VAE
    // -----------------------------------------------------------------------
    println!("Step 2: Encoding LOB snapshots with VAE...");

    let input_dim = num_levels * 4;
    let hidden_dim = 32;
    let latent_dim = 8;

    let vae = LobVae::new(input_dim, hidden_dim, latent_dim);

    // Encode first snapshot
    let features = real_snapshots[0].to_feature_vector(num_levels);
    println!("  Feature vector dimension: {}", features.len());

    let (mu, log_var) = vae.encode(&features);
    println!("  Latent mean (first 4): [{:.4}, {:.4}, {:.4}, {:.4}]",
        mu[0], mu[1], mu[2], mu[3]);
    println!("  Latent log-var (first 4): [{:.4}, {:.4}, {:.4}, {:.4}]",
        log_var[0], log_var[1], log_var[2], log_var[3]);

    // Full forward pass
    let (x_recon, mu, log_var) = vae.forward(&features);
    let loss = vae.loss(&features, &x_recon, &mu, &log_var);
    println!("  Initial VAE loss: {:.4}\n", loss);

    // -----------------------------------------------------------------------
    // Step 3: Train the generator and produce synthetic LOB states
    // -----------------------------------------------------------------------
    println!("Step 3: Training generator and generating synthetic LOB states...");

    let mut generator = LobGenerator::new(num_levels, hidden_dim, latent_dim);
    let losses = generator.train(&real_snapshots, 10, 0.001);

    println!("  Training losses over epochs:");
    for (i, loss) in losses.iter().enumerate() {
        println!("    Epoch {}: {:.4}", i + 1, loss);
    }

    // Generate synthetic snapshots
    let synthetic_snapshots = generator.generate_snapshots(50);
    println!("\n  Generated {} synthetic LOB snapshots", synthetic_snapshots.len());

    // Print a sample synthetic snapshot
    let sample = &synthetic_snapshots[0];
    println!("\n  Sample synthetic LOB snapshot:");
    println!("    Mid-price: {:.2}", sample.mid_price());
    println!("    Spread: {:.4}", sample.spread());
    println!("    Best bid: {:.2} @ {:.4}", sample.bids[0].price, sample.bids[0].quantity);
    println!("    Best ask: {:.2} @ {:.4}\n", sample.asks[0].price, sample.asks[0].quantity);

    // -----------------------------------------------------------------------
    // Step 4: Compare real vs synthetic statistics
    // -----------------------------------------------------------------------
    println!("Step 4: Comparing real vs synthetic statistics...");

    let report = LobValidator::validate(&real_snapshots, &synthetic_snapshots);
    println!("{}", report);

    // Additional analysis
    let real_spreads = LobValidator::spreads(&real_snapshots);
    let synth_spreads = LobValidator::spreads(&synthetic_snapshots);

    let (real_mean, real_std, _, _) = LobValidator::basic_stats(&real_spreads);
    let (synth_mean, synth_std, _, _) = LobValidator::basic_stats(&synth_spreads);

    println!("Spread comparison:");
    println!("  Real:      mean={:.4}, std={:.4}", real_mean, real_std);
    println!("  Synthetic: mean={:.4}, std={:.4}", synth_mean, synth_std);

    let real_bid_vols = LobValidator::bid_volumes(&real_snapshots, 5);
    let synth_bid_vols = LobValidator::bid_volumes(&synthetic_snapshots, 5);
    let (rv_mean, rv_std, _, _) = LobValidator::basic_stats(&real_bid_vols);
    let (sv_mean, sv_std, _, _) = LobValidator::basic_stats(&synth_bid_vols);

    println!("\nTop-5 bid volume comparison:");
    println!("  Real:      mean={:.4}, std={:.4}", rv_mean, rv_std);
    println!("  Synthetic: mean={:.4}, std={:.4}", sv_mean, sv_std);

    println!("\n=== Generative LOB Trading Example Complete ===");
}
