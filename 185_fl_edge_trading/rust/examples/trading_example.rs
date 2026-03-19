//! Edge Federated Learning Trading Example
//!
//! Simulates edge devices with different compute capabilities
//! training collaboratively on Bybit BTCUSDT market data.

use fl_edge_trading::*;

fn main() {
    println!("=== Edge Federated Learning for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch or generate market data
    // -----------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT kline data from Bybit...");
    let klines = match fetch_bybit_klines("BTCUSDT", "5", 200) {
        Ok(data) => {
            println!("    Fetched {} klines from Bybit API.", data.len());
            data
        }
        Err(e) => {
            println!("    Bybit API unavailable ({}), using synthetic data.", e);
            generate_synthetic_klines(200)
        }
    };

    println!(
        "    Price range: {:.2} - {:.2}",
        klines.iter().map(|k| k.low).fold(f64::MAX, f64::min),
        klines.iter().map(|k| k.high).fold(f64::MIN, f64::max),
    );

    // -----------------------------------------------------------------------
    // Step 2: Feature engineering
    // -----------------------------------------------------------------------
    println!("\n[2] Building features (window=10)...");
    let lookback_window = 10;
    let samples = build_features(&klines, lookback_window);
    println!("    Generated {} training samples.", samples.len());
    if let Some(s) = samples.first() {
        println!(
            "    Sample features: [ret={:.6}, vol={:.6}, mom={:.6}, vr={:.4}, range={:.6}]",
            s.features[0], s.features[1], s.features[2], s.features[3], s.features[4]
        );
    }

    // -----------------------------------------------------------------------
    // Step 3: Create heterogeneous edge devices
    // -----------------------------------------------------------------------
    println!("\n[3] Creating heterogeneous edge devices...");
    let num_colocation = 3;
    let num_mobile = 2;

    let mut coordinator = EdgeFLCoordinator::new(
        0.01,  // mu (proximal term)
        0.005, // learning rate
        0.5,   // top-k ratio (50% of gradients transmitted)
    );

    let mut devices =
        create_heterogeneous_devices(num_colocation, num_mobile, &coordinator.global_params);

    for d in &devices {
        println!(
            "    Device '{}': tier={:?}, max_epochs={}, param_fraction={:.0}%",
            d.id,
            d.tier,
            d.max_local_epochs,
            d.param_update_fraction * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // Step 4: Distribute data (non-IID)
    // -----------------------------------------------------------------------
    println!("\n[4] Distributing data across devices (non-IID)...");
    distribute_data_non_iid(&mut devices, &samples);
    for d in &devices {
        println!("    Device '{}': {} samples", d.id, d.data.len());
    }

    // -----------------------------------------------------------------------
    // Step 5: Run federated training
    // -----------------------------------------------------------------------
    let num_rounds = 15;
    println!("\n[5] Running FedProx training for {} rounds...", num_rounds);
    println!("    FedProx mu={}, lr={}, top-k ratio={}",
        coordinator.mu, coordinator.learning_rate, coordinator.top_k_ratio);
    println!();

    coordinator.train(&mut devices, num_rounds);

    // -----------------------------------------------------------------------
    // Step 6: Evaluate results
    // -----------------------------------------------------------------------
    println!("\n[6] Final Evaluation:");
    println!("    Global model parameters: {:?}", coordinator.global_params);
    println!();

    for d in &devices {
        let loss = d.evaluate();
        let acc = d.accuracy();
        println!(
            "    Device '{}' ({:?}): loss={:.6}, accuracy={:.2}%",
            d.id,
            d.tier,
            loss,
            acc * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // Step 7: Training loss curve
    // -----------------------------------------------------------------------
    println!("\n[7] Loss Curve:");
    for (i, loss) in coordinator.loss_history.iter().enumerate() {
        let bar_len = ((1.0 - loss.min(1.0)) * 40.0) as usize;
        let bar: String = std::iter::repeat('#').take(bar_len).collect();
        println!("    Round {:>2}: {:.6} |{}", i + 1, loss, bar);
    }

    // -----------------------------------------------------------------------
    // Step 8: Compression statistics
    // -----------------------------------------------------------------------
    println!("\n[8] Communication Efficiency:");
    let param_count = NUM_FEATURES + 1;
    let full_update_bytes = param_count * 4; // float32
    let compressed_components = (param_count as f64 * coordinator.top_k_ratio).ceil() as usize;
    let compressed_bytes = compressed_components * (4 + 1); // index(u32) + quantized(u8)
    let compression_ratio = full_update_bytes as f64 / compressed_bytes as f64;
    println!(
        "    Model parameters: {}",
        param_count
    );
    println!(
        "    Full update size: {} bytes",
        full_update_bytes
    );
    println!(
        "    Compressed update: {} bytes ({} components)",
        compressed_bytes, compressed_components
    );
    println!(
        "    Compression ratio: {:.1}x",
        compression_ratio
    );

    println!("\n=== Training Complete ===");
}
