//! # Quantum Boltzmann Trading Example
//!
//! Fetches BTCUSDT and ETHUSDT data from Bybit, discretizes returns into
//! binary features, trains a Quantum Boltzmann Machine on the joint return
//! distribution, generates synthetic scenarios, and compares statistics.

use anyhow::Result;
use ndarray::Array1;
use quantum_boltzmann_trading::*;

fn main() -> Result<()> {
    println!("=== Quantum Boltzmann Trading Example ===\n");

    // ── Step 1: Fetch data from Bybit ──────────────────────────────────
    println!("[1] Fetching market data from Bybit...");
    let client = BybitClient::new();

    let btc_klines = match client.fetch_klines("BTCUSDT", "60", 200) {
        Ok(k) => {
            println!("    BTCUSDT: {} candles fetched", k.len());
            k
        }
        Err(e) => {
            println!("    Warning: Could not fetch BTCUSDT: {}", e);
            println!("    Using synthetic data instead.\n");
            return run_with_synthetic_data();
        }
    };

    let eth_klines = match client.fetch_klines("ETHUSDT", "60", 200) {
        Ok(k) => {
            println!("    ETHUSDT: {} candles fetched", k.len());
            k
        }
        Err(e) => {
            println!("    Warning: Could not fetch ETHUSDT: {}", e);
            println!("    Using synthetic data instead.\n");
            return run_with_synthetic_data();
        }
    };

    // ── Step 2: Compute log returns ────────────────────────────────────
    println!("\n[2] Computing log returns...");
    let btc_returns = BybitClient::compute_log_returns(&btc_klines);
    let eth_returns = BybitClient::compute_log_returns(&eth_klines);

    // Align lengths
    let min_len = btc_returns.len().min(eth_returns.len());
    let btc_returns = &btc_returns[..min_len];
    let eth_returns = &eth_returns[..min_len];
    println!("    {} aligned return observations", min_len);

    let btc_stats = DistributionStats::from_data(btc_returns);
    let eth_stats = DistributionStats::from_data(eth_returns);
    println!("    BTC returns - mean: {:.6}, std: {:.6}", btc_stats.mean, btc_stats.std_dev);
    println!("    ETH returns - mean: {:.6}, std: {:.6}", eth_stats.mean, eth_stats.std_dev);

    // ── Step 3: Discretize and encode ──────────────────────────────────
    let n_bins = 4;
    println!("\n[3] Discretizing returns into {} bins with thermometer encoding...", n_bins);

    let btc_edges = compute_bin_edges(btc_returns, n_bins);
    let eth_edges = compute_bin_edges(eth_returns, n_bins);
    let bin_edges = vec![btc_edges, eth_edges];

    let encoded_data: Vec<Array1<f64>> = (0..min_len)
        .map(|t| encode_returns(&[btc_returns[t], eth_returns[t]], &bin_edges, n_bins))
        .collect();

    let n_visible = encoded_data[0].len();
    println!("    Binary feature vector length: {}", n_visible);
    println!("    Training samples: {}", encoded_data.len());

    // ── Step 4: Train Classical RBM (baseline) ─────────────────────────
    let n_hidden = 6;
    println!("\n[4] Training Classical RBM (baseline)...");
    println!("    Architecture: {} visible, {} hidden", n_visible, n_hidden);

    let mut classical_rbm = ClassicalRBM::new(n_visible, n_hidden, 0.05);
    classical_rbm.train_cd_k(&encoded_data, 1, 20);
    println!("    Classical RBM training complete (20 epochs, CD-1)");

    // ── Step 5: Train Quantum Boltzmann Machine ────────────────────────
    println!("\n[5] Training Quantum Boltzmann Machine...");
    let gamma = 1.5;
    let n_trotter = 4;
    println!("    Architecture: {} visible, {} hidden, gamma={}, trotter_slices={}",
             n_visible, n_hidden, gamma, n_trotter);

    let mut qbm = QuantumBoltzmannMachine::new(n_visible, n_hidden, gamma, n_trotter, 0.05);
    qbm.train(&encoded_data, 2, 20, 0.95);
    println!("    QBM training complete (20 epochs, QCD-2, gamma_decay=0.95)");
    println!("    Final gamma: {:.4}", qbm.gamma);

    // ── Step 6: Generate synthetic scenarios ───────────────────────────
    let n_gen = 500;
    println!("\n[6] Generating {} synthetic scenarios...", n_gen);

    let classical_samples = classical_rbm.generate_samples(n_gen, 50);
    let quantum_samples = qbm.generate_samples(n_gen, 10);

    // ── Step 7: Compare distributions ──────────────────────────────────
    println!("\n[7] Comparing real vs generated distributions:\n");

    let real_dist = compute_bin_distribution(&encoded_data, 2, n_bins);
    let classical_dist = compute_bin_distribution(&classical_samples, 2, n_bins);
    let quantum_dist = compute_bin_distribution(&quantum_samples, 2, n_bins);

    let asset_names = ["BTC", "ETH"];
    for (asset_idx, name) in asset_names.iter().enumerate() {
        println!("  {} bin distribution:", name);
        println!("    {:>8} {:>10} {:>10} {:>10}", "Bin", "Real", "Classical", "Quantum");
        for bin in 0..n_bins {
            println!(
                "    {:>8} {:>10.3} {:>10.3} {:>10.3}",
                bin,
                real_dist[asset_idx][bin],
                classical_dist[asset_idx][bin],
                quantum_dist[asset_idx][bin]
            );
        }
        println!();
    }

    // Compute KL-divergence-like metric (Jensen-Shannon for stability)
    println!("  Jensen-Shannon divergence from real distribution:");
    for (asset_idx, name) in asset_names.iter().enumerate() {
        let js_classical = jensen_shannon(&real_dist[asset_idx], &classical_dist[asset_idx]);
        let js_quantum = jensen_shannon(&real_dist[asset_idx], &quantum_dist[asset_idx]);
        println!("    {} - Classical RBM: {:.4}, QBM: {:.4}", name, js_classical, js_quantum);
    }

    // ── Step 8: Anomaly detection demo ─────────────────────────────────
    println!("\n[8] Anomaly detection using free energy:");
    let energies: Vec<f64> = encoded_data.iter().map(|v| qbm.free_energy(v)).collect();
    let energy_stats = DistributionStats::from_data(&energies);
    let threshold = energy_stats.mean + 2.0 * energy_stats.std_dev;

    println!("    Mean free energy: {:.4}", energy_stats.mean);
    println!("    Std free energy:  {:.4}", energy_stats.std_dev);
    println!("    Anomaly threshold (mean + 2*std): {:.4}", threshold);

    let anomalies: Vec<usize> = energies
        .iter()
        .enumerate()
        .filter(|(_, &e)| e > threshold)
        .map(|(i, _)| i)
        .collect();
    println!("    Anomalous observations: {} out of {}", anomalies.len(), encoded_data.len());

    println!("\n=== Example complete ===");
    Ok(())
}

/// Runs the example with synthetic data when Bybit API is unavailable.
fn run_with_synthetic_data() -> Result<()> {
    println!("=== Running with Synthetic Data ===\n");

    let mut rng = rand::thread_rng();
    use rand::Rng;

    // Generate correlated synthetic returns
    let n_samples = 200;
    let mut btc_returns = Vec::with_capacity(n_samples);
    let mut eth_returns = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let common = rng.gen_range(-0.02..0.02_f64);
        let btc_r = common + rng.gen_range(-0.01..0.01_f64);
        let eth_r = common * 1.2 + rng.gen_range(-0.015..0.015_f64);
        btc_returns.push(btc_r);
        eth_returns.push(eth_r);
    }

    let btc_stats = DistributionStats::from_data(&btc_returns);
    let eth_stats = DistributionStats::from_data(&eth_returns);
    println!("[Synthetic] BTC returns - mean: {:.6}, std: {:.6}", btc_stats.mean, btc_stats.std_dev);
    println!("[Synthetic] ETH returns - mean: {:.6}, std: {:.6}", eth_stats.mean, eth_stats.std_dev);

    let n_bins = 4;
    let btc_edges = compute_bin_edges(&btc_returns, n_bins);
    let eth_edges = compute_bin_edges(&eth_returns, n_bins);
    let bin_edges = vec![btc_edges, eth_edges];

    let encoded_data: Vec<Array1<f64>> = (0..n_samples)
        .map(|t| encode_returns(&[btc_returns[t], eth_returns[t]], &bin_edges, n_bins))
        .collect();

    let n_visible = encoded_data[0].len();
    let n_hidden = 6;

    println!("\nTraining Classical RBM...");
    let mut classical_rbm = ClassicalRBM::new(n_visible, n_hidden, 0.05);
    classical_rbm.train_cd_k(&encoded_data, 1, 20);

    println!("Training Quantum Boltzmann Machine...");
    let mut qbm = QuantumBoltzmannMachine::new(n_visible, n_hidden, 1.5, 4, 0.05);
    qbm.train(&encoded_data, 2, 20, 0.95);
    println!("Final gamma: {:.4}", qbm.gamma);

    let n_gen = 500;
    println!("\nGenerating {} synthetic scenarios...", n_gen);
    let classical_samples = classical_rbm.generate_samples(n_gen, 50);
    let quantum_samples = qbm.generate_samples(n_gen, 10);

    let real_dist = compute_bin_distribution(&encoded_data, 2, n_bins);
    let classical_dist = compute_bin_distribution(&classical_samples, 2, n_bins);
    let quantum_dist = compute_bin_distribution(&quantum_samples, 2, n_bins);

    println!("\nBin distribution comparison:");
    let asset_names = ["BTC", "ETH"];
    for (asset_idx, name) in asset_names.iter().enumerate() {
        println!("\n  {} bin distribution:", name);
        println!("    {:>8} {:>10} {:>10} {:>10}", "Bin", "Real", "Classical", "Quantum");
        for bin in 0..n_bins {
            println!(
                "    {:>8} {:>10.3} {:>10.3} {:>10.3}",
                bin,
                real_dist[asset_idx][bin],
                classical_dist[asset_idx][bin],
                quantum_dist[asset_idx][bin]
            );
        }
    }

    println!("\n  Jensen-Shannon divergence from real distribution:");
    for (asset_idx, name) in asset_names.iter().enumerate() {
        let js_classical = jensen_shannon(&real_dist[asset_idx], &classical_dist[asset_idx]);
        let js_quantum = jensen_shannon(&real_dist[asset_idx], &quantum_dist[asset_idx]);
        println!("    {} - Classical RBM: {:.4}, QBM: {:.4}", name, js_classical, js_quantum);
    }

    // Anomaly detection
    println!("\nAnomaly detection:");
    let energies: Vec<f64> = encoded_data.iter().map(|v| qbm.free_energy(v)).collect();
    let energy_stats = DistributionStats::from_data(&energies);
    let threshold = energy_stats.mean + 2.0 * energy_stats.std_dev;
    let anomalies = energies.iter().filter(|&&e| e > threshold).count();
    println!("    Anomalous observations: {} out of {}", anomalies, n_samples);

    println!("\n=== Example complete ===");
    Ok(())
}

/// Computes Jensen-Shannon divergence between two probability distributions.
fn jensen_shannon(p: &[f64], q: &[f64]) -> f64 {
    let epsilon = 1e-10;
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a, b)| (a + b) / 2.0).collect();

    let kl_pm: f64 = p
        .iter()
        .zip(m.iter())
        .map(|(&pi, &mi)| {
            if pi > epsilon && mi > epsilon {
                pi * (pi / mi).ln()
            } else {
                0.0
            }
        })
        .sum();

    let kl_qm: f64 = q
        .iter()
        .zip(m.iter())
        .map(|(&qi, &mi)| {
            if qi > epsilon && mi > epsilon {
                qi * (qi / mi).ln()
            } else {
                0.0
            }
        })
        .sum();

    (kl_pm + kl_qm) / 2.0
}
