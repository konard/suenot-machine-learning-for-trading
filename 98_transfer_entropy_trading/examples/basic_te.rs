//! Basic Transfer Entropy computation example.
//!
//! Demonstrates computing TE between simulated time series with
//! known lead-lag relationships.

use te_trading::prelude::*;

fn main() {
    println!("=== Basic Transfer Entropy Example ===\n");

    // Generate simulated data with lead-lag structure
    let mut gen = SimulatedDataGenerator::new(42);
    let (symbols, returns) = gen.generate_crypto_returns(500);

    println!("Generated {} periods for {} assets:", returns[0].len(), symbols.len());
    for (i, sym) in symbols.iter().enumerate() {
        let mean = returns[i].iter().sum::<f64>() / returns[i].len() as f64;
        let var: f64 = returns[i].iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / returns[i].len() as f64;
        println!("  {}: mean={:.6}, std={:.6}", sym, mean, var.sqrt());
    }

    // Compute pairwise TE using binning method
    println!("\n--- Binning Method (8 bins, history=3) ---\n");
    let estimator = TransferEntropyEstimator::new(
        TEMethod::Binning { n_bins: 8 },
        3,
    );

    for i in 0..symbols.len() {
        for j in 0..symbols.len() {
            if i != j {
                let te = estimator.compute_te(&returns[i], &returns[j]);
                println!("TE({} -> {}) = {:.4} bits", symbols[i], symbols[j], te);
            }
        }
    }

    // Compute net TE
    println!("\n--- Net Transfer Entropy ---\n");
    for i in 0..symbols.len() {
        let mut net_out = 0.0;
        for j in 0..symbols.len() {
            if i != j {
                net_out += estimator.compute_te(&returns[i], &returns[j])
                    - estimator.compute_te(&returns[j], &returns[i]);
            }
        }
        let role = if net_out > 0.0 { "Leader" } else { "Follower" };
        println!("{}: Net TE = {:+.4} ({})", symbols[i], net_out, role);
    }

    // Effective TE with significance test
    println!("\n--- Effective TE (BTC -> AVAX, 50 shuffles) ---\n");
    let (ete, p) = estimator.effective_te(&returns[0], &returns[3], 50);
    println!("ETE(BTC -> AVAX) = {:.4} bits, p-value = {:.4}", ete, p);

    // Compare with symbolic method
    println!("\n--- Symbolic Method (order=3, history=3) ---\n");
    let sym_estimator = TransferEntropyEstimator::new(
        TEMethod::Symbolic { order: 3 },
        3,
    );
    let te_bin = estimator.compute_te(&returns[0], &returns[3]);
    let te_sym = sym_estimator.compute_te(&returns[0], &returns[3]);
    println!("TE(BTC -> AVAX) binning:  {:.4} bits", te_bin);
    println!("TE(BTC -> AVAX) symbolic: {:.4} bits", te_sym);

    println!("\nDone!");
}
