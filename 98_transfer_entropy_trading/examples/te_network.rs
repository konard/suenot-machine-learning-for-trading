//! Transfer Entropy network construction example.
//!
//! Builds a directed information flow network across multiple assets
//! and identifies leaders and followers.

use te_trading::prelude::*;

fn main() {
    println!("=== TE Network Construction Example ===\n");

    // Generate simulated crypto returns with lead-lag
    let mut gen = SimulatedDataGenerator::new(42);
    let (symbols, returns) = gen.generate_crypto_returns(500);

    // Build TE network
    let return_refs: Vec<&[f64]> = returns.iter().map(|r| r.as_slice()).collect();
    let mut network = TENetwork::new(symbols.clone(), 3);
    network.compute_all_pairs(&return_refs);

    // Print TE matrix
    println!("Transfer Entropy Matrix (bits):\n");
    network.print_matrix();

    // Net TE
    println!("\nNet Transfer Entropy (outgoing - incoming):");
    let net = network.net_te();
    for (i, sym) in symbols.iter().enumerate() {
        let role = if net[i] > 0.0 { "Source" } else { "Sink" };
        println!("  {}: {:+.4} ({})", sym, net[i], role);
    }

    // Leaders and followers
    let (leaders, followers) = network.identify_leaders_followers();
    println!("\nLeaders:   {:?}", leaders);
    println!("Followers: {:?}", followers);

    // Strongest pairs
    if let Some((leader, score)) = network.strongest_leader() {
        println!("\nStrongest leader: {} (net TE = {:+.4})", leader, score);
    }
    if let Some((follower, score)) = network.strongest_follower() {
        println!("Strongest follower: {} (net TE = {:+.4})", follower, score);
    }

    // Multi-asset network with more assets
    println!("\n\n=== Extended Network (6 assets) ===\n");
    let mut gen2 = SimulatedDataGenerator::new(123);
    let all_returns = gen2.generate_lead_lag_returns(6, 500, 0.3, 0.02);
    let ext_symbols: Vec<String> = (0..6).map(|i| format!("Asset_{}", i)).collect();
    let ext_refs: Vec<&[f64]> = all_returns.iter().map(|r| r.as_slice()).collect();

    let mut ext_network = TENetwork::new(ext_symbols.clone(), 3);
    ext_network.compute_all_pairs(&ext_refs);

    println!("TE Matrix:");
    ext_network.print_matrix();

    let (leaders, followers) = ext_network.identify_leaders_followers();
    println!("\nLeaders:   {:?}", leaders);
    println!("Followers: {:?}", followers);

    println!("\nDone!");
}
