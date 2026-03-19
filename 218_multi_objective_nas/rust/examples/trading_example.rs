//! Trading example: Multi-Objective NAS with Bybit market data
//!
//! Fetches BTCUSDT kline data from Bybit, computes market statistics,
//! then runs NSGA-II to find Pareto-optimal architectures balancing
//! accuracy, inference speed, and model size.

use multi_objective_nas::{
    compute_returns, compute_volatility, fetch_bybit_klines, hypervolume_2d,
    multi_objective_search_with_seed, SearchConfig,
};

fn main() {
    println!("=== Multi-Objective NAS for Trading ===\n");

    // ------------------------------------------------------------------
    // 1. Fetch market data from Bybit
    // ------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT 15-minute klines from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "15", 200) {
        Ok(c) => {
            println!("    Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            eprintln!("    Warning: could not fetch Bybit data ({}). Using default volatility.", e);
            vec![]
        }
    };

    // ------------------------------------------------------------------
    // 2. Compute market statistics
    // ------------------------------------------------------------------
    let volatility = if candles.len() >= 2 {
        let returns = compute_returns(&candles);
        let vol = compute_volatility(&returns, 20);
        let avg_vol = if vol.is_empty() {
            0.02
        } else {
            vol.iter().sum::<f64>() / vol.len() as f64
        };
        println!("[2] Market statistics:");
        println!("    Returns computed: {}", returns.len());
        println!("    Average 20-period volatility: {:.6}", avg_vol);
        if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
            println!(
                "    Price range: {:.2} -> {:.2}",
                first.close, last.close
            );
        }
        avg_vol
    } else {
        println!("[2] Using default volatility: 0.02");
        0.02
    };

    // ------------------------------------------------------------------
    // 3. Configure and run multi-objective NAS
    // ------------------------------------------------------------------
    println!("\n[3] Running NSGA-II multi-objective NAS...");
    println!("    Objectives: (1) accuracy, (2) inference latency, (3) model size");

    let config = SearchConfig {
        population_size: 100,
        num_generations: 50,
        mutation_rate: 0.2,
        volatility,
    };

    println!(
        "    Population: {}, Generations: {}, Mutation rate: {:.1}%",
        config.population_size,
        config.num_generations,
        config.mutation_rate * 100.0,
    );

    let result = multi_objective_search_with_seed(&config, 42);

    // ------------------------------------------------------------------
    // 4. Display Pareto front
    // ------------------------------------------------------------------
    println!("\n[4] Pareto Front ({} architectures found):", result.pareto_front.len());
    println!(
        "    {:<6} {:<10} {:<14} {:<12} {:<8} {:<8} {:<6} {:<6}",
        "#", "Accuracy", "Latency (us)", "Size (M)", "Layers", "HidDim", "Attn", "Act"
    );
    println!("    {}", "-".repeat(78));

    let mut front_sorted = result.pareto_front.clone();
    front_sorted.sort_by(|a, b| {
        a.objectives[0]
            .partial_cmp(&b.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (i, point) in front_sorted.iter().enumerate() {
        let accuracy = -point.objectives[0]; // un-negate
        let latency = point.objectives[1];
        let size = point.objectives[2];
        let arch = &point.architecture;
        let act = format!("{:?}", arch.activation);
        println!(
            "    {:<6} {:<10.4} {:<14.2} {:<12.4} {:<8} {:<8} {:<6} {:<6}",
            i + 1,
            accuracy,
            latency,
            size,
            arch.num_layers,
            arch.hidden_dim,
            arch.attention_heads,
            act,
        );
    }

    // ------------------------------------------------------------------
    // 5. Hypervolume
    // ------------------------------------------------------------------
    println!("\n[5] Quality Metrics:");
    println!("    Hypervolume (accuracy x latency): {:.4}", result.hypervolume);

    // Also compute accuracy x model-size hypervolume
    let pts_acc_size: Vec<Vec<f64>> = result
        .pareto_front
        .iter()
        .map(|p| vec![p.objectives[0], p.objectives[2]])
        .collect();
    let hv_acc_size = hypervolume_2d(&pts_acc_size, &[0.0, 10.0]);
    println!("    Hypervolume (accuracy x size):    {:.4}", hv_acc_size);
    println!("    Generations run:                  {}", result.generations_run);

    // ------------------------------------------------------------------
    // 6. Trade-off analysis
    // ------------------------------------------------------------------
    println!("\n[6] Trade-off Analysis:");

    // Most accurate
    if let Some(best_acc) = front_sorted.first() {
        let acc = -best_acc.objectives[0];
        let lat = best_acc.objectives[1];
        let sz = best_acc.objectives[2];
        println!(
            "    Most accurate:  acc={:.4}, latency={:.1}us, size={:.4}M",
            acc, lat, sz
        );
    }

    // Fastest
    if let Some(fastest) = front_sorted
        .iter()
        .min_by(|a, b| a.objectives[1].partial_cmp(&b.objectives[1]).unwrap())
    {
        let acc = -fastest.objectives[0];
        let lat = fastest.objectives[1];
        let sz = fastest.objectives[2];
        println!(
            "    Fastest:        acc={:.4}, latency={:.1}us, size={:.4}M",
            acc, lat, sz
        );
    }

    // Smallest
    if let Some(smallest) = front_sorted
        .iter()
        .min_by(|a, b| a.objectives[2].partial_cmp(&b.objectives[2]).unwrap())
    {
        let acc = -smallest.objectives[0];
        let lat = smallest.objectives[1];
        let sz = smallest.objectives[2];
        println!(
            "    Smallest:       acc={:.4}, latency={:.1}us, size={:.4}M",
            acc, lat, sz
        );
    }

    // Best balanced (closest to ideal point)
    let ideal = [
        front_sorted.iter().map(|p| p.objectives[0]).fold(f64::INFINITY, f64::min),
        front_sorted.iter().map(|p| p.objectives[1]).fold(f64::INFINITY, f64::min),
        front_sorted.iter().map(|p| p.objectives[2]).fold(f64::INFINITY, f64::min),
    ];
    let nadir = [
        front_sorted.iter().map(|p| p.objectives[0]).fold(f64::NEG_INFINITY, f64::max),
        front_sorted.iter().map(|p| p.objectives[1]).fold(f64::NEG_INFINITY, f64::max),
        front_sorted.iter().map(|p| p.objectives[2]).fold(f64::NEG_INFINITY, f64::max),
    ];

    if let Some(balanced) = front_sorted.iter().min_by(|a, b| {
        let dist_a: f64 = a
            .objectives
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let range = (nadir[i] - ideal[i]).max(1e-12);
                ((o - ideal[i]) / range).powi(2)
            })
            .sum();
        let dist_b: f64 = b
            .objectives
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let range = (nadir[i] - ideal[i]).max(1e-12);
                ((o - ideal[i]) / range).powi(2)
            })
            .sum();
        dist_a.partial_cmp(&dist_b).unwrap()
    }) {
        let acc = -balanced.objectives[0];
        let lat = balanced.objectives[1];
        let sz = balanced.objectives[2];
        println!(
            "    Best balanced:  acc={:.4}, latency={:.1}us, size={:.4}M",
            acc, lat, sz
        );
        println!("\n    Recommended architecture:");
        println!("      Layers: {}", balanced.architecture.num_layers);
        println!("      Hidden dim: {}", balanced.architecture.hidden_dim);
        println!("      Attention heads: {}", balanced.architecture.attention_heads);
        println!("      Activation: {:?}", balanced.architecture.activation);
        println!("      Dropout: {:.1}", balanced.architecture.dropout);
        println!(
            "      Skip connections: {}",
            balanced.architecture.use_skip_connections
        );
        println!("      Batch norm: {}", balanced.architecture.use_batch_norm);
    }

    println!("\n=== Done ===");
}
