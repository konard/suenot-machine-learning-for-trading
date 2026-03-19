use multi_objective_rl_trading::*;

fn generate_synthetic_prices(n: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0; // Starting BTC-like price
    prices.push(price);
    for _ in 1..n {
        let ret = rng.gen_range(-0.03..0.03);
        price *= f64::max(1.0 + ret, 0.9);
        prices.push(price);
    }
    prices
}

fn main() {
    println!("=== Multi-Objective RL Trading ===\n");

    // Step 1: Try to fetch data from Bybit, fall back to synthetic
    println!("Step 1: Fetching BTCUSDT data from Bybit...");
    let client = BybitClient::new();
    let prices = match client.fetch_klines("BTCUSDT", "60", 200) {
        Ok(p) if p.len() >= 50 => {
            println!("  Fetched {} candles from Bybit.\n", p.len());
            p
        }
        Ok(_) => {
            println!("  Insufficient data from Bybit, using synthetic prices.\n");
            generate_synthetic_prices(200)
        }
        Err(e) => {
            println!("  Could not fetch from Bybit ({}), using synthetic prices.\n", e);
            generate_synthetic_prices(200)
        }
    };

    // Step 2: Define weight vectors for different risk preferences
    println!("Step 2: Training multi-objective RL agents...");
    let weight_sets: Vec<([f64; 3], &str)> = vec![
        ([0.8, 0.1, 0.1], "Aggressive (return-focused)"),
        ([0.5, 0.3, 0.2], "Balanced"),
        ([0.2, 0.5, 0.3], "Conservative (drawdown-focused)"),
        ([0.1, 0.3, 0.6], "Ultra-conservative (volatility-focused)"),
        ([0.33, 0.34, 0.33], "Equal-weighted"),
        ([0.6, 0.3, 0.1], "Growth with safety"),
        ([0.3, 0.1, 0.6], "Smooth returns"),
    ];

    let mut pareto_front = ParetoFront::new();
    let mut all_results: Vec<(&str, [f64; NUM_OBJECTIVES], [f64; NUM_OBJECTIVES])> = Vec::new();

    for (weights, name) in &weight_sets {
        let mut agent = ScalarizedQLearning::new(*weights, 0.1, 0.99, 0.3);
        agent.train_on_prices(&prices, 100);

        let objectives = PolicyEvaluator::evaluate(&agent, &prices);

        pareto_front.insert(ParetoSolution {
            objectives,
            weights: *weights,
        });

        all_results.push((name, *weights, objectives));
        println!("  Trained: {}", name);
    }

    // Step 3: Display results
    println!("\n\nStep 3: Policy Evaluation Results");
    println!("{:-<90}", "");
    println!(
        "{:<35} {:>15} {:>15} {:>15}",
        "Policy", "Avg Return", "Avg -Drawdown", "Avg -Volatility"
    );
    println!("{:-<90}", "");

    for (name, _weights, objectives) in &all_results {
        println!(
            "{:<35} {:>15.6} {:>15.6} {:>15.6}",
            name, objectives[0], objectives[1], objectives[2]
        );
    }
    println!("{:-<90}", "");

    // Step 4: Show Pareto front
    println!("\n\nStep 4: Pareto Front ({} non-dominated solutions)", pareto_front.len());
    println!("{:-<80}", "");
    for (i, sol) in pareto_front.solutions.iter().enumerate() {
        println!(
            "  Solution {}: objectives=[{:.6}, {:.6}, {:.6}], weights=[{:.2}, {:.2}, {:.2}]",
            i + 1,
            sol.objectives[0],
            sol.objectives[1],
            sol.objectives[2],
            sol.weights[0],
            sol.weights[1],
            sol.weights[2]
        );
    }

    // Step 5: Select policy based on risk preference
    println!("\n\nStep 5: Policy Selection for Different Risk Preferences");
    println!("{:-<80}", "");

    let preferences = vec![
        ([1.0, 0.0, 0.0], "Maximum return"),
        ([0.0, 1.0, 0.0], "Minimum drawdown"),
        ([0.0, 0.0, 1.0], "Minimum volatility"),
        ([0.4, 0.4, 0.2], "Balanced risk-return"),
    ];

    for (pref, desc) in &preferences {
        if let Some(selected) = pareto_front.select_policy(pref) {
            println!(
                "  {}: selected policy with objectives [{:.6}, {:.6}, {:.6}]",
                desc, selected.objectives[0], selected.objectives[1], selected.objectives[2]
            );
        }
    }

    // Step 6: Hypervolume
    let reference = [-0.01, -0.1];
    let hv = pareto_front.hypervolume_2d(reference);
    println!("\n\nStep 6: Pareto Front Quality");
    println!("  Hypervolume (2D, return vs drawdown): {:.8}", hv);
    println!("  Number of Pareto-optimal solutions: {}", pareto_front.len());

    println!("\n=== Done ===");
}
