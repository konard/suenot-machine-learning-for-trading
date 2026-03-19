use anyhow::Result;
use monte_carlo_tree_search::{fetch_bybit_klines, TradingState, MCTS};

fn main() -> Result<()> {
    println!("=== Monte Carlo Tree Search for Trading ===\n");

    // Fetch BTCUSDT hourly data from Bybit
    println!("Fetching BTCUSDT hourly data from Bybit...");
    let prices = match fetch_bybit_klines("BTCUSDT", "60", 100) {
        Ok(p) => {
            println!("Fetched {} price points", p.len());
            if let (Some(first), Some(last)) = (p.first(), p.last()) {
                println!("Price range: {:.2} -> {:.2}\n", first, last);
            }
            p
        }
        Err(e) => {
            println!("Could not fetch from Bybit: {}", e);
            println!("Using sample data instead.\n");
            // Fallback sample data simulating BTC price movement
            vec![
                50000.0, 50150.0, 50080.0, 50220.0, 50350.0, 50280.0, 50400.0, 50550.0,
                50480.0, 50620.0, 50700.0, 50650.0, 50800.0, 50750.0, 50900.0, 51000.0,
                50950.0, 51100.0, 51050.0, 51200.0,
            ]
        }
    };

    // Initialize trading state
    let initial_balance = 10000.0;
    let state = TradingState::new(prices, initial_balance);
    println!("Initial balance: ${:.2}", initial_balance);
    println!("Starting price: ${:.2}", state.current_price());
    println!();

    // Configure and run MCTS
    let exploration_constant = 1.414; // sqrt(2)
    let iterations = 5000;

    println!(
        "Running MCTS with {} iterations (c = {:.3})...\n",
        iterations, exploration_constant
    );

    let mut mcts = MCTS::new(exploration_constant, iterations);
    let best_action = mcts.search(&state);

    // Display results
    println!("--- MCTS Results ---\n");
    println!("Root node visits: {}", mcts.nodes[0].visits);
    println!("Tree size: {} nodes\n", mcts.nodes.len());

    println!("Action statistics:");
    println!("{:<10} {:>10} {:>15}", "Action", "Visits", "Avg Value");
    println!("{}", "-".repeat(37));

    let mut stats = mcts.root_children_stats();
    stats.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by visits (descending)

    for (action, visits, avg_value) in &stats {
        println!(
            "{:<10} {:>10} {:>15.6}",
            action.to_string(),
            visits,
            avg_value
        );
    }
    println!();

    match best_action {
        Some(action) => {
            println!("Best action: {}", action);
        }
        None => {
            println!("No action recommended (terminal state).");
        }
    }

    // Simulate a multi-step trading session
    println!("\n--- Multi-step Trading Session ---\n");
    let session_prices = state.prices[..state.prices.len().min(10)].to_vec();
    let mut current_state = TradingState::new(session_prices, initial_balance);

    for step in 0..5 {
        if current_state.is_terminal() {
            println!("Reached terminal state at step {}", step);
            break;
        }

        let mut step_mcts = MCTS::new(1.414, 2000);
        let action = step_mcts.search(&current_state);

        match action {
            Some(a) => {
                let prev_value = current_state.portfolio_value();
                current_state = current_state.apply_action(a);
                let new_value = current_state.portfolio_value();
                println!(
                    "Step {}: {} at ${:.2} | Balance: ${:.2} | Position: {:.4} | Portfolio: ${:.2} ({:+.2})",
                    step,
                    a,
                    current_state.prices[current_state.step.min(current_state.prices.len() - 1)],
                    current_state.balance,
                    current_state.position,
                    new_value,
                    new_value - prev_value
                );
            }
            None => {
                println!("Step {}: No action available", step);
                break;
            }
        }
    }

    let final_value = current_state.portfolio_value();
    println!(
        "\nFinal portfolio value: ${:.2} (PnL: {:+.2}, Return: {:+.2}%)",
        final_value,
        final_value - initial_balance,
        (final_value - initial_balance) / initial_balance * 100.0
    );

    Ok(())
}
