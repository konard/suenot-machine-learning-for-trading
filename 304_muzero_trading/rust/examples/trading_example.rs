//! # MuZero Trading Example
//!
//! Demonstrates MuZero for algorithmic trading:
//! 1. Fetch BTCUSDT data from Bybit
//! 2. Train MuZero on historical data using self-play
//! 3. Plan using the learned dynamics model (MCTS)
//! 4. Compare with a simple model-free DQN baseline

use muzero_trading::*;

fn main() {
    println!("=== MuZero Trading Example ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Get price data
    // -----------------------------------------------------------------------
    println!("[1] Fetching price data...");

    let prices = match fetch_bybit_klines_blocking("BTCUSDT", "15", 200) {
        Ok(klines) => {
            println!(
                "    Fetched {} candles from Bybit (BTCUSDT 15m)",
                klines.len()
            );
            if klines.len() >= 50 {
                klines_to_prices(&klines)
            } else {
                println!("    Not enough candles, using synthetic data.");
                generate_synthetic_prices(200, 65000.0)
            }
        }
        Err(e) => {
            println!("    Bybit API unavailable ({}), using synthetic data.", e);
            generate_synthetic_prices(200, 65000.0)
        }
    };

    println!(
        "    Price range: {:.2} - {:.2} ({} data points)\n",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        prices.len()
    );

    // -----------------------------------------------------------------------
    // Step 2: Setup environment and MuZero model
    // -----------------------------------------------------------------------
    println!("[2] Setting up MuZero model...");

    let window_size = 7;
    let obs_dim = window_size * 5 + 2; // 37
    let hidden_dim = 32;
    let num_simulations = 15;

    let mut model = MuZeroModel::new(obs_dim, hidden_dim);
    let mut buffer = ReplayBuffer::new(5000);

    println!("    Observation dim: {}", obs_dim);
    println!("    Hidden dim:      {}", hidden_dim);
    println!("    Actions:         {}", NUM_ACTIONS);
    println!("    MCTS sims:       {}\n", num_simulations);

    // -----------------------------------------------------------------------
    // Step 3: Training loop with self-play
    // -----------------------------------------------------------------------
    println!("[3] Training MuZero via self-play...");

    let num_episodes = 5;
    let batch_size = 16;
    let learning_rate = 0.001;

    for episode in 0..num_episodes {
        let mut env = TradingEnv::new(prices.clone(), window_size);

        // Self-play: generate training data
        let temperature = if episode < 3 { 1.0 } else { 0.5 };
        let transitions = self_play_episode(&model, &mut env, num_simulations, temperature);

        let episode_return: f64 = transitions.iter().map(|t| t.reward).sum();
        let num_transitions = transitions.len();

        // Add to replay buffer
        for t in transitions {
            buffer.push(t);
        }

        // Train on sampled batch
        if buffer.len() >= batch_size {
            let batch = buffer.sample(batch_size);
            let loss = train_step(&mut model, &batch, learning_rate);

            println!(
                "    Episode {}/{}: steps={}, return={:.6}, loss={:.6}, buffer={}",
                episode + 1,
                num_episodes,
                num_transitions,
                episode_return,
                loss,
                buffer.len()
            );
        } else {
            println!(
                "    Episode {}/{}: steps={}, return={:.6}, buffer={}",
                episode + 1,
                num_episodes,
                num_transitions,
                episode_return,
                buffer.len()
            );
        }
    }

    // Reanalysis: improve targets with current model
    println!("\n    Running reanalysis on replay buffer...");
    reanalyze(&model, &mut buffer, 5);
    println!("    Reanalysis complete.\n");

    // -----------------------------------------------------------------------
    // Step 4: Evaluate MuZero with MCTS planning
    // -----------------------------------------------------------------------
    println!("[4] Evaluating MuZero agent with MCTS planning...");

    let mut eval_env = TradingEnv::new(prices.clone(), window_size);
    let mut muzero_actions = Vec::new();
    let mut muzero_rewards = Vec::new();
    let mut step = 0;

    while !eval_env.is_done() {
        let obs = eval_env.get_observation();
        let root = run_mcts(&model, &obs, num_simulations);
        let (action_idx, _policy) = select_action_from_mcts(&root, 0.1); // low temperature for exploitation
        let action = TradingAction::from_index(action_idx);

        let (reward, _done) = eval_env.step(action);
        muzero_actions.push(action);
        muzero_rewards.push(reward);
        step += 1;

        if step <= 5 || step % 50 == 0 {
            println!(
                "    Step {:>4}: action={:?}, reward={:+.6}, position={:.1}, total_pnl={:+.6}",
                step, action, reward, eval_env.position, eval_env.total_pnl
            );
        }
    }

    let muzero_total_pnl = eval_env.total_pnl;
    let muzero_total_reward: f64 = muzero_rewards.iter().sum();
    println!("    MuZero total PnL:    {:+.6}", muzero_total_pnl);
    println!("    MuZero total reward:  {:+.6}\n", muzero_total_reward);

    // -----------------------------------------------------------------------
    // Step 5: Compare with model-free DQN baseline
    // -----------------------------------------------------------------------
    println!("[5] Comparing with model-free DQN baseline...");

    let dqn = SimpleDQN::new(obs_dim);
    let mut dqn_env = TradingEnv::new(prices.clone(), window_size);
    let mut dqn_rewards = Vec::new();

    while !dqn_env.is_done() {
        let obs = dqn_env.get_observation();
        let action_idx = dqn.select_action(&obs, 0.1); // epsilon-greedy
        let action = TradingAction::from_index(action_idx);
        let (reward, _done) = dqn_env.step(action);
        dqn_rewards.push(reward);
    }

    let dqn_total_pnl = dqn_env.total_pnl;
    let dqn_total_reward: f64 = dqn_rewards.iter().sum();
    println!("    DQN total PnL:       {:+.6}", dqn_total_pnl);
    println!("    DQN total reward:     {:+.6}\n", dqn_total_reward);

    // -----------------------------------------------------------------------
    // Step 6: Buy-and-hold baseline
    // -----------------------------------------------------------------------
    println!("[6] Buy-and-hold baseline...");
    let bnh_return = (prices.last().unwrap_or(&1.0) / prices[window_size]) - 1.0;
    println!("    Buy-and-hold return: {:+.6}\n", bnh_return);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("=== Summary ===");
    println!(
        "    MuZero (MCTS planning):  PnL = {:+.6}  |  Reward = {:+.6}",
        muzero_total_pnl, muzero_total_reward
    );
    println!(
        "    DQN (model-free):        PnL = {:+.6}  |  Reward = {:+.6}",
        dqn_total_pnl, dqn_total_reward
    );
    println!(
        "    Buy-and-Hold:            Return = {:+.6}",
        bnh_return
    );
    println!("\n    Note: MuZero benefits from more training episodes and simulations.");
    println!("    With sufficient training, MCTS planning typically outperforms reactive DQN.");

    // -----------------------------------------------------------------------
    // Demonstrate value scaling
    // -----------------------------------------------------------------------
    println!("\n[7] Value scaling demonstration...");
    let test_values = vec![0.0, 1.0, -1.0, 100.0, -50.0];
    let eps = 0.001;
    for v in test_values {
        let scaled = scale_value(v, eps);
        let recovered = inverse_scale_value(scaled, eps);
        println!(
            "    value={:>8.3}  scaled={:>8.5}  recovered={:>8.3}",
            v, scaled, recovered
        );
    }

    println!("\n=== MuZero Trading Example Complete ===");
    println!("DONE");
}
