use market_making_ml::*;
use rand::Rng;

fn main() {
    println!("=== Market Making ML - Trading Example ===\n");

    // --- Step 1: Fetch historical data from Bybit ---
    println!("Step 1: Fetching BTCUSDT kline data from Bybit...");
    let client = BybitClient::new();

    let candles = match client.fetch_klines("BTCUSDT", "1", 200) {
        Ok(c) => {
            println!("  Fetched {} candles", c.len());
            if let (Some(first), Some(last)) = (c.first(), c.last()) {
                println!(
                    "  Price range: {:.2} - {:.2}",
                    first.close, last.close
                );
            }
            c
        }
        Err(e) => {
            println!("  Could not fetch live data: {}", e);
            println!("  Using synthetic data for demonstration...\n");
            generate_synthetic_candles(200, 50000.0, 0.001)
        }
    };

    // --- Step 2: Compute volatility ---
    println!("\nStep 2: Volatility estimation");
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let rv = realized_volatility(&prices);
    let gk = garman_klass_volatility(&candles);
    println!("  Realized volatility:    {:.6}", rv);
    println!("  Garman-Klass volatility: {:.6}", gk);

    // --- Step 3: VPIN computation ---
    println!("\nStep 3: Flow toxicity (VPIN)");
    let vpin = compute_vpin(&candles, candles.last().map(|c| c.close).unwrap_or(50000.0) * 0.001);
    println!("  VPIN estimate: {:.4}", vpin);
    if vpin > 0.5 {
        println!("  WARNING: High flow toxicity detected!");
    } else {
        println!("  Flow toxicity is within normal range.");
    }

    // --- Step 4: Avellaneda-Stoikov optimal quoting ---
    println!("\nStep 4: Avellaneda-Stoikov optimal quotes");
    let as_config = AvellanedaStoikovConfig {
        gamma: 0.1,
        kappa: 1.5,
        time_horizon: 3600.0,
    };
    let model = AvellanedaStoikov::new(as_config.clone());
    let mid = candles.last().map(|c| c.close).unwrap_or(50000.0);

    println!("  Mid price: {:.2}", mid);
    for inv in [-3.0, -1.0, 0.0, 1.0, 3.0] {
        let (bid, ask) = model.compute_quotes(mid, inv, rv, 0.5);
        let spread_bps = (ask - bid) / mid * 10000.0;
        println!(
            "  Inventory {:+.0}: bid={:.2}, ask={:.2}, spread={:.1} bps",
            inv, bid, ask, spread_bps
        );
    }

    // --- Step 5: Adverse selection detection ---
    println!("\nStep 5: Adverse selection detection");
    let detector = AdverseSelectionDetector::default_detector();

    let scenarios = vec![
        ("Normal market", vec![0.2, 0.1, 0.5, 0.5, 0.1]),
        ("High VPIN", vec![0.9, -0.3, 1.5, 1.2, -0.6]),
        ("Strong imbalance", vec![0.3, 0.8, 0.3, 0.3, 0.9]),
    ];

    for (name, feat) in &scenarios {
        let features = ndarray::Array1::from(feat.clone());
        let prob = detector.predict(&features);
        let toxic = detector.is_toxic(&features);
        println!(
            "  {}: P(adverse)={:.3} [{}]",
            name,
            prob,
            if toxic { "TOXIC" } else { "SAFE" }
        );
    }

    // --- Step 6: RL agent training ---
    println!("\nStep 6: Training Q-learning market maker");
    let rl_config = MarketMakingRLConfig {
        epsilon: 0.3,
        learning_rate: 0.15,
        ..Default::default()
    };
    let mut agent = MarketMakingRL::new(rl_config);
    let mut rng = rand::thread_rng();

    let mut total_reward = 0.0;
    let episodes = 500;

    for ep in 0..episodes {
        let mut ep_reward = 0.0;
        let mut inv = 0.0_f64;

        for i in 20..candles.len() {
            let vol = realized_volatility(
                &candles[i - 20..i].iter().map(|c| c.close).collect::<Vec<_>>(),
            );
            let imb = if candles[i].close > candles[i].open { 0.3 } else { -0.3 };

            let features = MarketFeatures {
                inventory: inv,
                volatility: vol,
                ofi: imb,
                vpin: 0.3,
                spread: 0.001,
                imbalance: imb,
                time_remaining: (candles.len() - i) as f64 / candles.len() as f64,
            };

            let state = agent.featurize(&features, 0.05);
            let action = agent.select_action(&state, &mut rng);

            // Simulate: offset translates to spread multiplier
            let bid_offset = (action.bid_offset as f64 + 1.0) * vol * 0.5;
            let ask_offset = (action.ask_offset as f64 + 1.0) * vol * 0.5;
            let bid = candles[i].close * (1.0 - bid_offset);
            let ask = candles[i].close * (1.0 + ask_offset);

            // Simple fill simulation
            let mut pnl = 0.0;
            if candles[i].low <= bid && inv < 5.0 {
                inv += 1.0;
                pnl -= bid;
            }
            if candles[i].high >= ask && inv > -5.0 {
                inv -= 1.0;
                pnl += ask;
            }

            let reward = agent.compute_reward(pnl, inv);
            ep_reward += reward;

            // Next state
            let next_features = MarketFeatures {
                inventory: inv,
                ..features
            };
            let next_state = agent.featurize(&next_features, 0.05);
            agent.update(&state, &action, reward, &next_state);
        }

        total_reward += ep_reward;
        if (ep + 1) % 100 == 0 {
            println!(
                "  Episode {}/{}: avg reward = {:.4}",
                ep + 1,
                episodes,
                total_reward / (ep + 1) as f64
            );
        }
    }

    // --- Step 7: Full simulation backtest ---
    println!("\nStep 7: Running full Avellaneda-Stoikov backtest");
    let results = simulate_market_making(&candles, as_config, 10.0);
    let summary = compute_summary(&results);

    println!("  Total steps:    {}", summary.total_steps);
    println!("  Total fills:    {}", summary.total_fills);
    println!("  Final PnL:      {:.2}", summary.final_pnl);
    println!("  Max PnL:        {:.2}", summary.max_pnl);
    println!("  Max Drawdown:   {:.2}", summary.max_drawdown);
    println!("  Avg Spread:     {:.2}", summary.avg_spread);
    println!("  Avg |Inventory|: {:.2}", summary.avg_inventory);
    println!("  Max |Inventory|: {:.2}", summary.max_inventory);
    println!("  Sharpe Ratio:   {:.4}", summary.sharpe_ratio);

    // --- Step 8: Spread predictor demo ---
    println!("\nStep 8: Neural network spread predictor");
    let mut predictor = SpreadPredictor::new(5, 16, 8);

    // Generate training data from backtest results
    let train_data: Vec<(Vec<f64>, f64)> = results
        .iter()
        .map(|step| {
            let features = vec![
                step.inventory / 10.0,
                rv,
                step.spread / step.mid_price,
                0.5,
                step.inventory.signum() * 0.3,
            ];
            let target = step.spread / step.mid_price; // relative spread
            (features, target)
        })
        .collect();

    if !train_data.is_empty() {
        let loss = predictor.train(&train_data, 0.001, 100);
        println!("  Training loss (MSE): {:.8}", loss);

        let test_features = vec![0.0, rv, 0.001, 0.5, 0.0];
        let predicted_spread = predictor.predict(&test_features);
        println!("  Predicted optimal spread (relative): {:.6}", predicted_spread);
        println!("  Predicted optimal spread (absolute): {:.2}", predicted_spread * mid);
    }

    println!("\n=== Market Making ML Example Complete ===");
}

/// Generate synthetic candle data for demonstration when API is unavailable.
fn generate_synthetic_candles(n: usize, start_price: f64, volatility: f64) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = start_price;

    for i in 0..n {
        let ret = rng.gen_range(-volatility..volatility);
        let open = price;
        let close = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..volatility * 0.5));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..volatility * 0.5));
        let volume = rng.gen_range(1.0..20.0);

        candles.push(Candle {
            timestamp: 1700000000 + (i as u64) * 60000,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}
