use bayesian_optimization_trading::*;
use rand::Rng;

/// Generate synthetic price data resembling BTC price action.
/// Used as fallback when Bybit API is unavailable.
fn generate_synthetic_prices(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price = 50_000.0;

    for i in 0..n {
        // Trend component
        let trend = 0.0001 * (i as f64 / n as f64 - 0.5);
        // Cyclical component
        let cycle = (i as f64 * 0.02).sin() * 0.005;
        // Random walk
        let noise: f64 = rng.gen_range(-0.015..0.015);

        price *= 1.0 + trend + cycle + noise;
        prices.push(price);
    }

    prices
}

fn main() {
    println!("=== Chapter 321: Bayesian Optimization for Trading ===\n");

    // ── Step 1: Fetch data ──────────────────────────────────────────────
    println!("Step 1: Fetching BTCUSDT price data...");

    let prices = match BybitClient::new().fetch_close_prices("BTCUSDT", "60", 1000) {
        Ok(p) if p.len() >= 200 => {
            println!("  Fetched {} candles from Bybit API", p.len());
            p
        }
        _ => {
            println!("  Bybit API unavailable, using synthetic data");
            generate_synthetic_prices(1000)
        }
    };

    println!(
        "  Price range: {:.2} - {:.2}\n",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // ── Step 2: Define search space ─────────────────────────────────────
    let search_space = SearchSpace::new()
        .add(HyperParameter::new("fast_window", 5.0, 50.0, true))
        .add(HyperParameter::new("slow_window", 51.0, 200.0, true));

    let backtester = TradingBacktester::default();

    // ── Step 3: Default parameters baseline ─────────────────────────────
    println!("Step 2: Evaluating default parameters (fast=10, slow=50)...");
    let default_result = backtester.run(10, 50, &prices);
    println!("  Sharpe Ratio:  {:.4}", default_result.sharpe_ratio);
    println!("  Total Return:  {:.4}", default_result.total_return);
    println!("  Max Drawdown:  {:.4}", default_result.max_drawdown);
    println!("  Num Trades:    {}\n", default_result.n_trades);

    // ── Step 4: Random search baseline ──────────────────────────────────
    println!("Step 3: Running random search (30 evaluations)...");
    let mut rng = rand::thread_rng();
    let mut best_random_sharpe = f64::NEG_INFINITY;
    let mut best_random_params = (10_usize, 50_usize);

    for _ in 0..30 {
        let fast = rng.gen_range(5..=50);
        let slow = rng.gen_range(51..=200);
        let sharpe = backtester.sharpe_objective(fast, slow, &prices);
        if sharpe > best_random_sharpe {
            best_random_sharpe = sharpe;
            best_random_params = (fast, slow);
        }
    }

    let random_result = backtester.run(best_random_params.0, best_random_params.1, &prices);
    println!(
        "  Best params:   fast={}, slow={}",
        best_random_params.0, best_random_params.1
    );
    println!("  Sharpe Ratio:  {:.4}", random_result.sharpe_ratio);
    println!("  Total Return:  {:.4}", random_result.total_return);
    println!("  Max Drawdown:  {:.4}", random_result.max_drawdown);
    println!("  Num Trades:    {}\n", random_result.n_trades);

    // ── Step 5: Bayesian Optimization ───────────────────────────────────
    println!("Step 4: Running Bayesian Optimization (30 evaluations)...");

    let bounds = search_space.bounds();
    let mut optimizer = BayesianOptimizer::new(
        bounds.clone(),
        AcquisitionFunction::ExpectedImprovement,
    );
    optimizer.n_initial = 5;
    optimizer.n_candidates = 500;

    let bo_result = optimizer.optimize(
        |params| {
            let casted = search_space.cast_params(params);
            let fast = casted[0] as usize;
            let slow = casted[1] as usize;
            backtester.sharpe_objective(fast, slow, &prices)
        },
        30,
    );

    let bo_best_fast = search_space.cast_params(&bo_result.best_params)[0] as usize;
    let bo_best_slow = search_space.cast_params(&bo_result.best_params)[1] as usize;
    let bo_backtest = backtester.run(bo_best_fast, bo_best_slow, &prices);

    println!(
        "  Best params:   fast={}, slow={}",
        bo_best_fast, bo_best_slow
    );
    println!("  Sharpe Ratio:  {:.4}", bo_backtest.sharpe_ratio);
    println!("  Total Return:  {:.4}", bo_backtest.total_return);
    println!("  Max Drawdown:  {:.4}", bo_backtest.max_drawdown);
    println!("  Num Trades:    {}\n", bo_backtest.n_trades);

    // ── Step 6: UCB comparison ──────────────────────────────────────────
    println!("Step 5: Running BO with UCB acquisition (30 evaluations)...");

    let mut optimizer_ucb = BayesianOptimizer::new(
        bounds,
        AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 },
    );
    optimizer_ucb.n_initial = 5;
    optimizer_ucb.n_candidates = 500;

    let ucb_result = optimizer_ucb.optimize(
        |params| {
            let casted = search_space.cast_params(params);
            let fast = casted[0] as usize;
            let slow = casted[1] as usize;
            backtester.sharpe_objective(fast, slow, &prices)
        },
        30,
    );

    let ucb_best_fast = search_space.cast_params(&ucb_result.best_params)[0] as usize;
    let ucb_best_slow = search_space.cast_params(&ucb_result.best_params)[1] as usize;
    let ucb_backtest = backtester.run(ucb_best_fast, ucb_best_slow, &prices);

    println!(
        "  Best params:   fast={}, slow={}",
        ucb_best_fast, ucb_best_slow
    );
    println!("  Sharpe Ratio:  {:.4}", ucb_backtest.sharpe_ratio);
    println!("  Total Return:  {:.4}", ucb_backtest.total_return);
    println!("  Max Drawdown:  {:.4}", ucb_backtest.max_drawdown);
    println!("  Num Trades:    {}\n", ucb_backtest.n_trades);

    // ── Step 7: Summary ─────────────────────────────────────────────────
    println!("=== Summary ===");
    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>8}",
        "Method", "Sharpe", "Return", "MaxDD", "Trades"
    );
    println!("{}", "-".repeat(62));
    println!(
        "{:<20} {:>10.4} {:>12.4} {:>12.4} {:>8}",
        "Default (10/50)", default_result.sharpe_ratio, default_result.total_return,
        default_result.max_drawdown, default_result.n_trades
    );
    println!(
        "{:<20} {:>10.4} {:>12.4} {:>12.4} {:>8}",
        format!("Random ({}/{})", best_random_params.0, best_random_params.1),
        random_result.sharpe_ratio, random_result.total_return,
        random_result.max_drawdown, random_result.n_trades
    );
    println!(
        "{:<20} {:>10.4} {:>12.4} {:>12.4} {:>8}",
        format!("BO-EI ({}/{})", bo_best_fast, bo_best_slow),
        bo_backtest.sharpe_ratio, bo_backtest.total_return,
        bo_backtest.max_drawdown, bo_backtest.n_trades
    );
    println!(
        "{:<20} {:>10.4} {:>12.4} {:>12.4} {:>8}",
        format!("BO-UCB ({}/{})", ucb_best_fast, ucb_best_slow),
        ucb_backtest.sharpe_ratio, ucb_backtest.total_return,
        ucb_backtest.max_drawdown, ucb_backtest.n_trades
    );

    // ── Step 8: Convergence trace ───────────────────────────────────────
    println!("\n=== BO-EI Convergence Trace ===");
    let mut best_so_far = f64::NEG_INFINITY;
    for (i, val) in bo_result.all_values.iter().enumerate() {
        if *val > best_so_far {
            best_so_far = *val;
        }
        let bar_len = ((best_so_far + 10.0).max(0.0) * 2.0) as usize;
        let bar: String = "#".repeat(bar_len.min(40));
        println!("  Iter {:>2}: value={:>8.4}  best={:>8.4}  {}", i + 1, val, best_so_far, bar);
    }
}
