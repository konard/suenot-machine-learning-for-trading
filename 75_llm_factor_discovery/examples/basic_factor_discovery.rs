//! Basic factor discovery example
//!
//! Demonstrates how to use LLM to generate and evaluate alpha factors.
//!
//! Run with: cargo run --example basic_factor_discovery

use llm_factor_discovery::{
    discovery::{FactorCandidate, GenerationConfig, LlmClient},
    evaluation::{BacktestConfig, Backtester, IcCalculator},
    parser::FactorExpr,
    FactorDiscoverySystem,
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== LLM Factor Discovery Example ===\n");

    // Example 1: Parse and analyze factor expressions
    info!("Example 1: Parsing factor expressions");
    let expressions = vec![
        "rank(ts_mean(returns(close, 5), 20))",
        "zscore(correlation(returns(close, 1), volume, 60))",
        "ts_rank(ts_delta(close, 5), 20)",
        "rank(div(sub(high, low), close))",
    ];

    for expr_str in expressions {
        match FactorExpr::parse(expr_str) {
            Ok(expr) => {
                info!("  Expression: {}", expr_str);
                info!("    Parsed: {}", expr);
                info!("    Depth: {}", expr.depth());
                info!("    Variables: {:?}", expr.variables());
                info!("");
            }
            Err(e) => {
                info!("  Failed to parse '{}': {}", expr_str, e);
            }
        }
    }

    // Example 2: Calculate Information Coefficient
    info!("\nExample 2: IC Calculation");
    let ic_calc = IcCalculator::new();

    // Simulated factor values and forward returns
    let factor_values: Vec<f64> = (0..100)
        .map(|i| (i as f64 * 0.1).sin())
        .collect();

    let forward_returns: Vec<f64> = factor_values
        .iter()
        .map(|&f| f * 0.3 + rand_like(0.1)) // Factor explains 30% of returns
        .collect();

    if let Some(ic) = ic_calc.calculate_ic(&factor_values, &forward_returns) {
        info!("  Pearson IC: {:.4}", ic);
    }

    if let Some(rank_ic) = ic_calc.calculate_rank_ic(&factor_values, &forward_returns) {
        info!("  Rank IC: {:.4}", rank_ic);
    }

    // Calculate rolling IC
    let rolling_ic = ic_calc.rolling_ic(&factor_values, &forward_returns, 20);
    let ic_stats = ic_calc.ic_statistics(&rolling_ic);
    info!("  IC Mean: {:.4}", ic_stats.mean);
    info!("  IC Std: {:.4}", ic_stats.std);
    info!("  IC IR: {:.4}", ic_stats.ir);
    info!("  IC Hit Rate: {:.2}%", ic_stats.hit_rate * 100.0);

    // Example 3: Create factor candidates (manually, without LLM)
    info!("\nExample 3: Manual factor candidates");
    let candidates = vec![
        FactorCandidate {
            name: "momentum_5d".to_string(),
            expression: "rank(returns(close, 5))".to_string(),
            rationale: "Short-term momentum factor".to_string(),
            expected_ic: 0.03,
            expected_turnover: llm_factor_discovery::discovery::TurnoverLevel::High,
            hypothesis: "momentum".to_string(),
            iteration: 0,
        },
        FactorCandidate {
            name: "mean_reversion_20d".to_string(),
            expression: "rank(sub(0, ts_rank(close, 20)))".to_string(),
            rationale: "Mean reversion over 20 days".to_string(),
            expected_ic: 0.02,
            expected_turnover: llm_factor_discovery::discovery::TurnoverLevel::Medium,
            hypothesis: "mean_reversion".to_string(),
            iteration: 0,
        },
        FactorCandidate {
            name: "volume_momentum".to_string(),
            expression: "rank(ts_mean(mul(returns(close, 1), volume), 10))".to_string(),
            rationale: "Price movement weighted by volume".to_string(),
            expected_ic: 0.025,
            expected_turnover: llm_factor_discovery::discovery::TurnoverLevel::Medium,
            hypothesis: "volume".to_string(),
            iteration: 0,
        },
    ];

    for candidate in &candidates {
        info!("  Factor: {}", candidate.name);
        info!("    Expression: {}", candidate.expression);
        info!("    Expected IC: {:.3}", candidate.expected_ic);
        info!("");
    }

    // Example 4: Backtest factors (simulated)
    info!("\nExample 4: Backtesting factors");
    let backtest_config = BacktestConfig::default();
    let backtester = Backtester::new(backtest_config);

    for candidate in &candidates {
        if let Ok(expr) = FactorExpr::parse(&candidate.expression) {
            match backtester.run(&expr).await {
                Ok(result) => {
                    info!("  {} Results:", candidate.name);
                    info!("    IC: {:.4}", result.ic);
                    info!("    IC IR: {:.4}", result.ic_ir);
                    info!("    Turnover: {:.2}", result.turnover);
                    info!("    Quality Score: {:.4}", result.quality_score());
                    info!("");
                }
                Err(e) => {
                    info!("  Failed to backtest {}: {}", candidate.name, e);
                }
            }
        }
    }

    // Example 5: Using the LLM (requires API key)
    info!("\nExample 5: LLM-based discovery (requires OPENAI_API_KEY)");
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        info!("  API key found, attempting LLM-based discovery...");

        match LlmClient::new_openai(&api_key) {
            Ok(client) => {
                let system = FactorDiscoverySystem::new(client);

                // Generate factors for a hypothesis
                let hypothesis = "momentum reversal in oversold conditions";
                info!("  Generating factors for: '{}'", hypothesis);

                match system.generate_factors(hypothesis).await {
                    Ok(factors) => {
                        info!("  Generated {} factors:", factors.len());
                        for factor in &factors {
                            info!("    - {}: {}", factor.name, factor.expression);
                        }
                    }
                    Err(e) => {
                        info!("  Factor generation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                info!("  Failed to create LLM client: {}", e);
            }
        }
    } else {
        info!("  OPENAI_API_KEY not set, skipping LLM example");
        info!("  Set the environment variable to enable LLM-based discovery");
    }

    info!("\n=== Example Complete ===");

    Ok(())
}

/// Simple pseudo-random noise generator (for demo purposes)
fn rand_like(scale: f64) -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let normalized = (nanos as f64 / u32::MAX as f64) * 2.0 - 1.0;
    normalized * scale
}
