//! Integration tests for LLM Factor Discovery
//!
//! Tests the complete workflow from parsing to evaluation.

use llm_factor_discovery::{
    data::{MarketData, OhlcvBar, PriceData, TimeFrame},
    evaluation::{BacktestConfig, Backtester, IcCalculator},
    parser::{FactorExpr, FactorValidator},
};
use chrono::Utc;

/// Helper to create test OHLCV data
fn create_test_market_data(n: usize) -> MarketData {
    let mut data = MarketData::new("BTCUSDT".to_string(), TimeFrame::Day1);

    let mut price = 100.0;
    for i in 0..n {
        let change = 0.02 * ((i as f64 * 0.1).sin() + (i as f64 * 0.05).cos() * 0.5);
        price *= 1.0 + change;

        data.bars.push(OhlcvBar {
            timestamp: Utc::now(),
            open: price * 0.998,
            high: price * 1.005,
            low: price * 0.995,
            close: price,
            volume: 1000.0 * (1.0 + (i as f64 * 0.2).sin().abs()),
            turnover: price * 1000.0,
        });
    }

    data
}

#[test]
fn test_parser_basic_expressions() {
    let expressions = vec![
        "close",
        "rank(close)",
        "ts_mean(close, 20)",
        "rank(ts_mean(returns(close, 5), 20))",
    ];

    for expr in expressions {
        let result = FactorExpr::parse(expr);
        assert!(result.is_ok(), "Failed to parse: {}", expr);
    }
}

#[test]
fn test_parser_all_functions() {
    let functions = vec![
        // Cross-sectional
        ("rank(close)", "rank"),
        ("zscore(close)", "zscore"),
        ("scale(close)", "scale"),
        ("demean(close)", "demean"),
        // Time series
        ("ts_rank(close, 10)", "ts_rank"),
        ("ts_sum(close, 10)", "ts_sum"),
        ("ts_mean(close, 10)", "ts_mean"),
        ("ts_std(close, 10)", "ts_std"),
        ("ts_min(close, 10)", "ts_min"),
        ("ts_max(close, 10)", "ts_max"),
        ("ts_delay(close, 5)", "ts_delay"),
        ("ts_delta(close, 5)", "ts_delta"),
        // Price operations
        ("returns(close, 5)", "returns"),
        ("log_returns(close, 5)", "log_returns"),
        ("volatility(close, 20)", "volatility"),
        // Statistical
        ("correlation(close, volume, 20)", "correlation"),
        ("covariance(close, volume, 20)", "covariance"),
        // Math
        ("abs(close)", "abs"),
        ("sign(close)", "sign"),
        ("log(volume)", "log"),
        ("sqrt(volume)", "sqrt"),
        ("power(close, 2)", "power"),
        ("max(high, close)", "max"),
        ("min(low, close)", "min"),
    ];

    for (expr, name) in functions {
        let result = FactorExpr::parse(expr);
        assert!(result.is_ok(), "Failed to parse {} ({})", expr, name);
    }
}

#[test]
fn test_parser_nested_expressions() {
    let expr = FactorExpr::parse("rank(zscore(ts_mean(returns(close, 1), 10)))").unwrap();
    assert_eq!(expr.depth(), 5);

    let vars = expr.variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains(&"close".to_string()));
}

#[test]
fn test_parser_invalid_expressions() {
    let invalid = vec![
        "",                          // Empty
        "unknown_func(close)",       // Unknown function
        "rank(close",                // Unclosed paren
        "rank()",                    // Missing arg
        "ts_mean(close)",            // Missing window
    ];

    for expr in invalid {
        let result = FactorExpr::parse(expr);
        assert!(result.is_err(), "Should fail: {}", expr);
    }
}

#[test]
fn test_validator_known_variables() {
    let validator = FactorValidator::new();

    let valid_expr = FactorExpr::parse("rank(close)").unwrap();
    let result = validator.validate(&valid_expr);
    assert!(result.is_valid);

    let invalid_expr = FactorExpr::parse("rank(unknown_var)").unwrap();
    let result = validator.validate(&invalid_expr);
    assert!(!result.is_valid);
}

#[test]
fn test_validator_window_size() {
    let validator = FactorValidator::new();

    // Valid window
    let valid = FactorExpr::parse("ts_mean(close, 20)").unwrap();
    let result = validator.validate(&valid);
    assert!(result.is_valid);

    // Invalid negative window
    let invalid = FactorExpr::parse("ts_mean(close, -5)").unwrap();
    let result = validator.validate(&invalid);
    assert!(!result.is_valid);

    // Invalid too large window
    let too_large = FactorExpr::parse("ts_mean(close, 500)").unwrap();
    let result = validator.validate(&too_large);
    assert!(!result.is_valid);
}

#[test]
fn test_ic_calculation_perfect_correlation() {
    let calc = IcCalculator::new();

    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..100).map(|i| i as f64 * 2.0).collect();

    let ic = calc.calculate_ic(&x, &y).unwrap();
    assert!((ic - 1.0).abs() < 1e-10);

    let rank_ic = calc.calculate_rank_ic(&x, &y).unwrap();
    assert!((rank_ic - 1.0).abs() < 1e-10);
}

#[test]
fn test_ic_calculation_negative_correlation() {
    let calc = IcCalculator::new();

    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..100).map(|i| (100 - i) as f64).collect();

    let ic = calc.calculate_ic(&x, &y).unwrap();
    assert!((ic - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_ic_statistics() {
    let calc = IcCalculator::new();

    let ic_series = vec![0.03, 0.02, 0.04, 0.01, 0.02, 0.03, 0.02, 0.03];
    let stats = calc.ic_statistics(&ic_series);

    assert!(stats.mean > 0.02 && stats.mean < 0.03);
    assert!(stats.std > 0.0);
    assert!(stats.ir > 0.0);
    assert!(stats.hit_rate == 1.0); // All positive
}

#[test]
fn test_forward_returns_calculation() {
    let prices = vec![100.0, 110.0, 105.0, 115.0, 120.0];
    let returns = IcCalculator::calculate_forward_returns(&prices, 1);

    assert_eq!(returns.len(), 4);
    assert!((returns[0] - 0.1).abs() < 1e-10); // (110-100)/100
}

#[test]
fn test_market_data_returns() {
    let data = create_test_market_data(100);

    let returns = data.returns();
    assert_eq!(returns.len(), 99);

    let log_returns = data.log_returns();
    assert_eq!(log_returns.len(), 99);
}

#[test]
fn test_price_data_from_market_data() {
    let market_data = create_test_market_data(50);
    let price_data = PriceData::from_market_data(&market_data);

    assert_eq!(price_data.len(), 50);
    assert_eq!(price_data.close.len(), 50);
    assert_eq!(price_data.volume.len(), 50);

    assert!(price_data.get("close").is_some());
    assert!(price_data.get("volume").is_some());
    assert!(price_data.get("unknown").is_none());
}

#[test]
fn test_backtest_config_default() {
    let config = BacktestConfig::default();

    assert!(!config.universe.is_empty());
    assert!(!config.forward_periods.is_empty());
    assert!(config.min_data_points > 0);
}

#[tokio::test]
async fn test_backtester_simple_factor() {
    let config = BacktestConfig::default();
    let backtester = Backtester::new(config);

    let expr = FactorExpr::parse("rank(close)").unwrap();
    let result = backtester.run(&expr).await.unwrap();

    assert!(!result.factor_name.is_empty());
    assert!(!result.expression.is_empty());
    assert!(result.num_observations > 0);
}

#[test]
fn test_expression_display() {
    let expr = FactorExpr::parse("rank(ts_mean(close, 20))").unwrap();
    let display = expr.to_string();

    assert_eq!(display, "rank(ts_mean(close, 20))");
}

#[test]
fn test_complexity_scoring() {
    let validator = FactorValidator::new();

    let simple = FactorExpr::parse("rank(close)").unwrap();
    let complex = FactorExpr::parse("zscore(correlation(returns(close, 1), volume, 60))").unwrap();

    let simple_result = validator.validate(&simple);
    let complex_result = validator.validate(&complex);

    assert!(complex_result.complexity_score > simple_result.complexity_score);
}
