//! Alpha factor generation and evaluation module.
//!
//! Provides tools for creating, evaluating, and comparing alpha factors.

use crate::data::MarketData;
use crate::error::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Alpha factor definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaFactor {
    /// Unique name for the factor
    pub name: String,
    /// Mathematical expression
    pub expression: String,
    /// Human-readable description
    pub description: String,
    /// Confidence score (0-1)
    pub confidence: f64,
}

impl AlphaFactor {
    /// Create a new alpha factor.
    pub fn new(name: String, expression: String) -> Self {
        Self {
            name,
            expression,
            description: String::new(),
            confidence: 0.5,
        }
    }

    /// Create with full parameters.
    pub fn with_description(name: String, expression: String, description: String) -> Self {
        Self {
            name,
            expression,
            description,
            confidence: 0.5,
        }
    }
}

/// Metrics for evaluating alpha factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorMetrics {
    /// Information Coefficient (Pearson correlation with returns)
    pub ic: f64,
    /// Rank IC (Spearman correlation)
    pub rank_ic: f64,
    /// IC Information Ratio
    pub ic_ir: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// T-statistic for IC
    pub t_stat: f64,
    /// P-value for IC significance
    pub p_value: f64,
}

impl FactorMetrics {
    /// Calculate quality score (0-100).
    pub fn quality_score(&self) -> f64 {
        let mut score = 0.0;

        // IC score (0-30)
        score += (self.ic.abs() * 150.0).min(30.0);

        // IC-IR score (0-20)
        score += (self.ic_ir.abs() * 10.0).min(20.0);

        // Sharpe score (0-25)
        score += (self.sharpe_ratio.max(0.0) * 12.5).min(25.0);

        // Drawdown score (0-15)
        score += (15.0 - self.max_drawdown.abs() * 30.0).max(0.0);

        // Significance bonus (0-10)
        if self.p_value < 0.05 {
            score += 10.0;
        } else if self.p_value < 0.1 {
            score += 5.0;
        }

        score
    }

    /// Check if IC is statistically significant.
    pub fn is_significant(&self) -> bool {
        self.p_value < 0.05
    }
}

/// Alpha expression evaluator.
///
/// Parses and evaluates alpha factor expressions on market data.
pub struct AlphaEvaluator<'a> {
    data: &'a MarketData,
    variables: HashMap<String, Vec<f64>>,
}

impl<'a> AlphaEvaluator<'a> {
    /// Create a new evaluator for the given market data.
    pub fn new(data: &'a MarketData) -> Self {
        let mut variables = HashMap::new();

        // Add standard variables
        variables.insert("close".to_string(), data.close_prices());
        variables.insert("volume".to_string(), data.volumes());

        // Add OHLCV data
        let opens: Vec<f64> = data.candles.iter().map(|c| c.open).collect();
        let highs: Vec<f64> = data.candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = data.candles.iter().map(|c| c.low).collect();

        variables.insert("open".to_string(), opens);
        variables.insert("high".to_string(), highs);
        variables.insert("low".to_string(), lows);

        // Add computed variables
        variables.insert("returns".to_string(), data.returns());
        variables.insert("log_returns".to_string(), data.log_returns());

        Self { data, variables }
    }

    /// Validate an alpha expression.
    pub fn validate(&self, expression: &str) -> bool {
        // Check for dangerous patterns
        let dangerous = [
            "import", "exec", "eval", "__", "open(",
            "file(", "os.", "sys.", "subprocess",
        ];

        for pattern in &dangerous {
            if expression.to_lowercase().contains(pattern) {
                return false;
            }
        }

        // Check for valid function calls
        let func_re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(").unwrap();
        let valid_funcs = [
            "ts_mean", "ts_std", "ts_delta", "ts_delay", "ts_max", "ts_min",
            "ts_rank", "ts_sum", "ts_corr", "rank", "log", "abs", "sign", "sqrt",
        ];

        for cap in func_re.captures_iter(expression) {
            let func = &cap[1];
            if !valid_funcs.contains(&func) && func != "max" && func != "min" {
                return false;
            }
        }

        true
    }

    /// Evaluate an alpha factor expression.
    pub fn evaluate(&self, factor: &AlphaFactor) -> Result<Vec<f64>> {
        self.evaluate_expression(&factor.expression)
    }

    /// Evaluate an expression string.
    pub fn evaluate_expression(&self, expression: &str) -> Result<Vec<f64>> {
        if !self.validate(expression) {
            return Err(Error::InvalidExpression(expression.to_string()));
        }

        // Simple expression parser (supports common patterns)
        self.parse_and_evaluate(expression)
    }

    fn parse_and_evaluate(&self, expression: &str) -> Result<Vec<f64>> {
        let expr = expression.trim();

        // Handle negation
        if expr.starts_with("-1 *") || expr.starts_with("-1*") {
            let inner = expr.trim_start_matches("-1 *").trim_start_matches("-1*").trim();
            let values = self.parse_and_evaluate(inner)?;
            return Ok(values.iter().map(|v| -v).collect());
        }

        // Handle division
        if let Some(pos) = self.find_operator(expr, '/') {
            let left = self.parse_and_evaluate(&expr[..pos])?;
            let right = self.parse_and_evaluate(&expr[pos + 1..])?;
            return Ok(left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| if *b != 0.0 { a / b } else { f64::NAN })
                .collect());
        }

        // Handle subtraction
        if let Some(pos) = self.find_operator(expr, '-') {
            let left = self.parse_and_evaluate(&expr[..pos])?;
            let right = self.parse_and_evaluate(&expr[pos + 1..])?;
            return Ok(left.iter().zip(right.iter()).map(|(a, b)| a - b).collect());
        }

        // Handle addition
        if let Some(pos) = self.find_operator(expr, '+') {
            let left = self.parse_and_evaluate(&expr[..pos])?;
            let right = self.parse_and_evaluate(&expr[pos + 1..])?;
            return Ok(left.iter().zip(right.iter()).map(|(a, b)| a + b).collect());
        }

        // Handle multiplication
        if let Some(pos) = self.find_operator(expr, '*') {
            let left = self.parse_and_evaluate(&expr[..pos])?;
            let right = self.parse_and_evaluate(&expr[pos + 1..])?;
            return Ok(left.iter().zip(right.iter()).map(|(a, b)| a * b).collect());
        }

        // Handle parentheses
        if expr.starts_with('(') && expr.ends_with(')') {
            return self.parse_and_evaluate(&expr[1..expr.len() - 1]);
        }

        // Handle function calls
        if let Some(result) = self.evaluate_function(expr)? {
            return Ok(result);
        }

        // Handle variables
        if let Some(values) = self.variables.get(expr) {
            return Ok(values.clone());
        }

        // Handle numeric literals
        if let Ok(num) = expr.parse::<f64>() {
            let n = self.data.len();
            return Ok(vec![num; n]);
        }

        Err(Error::InvalidExpression(format!("Cannot parse: {}", expr)))
    }

    fn find_operator(&self, expr: &str, op: char) -> Option<usize> {
        let mut depth = 0;
        let chars: Vec<char> = expr.chars().collect();

        // Search from right to left for correct precedence
        for i in (0..chars.len()).rev() {
            match chars[i] {
                '(' => depth += 1,
                ')' => depth -= 1,
                c if c == op && depth == 0 && i > 0 => {
                    // Avoid matching negative numbers
                    if op == '-' {
                        let prev = chars[i - 1];
                        if prev == '(' || prev == '/' || prev == '*' || prev == '+' || prev == '-' {
                            continue;
                        }
                    }
                    return Some(i);
                }
                _ => {}
            }
        }
        None
    }

    fn evaluate_function(&self, expr: &str) -> Result<Option<Vec<f64>>> {
        let func_re = Regex::new(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$").unwrap();

        if let Some(caps) = func_re.captures(expr) {
            let func_name = &caps[1];
            let args_str = &caps[2];
            let args = self.split_args(args_str)?;

            match func_name {
                "ts_mean" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    let window: usize = args[1].trim().parse()
                        .map_err(|_| Error::InvalidExpression(format!("Invalid window: {}", args[1])))?;
                    return Ok(Some(Self::rolling_mean(&data, window)));
                }
                "ts_std" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    let window: usize = args[1].trim().parse()
                        .map_err(|_| Error::InvalidExpression(format!("Invalid window: {}", args[1])))?;
                    return Ok(Some(Self::rolling_std(&data, window)));
                }
                "ts_delta" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    let periods: usize = args[1].trim().parse()
                        .map_err(|_| Error::InvalidExpression(format!("Invalid periods: {}", args[1])))?;
                    return Ok(Some(Self::delta(&data, periods)));
                }
                "ts_delay" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    let periods: usize = args[1].trim().parse()
                        .map_err(|_| Error::InvalidExpression(format!("Invalid periods: {}", args[1])))?;
                    return Ok(Some(Self::delay(&data, periods)));
                }
                "ts_max" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    let window: usize = args[1].trim().parse()
                        .map_err(|_| Error::InvalidExpression(format!("Invalid window: {}", args[1])))?;
                    return Ok(Some(Self::rolling_max(&data, window)));
                }
                "ts_min" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    let window: usize = args[1].trim().parse()
                        .map_err(|_| Error::InvalidExpression(format!("Invalid window: {}", args[1])))?;
                    return Ok(Some(Self::rolling_min(&data, window)));
                }
                "rank" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    return Ok(Some(Self::rank(&data)));
                }
                "log" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    return Ok(Some(data.iter().map(|x| if *x > 0.0 { x.ln() } else { f64::NAN }).collect()));
                }
                "abs" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    return Ok(Some(data.iter().map(|x| x.abs()).collect()));
                }
                "sign" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    return Ok(Some(data.iter().map(|x| x.signum()).collect()));
                }
                "sqrt" => {
                    let data = self.parse_and_evaluate(&args[0])?;
                    return Ok(Some(data.iter().map(|x| if *x >= 0.0 { x.sqrt() } else { f64::NAN }).collect()));
                }
                _ => {}
            }
        }

        Ok(None)
    }

    fn split_args(&self, args_str: &str) -> Result<Vec<String>> {
        let mut args = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for c in args_str.chars() {
            match c {
                '(' => {
                    depth += 1;
                    current.push(c);
                }
                ')' => {
                    depth -= 1;
                    current.push(c);
                }
                ',' if depth == 0 => {
                    args.push(current.trim().to_string());
                    current = String::new();
                }
                _ => current.push(c),
            }
        }

        if !current.is_empty() {
            args.push(current.trim().to_string());
        }

        Ok(args)
    }

    // Rolling operations
    fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in (window - 1)..n {
            let sum: f64 = data[(i + 1 - window)..=i].iter().sum();
            result[i] = sum / window as f64;
        }

        result
    }

    fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in (window - 1)..n {
            let slice = &data[(i + 1 - window)..=i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    fn rolling_max(data: &[f64], window: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in (window - 1)..n {
            result[i] = data[(i + 1 - window)..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
        }

        result
    }

    fn rolling_min(data: &[f64], window: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in (window - 1)..n {
            result[i] = data[(i + 1 - window)..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
        }

        result
    }

    fn delta(data: &[f64], periods: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in periods..n {
            result[i] = data[i] - data[i - periods];
        }

        result
    }

    fn delay(data: &[f64], periods: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in periods..n {
            result[i] = data[i - periods];
        }

        result
    }

    fn rank(data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut indexed: Vec<(usize, f64)> = data
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(i, v)| (i, *v))
            .collect();

        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = vec![f64::NAN; n];
        let count = indexed.len() as f64;

        for (rank, (idx, _)) in indexed.iter().enumerate() {
            result[*idx] = (rank + 1) as f64 / count;
        }

        result
    }
}

/// Calculate Information Coefficient between factor and returns.
pub fn calculate_ic(factor: &[f64], returns: &[f64]) -> (f64, f64) {
    let n = factor.len().min(returns.len());
    let mut valid_pairs: Vec<(f64, f64)> = Vec::new();

    for i in 0..n {
        if !factor[i].is_nan() && !returns[i].is_nan() {
            valid_pairs.push((factor[i], returns[i]));
        }
    }

    if valid_pairs.len() < 3 {
        return (0.0, 1.0);
    }

    // Calculate Pearson correlation
    let n_valid = valid_pairs.len() as f64;
    let sum_x: f64 = valid_pairs.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = valid_pairs.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = valid_pairs.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = valid_pairs.iter().map(|(x, _)| x * x).sum();
    let sum_y2: f64 = valid_pairs.iter().map(|(_, y)| y * y).sum();

    let numerator = n_valid * sum_xy - sum_x * sum_y;
    let denominator = ((n_valid * sum_x2 - sum_x * sum_x) * (n_valid * sum_y2 - sum_y * sum_y)).sqrt();

    let ic = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    // Calculate p-value (approximation using t-distribution)
    let t_stat = if ic.abs() < 1.0 {
        ic * ((n_valid - 2.0) / (1.0 - ic * ic)).sqrt()
    } else {
        0.0
    };

    // Simplified p-value calculation (two-tailed)
    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));

    (ic, p_value)
}

/// Standard normal CDF approximation.
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Predefined alpha factors library.
pub fn predefined_factors() -> Vec<AlphaFactor> {
    vec![
        AlphaFactor::with_description(
            "momentum_5d".to_string(),
            "ts_delta(close, 5) / ts_delay(close, 5)".to_string(),
            "5-day price momentum (percentage change)".to_string(),
        ),
        AlphaFactor::with_description(
            "mean_reversion_20d".to_string(),
            "-1 * (close - ts_mean(close, 20)) / ts_std(close, 20)".to_string(),
            "Mean reversion z-score over 20 periods".to_string(),
        ),
        AlphaFactor::with_description(
            "volume_breakout".to_string(),
            "sign(volume / ts_mean(volume, 20) - 1)".to_string(),
            "Volume breakout signal (above average)".to_string(),
        ),
        AlphaFactor::with_description(
            "volatility_expansion".to_string(),
            "ts_std(close, 5) / ts_std(close, 20) - 1".to_string(),
            "Volatility expansion indicator".to_string(),
        ),
        AlphaFactor::with_description(
            "trend_strength".to_string(),
            "ts_mean(sign(ts_delta(close, 1)), 10)".to_string(),
            "Trend consistency (average sign of daily returns)".to_string(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_synthetic_data;

    #[test]
    fn test_evaluator_basic() {
        let data = generate_synthetic_data("TEST", 100, 42);
        let evaluator = AlphaEvaluator::new(&data);

        let result = evaluator.evaluate_expression("close").unwrap();
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_evaluator_ts_mean() {
        let data = generate_synthetic_data("TEST", 100, 42);
        let evaluator = AlphaEvaluator::new(&data);

        let result = evaluator.evaluate_expression("ts_mean(close, 10)").unwrap();
        assert_eq!(result.len(), 100);
        assert!(result[0..9].iter().all(|x| x.is_nan()));
        assert!(result[9..].iter().all(|x| !x.is_nan()));
    }

    #[test]
    fn test_evaluator_momentum() {
        let data = generate_synthetic_data("TEST", 100, 42);
        let evaluator = AlphaEvaluator::new(&data);

        let result = evaluator.evaluate_expression("ts_delta(close, 5) / ts_delay(close, 5)").unwrap();
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_validate_expression() {
        let data = generate_synthetic_data("TEST", 100, 42);
        let evaluator = AlphaEvaluator::new(&data);

        assert!(evaluator.validate("ts_mean(close, 10)"));
        assert!(evaluator.validate("rank(volume)"));
        assert!(!evaluator.validate("import os"));
        assert!(!evaluator.validate("eval(x)"));
    }

    #[test]
    fn test_calculate_ic() {
        let factor = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let returns = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let (ic, p_value) = calculate_ic(&factor, &returns);
        assert!(ic > 0.99); // Perfect positive correlation
        assert!(p_value < 0.01); // Highly significant
    }

    #[test]
    fn test_predefined_factors() {
        let factors = predefined_factors();
        assert!(!factors.is_empty());

        let data = generate_synthetic_data("TEST", 100, 42);
        let evaluator = AlphaEvaluator::new(&data);

        for factor in &factors {
            let result = evaluator.evaluate(factor);
            assert!(result.is_ok(), "Factor {} failed to evaluate", factor.name);
        }
    }
}
