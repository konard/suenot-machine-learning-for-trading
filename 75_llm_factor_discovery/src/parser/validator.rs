//! Factor expression validator
//!
//! Validates factor expressions for correctness and potential issues.

use super::expression::{FactorExpr, FactorFunction};
use thiserror::Error;

/// Validation errors
#[derive(Error, Debug, Clone)]
pub enum ValidationError {
    #[error("Unknown variable: {0}")]
    UnknownVariable(String),

    #[error("Invalid window size: {0} (must be positive)")]
    InvalidWindowSize(i64),

    #[error("Expression too complex: depth {0} exceeds maximum {1}")]
    TooComplex(usize, usize),

    #[error("Division by zero possible")]
    DivisionByZero,

    #[error("Negative value in sqrt")]
    NegativeSqrt,

    #[error("Window size too large: {0} (maximum: {1})")]
    WindowTooLarge(i64, i64),
}

/// Result of validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
    pub complexity_score: f64,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            complexity_score: 0.0,
        }
    }

    pub fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for factor validation
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Known variables in the dataset
    pub known_variables: Vec<String>,
    /// Maximum allowed expression depth
    pub max_depth: usize,
    /// Maximum allowed window size
    pub max_window_size: i64,
    /// Whether to allow unknown variables
    pub allow_unknown_variables: bool,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            known_variables: vec![
                "open".to_string(),
                "high".to_string(),
                "low".to_string(),
                "close".to_string(),
                "volume".to_string(),
                "vwap".to_string(),
                "returns".to_string(),
            ],
            max_depth: 10,
            max_window_size: 252,
            allow_unknown_variables: false,
        }
    }
}

/// Factor expression validator
pub struct FactorValidator {
    config: ValidatorConfig,
}

impl FactorValidator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidatorConfig::default(),
        }
    }

    /// Create a new validator with custom configuration
    pub fn with_config(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Validate a factor expression
    pub fn validate(&self, expr: &FactorExpr) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check depth
        let depth = expr.depth();
        if depth > self.config.max_depth {
            result.add_error(ValidationError::TooComplex(depth, self.config.max_depth));
        }

        // Calculate complexity score
        result.complexity_score = self.calculate_complexity(expr);

        // Validate recursively
        self.validate_recursive(expr, &mut result);

        result
    }

    fn validate_recursive(&self, expr: &FactorExpr, result: &mut ValidationResult) {
        match expr {
            FactorExpr::Number(n) => {
                if n.is_nan() {
                    result.add_warning("Expression contains NaN".to_string());
                }
                if n.is_infinite() {
                    result.add_warning("Expression contains infinity".to_string());
                }
            }
            FactorExpr::Variable(name) => {
                if !self.config.allow_unknown_variables
                    && !self.config.known_variables.contains(name)
                {
                    result.add_error(ValidationError::UnknownVariable(name.clone()));
                }
            }
            FactorExpr::Function { name, args } => {
                // Validate arguments recursively
                for arg in args {
                    self.validate_recursive(arg, result);
                }

                // Check window sizes for time series functions
                match name {
                    FactorFunction::TsRank
                    | FactorFunction::TsSum
                    | FactorFunction::TsMean
                    | FactorFunction::TsStd
                    | FactorFunction::TsMin
                    | FactorFunction::TsMax
                    | FactorFunction::TsArgmax
                    | FactorFunction::TsArgmin
                    | FactorFunction::TsDelay
                    | FactorFunction::TsDelta
                    | FactorFunction::TsDecay
                    | FactorFunction::Returns
                    | FactorFunction::LogReturns
                    | FactorFunction::Volatility => {
                        if let Some(FactorExpr::Number(window)) = args.get(1) {
                            let window = *window as i64;
                            if window <= 0 {
                                result.add_error(ValidationError::InvalidWindowSize(window));
                            } else if window > self.config.max_window_size {
                                result.add_error(ValidationError::WindowTooLarge(
                                    window,
                                    self.config.max_window_size,
                                ));
                            }
                        }
                    }
                    FactorFunction::Correlation | FactorFunction::Covariance => {
                        if let Some(FactorExpr::Number(window)) = args.get(2) {
                            let window = *window as i64;
                            if window <= 0 {
                                result.add_error(ValidationError::InvalidWindowSize(window));
                            } else if window > self.config.max_window_size {
                                result.add_error(ValidationError::WindowTooLarge(
                                    window,
                                    self.config.max_window_size,
                                ));
                            }
                            if window < 2 {
                                result.add_warning(
                                    "Correlation with window < 2 is meaningless".to_string(),
                                );
                            }
                        }
                    }
                    FactorFunction::Div => {
                        // Check for potential division by zero
                        if let Some(FactorExpr::Number(n)) = args.get(1) {
                            if n.abs() < 1e-10 {
                                result.add_error(ValidationError::DivisionByZero);
                            }
                        }
                    }
                    FactorFunction::Sqrt => {
                        // Check for potential negative values
                        if let Some(FactorExpr::Number(n)) = args.first() {
                            if *n < 0.0 {
                                result.add_error(ValidationError::NegativeSqrt);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn calculate_complexity(&self, expr: &FactorExpr) -> f64 {
        match expr {
            FactorExpr::Number(_) => 0.1,
            FactorExpr::Variable(_) => 0.2,
            FactorExpr::Function { name, args } => {
                let base_cost = match name {
                    // Simple operations
                    FactorFunction::Abs
                    | FactorFunction::Sign
                    | FactorFunction::Add
                    | FactorFunction::Sub
                    | FactorFunction::Mul
                    | FactorFunction::Div => 0.5,

                    // Cross-sectional operations
                    FactorFunction::Rank | FactorFunction::Zscore => 1.0,

                    // Time series operations
                    FactorFunction::TsSum
                    | FactorFunction::TsMean
                    | FactorFunction::TsMin
                    | FactorFunction::TsMax => 1.5,

                    FactorFunction::TsStd | FactorFunction::TsRank => 2.0,

                    // Complex operations
                    FactorFunction::Correlation | FactorFunction::Covariance => 3.0,

                    _ => 1.0,
                };

                let args_cost: f64 = args.iter().map(|a| self.calculate_complexity(a)).sum();

                base_cost + args_cost
            }
        }
    }
}

impl Default for FactorValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_expression() {
        let validator = FactorValidator::new();
        let expr = FactorExpr::parse("rank(close)").unwrap();
        let result = validator.validate(&expr);
        assert!(result.is_valid);
    }

    #[test]
    fn test_unknown_variable() {
        let validator = FactorValidator::new();
        let expr = FactorExpr::parse("rank(unknown_var)").unwrap();
        let result = validator.validate(&expr);
        assert!(!result.is_valid);
        assert!(matches!(
            result.errors.first(),
            Some(ValidationError::UnknownVariable(_))
        ));
    }

    #[test]
    fn test_invalid_window_size() {
        let validator = FactorValidator::new();
        let expr = FactorExpr::parse("ts_mean(close, -5)").unwrap();
        let result = validator.validate(&expr);
        assert!(!result.is_valid);
        assert!(matches!(
            result.errors.first(),
            Some(ValidationError::InvalidWindowSize(_))
        ));
    }

    #[test]
    fn test_complexity_calculation() {
        let validator = FactorValidator::new();

        let simple_expr = FactorExpr::parse("rank(close)").unwrap();
        let complex_expr =
            FactorExpr::parse("zscore(correlation(returns(close, 1), volume, 60))").unwrap();

        let simple_result = validator.validate(&simple_expr);
        let complex_result = validator.validate(&complex_expr);

        assert!(complex_result.complexity_score > simple_result.complexity_score);
    }
}
