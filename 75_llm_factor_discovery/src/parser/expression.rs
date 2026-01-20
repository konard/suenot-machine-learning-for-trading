//! Factor expression AST and parser
//!
//! This module defines the abstract syntax tree for factor expressions
//! and provides parsing capabilities.

use std::fmt;
use thiserror::Error;

/// Errors that can occur during factor expression parsing
#[derive(Error, Debug, Clone)]
pub enum FactorExprError {
    #[error("Unexpected token: {0}")]
    UnexpectedToken(String),

    #[error("Unknown function: {0}")]
    UnknownFunction(String),

    #[error("Invalid argument count for {function}: expected {expected}, got {got}")]
    InvalidArgCount {
        function: String,
        expected: usize,
        got: usize,
    },

    #[error("Unclosed parenthesis")]
    UnclosedParen,

    #[error("Empty expression")]
    EmptyExpression,

    #[error("Invalid number: {0}")]
    InvalidNumber(String),

    #[error("Expression too deep (max depth: {0})")]
    TooDeep(usize),

    #[error("Parse error at position {position}: {message}")]
    ParseError { position: usize, message: String },
}

/// Supported factor functions
#[derive(Debug, Clone, PartialEq)]
pub enum FactorFunction {
    // Cross-sectional operations
    Rank,
    Zscore,
    Scale,
    Demean,

    // Time series operations
    TsRank,
    TsSum,
    TsMean,
    TsStd,
    TsMin,
    TsMax,
    TsArgmax,
    TsArgmin,
    TsDelay,
    TsDelta,
    TsDecay,

    // Price-based
    Returns,
    LogReturns,
    Volatility,

    // Correlation/covariance
    Correlation,
    Covariance,

    // Mathematical operations
    Abs,
    Sign,
    Log,
    Sqrt,
    Power,

    // Comparison operations
    Max,
    Min,

    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,

    // Conditional
    IfElse,
}

impl FactorFunction {
    /// Parse function name from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rank" => Some(Self::Rank),
            "zscore" | "z_score" => Some(Self::Zscore),
            "scale" => Some(Self::Scale),
            "demean" => Some(Self::Demean),

            "ts_rank" | "tsrank" => Some(Self::TsRank),
            "ts_sum" | "tssum" => Some(Self::TsSum),
            "ts_mean" | "tsmean" | "ts_avg" => Some(Self::TsMean),
            "ts_std" | "tsstd" | "ts_stddev" => Some(Self::TsStd),
            "ts_min" | "tsmin" => Some(Self::TsMin),
            "ts_max" | "tsmax" => Some(Self::TsMax),
            "ts_argmax" | "tsargmax" => Some(Self::TsArgmax),
            "ts_argmin" | "tsargmin" => Some(Self::TsArgmin),
            "ts_delay" | "delay" | "lag" => Some(Self::TsDelay),
            "ts_delta" | "delta" => Some(Self::TsDelta),
            "ts_decay" | "decay_linear" => Some(Self::TsDecay),

            "returns" | "return" | "ret" => Some(Self::Returns),
            "log_returns" | "logreturns" | "logret" => Some(Self::LogReturns),
            "volatility" | "vol" | "std" => Some(Self::Volatility),

            "correlation" | "corr" => Some(Self::Correlation),
            "covariance" | "cov" => Some(Self::Covariance),

            "abs" => Some(Self::Abs),
            "sign" | "signum" => Some(Self::Sign),
            "log" | "ln" => Some(Self::Log),
            "sqrt" => Some(Self::Sqrt),
            "power" | "pow" => Some(Self::Power),

            "max" => Some(Self::Max),
            "min" => Some(Self::Min),

            "add" => Some(Self::Add),
            "sub" => Some(Self::Sub),
            "mul" => Some(Self::Mul),
            "div" => Some(Self::Div),

            "if" | "ifelse" | "if_else" => Some(Self::IfElse),

            _ => None,
        }
    }

    /// Get the expected number of arguments for this function
    pub fn arg_count(&self) -> (usize, usize) {
        // Returns (min_args, max_args)
        match self {
            // Unary operations
            Self::Rank | Self::Zscore | Self::Scale | Self::Demean => (1, 1),
            Self::Abs | Self::Sign | Self::Log | Self::Sqrt => (1, 1),

            // Binary operations
            Self::Add | Self::Sub | Self::Mul | Self::Div => (2, 2),
            Self::Max | Self::Min => (2, 2),
            Self::Power => (2, 2),

            // Time series operations with window
            Self::TsRank
            | Self::TsSum
            | Self::TsMean
            | Self::TsStd
            | Self::TsMin
            | Self::TsMax
            | Self::TsArgmax
            | Self::TsArgmin
            | Self::TsDelay
            | Self::TsDelta
            | Self::TsDecay => (2, 2),

            // Price operations with period
            Self::Returns | Self::LogReturns | Self::Volatility => (2, 2),

            // Correlation with window
            Self::Correlation | Self::Covariance => (3, 3),

            // Conditional
            Self::IfElse => (3, 3),
        }
    }
}

impl fmt::Display for FactorFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rank => write!(f, "rank"),
            Self::Zscore => write!(f, "zscore"),
            Self::Scale => write!(f, "scale"),
            Self::Demean => write!(f, "demean"),
            Self::TsRank => write!(f, "ts_rank"),
            Self::TsSum => write!(f, "ts_sum"),
            Self::TsMean => write!(f, "ts_mean"),
            Self::TsStd => write!(f, "ts_std"),
            Self::TsMin => write!(f, "ts_min"),
            Self::TsMax => write!(f, "ts_max"),
            Self::TsArgmax => write!(f, "ts_argmax"),
            Self::TsArgmin => write!(f, "ts_argmin"),
            Self::TsDelay => write!(f, "ts_delay"),
            Self::TsDelta => write!(f, "ts_delta"),
            Self::TsDecay => write!(f, "ts_decay"),
            Self::Returns => write!(f, "returns"),
            Self::LogReturns => write!(f, "log_returns"),
            Self::Volatility => write!(f, "volatility"),
            Self::Correlation => write!(f, "correlation"),
            Self::Covariance => write!(f, "covariance"),
            Self::Abs => write!(f, "abs"),
            Self::Sign => write!(f, "sign"),
            Self::Log => write!(f, "log"),
            Self::Sqrt => write!(f, "sqrt"),
            Self::Power => write!(f, "power"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::Mul => write!(f, "mul"),
            Self::Div => write!(f, "div"),
            Self::IfElse => write!(f, "if_else"),
        }
    }
}

/// A factor expression AST node
#[derive(Debug, Clone, PartialEq)]
pub enum FactorExpr {
    /// A numeric constant
    Number(f64),

    /// A variable reference (e.g., "close", "volume", "high")
    Variable(String),

    /// A function call with arguments
    Function {
        name: FactorFunction,
        args: Vec<FactorExpr>,
    },
}

impl FactorExpr {
    /// Parse a factor expression from a string
    pub fn parse(input: &str) -> Result<Self, FactorExprError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(FactorExprError::EmptyExpression);
        }

        Self::parse_expr(input, 0)
    }

    fn parse_expr(input: &str, depth: usize) -> Result<Self, FactorExprError> {
        if depth > crate::MAX_EXPRESSION_DEPTH {
            return Err(FactorExprError::TooDeep(crate::MAX_EXPRESSION_DEPTH));
        }

        let input = input.trim();

        // Check if it's a number
        if let Ok(n) = input.parse::<f64>() {
            return Ok(FactorExpr::Number(n));
        }

        // Check if it's an integer (for window sizes)
        if let Ok(n) = input.parse::<i64>() {
            return Ok(FactorExpr::Number(n as f64));
        }

        // Check if it's a function call
        if let Some(paren_pos) = input.find('(') {
            let func_name = input[..paren_pos].trim();

            // Find matching closing parenthesis
            let close_paren = Self::find_matching_paren(input, paren_pos)?;

            if close_paren != input.len() - 1 {
                return Err(FactorExprError::ParseError {
                    position: close_paren,
                    message: "Unexpected characters after closing parenthesis".to_string(),
                });
            }

            let func = FactorFunction::from_str(func_name)
                .ok_or_else(|| FactorExprError::UnknownFunction(func_name.to_string()))?;

            let args_str = &input[paren_pos + 1..close_paren];
            let args = Self::parse_args(args_str, depth + 1)?;

            // Validate argument count
            let (min_args, max_args) = func.arg_count();
            if args.len() < min_args || args.len() > max_args {
                return Err(FactorExprError::InvalidArgCount {
                    function: func.to_string(),
                    expected: min_args,
                    got: args.len(),
                });
            }

            return Ok(FactorExpr::Function { name: func, args });
        }

        // It's a variable
        if Self::is_valid_variable_name(input) {
            return Ok(FactorExpr::Variable(input.to_string()));
        }

        Err(FactorExprError::UnexpectedToken(input.to_string()))
    }

    fn find_matching_paren(input: &str, open_pos: usize) -> Result<usize, FactorExprError> {
        let mut depth = 0;
        for (i, c) in input[open_pos..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(open_pos + i);
                    }
                }
                _ => {}
            }
        }
        Err(FactorExprError::UnclosedParen)
    }

    fn parse_args(args_str: &str, depth: usize) -> Result<Vec<FactorExpr>, FactorExprError> {
        if args_str.trim().is_empty() {
            return Ok(vec![]);
        }

        let mut args = vec![];
        let mut current_arg = String::new();
        let mut paren_depth = 0;

        for c in args_str.chars() {
            match c {
                '(' => {
                    paren_depth += 1;
                    current_arg.push(c);
                }
                ')' => {
                    paren_depth -= 1;
                    current_arg.push(c);
                }
                ',' if paren_depth == 0 => {
                    let arg = Self::parse_expr(&current_arg, depth)?;
                    args.push(arg);
                    current_arg.clear();
                }
                _ => {
                    current_arg.push(c);
                }
            }
        }

        // Don't forget the last argument
        if !current_arg.trim().is_empty() {
            let arg = Self::parse_expr(&current_arg, depth)?;
            args.push(arg);
        }

        Ok(args)
    }

    fn is_valid_variable_name(name: &str) -> bool {
        if name.is_empty() {
            return false;
        }

        let first = name.chars().next().unwrap();
        if !first.is_alphabetic() && first != '_' {
            return false;
        }

        name.chars()
            .all(|c| c.is_alphanumeric() || c == '_')
    }

    /// Get all variables referenced in this expression
    pub fn variables(&self) -> Vec<String> {
        match self {
            FactorExpr::Number(_) => vec![],
            FactorExpr::Variable(name) => vec![name.clone()],
            FactorExpr::Function { args, .. } => {
                args.iter()
                    .flat_map(|arg| arg.variables())
                    .collect()
            }
        }
    }

    /// Get the depth of this expression tree
    pub fn depth(&self) -> usize {
        match self {
            FactorExpr::Number(_) | FactorExpr::Variable(_) => 1,
            FactorExpr::Function { args, .. } => {
                1 + args.iter().map(|arg| arg.depth()).max().unwrap_or(0)
            }
        }
    }
}

impl fmt::Display for FactorExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FactorExpr::Number(n) => {
                if n.fract() == 0.0 && n.abs() < 1e10 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            FactorExpr::Variable(name) => write!(f, "{}", name),
            FactorExpr::Function { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_number() {
        let expr = FactorExpr::parse("42").unwrap();
        assert_eq!(expr, FactorExpr::Number(42.0));
    }

    #[test]
    fn test_parse_variable() {
        let expr = FactorExpr::parse("close").unwrap();
        assert_eq!(expr, FactorExpr::Variable("close".to_string()));
    }

    #[test]
    fn test_parse_simple_function() {
        let expr = FactorExpr::parse("rank(close)").unwrap();
        assert!(matches!(
            expr,
            FactorExpr::Function {
                name: FactorFunction::Rank,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_nested_function() {
        let expr = FactorExpr::parse("rank(ts_mean(returns(close, 5), 20))").unwrap();
        assert!(matches!(
            expr,
            FactorExpr::Function {
                name: FactorFunction::Rank,
                ..
            }
        ));
        assert_eq!(expr.depth(), 4);
    }

    #[test]
    fn test_parse_correlation() {
        let expr = FactorExpr::parse("correlation(returns(close, 1), volume, 60)").unwrap();
        let vars = expr.variables();
        assert!(vars.contains(&"close".to_string()));
        assert!(vars.contains(&"volume".to_string()));
    }

    #[test]
    fn test_unknown_function() {
        let result = FactorExpr::parse("unknown_func(close)");
        assert!(matches!(result, Err(FactorExprError::UnknownFunction(_))));
    }

    #[test]
    fn test_display() {
        let expr = FactorExpr::parse("rank(ts_mean(close, 20))").unwrap();
        assert_eq!(expr.to_string(), "rank(ts_mean(close, 20))");
    }
}
