//! Factor expression parser module
//!
//! This module provides tools for parsing factor expressions like:
//! - `rank(ts_mean(returns(close, 5), 20))`
//! - `zscore(correlation(returns(close, 1), volume, 60))`
//!
//! The parser converts string expressions into an AST that can be evaluated.

mod expression;
mod lexer;
mod validator;

pub use expression::{FactorExpr, FactorExprError, FactorFunction};
pub use lexer::{FactorLexer, Token, TokenKind};
pub use validator::{FactorValidator, ValidationResult, ValidationError};
