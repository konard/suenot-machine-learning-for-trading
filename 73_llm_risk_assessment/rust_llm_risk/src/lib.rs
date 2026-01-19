//! LLM Risk Assessment for Trading
//!
//! This crate provides tools for assessing financial risk using
//! Large Language Models (LLMs) for cryptocurrency and stock trading.
//!
//! # Features
//!
//! - Fetch market data from Bybit and other exchanges
//! - Process news and text data for risk signals
//! - Generate risk scores using LLM APIs
//! - Build trading strategies based on risk assessments
//! - Backtest strategies with historical data

pub mod data;
pub mod llm;
pub mod risk;
pub mod strategy;
pub mod utils;

pub use data::{BybitClient, OHLCV, MultiSymbolData};
pub use llm::{LLMClient, LLMProvider};
pub use risk::{RiskAssessor, RiskScore, RiskLevel};
pub use strategy::{RiskBasedTrader, TradingSignal};
