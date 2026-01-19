//! Trading Agents Module
//!
//! This module provides various agent types that participate in the market simulation:
//! - Value Investor: Buys below fundamental value, sells above
//! - Momentum Trader: Follows price trends
//! - Market Maker: Provides liquidity by quoting bid/ask

mod base;
mod value;
mod momentum;
mod market_maker;

pub use base::{Agent, AgentDecision, Action, MarketState};
pub use value::ValueInvestor;
pub use momentum::MomentumTrader;
pub use market_maker::MarketMaker;
