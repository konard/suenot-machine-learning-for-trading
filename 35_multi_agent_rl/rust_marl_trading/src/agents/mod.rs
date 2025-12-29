//! Trading agents module
//!
//! Provides different types of trading agents for multi-agent simulation.

mod base;
mod trend_following;
mod mean_reversion;
mod noise_trader;
mod market_maker;
mod informed;
mod rl_agent;

pub use base::{Agent, AgentAction, AgentId, AgentState};
pub use trend_following::TrendFollowingAgent;
pub use mean_reversion::MeanReversionAgent;
pub use noise_trader::NoiseTrader;
pub use market_maker::MarketMakerAgent;
pub use informed::InformedTrader;
pub use rl_agent::RLAgent;
