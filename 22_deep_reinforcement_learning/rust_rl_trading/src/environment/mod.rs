//! Trading environment module implementing OpenAI Gym-like interface.

mod trading_env;
mod trading_state;

pub use trading_env::TradingEnvironment;
pub use trading_state::{TradingAction, TradingState};
