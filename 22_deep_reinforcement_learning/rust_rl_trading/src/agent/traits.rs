//! Agent trait definition.

use crate::environment::{TradingAction, TradingState};
use anyhow::Result;

/// Trait for RL agents
pub trait Agent {
    /// Select an action given the current state
    fn select_action(&self, state: &TradingState, epsilon: f64) -> TradingAction;

    /// Learn from experience
    fn learn(&mut self, experiences: &[(TradingState, TradingAction, f64, TradingState, bool)]);

    /// Get the current epsilon value
    fn get_epsilon(&self) -> f64;

    /// Decay epsilon
    fn decay_epsilon(&mut self);

    /// Save the agent to a file
    fn save(&self, path: &str) -> Result<()>;

    /// Load the agent from a file
    fn load(&mut self, path: &str) -> Result<()>;

    /// Get agent name
    fn name(&self) -> &str;
}
