//! Trading state and action definitions.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trading actions available to the agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradingAction {
    /// Short position (bet on price decrease)
    Short = 0,
    /// Hold / neutral position
    Hold = 1,
    /// Long position (bet on price increase)
    Long = 2,
}

impl TradingAction {
    /// Get all possible actions
    pub fn all() -> [TradingAction; 3] {
        [TradingAction::Short, TradingAction::Hold, TradingAction::Long]
    }

    /// Number of possible actions
    pub fn count() -> usize {
        3
    }

    /// Convert from index
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(TradingAction::Short),
            1 => Some(TradingAction::Hold),
            2 => Some(TradingAction::Long),
            _ => None,
        }
    }

    /// Convert to index
    pub fn to_index(self) -> usize {
        self as usize
    }

    /// Get position multiplier (-1 for short, 0 for hold, 1 for long)
    pub fn position(&self) -> f64 {
        match self {
            TradingAction::Short => -1.0,
            TradingAction::Hold => 0.0,
            TradingAction::Long => 1.0,
        }
    }
}

impl std::fmt::Display for TradingAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingAction::Short => write!(f, "SHORT"),
            TradingAction::Hold => write!(f, "HOLD"),
            TradingAction::Long => write!(f, "LONG"),
        }
    }
}

/// Complete trading state including market data and portfolio info
#[derive(Debug, Clone)]
pub struct TradingState {
    /// Market features (from MarketData)
    pub market_features: Array1<f64>,
    /// Current position (-1, 0, 1)
    pub position: f64,
    /// Unrealized PnL (percentage)
    pub unrealized_pnl: f64,
    /// Time in current position (normalized)
    pub time_in_position: f64,
    /// Current step in episode (normalized)
    pub episode_progress: f64,
}

impl TradingState {
    /// Create a new trading state
    pub fn new(
        market_features: Array1<f64>,
        position: f64,
        unrealized_pnl: f64,
        time_in_position: f64,
        episode_progress: f64,
    ) -> Self {
        Self {
            market_features,
            position,
            unrealized_pnl,
            time_in_position,
            episode_progress,
        }
    }

    /// Convert state to a flat array for neural network input
    pub fn to_array(&self) -> Array1<f64> {
        let mut features = self.market_features.to_vec();
        features.push(self.position);
        features.push(self.unrealized_pnl.clamp(-1.0, 1.0));
        features.push(self.time_in_position);
        features.push(self.episode_progress);
        Array1::from_vec(features)
    }

    /// Get the size of the state vector
    pub fn size(market_feature_size: usize) -> usize {
        market_feature_size + 4 // market features + position + unrealized_pnl + time_in_pos + progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_conversion() {
        assert_eq!(TradingAction::from_index(0), Some(TradingAction::Short));
        assert_eq!(TradingAction::from_index(1), Some(TradingAction::Hold));
        assert_eq!(TradingAction::from_index(2), Some(TradingAction::Long));
        assert_eq!(TradingAction::from_index(3), None);

        assert_eq!(TradingAction::Short.to_index(), 0);
        assert_eq!(TradingAction::Hold.to_index(), 1);
        assert_eq!(TradingAction::Long.to_index(), 2);
    }

    #[test]
    fn test_position() {
        assert_eq!(TradingAction::Short.position(), -1.0);
        assert_eq!(TradingAction::Hold.position(), 0.0);
        assert_eq!(TradingAction::Long.position(), 1.0);
    }

    #[test]
    fn test_state_to_array() {
        let market_features = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        let state = TradingState::new(market_features, 1.0, 0.05, 0.1, 0.5);

        let array = state.to_array();
        assert_eq!(array.len(), 11); // 7 market features + 4 state features
    }
}
