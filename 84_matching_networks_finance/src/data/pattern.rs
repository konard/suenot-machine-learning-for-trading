//! Market Pattern definitions
//!
//! Defines the market patterns that the Matching Network learns to recognize.

use std::fmt;

/// Market patterns for classification
///
/// These patterns represent distinct market behaviors that traders
/// commonly look for when making decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MarketPattern {
    /// Price continues in the direction of the existing trend
    TrendContinuation = 0,
    /// Price reverses from the existing trend
    TrendReversal = 1,
    /// Price breaks through a support/resistance level
    Breakout = 2,
    /// False breakout that quickly reverses
    FalseBreakout = 3,
    /// Price moves sideways in a range
    Consolidation = 4,
}

impl MarketPattern {
    /// Get all pattern variants
    pub fn all() -> &'static [MarketPattern] {
        &[
            MarketPattern::TrendContinuation,
            MarketPattern::TrendReversal,
            MarketPattern::Breakout,
            MarketPattern::FalseBreakout,
            MarketPattern::Consolidation,
        ]
    }

    /// Get the number of patterns
    pub fn count() -> usize {
        5
    }

    /// Convert to class index
    pub fn to_index(self) -> usize {
        self as usize
    }

    /// Convert from class index
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(MarketPattern::TrendContinuation),
            1 => Some(MarketPattern::TrendReversal),
            2 => Some(MarketPattern::Breakout),
            3 => Some(MarketPattern::FalseBreakout),
            4 => Some(MarketPattern::Consolidation),
            _ => None,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            MarketPattern::TrendContinuation => {
                "Price continues in the direction of the existing trend"
            }
            MarketPattern::TrendReversal => {
                "Price reverses from the existing trend direction"
            }
            MarketPattern::Breakout => {
                "Price breaks through a support or resistance level"
            }
            MarketPattern::FalseBreakout => {
                "Breakout attempt that fails and reverses"
            }
            MarketPattern::Consolidation => {
                "Price moves sideways within a range"
            }
        }
    }

    /// Get the typical trading action for this pattern
    pub fn typical_action(&self) -> TradingAction {
        match self {
            MarketPattern::TrendContinuation => TradingAction::FollowTrend,
            MarketPattern::TrendReversal => TradingAction::ReversePosition,
            MarketPattern::Breakout => TradingAction::EnterBreakout,
            MarketPattern::FalseBreakout => TradingAction::FadeBreakout,
            MarketPattern::Consolidation => TradingAction::WaitOrRange,
        }
    }
}

impl fmt::Display for MarketPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            MarketPattern::TrendContinuation => "Trend Continuation",
            MarketPattern::TrendReversal => "Trend Reversal",
            MarketPattern::Breakout => "Breakout",
            MarketPattern::FalseBreakout => "False Breakout",
            MarketPattern::Consolidation => "Consolidation",
        };
        write!(f, "{}", name)
    }
}

/// Trading actions associated with patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAction {
    /// Follow the current trend (buy in uptrend, sell in downtrend)
    FollowTrend,
    /// Reverse position (sell if long, buy if short)
    ReversePosition,
    /// Enter in the direction of the breakout
    EnterBreakout,
    /// Fade the breakout (enter opposite direction)
    FadeBreakout,
    /// Wait for clearer signal or trade the range
    WaitOrRange,
}

impl fmt::Display for TradingAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            TradingAction::FollowTrend => "Follow Trend",
            TradingAction::ReversePosition => "Reverse Position",
            TradingAction::EnterBreakout => "Enter Breakout",
            TradingAction::FadeBreakout => "Fade Breakout",
            TradingAction::WaitOrRange => "Wait/Range Trade",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_count() {
        assert_eq!(MarketPattern::count(), 5);
        assert_eq!(MarketPattern::all().len(), 5);
    }

    #[test]
    fn test_pattern_index_conversion() {
        for pattern in MarketPattern::all() {
            let index = pattern.to_index();
            let recovered = MarketPattern::from_index(index).unwrap();
            assert_eq!(*pattern, recovered);
        }
    }

    #[test]
    fn test_invalid_index() {
        assert!(MarketPattern::from_index(100).is_none());
    }

    #[test]
    fn test_pattern_display() {
        assert_eq!(format!("{}", MarketPattern::Breakout), "Breakout");
    }

    #[test]
    fn test_typical_actions() {
        assert_eq!(
            MarketPattern::TrendContinuation.typical_action(),
            TradingAction::FollowTrend
        );
        assert_eq!(
            MarketPattern::FalseBreakout.typical_action(),
            TradingAction::FadeBreakout
        );
    }
}
