//! Trading signal generation based on Transfer Entropy analysis.

/// Trading signal type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Go long (buy).
    Long,
    /// Go short (sell).
    Short,
    /// No position.
    Neutral,
}

impl TradingSignal {
    /// Convert signal to position size (-1, 0, or 1).
    pub fn position(&self) -> f64 {
        match self {
            TradingSignal::Long => 1.0,
            TradingSignal::Short => -1.0,
            TradingSignal::Neutral => 0.0,
        }
    }
}

/// Generate a trading signal for a follower based on leader's return.
///
/// # Arguments
/// * `leader_return` - Most recent return of the leader asset.
/// * `te_strength` - Transfer Entropy from leader to follower (higher = more confidence).
/// * `return_threshold` - Minimum leader return magnitude to trigger a signal.
/// * `te_threshold` - Minimum TE to consider the pair valid.
pub fn generate_signal(
    leader_return: f64,
    te_strength: f64,
    return_threshold: f64,
    te_threshold: f64,
) -> TradingSignal {
    // Only trade if TE is above threshold (valid leader-follower pair)
    if te_strength < te_threshold {
        return TradingSignal::Neutral;
    }

    if leader_return > return_threshold {
        TradingSignal::Long
    } else if leader_return < -return_threshold {
        TradingSignal::Short
    } else {
        TradingSignal::Neutral
    }
}
