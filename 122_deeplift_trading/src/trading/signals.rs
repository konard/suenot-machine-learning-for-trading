//! Trading signal generation.

/// Trading signal types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Neutral / hold
    Neutral,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl TradingSignal {
    /// Convert prediction value to signal.
    pub fn from_prediction(prediction: f64, threshold: f64) -> Self {
        if prediction > threshold * 2.0 {
            TradingSignal::StrongBuy
        } else if prediction > threshold {
            TradingSignal::Buy
        } else if prediction < -threshold * 2.0 {
            TradingSignal::StrongSell
        } else if prediction < -threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Neutral
        }
    }

    /// Get position size (-1, 0, or 1).
    pub fn position(&self) -> i32 {
        match self {
            TradingSignal::StrongBuy | TradingSignal::Buy => 1,
            TradingSignal::StrongSell | TradingSignal::Sell => -1,
            TradingSignal::Neutral => 0,
        }
    }

    /// Get position size with confidence (-1.0 to 1.0).
    pub fn position_with_confidence(&self) -> f64 {
        match self {
            TradingSignal::StrongBuy => 1.0,
            TradingSignal::Buy => 0.5,
            TradingSignal::Neutral => 0.0,
            TradingSignal::Sell => -0.5,
            TradingSignal::StrongSell => -1.0,
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &str {
        match self {
            TradingSignal::StrongBuy => "Strong Buy",
            TradingSignal::Buy => "Buy",
            TradingSignal::Neutral => "Neutral",
            TradingSignal::Sell => "Sell",
            TradingSignal::StrongSell => "Strong Sell",
        }
    }
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_prediction() {
        let threshold = 0.01;

        assert_eq!(
            TradingSignal::from_prediction(0.05, threshold),
            TradingSignal::StrongBuy
        );
        assert_eq!(
            TradingSignal::from_prediction(0.015, threshold),
            TradingSignal::Buy
        );
        assert_eq!(
            TradingSignal::from_prediction(0.005, threshold),
            TradingSignal::Neutral
        );
        assert_eq!(
            TradingSignal::from_prediction(-0.015, threshold),
            TradingSignal::Sell
        );
        assert_eq!(
            TradingSignal::from_prediction(-0.05, threshold),
            TradingSignal::StrongSell
        );
    }

    #[test]
    fn test_signal_position() {
        assert_eq!(TradingSignal::StrongBuy.position(), 1);
        assert_eq!(TradingSignal::Neutral.position(), 0);
        assert_eq!(TradingSignal::StrongSell.position(), -1);
    }
}
