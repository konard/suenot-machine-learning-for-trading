//! Trading signals and signal generation.

use crate::MarketRegime;
use serde::{Deserialize, Serialize};

/// Trading signal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Signal {
    /// Strong buy signal
    StrongBuy,
    /// Weak buy signal
    Buy,
    /// Hold current position
    Hold,
    /// Weak sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl Signal {
    /// Get position sizing multiplier for this signal.
    pub fn position_multiplier(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Hold => 0.0,
            Signal::Sell => -0.5,
            Signal::StrongSell => -1.0,
        }
    }

    /// Check if this is a buy signal.
    pub fn is_buy(&self) -> bool {
        matches!(self, Signal::StrongBuy | Signal::Buy)
    }

    /// Check if this is a sell signal.
    pub fn is_sell(&self) -> bool {
        matches!(self, Signal::StrongSell | Signal::Sell)
    }

    /// Convert from market regime prediction.
    pub fn from_regime(regime: MarketRegime) -> Self {
        match regime {
            MarketRegime::StrongUptrend => Signal::StrongBuy,
            MarketRegime::WeakUptrend => Signal::Buy,
            MarketRegime::Sideways => Signal::Hold,
            MarketRegime::WeakDowntrend => Signal::Sell,
            MarketRegime::StrongDowntrend => Signal::StrongSell,
        }
    }

    /// Convert from regime with confidence adjustment.
    pub fn from_regime_with_confidence(regime: MarketRegime, confidence: f64) -> Self {
        // Low confidence -> Hold
        if confidence < 0.3 {
            return Signal::Hold;
        }

        // Medium confidence -> weaker signals
        if confidence < 0.6 {
            return match regime {
                MarketRegime::StrongUptrend => Signal::Buy,
                MarketRegime::WeakUptrend => Signal::Hold,
                MarketRegime::Sideways => Signal::Hold,
                MarketRegime::WeakDowntrend => Signal::Hold,
                MarketRegime::StrongDowntrend => Signal::Sell,
            };
        }

        // High confidence -> full signals
        Signal::from_regime(regime)
    }
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::StrongBuy => write!(f, "STRONG BUY ⬆️⬆️"),
            Signal::Buy => write!(f, "BUY ⬆️"),
            Signal::Hold => write!(f, "HOLD ➡️"),
            Signal::Sell => write!(f, "SELL ⬇️"),
            Signal::StrongSell => write!(f, "STRONG SELL ⬇️⬇️"),
        }
    }
}

/// A timestamped trading signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedSignal {
    /// The signal
    pub signal: Signal,
    /// Timestamp (Unix millis)
    pub timestamp: i64,
    /// Asset symbol
    pub symbol: String,
    /// Predicted regime
    pub regime: MarketRegime,
    /// Confidence score
    pub confidence: f64,
    /// Current price at signal generation
    pub price: f64,
}

impl TimestampedSignal {
    /// Create a new timestamped signal.
    pub fn new(
        signal: Signal,
        symbol: String,
        regime: MarketRegime,
        confidence: f64,
        price: f64,
    ) -> Self {
        Self {
            signal,
            timestamp: chrono::Utc::now().timestamp_millis(),
            symbol,
            regime,
            confidence,
            price,
        }
    }
}

/// Signal filter for risk management.
#[derive(Debug, Clone)]
pub struct SignalFilter {
    /// Minimum confidence to emit signal
    pub min_confidence: f64,
    /// Maximum consecutive same-direction signals
    pub max_consecutive: usize,
    /// Cooldown period between signals (seconds)
    pub cooldown_secs: u64,
    /// Last signal timestamp
    last_signal_time: Option<i64>,
    /// Count of consecutive signals
    consecutive_count: usize,
    /// Last signal direction
    last_direction: Option<bool>, // true = buy, false = sell
}

impl Default for SignalFilter {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_consecutive: 3,
            cooldown_secs: 60,
            last_signal_time: None,
            consecutive_count: 0,
            last_direction: None,
        }
    }
}

impl SignalFilter {
    /// Create a new signal filter.
    pub fn new(min_confidence: f64, max_consecutive: usize, cooldown_secs: u64) -> Self {
        Self {
            min_confidence,
            max_consecutive,
            cooldown_secs,
            last_signal_time: None,
            consecutive_count: 0,
            last_direction: None,
        }
    }

    /// Filter a signal based on rules.
    pub fn filter(&mut self, signal: Signal, confidence: f64) -> Signal {
        // Check confidence threshold
        if confidence < self.min_confidence && signal != Signal::Hold {
            return Signal::Hold;
        }

        // Check cooldown
        let now = chrono::Utc::now().timestamp();
        if let Some(last_time) = self.last_signal_time {
            if (now - last_time) < self.cooldown_secs as i64 {
                return Signal::Hold;
            }
        }

        // Check consecutive signals
        let is_buy = signal.is_buy();
        let is_sell = signal.is_sell();

        if is_buy || is_sell {
            let current_direction = is_buy;

            if self.last_direction == Some(current_direction) {
                self.consecutive_count += 1;
                if self.consecutive_count > self.max_consecutive {
                    return Signal::Hold;
                }
            } else {
                self.consecutive_count = 1;
                self.last_direction = Some(current_direction);
            }

            self.last_signal_time = Some(now);
        }

        signal
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.last_signal_time = None;
        self.consecutive_count = 0;
        self.last_direction = None;
    }
}

/// Signal aggregator for combining multiple signals.
#[derive(Debug, Clone, Default)]
pub struct SignalAggregator {
    signals: Vec<(Signal, f64)>, // (signal, weight)
}

impl SignalAggregator {
    /// Create a new aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a signal with weight.
    pub fn add(&mut self, signal: Signal, weight: f64) {
        self.signals.push((signal, weight));
    }

    /// Get aggregated signal.
    pub fn aggregate(&self) -> Signal {
        if self.signals.is_empty() {
            return Signal::Hold;
        }

        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;

        for (signal, weight) in &self.signals {
            let score = signal.position_multiplier();
            weighted_score += score * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return Signal::Hold;
        }

        let avg_score = weighted_score / total_weight;

        if avg_score > 0.7 {
            Signal::StrongBuy
        } else if avg_score > 0.2 {
            Signal::Buy
        } else if avg_score < -0.7 {
            Signal::StrongSell
        } else if avg_score < -0.2 {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    /// Clear all signals.
    pub fn clear(&mut self) {
        self.signals.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_regime() {
        assert_eq!(
            Signal::from_regime(MarketRegime::StrongUptrend),
            Signal::StrongBuy
        );
        assert_eq!(Signal::from_regime(MarketRegime::Sideways), Signal::Hold);
        assert_eq!(
            Signal::from_regime(MarketRegime::StrongDowntrend),
            Signal::StrongSell
        );
    }

    #[test]
    fn test_signal_aggregator() {
        let mut agg = SignalAggregator::new();
        agg.add(Signal::StrongBuy, 1.0);
        agg.add(Signal::Buy, 1.0);
        agg.add(Signal::Hold, 0.5);

        let result = agg.aggregate();
        assert!(result.is_buy());
    }

    #[test]
    fn test_signal_filter() {
        let mut filter = SignalFilter::new(0.5, 2, 0);

        // Low confidence should be filtered
        let result = filter.filter(Signal::Buy, 0.3);
        assert_eq!(result, Signal::Hold);

        // High confidence should pass
        let result = filter.filter(Signal::Buy, 0.8);
        assert_eq!(result, Signal::Buy);
    }
}
