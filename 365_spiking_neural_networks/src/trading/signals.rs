//! Trading signals and market regime detection

use std::collections::VecDeque;

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingSignal {
    /// Strong buy signal
    StrongBuy,
    /// Regular buy signal
    Buy,
    /// Hold position
    Hold,
    /// Regular sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl TradingSignal {
    /// Convert to position sizing factor (-1.0 to 1.0)
    pub fn to_position_factor(&self) -> f64 {
        match self {
            TradingSignal::StrongBuy => 1.0,
            TradingSignal::Buy => 0.5,
            TradingSignal::Hold => 0.0,
            TradingSignal::Sell => -0.5,
            TradingSignal::StrongSell => -1.0,
        }
    }

    /// Create from numeric value (-1.0 to 1.0)
    pub fn from_value(value: f64) -> Self {
        if value > 0.6 {
            TradingSignal::StrongBuy
        } else if value > 0.2 {
            TradingSignal::Buy
        } else if value > -0.2 {
            TradingSignal::Hold
        } else if value > -0.6 {
            TradingSignal::Sell
        } else {
            TradingSignal::StrongSell
        }
    }

    /// Check if signal suggests going long
    pub fn is_bullish(&self) -> bool {
        matches!(self, TradingSignal::StrongBuy | TradingSignal::Buy)
    }

    /// Check if signal suggests going short
    pub fn is_bearish(&self) -> bool {
        matches!(self, TradingSignal::StrongSell | TradingSignal::Sell)
    }
}

/// Signal strength indicator
#[derive(Debug, Clone, Copy)]
pub struct SignalStrength {
    /// Raw signal value (-1.0 to 1.0)
    pub value: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Number of confirming indicators
    pub confirmations: usize,
}

impl SignalStrength {
    /// Create a new signal strength
    pub fn new(value: f64, confidence: f64) -> Self {
        Self {
            value: value.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            confirmations: 0,
        }
    }

    /// Add a confirmation
    pub fn with_confirmation(mut self) -> Self {
        self.confirmations += 1;
        self
    }

    /// Get effective signal (value weighted by confidence)
    pub fn effective_signal(&self) -> f64 {
        self.value * self.confidence
    }

    /// Convert to trading signal
    pub fn to_signal(&self) -> TradingSignal {
        TradingSignal::from_value(self.effective_signal())
    }

    /// Check if signal is actionable (high enough confidence)
    pub fn is_actionable(&self, min_confidence: f64) -> bool {
        self.confidence >= min_confidence
    }
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Strong upward trend
    BullTrend,
    /// Weak upward movement
    WeakBull,
    /// Sideways/ranging market
    Ranging,
    /// Weak downward movement
    WeakBear,
    /// Strong downward trend
    BearTrend,
    /// High volatility regime
    HighVolatility,
    /// Low volatility regime
    LowVolatility,
}

impl MarketRegime {
    /// Get suggested position sizing multiplier
    pub fn position_multiplier(&self) -> f64 {
        match self {
            MarketRegime::BullTrend => 1.2,
            MarketRegime::WeakBull => 1.0,
            MarketRegime::Ranging => 0.5,
            MarketRegime::WeakBear => 0.8,
            MarketRegime::BearTrend => 0.6,
            MarketRegime::HighVolatility => 0.4,
            MarketRegime::LowVolatility => 0.8,
        }
    }

    /// Check if trending market
    pub fn is_trending(&self) -> bool {
        matches!(self, MarketRegime::BullTrend | MarketRegime::BearTrend)
    }
}

/// Market regime detector using spike patterns
#[derive(Debug)]
pub struct RegimeDetector {
    /// Recent price changes
    price_changes: VecDeque<f64>,
    /// Recent volatility measurements
    volatility: VecDeque<f64>,
    /// Window size for detection
    window_size: usize,
    /// Volatility threshold
    vol_threshold: f64,
    /// Trend threshold
    trend_threshold: f64,
}

impl RegimeDetector {
    /// Create a new regime detector
    pub fn new(window_size: usize) -> Self {
        Self {
            price_changes: VecDeque::with_capacity(window_size),
            volatility: VecDeque::with_capacity(window_size),
            window_size,
            vol_threshold: 0.02,  // 2% volatility threshold
            trend_threshold: 0.01, // 1% trend threshold
        }
    }

    /// Update with new price data
    pub fn update(&mut self, price_change: f64) {
        if self.price_changes.len() >= self.window_size {
            self.price_changes.pop_front();
        }
        self.price_changes.push_back(price_change);

        // Update volatility
        if self.price_changes.len() >= 2 {
            let vol = price_change.abs();
            if self.volatility.len() >= self.window_size {
                self.volatility.pop_front();
            }
            self.volatility.push_back(vol);
        }
    }

    /// Detect current market regime
    pub fn detect(&self) -> MarketRegime {
        if self.price_changes.is_empty() {
            return MarketRegime::Ranging;
        }

        // Calculate trend
        let trend: f64 = self.price_changes.iter().sum::<f64>()
            / self.price_changes.len() as f64;

        // Calculate volatility
        let avg_vol: f64 = if self.volatility.is_empty() {
            0.0
        } else {
            self.volatility.iter().sum::<f64>() / self.volatility.len() as f64
        };

        // Classify regime
        if avg_vol > self.vol_threshold * 2.0 {
            MarketRegime::HighVolatility
        } else if avg_vol < self.vol_threshold * 0.5 {
            MarketRegime::LowVolatility
        } else if trend > self.trend_threshold * 2.0 {
            MarketRegime::BullTrend
        } else if trend > self.trend_threshold {
            MarketRegime::WeakBull
        } else if trend < -self.trend_threshold * 2.0 {
            MarketRegime::BearTrend
        } else if trend < -self.trend_threshold {
            MarketRegime::WeakBear
        } else {
            MarketRegime::Ranging
        }
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.price_changes.clear();
        self.volatility.clear();
    }
}

/// Signal aggregator for combining multiple SNN outputs
#[derive(Debug)]
pub struct SignalAggregator {
    /// Signal history
    signals: VecDeque<f64>,
    /// Weights for each signal source
    weights: Vec<f64>,
    /// Smoothing window
    smoothing_window: usize,
}

impl SignalAggregator {
    /// Create a new aggregator
    pub fn new(num_sources: usize, smoothing_window: usize) -> Self {
        Self {
            signals: VecDeque::with_capacity(smoothing_window),
            weights: vec![1.0 / num_sources as f64; num_sources],
            smoothing_window,
        }
    }

    /// Set custom weights for signal sources
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        // Normalize weights
        let sum: f64 = weights.iter().sum();
        self.weights = weights.into_iter().map(|w| w / sum).collect();
        self
    }

    /// Aggregate signals from multiple sources
    pub fn aggregate(&mut self, signals: &[f64]) -> SignalStrength {
        assert_eq!(signals.len(), self.weights.len());

        // Weighted average
        let weighted_signal: f64 = signals.iter()
            .zip(self.weights.iter())
            .map(|(&s, &w)| s * w)
            .sum();

        // Add to history for smoothing
        if self.signals.len() >= self.smoothing_window {
            self.signals.pop_front();
        }
        self.signals.push_back(weighted_signal);

        // Smoothed signal
        let smoothed = self.signals.iter().sum::<f64>() / self.signals.len() as f64;

        // Calculate confidence based on signal agreement
        let variance: f64 = signals.iter()
            .map(|&s| (s - weighted_signal).powi(2))
            .sum::<f64>() / signals.len() as f64;

        let confidence = 1.0 / (1.0 + variance.sqrt() * 5.0);

        SignalStrength::new(smoothed, confidence)
    }

    /// Reset aggregator state
    pub fn reset(&mut self) {
        self.signals.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_signal() {
        assert_eq!(TradingSignal::from_value(0.8), TradingSignal::StrongBuy);
        assert_eq!(TradingSignal::from_value(0.3), TradingSignal::Buy);
        assert_eq!(TradingSignal::from_value(0.0), TradingSignal::Hold);
        assert_eq!(TradingSignal::from_value(-0.3), TradingSignal::Sell);
        assert_eq!(TradingSignal::from_value(-0.8), TradingSignal::StrongSell);
    }

    #[test]
    fn test_signal_strength() {
        let signal = SignalStrength::new(0.8, 0.9);
        assert!((signal.effective_signal() - 0.72).abs() < 1e-10);
        assert!(signal.is_actionable(0.5));
        assert!(!signal.is_actionable(0.95));
    }

    #[test]
    fn test_regime_detection() {
        let mut detector = RegimeDetector::new(10);

        // Simulate uptrend
        for _ in 0..10 {
            detector.update(0.03);
        }
        assert_eq!(detector.detect(), MarketRegime::BullTrend);

        // Reset and simulate downtrend
        detector.reset();
        for _ in 0..10 {
            detector.update(-0.03);
        }
        assert_eq!(detector.detect(), MarketRegime::BearTrend);
    }

    #[test]
    fn test_signal_aggregator() {
        let mut aggregator = SignalAggregator::new(3, 5);

        let signals = vec![0.8, 0.7, 0.9];
        let result = aggregator.aggregate(&signals);

        assert!(result.value > 0.7);
        assert!(result.confidence > 0.5);
    }
}
