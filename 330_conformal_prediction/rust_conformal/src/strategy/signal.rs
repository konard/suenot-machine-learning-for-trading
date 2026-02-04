//! Trading signal generation with conformal prediction

use crate::conformal::predictor::PredictionInterval;

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    Long,
    Short,
    Hold,
}

impl SignalType {
    /// Get numeric value for position sizing
    pub fn value(&self) -> f64 {
        match self {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            SignalType::Hold => 0.0,
        }
    }
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Long => write!(f, "LONG"),
            SignalType::Short => write!(f, "SHORT"),
            SignalType::Hold => write!(f, "HOLD"),
        }
    }
}

/// Trading signal with uncertainty information
#[derive(Debug, Clone)]
pub struct Signal {
    /// Timestamp
    pub timestamp: i64,
    /// Signal type
    pub signal_type: SignalType,
    /// Point prediction
    pub point_prediction: f64,
    /// Lower bound of interval
    pub lower_bound: f64,
    /// Upper bound of interval
    pub upper_bound: f64,
    /// Interval width
    pub interval_width: f64,
    /// Confidence (inverse of normalized width)
    pub confidence: f64,
    /// Position size
    pub position_size: f64,
}

impl Signal {
    /// Create new signal
    pub fn new(
        timestamp: i64,
        signal_type: SignalType,
        interval: &PredictionInterval,
        confidence: f64,
        position_size: f64,
    ) -> Self {
        Self {
            timestamp,
            signal_type,
            point_prediction: interval.point,
            lower_bound: interval.lower,
            upper_bound: interval.upper,
            interval_width: interval.width,
            confidence,
            position_size,
        }
    }

    /// Create hold signal
    pub fn hold(timestamp: i64) -> Self {
        Self {
            timestamp,
            signal_type: SignalType::Hold,
            point_prediction: 0.0,
            lower_bound: 0.0,
            upper_bound: 0.0,
            interval_width: 0.0,
            confidence: 0.0,
            position_size: 0.0,
        }
    }
}

/// Signal generator using conformal prediction
pub struct SignalGenerator {
    /// Minimum confidence to trade
    min_confidence: f64,
    /// Maximum position size
    max_position: f64,
    /// Volatility threshold for trading
    volatility_threshold: f64,
    /// Historical volatility for normalization
    historical_volatility: f64,
    /// Position scaling method
    position_scaling: PositionScaling,
}

/// Position scaling method
#[derive(Debug, Clone, Copy)]
pub enum PositionScaling {
    /// Fixed position size
    Fixed,
    /// Inverse of interval width
    InverseWidth,
    /// Kelly-inspired sizing
    Kelly,
}

impl SignalGenerator {
    /// Create new signal generator
    pub fn new(min_confidence: f64, max_position: f64) -> Self {
        Self {
            min_confidence: min_confidence.clamp(0.1, 0.9),
            max_position: max_position.clamp(0.1, 2.0),
            volatility_threshold: 0.05,
            historical_volatility: 0.02, // Default 2%
            position_scaling: PositionScaling::InverseWidth,
        }
    }

    /// Set historical volatility
    pub fn with_historical_volatility(mut self, vol: f64) -> Self {
        self.historical_volatility = vol.max(0.001);
        self
    }

    /// Set volatility threshold
    pub fn with_volatility_threshold(mut self, threshold: f64) -> Self {
        self.volatility_threshold = threshold;
        self
    }

    /// Set position scaling method
    pub fn with_position_scaling(mut self, scaling: PositionScaling) -> Self {
        self.position_scaling = scaling;
        self
    }

    /// Generate signal from prediction interval
    pub fn generate(&self, timestamp: i64, interval: &PredictionInterval) -> Signal {
        // Normalize interval width
        let normalized_width = interval.width / self.historical_volatility;

        // Calculate confidence
        let confidence = 1.0 / (1.0 + normalized_width);

        // Check confidence threshold
        if confidence < self.min_confidence {
            return Signal::hold(timestamp);
        }

        // Check volatility threshold
        if normalized_width > self.volatility_threshold * 10.0 {
            return Signal::hold(timestamp);
        }

        // Determine signal type
        let signal_type = self.determine_signal_type(interval, confidence);

        // Calculate position size
        let position_size = self.calculate_position_size(signal_type, interval, confidence);

        Signal::new(timestamp, signal_type, interval, confidence, position_size)
    }

    /// Determine signal type based on interval
    fn determine_signal_type(&self, interval: &PredictionInterval, confidence: f64) -> SignalType {
        // Strong signal: entire interval is positive or negative
        if interval.lower > 0.0 {
            return SignalType::Long;
        }

        if interval.upper < 0.0 {
            return SignalType::Short;
        }

        // Weaker signal based on point prediction
        let threshold = 0.001; // 0.1% threshold

        if interval.point > threshold && confidence > 0.6 {
            return SignalType::Long;
        }

        if interval.point < -threshold && confidence > 0.6 {
            return SignalType::Short;
        }

        SignalType::Hold
    }

    /// Calculate position size based on uncertainty
    fn calculate_position_size(
        &self,
        signal_type: SignalType,
        interval: &PredictionInterval,
        confidence: f64,
    ) -> f64 {
        if signal_type == SignalType::Hold {
            return 0.0;
        }

        let direction = signal_type.value();

        let position = match self.position_scaling {
            PositionScaling::Fixed => self.max_position,

            PositionScaling::InverseWidth => {
                let scale = confidence.powi(2);
                self.max_position * scale
            }

            PositionScaling::Kelly => {
                let expected_return = interval.point.abs();
                let uncertainty = interval.width / 2.0;

                if uncertainty > 1e-10 {
                    let kelly = expected_return / (uncertainty * uncertainty);
                    let kelly_fraction = kelly.min(1.0);
                    self.max_position * kelly_fraction * confidence
                } else {
                    self.max_position * confidence
                }
            }
        };

        (position * direction).clamp(-self.max_position, self.max_position)
    }

    /// Generate signals for multiple intervals
    pub fn generate_batch(
        &self,
        timestamps: &[i64],
        intervals: &[PredictionInterval],
    ) -> Vec<Signal> {
        timestamps
            .iter()
            .zip(intervals.iter())
            .map(|(&ts, interval)| self.generate(ts, interval))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::new(0.4, 1.0)
            .with_historical_volatility(0.02);

        // Strong long signal (interval entirely positive)
        let interval = PredictionInterval::new(0.05, 0.02, 0.08, 0.1);
        let signal = generator.generate(1000, &interval);

        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_hold_signal() {
        let generator = SignalGenerator::new(0.8, 1.0); // High confidence threshold

        // Uncertain signal (wide interval crossing zero)
        let interval = PredictionInterval::new(0.001, -0.1, 0.1, 0.1);
        let signal = generator.generate(1000, &interval);

        assert_eq!(signal.signal_type, SignalType::Hold);
        assert_eq!(signal.position_size, 0.0);
    }
}
