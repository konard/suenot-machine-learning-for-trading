//! Trading signal generation

use std::collections::HashMap;
use ndarray::Array1;

/// Signal type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Neutral/hold signal
    Neutral,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl SignalType {
    /// Convert to numeric value (-1 to 1)
    pub fn to_value(&self) -> f64 {
        match self {
            SignalType::StrongBuy => 1.0,
            SignalType::Buy => 0.5,
            SignalType::Neutral => 0.0,
            SignalType::Sell => -0.5,
            SignalType::StrongSell => -1.0,
        }
    }

    /// Create from probability distribution
    pub fn from_probs(p_down: f64, p_neutral: f64, p_up: f64) -> Self {
        if p_up > 0.6 {
            SignalType::StrongBuy
        } else if p_up > 0.45 && p_up > p_down {
            SignalType::Buy
        } else if p_down > 0.6 {
            SignalType::StrongSell
        } else if p_down > 0.45 && p_down > p_up {
            SignalType::Sell
        } else {
            SignalType::Neutral
        }
    }
}

/// Trading signal
#[derive(Debug, Clone)]
pub struct Signal {
    /// Symbol
    pub symbol: String,
    /// Signal type
    pub signal_type: SignalType,
    /// Signal strength (0-1)
    pub strength: f64,
    /// Confidence (0-1)
    pub confidence: f64,
    /// Expected return
    pub expected_return: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Source (e.g., metapath name)
    pub source: String,
}

impl Signal {
    /// Create a new signal
    pub fn new(symbol: impl Into<String>, signal_type: SignalType, strength: f64) -> Self {
        Self {
            symbol: symbol.into(),
            signal_type,
            strength,
            confidence: 1.0,
            expected_return: 0.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            source: "default".to_string(),
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set expected return
    pub fn with_expected_return(mut self, expected_return: f64) -> Self {
        self.expected_return = expected_return;
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Get weighted signal value
    pub fn weighted_value(&self) -> f64 {
        self.signal_type.to_value() * self.strength * self.confidence
    }
}

/// Metapath-based signal
#[derive(Debug, Clone)]
pub struct MetapathSignal {
    /// Metapath name
    pub metapath_name: String,
    /// Target symbol
    pub symbol: String,
    /// Related symbols in the path
    pub related_symbols: Vec<String>,
    /// Signal type
    pub signal_type: SignalType,
    /// Path weight
    pub weight: f64,
    /// Confidence
    pub confidence: f64,
}

impl MetapathSignal {
    /// Create a new metapath signal
    pub fn new(
        metapath_name: impl Into<String>,
        symbol: impl Into<String>,
        signal_type: SignalType,
        weight: f64,
    ) -> Self {
        Self {
            metapath_name: metapath_name.into(),
            symbol: symbol.into(),
            related_symbols: Vec::new(),
            signal_type,
            weight,
            confidence: 1.0,
        }
    }

    /// Add related symbols
    pub fn with_related(mut self, related: Vec<String>) -> Self {
        self.related_symbols = related;
        self
    }

    /// Convert to base signal
    pub fn to_signal(&self) -> Signal {
        Signal::new(&self.symbol, self.signal_type, self.weight)
            .with_confidence(self.confidence)
            .with_source(&self.metapath_name)
    }
}

/// Signal aggregator for combining multiple signals
#[derive(Debug)]
pub struct SignalAggregator {
    /// Weight for each signal source
    source_weights: HashMap<String, f64>,
    /// Historical accuracy per source
    source_accuracy: HashMap<String, f64>,
    /// Decay factor for old signals
    decay_factor: f64,
}

impl SignalAggregator {
    /// Create a new signal aggregator
    pub fn new() -> Self {
        Self {
            source_weights: HashMap::new(),
            source_accuracy: HashMap::new(),
            decay_factor: 0.95,
        }
    }

    /// Set weight for a signal source
    pub fn set_weight(&mut self, source: impl Into<String>, weight: f64) {
        self.source_weights.insert(source.into(), weight);
    }

    /// Set accuracy for a signal source
    pub fn set_accuracy(&mut self, source: impl Into<String>, accuracy: f64) {
        self.source_accuracy.insert(source.into(), accuracy);
    }

    /// Aggregate multiple signals for a symbol
    pub fn aggregate(&self, signals: &[Signal]) -> Option<Signal> {
        if signals.is_empty() {
            return None;
        }

        let symbol = signals[0].symbol.clone();

        // Calculate weighted sum
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut max_confidence = 0.0;

        for signal in signals {
            let source_weight = self.source_weights
                .get(&signal.source)
                .copied()
                .unwrap_or(1.0);
            let accuracy = self.source_accuracy
                .get(&signal.source)
                .copied()
                .unwrap_or(0.5);

            let weight = source_weight * accuracy * signal.confidence;
            weighted_sum += signal.weighted_value() * weight;
            total_weight += weight;
            max_confidence = max_confidence.max(signal.confidence);
        }

        if total_weight == 0.0 {
            return None;
        }

        let avg_value = weighted_sum / total_weight;

        // Convert to signal type
        let signal_type = if avg_value > 0.5 {
            SignalType::StrongBuy
        } else if avg_value > 0.1 {
            SignalType::Buy
        } else if avg_value < -0.5 {
            SignalType::StrongSell
        } else if avg_value < -0.1 {
            SignalType::Sell
        } else {
            SignalType::Neutral
        };

        Some(Signal::new(&symbol, signal_type, avg_value.abs())
            .with_confidence(max_confidence)
            .with_source("aggregated"))
    }

    /// Aggregate signals grouped by symbol
    pub fn aggregate_by_symbol(&self, signals: &[Signal]) -> HashMap<String, Signal> {
        let mut by_symbol: HashMap<String, Vec<&Signal>> = HashMap::new();

        for signal in signals {
            by_symbol.entry(signal.symbol.clone())
                .or_default()
                .push(signal);
        }

        by_symbol
            .into_iter()
            .filter_map(|(symbol, sigs)| {
                let owned: Vec<Signal> = sigs.into_iter().cloned().collect();
                self.aggregate(&owned).map(|s| (symbol, s))
            })
            .collect()
    }
}

impl Default for SignalAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type() {
        assert_eq!(SignalType::StrongBuy.to_value(), 1.0);
        assert_eq!(SignalType::Neutral.to_value(), 0.0);
        assert_eq!(SignalType::StrongSell.to_value(), -1.0);
    }

    #[test]
    fn test_signal_from_probs() {
        assert_eq!(SignalType::from_probs(0.1, 0.2, 0.7), SignalType::StrongBuy);
        assert_eq!(SignalType::from_probs(0.7, 0.2, 0.1), SignalType::StrongSell);
        assert_eq!(SignalType::from_probs(0.3, 0.4, 0.3), SignalType::Neutral);
    }

    #[test]
    fn test_signal_aggregator() {
        let aggregator = SignalAggregator::new();

        let signals = vec![
            Signal::new("BTCUSDT", SignalType::Buy, 0.6),
            Signal::new("BTCUSDT", SignalType::StrongBuy, 0.8),
        ];

        let result = aggregator.aggregate(&signals);
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.symbol, "BTCUSDT");
    }
}
