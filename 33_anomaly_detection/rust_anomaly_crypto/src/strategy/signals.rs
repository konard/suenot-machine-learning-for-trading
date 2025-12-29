//! Trading signals based on anomaly detection

use crate::anomaly::AnomalyResult;

/// Configuration for signal generation
#[derive(Clone, Debug)]
pub struct SignalConfig {
    /// Threshold for reducing position (normalized score)
    pub reduce_threshold: f64,
    /// Threshold for emergency exit
    pub exit_threshold: f64,
    /// Threshold for contrarian entry after anomaly
    pub entry_threshold: f64,
    /// Minimum anomaly score reduction to consider resolved
    pub resolution_ratio: f64,
    /// Enable contrarian trading (buy after crash, sell after spike)
    pub enable_contrarian: bool,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            reduce_threshold: 0.7,
            exit_threshold: 1.5,
            entry_threshold: 1.0,
            resolution_ratio: 0.5,
            enable_contrarian: true,
        }
    }
}

/// Trading signal type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Signal {
    /// No action
    Hold,
    /// Reduce position size
    ReducePosition,
    /// Exit all positions (emergency)
    ExitAll,
    /// Enter long position (contrarian after crash)
    EntryLong,
    /// Enter short position (contrarian after spike)
    EntryShort,
}

impl Signal {
    /// Check if signal is an entry signal
    pub fn is_entry(&self) -> bool {
        matches!(self, Signal::EntryLong | Signal::EntryShort)
    }

    /// Check if signal is an exit/reduce signal
    pub fn is_exit(&self) -> bool {
        matches!(self, Signal::ReducePosition | Signal::ExitAll)
    }
}

/// Signal with additional metadata
#[derive(Clone, Debug)]
pub struct TradingSignal {
    /// The signal type
    pub signal: Signal,
    /// Anomaly score that triggered the signal
    pub anomaly_score: f64,
    /// Confidence in the signal (0-1)
    pub confidence: f64,
    /// Suggested position adjustment (e.g., -0.5 = reduce by 50%)
    pub position_adjustment: f64,
    /// Reason for the signal
    pub reason: String,
}

impl TradingSignal {
    /// Create a hold signal
    pub fn hold(anomaly_score: f64) -> Self {
        Self {
            signal: Signal::Hold,
            anomaly_score,
            confidence: 1.0,
            position_adjustment: 0.0,
            reason: "No anomaly detected".to_string(),
        }
    }

    /// Create a reduce position signal
    pub fn reduce(anomaly_score: f64, reduction_pct: f64) -> Self {
        Self {
            signal: Signal::ReducePosition,
            anomaly_score,
            confidence: anomaly_score.min(1.0),
            position_adjustment: -reduction_pct,
            reason: format!("High anomaly score: {:.2}", anomaly_score),
        }
    }

    /// Create an exit all signal
    pub fn exit_all(anomaly_score: f64) -> Self {
        Self {
            signal: Signal::ExitAll,
            anomaly_score,
            confidence: 1.0,
            position_adjustment: -1.0,
            reason: format!("Critical anomaly: {:.2}", anomaly_score),
        }
    }

    /// Create a contrarian long entry signal
    pub fn entry_long(anomaly_score: f64, confidence: f64) -> Self {
        Self {
            signal: Signal::EntryLong,
            anomaly_score,
            confidence,
            position_adjustment: confidence, // Scale entry by confidence
            reason: "Contrarian entry after down anomaly".to_string(),
        }
    }

    /// Create a contrarian short entry signal
    pub fn entry_short(anomaly_score: f64, confidence: f64) -> Self {
        Self {
            signal: Signal::EntryShort,
            anomaly_score,
            confidence,
            position_adjustment: -confidence,
            reason: "Contrarian entry after up anomaly".to_string(),
        }
    }
}

/// Signal generator from anomaly results
pub struct SignalGenerator {
    config: SignalConfig,
    /// Previous anomaly score for detecting resolution
    prev_score: Option<f64>,
    /// Previous return for determining anomaly direction
    prev_return: Option<f64>,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalConfig) -> Self {
        Self {
            config,
            prev_score: None,
            prev_return: None,
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(SignalConfig::default())
    }

    /// Reset the generator state
    pub fn reset(&mut self) {
        self.prev_score = None;
        self.prev_return = None;
    }

    /// Generate signal from current anomaly score and return
    pub fn generate(&mut self, anomaly_score: f64, current_return: f64) -> TradingSignal {
        let signal = self.compute_signal(anomaly_score, current_return);

        // Update state
        self.prev_score = Some(anomaly_score);
        self.prev_return = Some(current_return);

        signal
    }

    /// Generate signals for an entire series
    pub fn generate_series(
        &mut self,
        result: &AnomalyResult,
        returns: &[f64],
    ) -> Vec<TradingSignal> {
        assert_eq!(result.normalized_scores.len(), returns.len());

        self.reset();

        result
            .normalized_scores
            .iter()
            .zip(returns.iter())
            .map(|(&score, &ret)| self.generate(score, ret))
            .collect()
    }

    fn compute_signal(&self, anomaly_score: f64, current_return: f64) -> TradingSignal {
        // Check for emergency exit
        if anomaly_score > self.config.exit_threshold {
            return TradingSignal::exit_all(anomaly_score);
        }

        // Check for position reduction
        if anomaly_score > self.config.reduce_threshold {
            let reduction = (anomaly_score - self.config.reduce_threshold)
                / (self.config.exit_threshold - self.config.reduce_threshold);
            return TradingSignal::reduce(anomaly_score, reduction.min(0.5));
        }

        // Check for contrarian entry opportunity
        if self.config.enable_contrarian {
            if let (Some(prev_score), Some(prev_return)) = (self.prev_score, self.prev_return) {
                // Anomaly resolved: was above entry threshold, now significantly lower
                let anomaly_resolved = prev_score > self.config.entry_threshold
                    && anomaly_score < prev_score * self.config.resolution_ratio;

                if anomaly_resolved {
                    let confidence = 1.0 - anomaly_score / self.config.entry_threshold;

                    if prev_return < 0.0 {
                        // Previous anomaly was downward -> contrarian long
                        return TradingSignal::entry_long(anomaly_score, confidence);
                    } else if prev_return > 0.0 {
                        // Previous anomaly was upward -> contrarian short
                        return TradingSignal::entry_short(anomaly_score, confidence);
                    }
                }
            }
        }

        TradingSignal::hold(anomaly_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_signal() {
        let mut generator = SignalGenerator::default_config();

        let signal = generator.generate(2.0, -0.05);
        assert_eq!(signal.signal, Signal::ExitAll);
    }

    #[test]
    fn test_reduce_signal() {
        let mut generator = SignalGenerator::default_config();

        let signal = generator.generate(1.0, 0.01);
        assert_eq!(signal.signal, Signal::ReducePosition);
    }

    #[test]
    fn test_contrarian_entry() {
        let mut generator = SignalGenerator::default_config();

        // First: high anomaly with negative return
        generator.generate(1.5, -0.10);

        // Then: anomaly resolves
        let signal = generator.generate(0.3, 0.01);
        assert_eq!(signal.signal, Signal::EntryLong);
    }

    #[test]
    fn test_hold_signal() {
        let mut generator = SignalGenerator::default_config();

        let signal = generator.generate(0.3, 0.01);
        assert_eq!(signal.signal, Signal::Hold);
    }
}
