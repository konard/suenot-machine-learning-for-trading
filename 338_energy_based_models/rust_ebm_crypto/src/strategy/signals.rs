//! Trading signals based on EBM energy scores

use crate::ebm::{MarketRegime, OnlineEnergyEstimator};

/// Trading signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Enter long position
    Long,
    /// Enter short position
    Short,
    /// Exit all positions
    Exit,
    /// Reduce position size
    ReducePosition,
    /// Hold current position
    Hold,
}

impl SignalType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalType::Long => "LONG",
            SignalType::Short => "SHORT",
            SignalType::Exit => "EXIT",
            SignalType::ReducePosition => "REDUCE",
            SignalType::Hold => "HOLD",
        }
    }
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal type
    pub signal_type: SignalType,
    /// Confidence (0-1)
    pub confidence: f64,
    /// Suggested position size multiplier (0-1)
    pub position_scale: f64,
    /// Raw energy value
    pub energy: f64,
    /// Normalized energy (in standard deviations)
    pub normalized_energy: f64,
    /// Detected market regime
    pub regime: MarketRegime,
    /// Timestamp
    pub timestamp: i64,
    /// Reason for signal
    pub reason: String,
}

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Threshold for anomaly detection (in standard deviations)
    pub anomaly_threshold: f64,
    /// Threshold for position reduction
    pub reduce_threshold: f64,
    /// Threshold for emergency exit
    pub exit_threshold: f64,
    /// Lookback period for contrarian signals
    pub contrarian_lookback: usize,
    /// Minimum confidence for signals
    pub min_confidence: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            anomaly_threshold: 2.0,
            reduce_threshold: 1.5,
            exit_threshold: 3.0,
            contrarian_lookback: 5,
            min_confidence: 0.6,
        }
    }
}

/// EBM-based signal generator
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Configuration
    pub config: SignalConfig,
    /// Recent energy history
    energy_history: Vec<f64>,
    /// Recent return history
    return_history: Vec<f64>,
    /// Maximum history size
    max_history: usize,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalConfig) -> Self {
        Self {
            config,
            energy_history: Vec::new(),
            return_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Generate signal based on energy and return
    pub fn generate(
        &mut self,
        normalized_energy: f64,
        raw_energy: f64,
        current_return: f64,
        regime: MarketRegime,
        timestamp: i64,
    ) -> TradingSignal {
        // Update history
        self.energy_history.push(normalized_energy);
        self.return_history.push(current_return);

        if self.energy_history.len() > self.max_history {
            self.energy_history.remove(0);
            self.return_history.remove(0);
        }

        // Calculate position scale
        let position_scale = self.calculate_position_scale(normalized_energy);

        // Determine signal type
        let (signal_type, confidence, reason) =
            self.determine_signal(normalized_energy, current_return, regime);

        TradingSignal {
            signal_type,
            confidence,
            position_scale,
            energy: raw_energy,
            normalized_energy,
            regime,
            timestamp,
            reason,
        }
    }

    /// Calculate position scale based on energy
    fn calculate_position_scale(&self, normalized_energy: f64) -> f64 {
        if normalized_energy < 0.0 {
            1.0 // Low energy = full position
        } else if normalized_energy < 1.0 {
            1.0 - 0.2 * normalized_energy
        } else if normalized_energy < 2.0 {
            0.8 - 0.3 * (normalized_energy - 1.0)
        } else if normalized_energy < 3.0 {
            0.5 - 0.3 * (normalized_energy - 2.0)
        } else {
            0.1 // Minimum position in crisis
        }
        .max(0.0)
    }

    /// Determine signal type and confidence
    fn determine_signal(
        &self,
        normalized_energy: f64,
        current_return: f64,
        regime: MarketRegime,
    ) -> (SignalType, f64, String) {
        // Emergency exit on extreme energy
        if normalized_energy > self.config.exit_threshold {
            return (
                SignalType::Exit,
                0.95,
                format!(
                    "Emergency exit: energy {:.2} > {:.2}",
                    normalized_energy, self.config.exit_threshold
                ),
            );
        }

        // Reduce position on high energy
        if normalized_energy > self.config.reduce_threshold {
            return (
                SignalType::ReducePosition,
                0.8,
                format!(
                    "High energy warning: {:.2} > {:.2}",
                    normalized_energy, self.config.reduce_threshold
                ),
            );
        }

        // Check for contrarian signal (energy spike resolved)
        if let Some(contrarian) = self.check_contrarian_signal() {
            return contrarian;
        }

        // Normal conditions - hold or follow trend
        (
            SignalType::Hold,
            0.5,
            format!("Normal conditions, regime: {}", regime.as_str()),
        )
    }

    /// Check for contrarian entry opportunity
    fn check_contrarian_signal(&self) -> Option<(SignalType, f64, String)> {
        if self.energy_history.len() < self.config.contrarian_lookback {
            return None;
        }

        let recent = &self.energy_history[self.energy_history.len() - self.config.contrarian_lookback..];
        let recent_returns =
            &self.return_history[self.return_history.len() - self.config.contrarian_lookback..];

        // Check if there was a recent energy spike that has now resolved
        let max_recent = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let current = *self.energy_history.last().unwrap();

        if max_recent > self.config.anomaly_threshold && current < self.config.anomaly_threshold * 0.5
        {
            // Energy spike resolved - look for contrarian entry
            let sum_returns: f64 = recent_returns.iter().sum();

            if sum_returns < -0.02 {
                // Down move during spike - go long
                return Some((
                    SignalType::Long,
                    0.7,
                    format!(
                        "Contrarian long: energy spike resolved, down {:.2}%",
                        sum_returns * 100.0
                    ),
                ));
            } else if sum_returns > 0.02 {
                // Up move during spike - go short
                return Some((
                    SignalType::Short,
                    0.7,
                    format!(
                        "Contrarian short: energy spike resolved, up {:.2}%",
                        sum_returns * 100.0
                    ),
                ));
            }
        }

        None
    }

    /// Get current energy trend
    pub fn energy_trend(&self) -> EnergyTrend {
        if self.energy_history.len() < 5 {
            return EnergyTrend::Unknown;
        }

        let recent = &self.energy_history[self.energy_history.len() - 5..];
        let first_half: f64 = recent[..2].iter().sum::<f64>() / 2.0;
        let second_half: f64 = recent[3..].iter().sum::<f64>() / 2.0;

        let change = second_half - first_half;

        if change > 0.5 {
            EnergyTrend::Increasing
        } else if change < -0.5 {
            EnergyTrend::Decreasing
        } else {
            EnergyTrend::Stable
        }
    }
}

/// Energy trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyTrend {
    /// Energy is increasing (becoming more anomalous)
    Increasing,
    /// Energy is stable
    Stable,
    /// Energy is decreasing (returning to normal)
    Decreasing,
    /// Not enough data
    Unknown,
}

impl EnergyTrend {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            EnergyTrend::Increasing => "INCREASING",
            EnergyTrend::Stable => "STABLE",
            EnergyTrend::Decreasing => "DECREASING",
            EnergyTrend::Unknown => "UNKNOWN",
        }
    }
}

/// Signal aggregator for combining multiple signal sources
#[derive(Debug, Clone)]
pub struct SignalAggregator {
    /// Weights for different signal sources
    pub weights: Vec<f64>,
    /// Signal history
    signals: Vec<TradingSignal>,
    /// Maximum history size
    max_history: usize,
}

impl SignalAggregator {
    /// Create a new aggregator with equal weights
    pub fn new(n_sources: usize) -> Self {
        Self {
            weights: vec![1.0 / n_sources as f64; n_sources],
            signals: Vec::new(),
            max_history: 100,
        }
    }

    /// Add a signal to history
    pub fn add_signal(&mut self, signal: TradingSignal) {
        self.signals.push(signal);
        if self.signals.len() > self.max_history {
            self.signals.remove(0);
        }
    }

    /// Get weighted average confidence
    pub fn average_confidence(&self) -> f64 {
        if self.signals.is_empty() {
            return 0.5;
        }

        let n = self.signals.len().min(10);
        let recent = &self.signals[self.signals.len() - n..];
        recent.iter().map(|s| s.confidence).sum::<f64>() / n as f64
    }

    /// Get most frequent signal type
    pub fn dominant_signal(&self) -> Option<SignalType> {
        if self.signals.is_empty() {
            return None;
        }

        let n = self.signals.len().min(5);
        let recent = &self.signals[self.signals.len() - n..];

        let mut counts = [0i32; 5]; // Long, Short, Exit, Reduce, Hold

        for signal in recent {
            match signal.signal_type {
                SignalType::Long => counts[0] += 1,
                SignalType::Short => counts[1] += 1,
                SignalType::Exit => counts[2] += 1,
                SignalType::ReducePosition => counts[3] += 1,
                SignalType::Hold => counts[4] += 1,
            }
        }

        let max_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)?;

        Some(match max_idx {
            0 => SignalType::Long,
            1 => SignalType::Short,
            2 => SignalType::Exit,
            3 => SignalType::ReducePosition,
            _ => SignalType::Hold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generator() {
        let config = SignalConfig::default();
        let mut generator = SignalGenerator::new(config);

        // Normal conditions
        let signal = generator.generate(0.5, 1.0, 0.001, MarketRegime::Normal, 1000);
        assert_eq!(signal.signal_type, SignalType::Hold);
        assert!(signal.position_scale > 0.8);

        // High energy
        let signal = generator.generate(2.5, 5.0, -0.02, MarketRegime::Elevated, 2000);
        assert_eq!(signal.signal_type, SignalType::ReducePosition);
        assert!(signal.position_scale < 0.5);

        // Extreme energy
        let signal = generator.generate(4.0, 10.0, -0.05, MarketRegime::Crisis, 3000);
        assert_eq!(signal.signal_type, SignalType::Exit);
    }

    #[test]
    fn test_position_scale() {
        let config = SignalConfig::default();
        let generator = SignalGenerator::new(config);

        // Low energy = high position
        assert!(generator.calculate_position_scale(-1.0) == 1.0);

        // High energy = low position
        assert!(generator.calculate_position_scale(3.0) < 0.3);
    }
}
