//! Trading signal generation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::bnn::uncertainty::UncertaintyEstimate;

/// Trading signal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Long (buy) signal
    Long,
    /// Short (sell) signal
    Short,
    /// Neutral (no action)
    Neutral,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Long => write!(f, "LONG"),
            SignalType::Short => write!(f, "SHORT"),
            SignalType::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// Trading signal with uncertainty information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Timestamp of signal generation
    pub timestamp: DateTime<Utc>,
    /// Signal type (long/short/neutral)
    pub signal_type: SignalType,
    /// Predicted probability for the signal direction
    pub probability: f64,
    /// Epistemic (model) uncertainty
    pub uncertainty: f64,
    /// Confidence (1 / (1 + uncertainty))
    pub confidence: f64,
    /// Recommended position size (0 to 1)
    pub position_size: f64,
    /// Expected return (if available)
    pub expected_return: Option<f64>,
    /// Symbol
    pub symbol: Option<String>,
}

impl Signal {
    /// Create a new signal.
    pub fn new(
        timestamp: DateTime<Utc>,
        signal_type: SignalType,
        probability: f64,
        uncertainty: f64,
        position_size: f64,
    ) -> Self {
        let confidence = 1.0 / (1.0 + uncertainty);
        Self {
            timestamp,
            signal_type,
            probability,
            uncertainty,
            confidence,
            position_size,
            expected_return: None,
            symbol: None,
        }
    }

    /// Set expected return.
    pub fn with_expected_return(mut self, expected_return: f64) -> Self {
        self.expected_return = Some(expected_return);
        self
    }

    /// Set symbol.
    pub fn with_symbol(mut self, symbol: String) -> Self {
        self.symbol = Some(symbol);
        self
    }

    /// Check if this is a tradeable signal (not neutral, has position size).
    pub fn is_tradeable(&self) -> bool {
        self.signal_type != SignalType::Neutral && self.position_size > 0.0
    }
}

/// Signal generator from Bayesian predictions.
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Minimum probability for generating a signal
    pub prob_threshold: f64,
    /// Minimum confidence for trading
    pub conf_threshold: f64,
    /// Maximum position size
    pub max_position_size: f64,
    /// Whether to use Kelly criterion for sizing
    pub use_kelly: bool,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            prob_threshold: 0.55,
            conf_threshold: 0.3,
            max_position_size: 0.25,
            use_kelly: true,
        }
    }
}

impl SignalGenerator {
    /// Create a new signal generator.
    pub fn new(prob_threshold: f64, conf_threshold: f64) -> Self {
        Self {
            prob_threshold,
            conf_threshold,
            ..Default::default()
        }
    }

    /// Generate signal from uncertainty estimate.
    pub fn generate(&self, estimate: &UncertaintyEstimate, timestamp: DateTime<Utc>) -> Signal {
        // Get predicted probabilities (assume softmax output)
        let probs = &estimate.mean;
        if probs.len() < 3 {
            return Signal::new(
                timestamp,
                SignalType::Neutral,
                0.0,
                estimate.mean_epistemic(),
                0.0,
            );
        }

        let prob_up = probs[0];
        let prob_neutral = probs[1];
        let prob_down = probs[2];

        let uncertainty = estimate.mean_epistemic();
        let confidence = estimate.mean_confidence();

        // Determine signal type
        let (signal_type, probability) = if prob_up > self.prob_threshold && prob_up > prob_down {
            (SignalType::Long, prob_up)
        } else if prob_down > self.prob_threshold && prob_down > prob_up {
            (SignalType::Short, prob_down)
        } else {
            (SignalType::Neutral, prob_neutral)
        };

        // Calculate position size
        let position_size = if signal_type == SignalType::Neutral
            || confidence < self.conf_threshold
        {
            0.0
        } else if self.use_kelly {
            self.kelly_position_size(probability, confidence, uncertainty)
        } else {
            confidence * self.max_position_size
        };

        Signal::new(timestamp, signal_type, probability, uncertainty, position_size)
    }

    /// Calculate position size using modified Kelly Criterion.
    fn kelly_position_size(
        &self,
        prob_win: f64,
        confidence: f64,
        uncertainty: f64,
    ) -> f64 {
        let win_loss_ratio = 2.0; // Assume 2:1 reward/risk

        let b = win_loss_ratio;
        let q = 1.0 - prob_win;

        // Standard Kelly
        let kelly = (prob_win * b - q) / b;

        // Uncertainty penalty
        let uncertainty_penalty = uncertainty.sqrt();
        let adjusted_kelly = kelly * (1.0 - uncertainty_penalty) * confidence;

        // Constrain to valid range
        adjusted_kelly.max(0.0).min(self.max_position_size)
    }
}
