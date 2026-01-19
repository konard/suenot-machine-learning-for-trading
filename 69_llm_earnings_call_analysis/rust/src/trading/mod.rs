//! Trading signal generation module
//!
//! Generates trading signals based on earnings call analysis.

use crate::analyzer::{EarningsAnalysis, EarningsAnalyzer};

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalType {
    StrongBuy,
    Buy,
    Neutral,
    Sell,
    StrongSell,
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub signal_type: SignalType,
    pub strength: f64,
    pub confidence: f64,
    pub reasons: Vec<String>,
}

/// Configuration for signal generation
#[derive(Debug, Clone)]
pub struct SignalConfig {
    pub strong_threshold: f64,
    pub weak_threshold: f64,
    pub confidence_weight: f64,
    pub guidance_weight: f64,
    pub qa_weight: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            strong_threshold: 0.3,
            weak_threshold: 0.1,
            confidence_weight: 0.3,
            guidance_weight: 0.25,
            qa_weight: 0.15,
        }
    }
}

/// Signal generator for earnings calls
pub struct SignalGenerator {
    analyzer: EarningsAnalyzer,
    config: SignalConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new() -> Self {
        Self {
            analyzer: EarningsAnalyzer::new(),
            config: SignalConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SignalConfig) -> Self {
        Self {
            analyzer: EarningsAnalyzer::new(),
            config,
        }
    }

    /// Generate trading signal from transcript
    pub fn generate_signal(&self, transcript: &str) -> TradingSignal {
        let analysis = self.analyzer.analyze(transcript);
        self.signal_from_analysis(&analysis)
    }

    /// Generate signal from pre-computed analysis
    pub fn signal_from_analysis(&self, analysis: &EarningsAnalysis) -> TradingSignal {
        let mut reasons = Vec::new();

        // Calculate base score from overall sentiment
        let mut score = analysis.overall_sentiment.net_sentiment;

        // Adjust for confidence
        let confidence_adjustment = (analysis.confidence.overall - 0.5)
            * self.config.confidence_weight;
        score += confidence_adjustment;

        if analysis.confidence.overall > 0.6 {
            reasons.push("High management confidence".to_string());
        } else if analysis.confidence.overall < 0.4 {
            reasons.push("Low management confidence".to_string());
        }

        // Adjust for guidance direction
        let guidance_adjustment = match analysis.guidance.direction.as_str() {
            "raised" => {
                reasons.push("Raised guidance".to_string());
                self.config.guidance_weight
            }
            "lowered" => {
                reasons.push("Lowered guidance".to_string());
                -self.config.guidance_weight
            }
            _ => 0.0,
        };
        score += guidance_adjustment;

        // Adjust for Q&A quality
        let qa_adjustment = if analysis.qa_quality.responsiveness > 0.6 {
            reasons.push("Strong Q&A responses".to_string());
            self.config.qa_weight
        } else if analysis.qa_quality.responsiveness < 0.4 {
            reasons.push("Weak Q&A responses".to_string());
            -self.config.qa_weight
        } else {
            0.0
        };
        score += qa_adjustment;

        // Add sentiment-based reasons
        if analysis.overall_sentiment.positive_score > 0.05 {
            reasons.push(format!(
                "Positive sentiment ({:.1}%)",
                analysis.overall_sentiment.positive_score * 100.0
            ));
        }
        if analysis.overall_sentiment.negative_score > 0.05 {
            reasons.push(format!(
                "Negative sentiment ({:.1}%)",
                analysis.overall_sentiment.negative_score * 100.0
            ));
        }
        if analysis.overall_sentiment.hedging_score > 0.03 {
            reasons.push("Hedging language detected".to_string());
        }

        // Determine signal type
        let signal_type = if score > self.config.strong_threshold {
            SignalType::StrongBuy
        } else if score > self.config.weak_threshold {
            SignalType::Buy
        } else if score < -self.config.strong_threshold {
            SignalType::StrongSell
        } else if score < -self.config.weak_threshold {
            SignalType::Sell
        } else {
            SignalType::Neutral
        };

        // Calculate confidence in the signal
        let signal_confidence = (score.abs() / self.config.strong_threshold).min(1.0)
            * analysis.confidence.overall;

        TradingSignal {
            signal_type,
            strength: score,
            confidence: signal_confidence,
            reasons,
        }
    }

    /// Get the underlying analyzer
    pub fn analyzer(&self) -> &EarningsAnalyzer {
        &self.analyzer
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::StrongBuy => write!(f, "STRONG BUY"),
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Neutral => write!(f, "NEUTRAL"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::StrongSell => write!(f, "STRONG SELL"),
        }
    }
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Signal: {}", self.signal_type)?;
        writeln!(f, "Strength: {:.3}", self.strength)?;
        writeln!(f, "Confidence: {:.1}%", self.confidence * 100.0)?;
        writeln!(f, "Reasons:")?;
        for reason in &self.reasons {
            writeln!(f, "  - {}", reason)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bullish_signal() {
        let generator = SignalGenerator::new();

        let transcript = r#"
John Smith - CEO:
We are pleased to report exceptional results this quarter.
Revenue grew 30% year over year, exceeding expectations.
We are raising guidance for the full year.
We are confident in our growth trajectory.

Question-and-Answer Session

Analyst - Goldman Sachs:
Can you elaborate on the guidance raise?

John Smith - CEO:
Absolutely. We have strong visibility into our pipeline
and are committed to delivering sustained growth.
        "#;

        let signal = generator.generate_signal(transcript);

        assert!(matches!(
            signal.signal_type,
            SignalType::StrongBuy | SignalType::Buy
        ));
        assert!(signal.strength > 0.0);
    }

    #[test]
    fn test_bearish_signal() {
        let generator = SignalGenerator::new();

        let transcript = r#"
Jane Doe - CFO:
We faced significant challenges this quarter.
Revenue declined due to headwinds and market weakness.
We are lowering guidance given the uncertain outlook.
Results were disappointing.

Question-and-Answer Session

Analyst - Morgan Stanley:
What is driving the weakness?

Jane Doe - CFO:
We are uncertain about the timeline for recovery.
Market conditions remain challenging.
        "#;

        let signal = generator.generate_signal(transcript);

        assert!(matches!(
            signal.signal_type,
            SignalType::StrongSell | SignalType::Sell
        ));
        assert!(signal.strength < 0.0);
    }

    #[test]
    fn test_signal_display() {
        let signal = TradingSignal {
            signal_type: SignalType::Buy,
            strength: 0.25,
            confidence: 0.75,
            reasons: vec!["Test reason".to_string()],
        };

        let display = format!("{}", signal);
        assert!(display.contains("BUY"));
        assert!(display.contains("0.25"));
    }
}
