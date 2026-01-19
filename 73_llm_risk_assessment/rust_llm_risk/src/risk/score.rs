//! Risk scoring data structures.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Risk level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Very low risk (score 1-2)
    VeryLow,
    /// Low risk (score 3-4)
    Low,
    /// Moderate risk (score 5-6)
    Moderate,
    /// High risk (score 7-8)
    High,
    /// Severe risk (score 9-10)
    Severe,
}

impl RiskLevel {
    /// Create a risk level from a numeric score.
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s <= 2.0 => RiskLevel::VeryLow,
            s if s <= 4.0 => RiskLevel::Low,
            s if s <= 6.0 => RiskLevel::Moderate,
            s if s <= 8.0 => RiskLevel::High,
            _ => RiskLevel::Severe,
        }
    }

    /// Get the recommended position multiplier for this risk level.
    pub fn position_multiplier(&self) -> f64 {
        match self {
            RiskLevel::VeryLow => 1.0,
            RiskLevel::Low => 0.8,
            RiskLevel::Moderate => 0.5,
            RiskLevel::High => 0.25,
            RiskLevel::Severe => 0.0,
        }
    }
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskLevel::VeryLow => write!(f, "Very Low"),
            RiskLevel::Low => write!(f, "Low"),
            RiskLevel::Moderate => write!(f, "Moderate"),
            RiskLevel::High => write!(f, "High"),
            RiskLevel::Severe => write!(f, "Severe"),
        }
    }
}

/// Individual risk dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDimension {
    /// Name of the dimension.
    pub name: String,
    /// Score (1-10).
    pub score: f64,
    /// Key factors identified.
    pub factors: Vec<String>,
    /// Confidence level.
    pub confidence: Confidence,
}

impl RiskDimension {
    pub fn new(name: &str, score: f64) -> Self {
        Self {
            name: name.to_string(),
            score: score.clamp(1.0, 10.0),
            factors: Vec::new(),
            confidence: Confidence::Medium,
        }
    }

    pub fn with_factors(mut self, factors: Vec<String>) -> Self {
        self.factors = factors;
        self
    }

    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Confidence level for risk assessments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    Low,
    Medium,
    High,
}

impl Default for Confidence {
    fn default() -> Self {
        Confidence::Medium
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Confidence::Low => write!(f, "low"),
            Confidence::Medium => write!(f, "medium"),
            Confidence::High => write!(f, "high"),
        }
    }
}

/// Risk trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskDirection {
    Increasing,
    Stable,
    Decreasing,
}

impl Default for RiskDirection {
    fn default() -> Self {
        RiskDirection::Stable
    }
}

impl fmt::Display for RiskDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskDirection::Increasing => write!(f, "increasing"),
            RiskDirection::Stable => write!(f, "stable"),
            RiskDirection::Decreasing => write!(f, "decreasing"),
        }
    }
}

/// Complete risk score with all dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskScore {
    /// Market risk score (1-10).
    pub market_risk: f64,
    /// Credit risk score (1-10).
    pub credit_risk: f64,
    /// Liquidity risk score (1-10).
    pub liquidity_risk: f64,
    /// Operational risk score (1-10).
    pub operational_risk: f64,
    /// Regulatory risk score (1-10).
    pub regulatory_risk: f64,
    /// Sentiment risk score (1-10).
    pub sentiment_risk: f64,
    /// Overall weighted score.
    pub overall_score: f64,
    /// Confidence in the assessment.
    pub confidence: Confidence,
    /// Direction of risk trend.
    pub direction: RiskDirection,
    /// Key factors identified across all dimensions.
    pub key_factors: Vec<String>,
    /// Timestamp of the assessment.
    pub timestamp: i64,
    /// Symbol/asset being assessed.
    pub symbol: Option<String>,
}

impl Default for RiskScore {
    fn default() -> Self {
        Self {
            market_risk: 5.0,
            credit_risk: 5.0,
            liquidity_risk: 5.0,
            operational_risk: 5.0,
            regulatory_risk: 5.0,
            sentiment_risk: 5.0,
            overall_score: 5.0,
            confidence: Confidence::Medium,
            direction: RiskDirection::Stable,
            key_factors: Vec::new(),
            timestamp: 0,
            symbol: None,
        }
    }
}

impl RiskScore {
    /// Create a new risk score with all dimensions.
    pub fn new(
        market_risk: f64,
        credit_risk: f64,
        liquidity_risk: f64,
        operational_risk: f64,
        regulatory_risk: f64,
        sentiment_risk: f64,
    ) -> Self {
        let mut score = Self {
            market_risk: market_risk.clamp(1.0, 10.0),
            credit_risk: credit_risk.clamp(1.0, 10.0),
            liquidity_risk: liquidity_risk.clamp(1.0, 10.0),
            operational_risk: operational_risk.clamp(1.0, 10.0),
            regulatory_risk: regulatory_risk.clamp(1.0, 10.0),
            sentiment_risk: sentiment_risk.clamp(1.0, 10.0),
            ..Default::default()
        };
        score.overall_score = score.calculate_overall();
        score.timestamp = chrono::Utc::now().timestamp_millis();
        score
    }

    /// Calculate the weighted overall score.
    fn calculate_overall(&self) -> f64 {
        // Default weights (can be customized)
        let weights = [
            (self.market_risk, 0.25),      // Market risk most important
            (self.credit_risk, 0.15),      // Credit risk
            (self.liquidity_risk, 0.15),   // Liquidity risk
            (self.operational_risk, 0.15), // Operational risk
            (self.regulatory_risk, 0.15),  // Regulatory risk
            (self.sentiment_risk, 0.15),   // Sentiment risk
        ];

        let weighted_sum: f64 = weights.iter().map(|(score, weight)| score * weight).sum();
        weighted_sum.clamp(1.0, 10.0)
    }

    /// Get the risk level based on overall score.
    pub fn risk_level(&self) -> RiskLevel {
        RiskLevel::from_score(self.overall_score)
    }

    /// Set confidence level.
    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set risk direction.
    pub fn with_direction(mut self, direction: RiskDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set key factors.
    pub fn with_factors(mut self, factors: Vec<String>) -> Self {
        self.key_factors = factors;
        self
    }

    /// Set symbol.
    pub fn with_symbol(mut self, symbol: String) -> Self {
        self.symbol = Some(symbol);
        self
    }

    /// Get all dimension scores as a vector.
    pub fn dimension_scores(&self) -> Vec<(&str, f64)> {
        vec![
            ("Market", self.market_risk),
            ("Credit", self.credit_risk),
            ("Liquidity", self.liquidity_risk),
            ("Operational", self.operational_risk),
            ("Regulatory", self.regulatory_risk),
            ("Sentiment", self.sentiment_risk),
        ]
    }

    /// Get the highest risk dimension.
    pub fn highest_risk_dimension(&self) -> (&str, f64) {
        self.dimension_scores()
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("Unknown", 5.0))
    }

    /// Get the lowest risk dimension.
    pub fn lowest_risk_dimension(&self) -> (&str, f64) {
        self.dimension_scores()
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("Unknown", 5.0))
    }

    /// Check if any dimension exceeds a threshold.
    pub fn has_high_risk_dimension(&self, threshold: f64) -> bool {
        self.dimension_scores()
            .iter()
            .any(|(_, score)| *score >= threshold)
    }

    /// Get recommended position size multiplier.
    pub fn position_multiplier(&self) -> f64 {
        let base = self.risk_level().position_multiplier();

        // Adjust based on confidence
        let confidence_adj = match self.confidence {
            Confidence::High => 1.0,
            Confidence::Medium => 0.85,
            Confidence::Low => 0.6,
        };

        // Adjust based on direction
        let direction_adj = match self.direction {
            RiskDirection::Decreasing => 1.1,
            RiskDirection::Stable => 1.0,
            RiskDirection::Increasing => 0.8,
        };

        (base * confidence_adj * direction_adj).clamp(0.0, 1.0)
    }
}

impl fmt::Display for RiskScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Risk Assessment")?;
        writeln!(f, "===============")?;
        if let Some(ref symbol) = self.symbol {
            writeln!(f, "Symbol: {}", symbol)?;
        }
        writeln!(f, "Overall Score: {:.1}/10 ({})", self.overall_score, self.risk_level())?;
        writeln!(f, "Direction: {}", self.direction)?;
        writeln!(f, "Confidence: {}", self.confidence)?;
        writeln!(f)?;
        writeln!(f, "Dimensions:")?;
        writeln!(f, "  Market:      {:.1}/10", self.market_risk)?;
        writeln!(f, "  Credit:      {:.1}/10", self.credit_risk)?;
        writeln!(f, "  Liquidity:   {:.1}/10", self.liquidity_risk)?;
        writeln!(f, "  Operational: {:.1}/10", self.operational_risk)?;
        writeln!(f, "  Regulatory:  {:.1}/10", self.regulatory_risk)?;
        writeln!(f, "  Sentiment:   {:.1}/10", self.sentiment_risk)?;

        if !self.key_factors.is_empty() {
            writeln!(f)?;
            writeln!(f, "Key Factors:")?;
            for factor in &self.key_factors {
                writeln!(f, "  - {}", factor)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_from_score() {
        assert_eq!(RiskLevel::from_score(1.0), RiskLevel::VeryLow);
        assert_eq!(RiskLevel::from_score(3.5), RiskLevel::Low);
        assert_eq!(RiskLevel::from_score(5.5), RiskLevel::Moderate);
        assert_eq!(RiskLevel::from_score(7.5), RiskLevel::High);
        assert_eq!(RiskLevel::from_score(9.5), RiskLevel::Severe);
    }

    #[test]
    fn test_risk_score_creation() {
        let score = RiskScore::new(4.0, 3.0, 2.0, 5.0, 6.0, 4.0);

        assert!(score.overall_score > 0.0);
        assert!(score.overall_score <= 10.0);
        assert_eq!(score.risk_level(), RiskLevel::Low);
    }

    #[test]
    fn test_dimension_analysis() {
        let score = RiskScore::new(3.0, 4.0, 2.0, 8.0, 5.0, 4.0);

        let (highest_name, highest_score) = score.highest_risk_dimension();
        assert_eq!(highest_name, "Operational");
        assert_eq!(highest_score, 8.0);

        let (lowest_name, lowest_score) = score.lowest_risk_dimension();
        assert_eq!(lowest_name, "Liquidity");
        assert_eq!(lowest_score, 2.0);

        assert!(score.has_high_risk_dimension(7.0));
        assert!(!score.has_high_risk_dimension(9.0));
    }

    #[test]
    fn test_position_multiplier() {
        let low_risk = RiskScore::new(2.0, 2.0, 2.0, 2.0, 2.0, 2.0);
        let high_risk = RiskScore::new(8.0, 8.0, 8.0, 8.0, 8.0, 8.0);

        assert!(low_risk.position_multiplier() > high_risk.position_multiplier());
    }
}
