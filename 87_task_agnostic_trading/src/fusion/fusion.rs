//! Decision fusion for combining multi-task predictions into trading signals

use crate::tasks::{
    MultiTaskPrediction,
    Direction,
    MarketRegime,
};
use serde::{Deserialize, Serialize};

/// Fusion method for combining predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionMethod {
    /// Simple voting based on task predictions
    Voting,
    /// Weighted combination based on confidence
    WeightedConfidence,
    /// Bayesian combination
    Bayesian,
    /// Rule-based fusion
    RuleBased,
}

impl Default for FusionMethod {
    fn default() -> Self {
        Self::WeightedConfidence
    }
}

/// Configuration for decision fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Fusion method to use
    pub method: FusionMethod,
    /// Minimum confidence threshold for trading
    pub min_confidence: f64,
    /// Minimum task agreement ratio
    pub min_agreement: f64,
    /// Weight for direction prediction
    pub direction_weight: f64,
    /// Weight for volatility prediction
    pub volatility_weight: f64,
    /// Weight for regime prediction
    pub regime_weight: f64,
    /// Weight for returns prediction
    pub returns_weight: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::WeightedConfidence,
            min_confidence: 0.6,
            min_agreement: 0.5,
            direction_weight: 0.3,
            volatility_weight: 0.2,
            regime_weight: 0.2,
            returns_weight: 0.3,
        }
    }
}

/// Trading decision type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingDecision {
    /// Open long position
    Long,
    /// Open short position
    Short,
    /// Close existing positions / stay flat
    Flat,
    /// Hold current position
    Hold,
}

impl std::fmt::Display for TradingDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingDecision::Long => write!(f, "LONG"),
            TradingDecision::Short => write!(f, "SHORT"),
            TradingDecision::Flat => write!(f, "FLAT"),
            TradingDecision::Hold => write!(f, "HOLD"),
        }
    }
}

/// Confidence breakdown for a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionConfidence {
    /// Overall confidence score
    pub overall: f64,
    /// Confidence from direction prediction
    pub direction_confidence: f64,
    /// Confidence from volatility prediction
    pub volatility_confidence: f64,
    /// Confidence from regime prediction
    pub regime_confidence: f64,
    /// Confidence from returns prediction
    pub returns_confidence: f64,
    /// Task agreement score (0-1)
    pub task_agreement: f64,
    /// Risk-adjusted confidence
    pub risk_adjusted: f64,
}

/// Decision fusion result
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// The trading decision
    pub decision: TradingDecision,
    /// Confidence information
    pub confidence: DecisionConfidence,
    /// Suggested position size (0-1, relative to max)
    pub position_size: f64,
    /// Reasoning for the decision
    pub reasoning: Vec<String>,
}

/// Decision fusion for combining multi-task predictions
pub struct DecisionFusion {
    config: FusionConfig,
}

impl DecisionFusion {
    /// Create a new decision fusion
    pub fn new(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_fusion() -> Self {
        Self::new(FusionConfig::default())
    }

    /// Fuse predictions into a trading decision
    pub fn fuse(&self, prediction: &MultiTaskPrediction) -> FusionResult {
        match self.config.method {
            FusionMethod::Voting => self.voting_fusion(prediction),
            FusionMethod::WeightedConfidence => self.weighted_fusion(prediction),
            FusionMethod::Bayesian => self.bayesian_fusion(prediction),
            FusionMethod::RuleBased => self.rule_based_fusion(prediction),
        }
    }

    /// Simple voting-based fusion
    fn voting_fusion(&self, prediction: &MultiTaskPrediction) -> FusionResult {
        let mut long_votes = 0;
        let mut short_votes = 0;
        let mut flat_votes = 0;
        let mut reasoning = Vec::new();

        // Direction vote
        if let Some(ref dir) = prediction.direction {
            match dir.direction {
                Direction::Up => {
                    long_votes += 1;
                    reasoning.push(format!("Direction: Up ({:.1}%)", dir.confidence * 100.0));
                }
                Direction::Down => {
                    short_votes += 1;
                    reasoning.push(format!("Direction: Down ({:.1}%)", dir.confidence * 100.0));
                }
                Direction::Sideways => {
                    flat_votes += 1;
                    reasoning.push(format!("Direction: Sideways ({:.1}%)", dir.confidence * 100.0));
                }
            }
        }

        // Returns vote
        if let Some(ref ret) = prediction.returns {
            if ret.is_bullish() {
                long_votes += 1;
                reasoning.push(format!("Returns: Bullish ({:.2}%)", ret.return_pct));
            } else if ret.is_bearish() {
                short_votes += 1;
                reasoning.push(format!("Returns: Bearish ({:.2}%)", ret.return_pct));
            } else {
                flat_votes += 1;
                reasoning.push("Returns: Neutral".to_string());
            }
        }

        // Regime vote
        if let Some(ref regime) = prediction.regime {
            match regime.regime {
                MarketRegime::Trending => {
                    // Trending supports directional trades
                    if long_votes > short_votes {
                        long_votes += 1;
                    } else {
                        short_votes += 1;
                    }
                    reasoning.push(format!("Regime: Trending (risk: {})", regime.risk_level));
                }
                MarketRegime::Crash => {
                    short_votes += 2; // Strong bearish signal
                    reasoning.push("Regime: CRASH - bearish bias".to_string());
                }
                MarketRegime::Volatile => {
                    flat_votes += 1;
                    reasoning.push("Regime: Volatile - reduced conviction".to_string());
                }
                _ => {
                    reasoning.push(format!("Regime: {}", regime.regime));
                }
            }
        }

        // Determine decision
        let total_votes = long_votes + short_votes + flat_votes;
        let (decision, winning_votes) = if long_votes > short_votes && long_votes > flat_votes {
            (TradingDecision::Long, long_votes)
        } else if short_votes > long_votes && short_votes > flat_votes {
            (TradingDecision::Short, short_votes)
        } else {
            (TradingDecision::Flat, flat_votes)
        };

        let agreement = if total_votes > 0 {
            winning_votes as f64 / total_votes as f64
        } else {
            0.0
        };

        let confidence = self.compute_confidence(prediction, agreement);

        // Position size based on agreement
        let position_size = if confidence.overall >= self.config.min_confidence {
            (agreement * confidence.overall).min(1.0)
        } else {
            0.0
        };

        FusionResult {
            decision: if position_size > 0.0 { decision } else { TradingDecision::Hold },
            confidence,
            position_size,
            reasoning,
        }
    }

    /// Weighted confidence fusion
    fn weighted_fusion(&self, prediction: &MultiTaskPrediction) -> FusionResult {
        let mut bullish_score = 0.0;
        let mut bearish_score = 0.0;
        let mut total_weight = 0.0;
        let mut reasoning = Vec::new();

        // Direction contribution
        if let Some(ref dir) = prediction.direction {
            let weight = self.config.direction_weight * dir.confidence;
            match dir.direction {
                Direction::Up => bullish_score += weight,
                Direction::Down => bearish_score += weight,
                Direction::Sideways => {} // Neutral
            }
            total_weight += self.config.direction_weight;
            reasoning.push(format!("Direction: {} ({:.1}%)", dir.direction, dir.confidence * 100.0));
        }

        // Returns contribution
        if let Some(ref ret) = prediction.returns {
            let weight = self.config.returns_weight * ret.confidence;
            if ret.return_pct > 0.0 {
                bullish_score += weight * (ret.return_pct / 5.0).min(1.0); // Normalize
            } else {
                bearish_score += weight * (ret.return_pct.abs() / 5.0).min(1.0);
            }
            total_weight += self.config.returns_weight;
            reasoning.push(format!("Expected return: {:.2}%", ret.return_pct));
        }

        // Volatility adjustment
        if let Some(ref vol) = prediction.volatility {
            // High volatility reduces position size, not direction
            let vol_factor = 1.0 / (1.0 + vol.volatility_pct / 5.0);
            bullish_score *= vol_factor;
            bearish_score *= vol_factor;
            reasoning.push(format!("Volatility: {:.2}% ({})", vol.volatility_pct, vol.level));
        }

        // Regime adjustment
        if let Some(ref regime) = prediction.regime {
            let risk_factor = 1.0 / (regime.risk_level as f64 / 2.0);
            bullish_score *= risk_factor;
            bearish_score *= risk_factor;

            // Special handling for crash regime
            if regime.regime == MarketRegime::Crash {
                bearish_score *= 2.0;
                reasoning.push("CRASH regime detected - bearish bias".to_string());
            }
            reasoning.push(format!("Regime: {} (risk {})", regime.regime, regime.risk_level));
        }

        // Determine decision
        let net_score = if total_weight > 0.0 {
            (bullish_score - bearish_score) / total_weight
        } else {
            0.0
        };

        let decision = if net_score > 0.2 {
            TradingDecision::Long
        } else if net_score < -0.2 {
            TradingDecision::Short
        } else {
            TradingDecision::Flat
        };

        let agreement = 1.0 - (bullish_score - bearish_score).abs() / (bullish_score + bearish_score + 0.01);
        let confidence = self.compute_confidence(prediction, 1.0 - agreement);

        let position_size = if confidence.overall >= self.config.min_confidence {
            net_score.abs().min(1.0)
        } else {
            0.0
        };

        FusionResult {
            decision: if position_size > 0.0 { decision } else { TradingDecision::Hold },
            confidence,
            position_size,
            reasoning,
        }
    }

    /// Bayesian fusion (simplified)
    fn bayesian_fusion(&self, prediction: &MultiTaskPrediction) -> FusionResult {
        // Use weighted fusion as base, then apply Bayesian update
        let base_result = self.weighted_fusion(prediction);

        // Prior: assume 50/50 for up/down
        let mut prior_up = 0.5;
        let mut prior_down = 0.5;

        // Update based on each prediction
        if let Some(ref dir) = prediction.direction {
            let likelihood = dir.confidence;
            match dir.direction {
                Direction::Up => prior_up *= likelihood,
                Direction::Down => prior_down *= likelihood,
                Direction::Sideways => {
                    prior_up *= 0.5;
                    prior_down *= 0.5;
                }
            }
        }

        // Normalize
        let total = prior_up + prior_down;
        if total > 0.0 {
            prior_up /= total;
            prior_down /= total;
        }

        let decision = if prior_up > 0.6 {
            TradingDecision::Long
        } else if prior_down > 0.6 {
            TradingDecision::Short
        } else {
            TradingDecision::Flat
        };

        FusionResult {
            decision,
            confidence: base_result.confidence,
            position_size: prior_up.max(prior_down) - 0.5,
            reasoning: base_result.reasoning,
        }
    }

    /// Rule-based fusion
    fn rule_based_fusion(&self, prediction: &MultiTaskPrediction) -> FusionResult {
        let mut reasoning = Vec::new();

        // Rule 1: Never trade in crash regime
        if let Some(ref regime) = prediction.regime {
            if regime.regime == MarketRegime::Crash && regime.confidence > 0.6 {
                reasoning.push("RULE: Crash regime detected - stay flat".to_string());
                return FusionResult {
                    decision: TradingDecision::Flat,
                    confidence: self.compute_confidence(prediction, 0.0),
                    position_size: 0.0,
                    reasoning,
                };
            }
        }

        // Rule 2: Require direction and returns to agree
        let dir_bullish = prediction.direction.as_ref()
            .map(|d| d.direction == Direction::Up && d.confidence > 0.6)
            .unwrap_or(false);
        let dir_bearish = prediction.direction.as_ref()
            .map(|d| d.direction == Direction::Down && d.confidence > 0.6)
            .unwrap_or(false);

        let returns_bullish = prediction.returns.as_ref()
            .map(|r| r.is_bullish() && r.confidence > 0.5)
            .unwrap_or(false);
        let returns_bearish = prediction.returns.as_ref()
            .map(|r| r.is_bearish() && r.confidence > 0.5)
            .unwrap_or(false);

        let decision = if dir_bullish && returns_bullish {
            reasoning.push("RULE: Direction and returns both bullish".to_string());
            TradingDecision::Long
        } else if dir_bearish && returns_bearish {
            reasoning.push("RULE: Direction and returns both bearish".to_string());
            TradingDecision::Short
        } else {
            reasoning.push("RULE: No agreement - stay flat".to_string());
            TradingDecision::Flat
        };

        // Rule 3: Reduce size in high volatility
        let vol_adjustment = prediction.volatility.as_ref()
            .map(|v| if v.volatility_pct > 3.0 { 0.5 } else { 1.0 })
            .unwrap_or(1.0);

        if vol_adjustment < 1.0 {
            reasoning.push("RULE: High volatility - reduced position".to_string());
        }

        let base_size = if decision != TradingDecision::Flat { 0.8 } else { 0.0 };
        let position_size = base_size * vol_adjustment;

        FusionResult {
            decision,
            confidence: self.compute_confidence(prediction, if decision != TradingDecision::Flat { 1.0 } else { 0.0 }),
            position_size,
            reasoning,
        }
    }

    /// Compute confidence breakdown
    fn compute_confidence(&self, prediction: &MultiTaskPrediction, agreement: f64) -> DecisionConfidence {
        let dir_conf = prediction.direction.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let vol_conf = prediction.volatility.as_ref().map(|v| v.confidence).unwrap_or(0.0);
        let regime_conf = prediction.regime.as_ref().map(|r| r.confidence).unwrap_or(0.0);
        let returns_conf = prediction.returns.as_ref().map(|r| r.confidence).unwrap_or(0.0);

        let overall = (
            dir_conf * self.config.direction_weight +
            vol_conf * self.config.volatility_weight +
            regime_conf * self.config.regime_weight +
            returns_conf * self.config.returns_weight
        ) / (self.config.direction_weight + self.config.volatility_weight +
             self.config.regime_weight + self.config.returns_weight);

        // Risk adjustment based on regime
        let risk_factor = prediction.regime.as_ref()
            .map(|r| 1.0 / (r.risk_level as f64 / 2.0))
            .unwrap_or(1.0);

        DecisionConfidence {
            overall,
            direction_confidence: dir_conf,
            volatility_confidence: vol_conf,
            regime_confidence: regime_conf,
            returns_confidence: returns_conf,
            task_agreement: agreement,
            risk_adjusted: overall * risk_factor * agreement,
        }
    }

    /// Get config
    pub fn config(&self) -> &FusionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{
        DirectionPrediction,
        VolatilityPrediction, VolatilityLevel,
        RegimePrediction,
        ReturnsPrediction,
    };

    fn create_bullish_prediction() -> MultiTaskPrediction {
        let mut pred = MultiTaskPrediction::new();

        pred.direction = Some(DirectionPrediction {
            direction: Direction::Up,
            confidence: 0.8,
            probabilities: [0.8, 0.1, 0.1],
        });

        pred.returns = Some(ReturnsPrediction {
            return_pct: 2.5,
            confidence: 0.7,
            lower_bound: 1.0,
            upper_bound: 4.0,
            risk_adjusted: 1.5,
        });

        pred.volatility = Some(VolatilityPrediction {
            volatility_pct: 2.0,
            level: VolatilityLevel::Medium,
            confidence: 0.6,
            lower_bound: 1.0,
            upper_bound: 3.0,
        });

        pred.regime = Some(RegimePrediction::from_probabilities(&[0.7, 0.1, 0.1, 0.05, 0.05]));

        pred
    }

    #[test]
    fn test_voting_fusion() {
        let fusion = DecisionFusion::new(FusionConfig {
            method: FusionMethod::Voting,
            ..Default::default()
        });

        let pred = create_bullish_prediction();
        let result = fusion.fuse(&pred);

        assert_eq!(result.decision, TradingDecision::Long);
        assert!(result.position_size > 0.0);
    }

    #[test]
    fn test_weighted_fusion() {
        let fusion = DecisionFusion::default_fusion();
        let pred = create_bullish_prediction();
        let result = fusion.fuse(&pred);

        assert!(result.confidence.overall > 0.5);
        assert!(!result.reasoning.is_empty());
    }

    #[test]
    fn test_rule_based_crash() {
        let fusion = DecisionFusion::new(FusionConfig {
            method: FusionMethod::RuleBased,
            ..Default::default()
        });

        let mut pred = create_bullish_prediction();
        // Override regime to crash
        pred.regime = Some(RegimePrediction {
            regime: MarketRegime::Crash,
            confidence: 0.8,
            probabilities: vec![0.05, 0.05, 0.05, 0.8, 0.05],
            risk_level: 5,
            recommendation: "Stay out".to_string(),
        });

        let result = fusion.fuse(&pred);
        assert_eq!(result.decision, TradingDecision::Flat);
    }
}
