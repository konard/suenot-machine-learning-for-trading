//! Portfolio construction from signals

use super::signal::{Signal, SignalType};
use std::collections::HashMap;

/// Portfolio position
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub weight: f64,
    pub direction: i32,  // 1 = long, -1 = short
}

/// Portfolio allocation
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub positions: Vec<Position>,
    pub cash_weight: f64,
}

impl Portfolio {
    /// Create empty portfolio
    pub fn empty() -> Self {
        Self {
            positions: vec![],
            cash_weight: 1.0,
        }
    }

    /// Get total invested weight
    pub fn invested_weight(&self) -> f64 {
        self.positions.iter().map(|p| p.weight.abs()).sum()
    }

    /// Get long exposure
    pub fn long_exposure(&self) -> f64 {
        self.positions
            .iter()
            .filter(|p| p.direction > 0)
            .map(|p| p.weight)
            .sum()
    }

    /// Get short exposure
    pub fn short_exposure(&self) -> f64 {
        self.positions
            .iter()
            .filter(|p| p.direction < 0)
            .map(|p| p.weight.abs())
            .sum()
    }

    /// Get net exposure
    pub fn net_exposure(&self) -> f64 {
        self.long_exposure() - self.short_exposure()
    }
}

/// Portfolio builder configuration
#[derive(Debug, Clone)]
pub struct PortfolioBuilderConfig {
    /// Maximum weight per position
    pub max_position_weight: f64,
    /// Minimum weight per position
    pub min_position_weight: f64,
    /// Maximum total exposure
    pub max_total_exposure: f64,
    /// Maximum number of positions
    pub max_positions: usize,
    /// Whether to allow short positions
    pub allow_shorts: bool,
}

impl Default for PortfolioBuilderConfig {
    fn default() -> Self {
        Self {
            max_position_weight: 0.2,
            min_position_weight: 0.02,
            max_total_exposure: 1.0,
            max_positions: 10,
            allow_shorts: true,
        }
    }
}

/// Portfolio builder
pub struct PortfolioBuilder {
    config: PortfolioBuilderConfig,
}

impl Default for PortfolioBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PortfolioBuilder {
    pub fn new() -> Self {
        Self {
            config: PortfolioBuilderConfig::default(),
        }
    }

    pub fn with_config(config: PortfolioBuilderConfig) -> Self {
        Self { config }
    }

    /// Build portfolio from signals
    pub fn build(&self, signals: &[Signal]) -> Portfolio {
        if signals.is_empty() {
            return Portfolio::empty();
        }

        // Filter actionable signals
        let mut actionable: Vec<&Signal> = signals
            .iter()
            .filter(|s| s.is_actionable())
            .filter(|s| self.config.allow_shorts || s.signal_type == SignalType::Long)
            .collect();

        // Sort by confidence
        actionable.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Take top signals
        actionable.truncate(self.config.max_positions);

        if actionable.is_empty() {
            return Portfolio::empty();
        }

        // Calculate weights based on confidence
        let total_confidence: f64 = actionable.iter().map(|s| s.confidence).sum();
        let mut positions = Vec::new();
        let mut total_weight = 0.0;

        for signal in actionable {
            // Weight proportional to confidence
            let raw_weight = signal.confidence / total_confidence * self.config.max_total_exposure;

            // Apply position limits
            let weight = raw_weight
                .min(self.config.max_position_weight)
                .max(self.config.min_position_weight);

            // Check if adding this position would exceed max exposure
            if total_weight + weight > self.config.max_total_exposure {
                break;
            }

            positions.push(Position {
                symbol: signal.symbol.clone(),
                weight,
                direction: signal.direction(),
            });

            total_weight += weight;
        }

        Portfolio {
            positions,
            cash_weight: 1.0 - total_weight,
        }
    }

    /// Build equal-weighted portfolio from signals
    pub fn build_equal_weight(&self, signals: &[Signal]) -> Portfolio {
        let actionable: Vec<&Signal> = signals
            .iter()
            .filter(|s| s.is_actionable())
            .filter(|s| self.config.allow_shorts || s.signal_type == SignalType::Long)
            .take(self.config.max_positions)
            .collect();

        if actionable.is_empty() {
            return Portfolio::empty();
        }

        let weight = (self.config.max_total_exposure / actionable.len() as f64)
            .min(self.config.max_position_weight);

        let positions: Vec<Position> = actionable
            .iter()
            .map(|s| Position {
                symbol: s.symbol.clone(),
                weight,
                direction: s.direction(),
            })
            .collect();

        let total_weight: f64 = positions.iter().map(|p| p.weight).sum();

        Portfolio {
            positions,
            cash_weight: 1.0 - total_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_builder() {
        let signals = vec![
            Signal::new("BTCUSDT", SignalType::Long, 0.8, 0.02),
            Signal::new("ETHUSDT", SignalType::Long, 0.7, 0.015),
            Signal::new("SOLUSDT", SignalType::Short, 0.6, -0.01),
        ];

        let builder = PortfolioBuilder::new();
        let portfolio = builder.build(&signals);

        assert_eq!(portfolio.positions.len(), 3);
        assert!(portfolio.invested_weight() <= 1.0);
        assert!(portfolio.cash_weight >= 0.0);
    }

    #[test]
    fn test_equal_weight() {
        let signals = vec![
            Signal::new("BTCUSDT", SignalType::Long, 0.8, 0.02),
            Signal::new("ETHUSDT", SignalType::Long, 0.6, 0.01),
        ];

        let builder = PortfolioBuilder::new();
        let portfolio = builder.build_equal_weight(&signals);

        assert_eq!(portfolio.positions.len(), 2);
        assert!((portfolio.positions[0].weight - portfolio.positions[1].weight).abs() < 0.01);
    }
}
