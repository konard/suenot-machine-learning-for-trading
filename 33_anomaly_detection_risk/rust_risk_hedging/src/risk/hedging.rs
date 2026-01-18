//! Hedging strategies and instruments
//!
//! Implements hedging decision logic based on anomaly scores

use crate::anomaly::AnomalyLevel;
use std::collections::HashMap;

/// Hedging instrument type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HedgeInstrument {
    /// Stablecoin (USDT, USDC)
    Stablecoin,
    /// Short position on main asset
    ShortPosition,
    /// Put options
    PutOption,
    /// Inverse perpetual
    InversePerpetual,
    /// Cross-margin collateral
    CrossCollateral,
}

impl HedgeInstrument {
    /// Get estimated annual cost (as percentage)
    pub fn annual_cost(&self) -> f64 {
        match self {
            HedgeInstrument::Stablecoin => 0.0, // Opportunity cost only
            HedgeInstrument::ShortPosition => 0.03, // ~3% funding
            HedgeInstrument::PutOption => 0.05, // ~5% premium decay
            HedgeInstrument::InversePerpetual => 0.04,
            HedgeInstrument::CrossCollateral => 0.02,
        }
    }

    /// Get execution speed (1-10, 10 = fastest)
    pub fn execution_speed(&self) -> u8 {
        match self {
            HedgeInstrument::Stablecoin => 10, // Instant swap
            HedgeInstrument::ShortPosition => 9,
            HedgeInstrument::PutOption => 5, // May have liquidity issues
            HedgeInstrument::InversePerpetual => 8,
            HedgeInstrument::CrossCollateral => 7,
        }
    }

    /// Get capital efficiency (how much protection per dollar)
    pub fn capital_efficiency(&self) -> f64 {
        match self {
            HedgeInstrument::Stablecoin => 1.0, // 1:1
            HedgeInstrument::ShortPosition => 5.0, // 5x leverage typical
            HedgeInstrument::PutOption => 10.0, // High leverage
            HedgeInstrument::InversePerpetual => 5.0,
            HedgeInstrument::CrossCollateral => 2.0,
        }
    }
}

/// Hedge allocation
#[derive(Debug, Clone)]
pub struct HedgeAllocation {
    /// Allocations by instrument (percentage of portfolio)
    pub allocations: HashMap<HedgeInstrument, f64>,
    /// Total hedge percentage
    pub total_hedge_pct: f64,
    /// Estimated annual cost
    pub estimated_annual_cost: f64,
    /// Reason for this allocation
    pub reason: String,
}

impl HedgeAllocation {
    /// Create empty allocation
    pub fn none() -> Self {
        Self {
            allocations: HashMap::new(),
            total_hedge_pct: 0.0,
            estimated_annual_cost: 0.0,
            reason: "No hedging required".into(),
        }
    }

    /// Create allocation for light hedge
    pub fn light() -> Self {
        let mut allocations = HashMap::new();
        allocations.insert(HedgeInstrument::Stablecoin, 0.03);
        allocations.insert(HedgeInstrument::ShortPosition, 0.02);

        Self {
            allocations,
            total_hedge_pct: 0.05,
            estimated_annual_cost: 0.02 * 0.03, // 2% of 3% funding
            reason: "Light hedge for elevated risk".into(),
        }
    }

    /// Create allocation for medium hedge
    pub fn medium() -> Self {
        let mut allocations = HashMap::new();
        allocations.insert(HedgeInstrument::Stablecoin, 0.05);
        allocations.insert(HedgeInstrument::ShortPosition, 0.03);
        allocations.insert(HedgeInstrument::PutOption, 0.02);

        Self {
            allocations,
            total_hedge_pct: 0.10,
            estimated_annual_cost: 0.03 * 0.03 + 0.02 * 0.05,
            reason: "Medium hedge for high risk".into(),
        }
    }

    /// Create allocation for heavy hedge
    pub fn heavy() -> Self {
        let mut allocations = HashMap::new();
        allocations.insert(HedgeInstrument::Stablecoin, 0.10);
        allocations.insert(HedgeInstrument::ShortPosition, 0.05);
        allocations.insert(HedgeInstrument::PutOption, 0.03);
        allocations.insert(HedgeInstrument::InversePerpetual, 0.02);

        Self {
            allocations,
            total_hedge_pct: 0.20,
            estimated_annual_cost: 0.05 * 0.03 + 0.03 * 0.05 + 0.02 * 0.04,
            reason: "Heavy hedge for extreme risk".into(),
        }
    }

    /// Get dollar amounts for each instrument
    pub fn dollar_amounts(&self, portfolio_value: f64) -> HashMap<HedgeInstrument, f64> {
        self.allocations
            .iter()
            .map(|(&instrument, &pct)| (instrument, pct * portfolio_value))
            .collect()
    }

    /// Format for display
    pub fn format(&self) -> String {
        let mut lines = vec![format!("Total Hedge: {:.1}%", self.total_hedge_pct * 100.0)];

        for (instrument, pct) in &self.allocations {
            lines.push(format!("  {:?}: {:.1}%", instrument, pct * 100.0));
        }

        lines.push(format!(
            "Estimated Annual Cost: {:.2}%",
            self.estimated_annual_cost * 100.0
        ));
        lines.push(format!("Reason: {}", self.reason));

        lines.join("\n")
    }
}

/// Hedging strategy configuration
#[derive(Debug, Clone)]
pub struct HedgingStrategy {
    /// Threshold for light hedge
    pub light_threshold: f64,
    /// Threshold for medium hedge
    pub medium_threshold: f64,
    /// Threshold for heavy hedge
    pub heavy_threshold: f64,
    /// Maximum hedge percentage
    pub max_hedge_pct: f64,
    /// Preferred instruments
    pub preferred_instruments: Vec<HedgeInstrument>,
}

impl Default for HedgingStrategy {
    fn default() -> Self {
        Self {
            light_threshold: 0.5,  // 50% anomaly score
            medium_threshold: 0.7, // 70%
            heavy_threshold: 0.9,  // 90%
            max_hedge_pct: 0.25,
            preferred_instruments: vec![
                HedgeInstrument::Stablecoin,
                HedgeInstrument::ShortPosition,
                HedgeInstrument::PutOption,
            ],
        }
    }
}

impl HedgingStrategy {
    /// Decide hedge allocation based on anomaly score
    pub fn decide(&self, anomaly_score: f64, portfolio_value: f64) -> HedgeAllocation {
        let level = AnomalyLevel::from_score(anomaly_score);

        let mut allocation = match level {
            AnomalyLevel::Normal => HedgeAllocation::none(),
            AnomalyLevel::Elevated => HedgeAllocation::light(),
            AnomalyLevel::High => HedgeAllocation::medium(),
            AnomalyLevel::Extreme => HedgeAllocation::heavy(),
        };

        // Cap at maximum
        if allocation.total_hedge_pct > self.max_hedge_pct {
            let scale = self.max_hedge_pct / allocation.total_hedge_pct;
            for pct in allocation.allocations.values_mut() {
                *pct *= scale;
            }
            allocation.total_hedge_pct = self.max_hedge_pct;
        }

        allocation.reason = format!(
            "{} (Score: {:.2}, Level: {:?})",
            allocation.reason, anomaly_score, level
        );

        allocation
    }

    /// Calculate expected protection during a crisis
    pub fn expected_protection(&self, allocation: &HedgeAllocation, drawdown: f64) -> f64 {
        // Simple model: hedge reduces drawdown proportionally
        let hedge_effectiveness = 0.8; // 80% of theoretical protection
        let protected = allocation.total_hedge_pct * hedge_effectiveness * drawdown;
        protected
    }

    /// Calculate hedge efficiency ratio
    pub fn hedge_efficiency(&self, allocation: &HedgeAllocation, expected_drawdown: f64) -> f64 {
        let protection = self.expected_protection(allocation, expected_drawdown);
        let cost = allocation.estimated_annual_cost;

        if cost < 1e-10 {
            return f64::INFINITY;
        }

        protection / cost
    }
}

/// Dynamic hedge sizing based on volatility
#[derive(Debug, Clone)]
pub struct VolatilityScaledHedge {
    /// Base strategy
    pub base: HedgingStrategy,
    /// Target volatility for scaling
    pub target_vol: f64,
    /// Maximum scale factor
    pub max_scale: f64,
}

impl Default for VolatilityScaledHedge {
    fn default() -> Self {
        Self {
            base: HedgingStrategy::default(),
            target_vol: 2.0, // 2% daily volatility
            max_scale: 2.0,
        }
    }
}

impl VolatilityScaledHedge {
    /// Decide with volatility scaling
    pub fn decide(
        &self,
        anomaly_score: f64,
        current_vol: f64,
        portfolio_value: f64,
    ) -> HedgeAllocation {
        let mut allocation = self.base.decide(anomaly_score, portfolio_value);

        // Scale by volatility ratio
        let vol_ratio = (current_vol / self.target_vol).min(self.max_scale);

        if vol_ratio > 1.0 {
            for pct in allocation.allocations.values_mut() {
                *pct *= vol_ratio;
            }
            allocation.total_hedge_pct *= vol_ratio;
            allocation.estimated_annual_cost *= vol_ratio;

            // Cap at maximum
            if allocation.total_hedge_pct > self.base.max_hedge_pct {
                let scale = self.base.max_hedge_pct / allocation.total_hedge_pct;
                for pct in allocation.allocations.values_mut() {
                    *pct *= scale;
                }
                allocation.total_hedge_pct = self.base.max_hedge_pct;
            }

            allocation.reason = format!(
                "{} [Vol scaled {:.1}x]",
                allocation.reason, vol_ratio
            );
        }

        allocation
    }
}

/// Hedge execution plan
#[derive(Debug, Clone)]
pub struct HedgeExecutionPlan {
    /// Target allocation
    pub target: HedgeAllocation,
    /// Current positions
    pub current: HashMap<HedgeInstrument, f64>,
    /// Actions needed
    pub actions: Vec<HedgeAction>,
}

/// Single hedge action
#[derive(Debug, Clone)]
pub struct HedgeAction {
    pub instrument: HedgeInstrument,
    pub action_type: HedgeActionType,
    pub amount: f64,
    pub priority: u8,
}

/// Type of hedge action
#[derive(Debug, Clone, Copy)]
pub enum HedgeActionType {
    Open,
    Increase,
    Decrease,
    Close,
}

impl HedgeExecutionPlan {
    /// Create execution plan from current and target states
    pub fn create(
        target: HedgeAllocation,
        current: HashMap<HedgeInstrument, f64>,
        portfolio_value: f64,
    ) -> Self {
        let mut actions = Vec::new();

        let target_amounts = target.dollar_amounts(portfolio_value);

        // Check each instrument
        for (&instrument, &target_amount) in &target_amounts {
            let current_amount = *current.get(&instrument).unwrap_or(&0.0);
            let diff = target_amount - current_amount;

            if diff.abs() > portfolio_value * 0.001 {
                // > 0.1% difference
                let action_type = if current_amount < 1e-10 && target_amount > 0.0 {
                    HedgeActionType::Open
                } else if diff > 0.0 {
                    HedgeActionType::Increase
                } else if target_amount < 1e-10 {
                    HedgeActionType::Close
                } else {
                    HedgeActionType::Decrease
                };

                actions.push(HedgeAction {
                    instrument,
                    action_type,
                    amount: diff.abs(),
                    priority: instrument.execution_speed(),
                });
            }
        }

        // Check for positions to close
        for (&instrument, &current_amount) in &current {
            if current_amount > 0.0 && !target_amounts.contains_key(&instrument) {
                actions.push(HedgeAction {
                    instrument,
                    action_type: HedgeActionType::Close,
                    amount: current_amount,
                    priority: 10, // High priority to close unnecessary hedges
                });
            }
        }

        // Sort by priority (higher first)
        actions.sort_by(|a, b| b.priority.cmp(&a.priority));

        Self {
            target,
            current,
            actions,
        }
    }

    /// Format execution plan
    pub fn format(&self) -> String {
        let mut lines = vec!["=== Hedge Execution Plan ===".to_string()];

        if self.actions.is_empty() {
            lines.push("No actions needed".to_string());
        } else {
            for action in &self.actions {
                lines.push(format!(
                    "{:?} {:?}: ${:.2} (priority: {})",
                    action.action_type, action.instrument, action.amount, action.priority
                ));
            }
        }

        lines.push("\nTarget allocation:".to_string());
        lines.push(self.target.format());

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hedge_allocation() {
        let alloc = HedgeAllocation::medium();
        assert!((alloc.total_hedge_pct - 0.10).abs() < 0.01);

        let amounts = alloc.dollar_amounts(100_000.0);
        let total: f64 = amounts.values().sum();
        assert!((total - 10_000.0).abs() < 100.0);
    }

    #[test]
    fn test_hedging_strategy() {
        let strategy = HedgingStrategy::default();

        // Normal conditions
        let alloc_normal = strategy.decide(0.3, 100_000.0);
        assert_eq!(alloc_normal.total_hedge_pct, 0.0);

        // High risk
        let alloc_high = strategy.decide(0.8, 100_000.0);
        assert!(alloc_high.total_hedge_pct >= 0.05);

        // Extreme risk
        let alloc_extreme = strategy.decide(0.95, 100_000.0);
        assert!(alloc_extreme.total_hedge_pct >= 0.10);
    }

    #[test]
    fn test_execution_plan() {
        let target = HedgeAllocation::light();
        let current = HashMap::new();

        let plan = HedgeExecutionPlan::create(target, current, 100_000.0);

        assert!(!plan.actions.is_empty());
    }
}
