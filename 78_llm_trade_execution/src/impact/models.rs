//! Market impact models.

use serde::{Deserialize, Serialize};

/// Market impact component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactComponent {
    /// Permanent impact (affects future prices)
    pub permanent: f64,
    /// Temporary impact (affects only this trade)
    pub temporary: f64,
    /// Total impact
    pub total: f64,
}

impl ImpactComponent {
    /// Create a new impact component
    pub fn new(permanent: f64, temporary: f64) -> Self {
        Self {
            permanent,
            temporary,
            total: permanent + temporary,
        }
    }

    /// Get impact in basis points
    pub fn as_bps(&self) -> f64 {
        self.total * 10000.0
    }
}

/// Permanent impact parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermanentImpact {
    /// Linear coefficient (gamma)
    pub gamma: f64,
    /// Exponent (typically 0.5-1.0)
    pub exponent: f64,
}

impl Default for PermanentImpact {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            exponent: 0.5,
        }
    }
}

impl PermanentImpact {
    /// Calculate permanent impact for a given trade fraction
    pub fn calculate(&self, fraction: f64) -> f64 {
        self.gamma * fraction.powf(self.exponent)
    }
}

/// Temporary impact parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporaryImpact {
    /// Linear coefficient (eta)
    pub eta: f64,
    /// Exponent (typically 0.5-1.0)
    pub exponent: f64,
}

impl Default for TemporaryImpact {
    fn default() -> Self {
        Self {
            eta: 0.2,
            exponent: 0.6,
        }
    }
}

impl TemporaryImpact {
    /// Calculate temporary impact for a given trade rate
    pub fn calculate(&self, rate: f64) -> f64 {
        self.eta * rate.powf(self.exponent)
    }
}

/// Almgren-Chriss model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlmgrenChrissParams {
    /// Daily volatility (sigma)
    pub sigma: f64,
    /// Daily trading volume (average)
    pub daily_volume: f64,
    /// Permanent impact parameter (gamma)
    pub gamma: f64,
    /// Temporary impact parameter (eta)
    pub eta: f64,
    /// Risk aversion parameter (lambda)
    pub lambda: f64,
    /// Bid-ask spread (as fraction)
    pub spread: f64,
}

impl Default for AlmgrenChrissParams {
    fn default() -> Self {
        Self {
            sigma: 0.02,        // 2% daily volatility
            daily_volume: 1e6,  // 1M daily volume
            gamma: 0.1,         // Permanent impact coefficient
            eta: 0.2,           // Temporary impact coefficient
            lambda: 1e-6,       // Risk aversion
            spread: 0.0005,     // 5 bps spread
        }
    }
}

impl AlmgrenChrissParams {
    /// Create parameters for a liquid asset
    pub fn liquid() -> Self {
        Self {
            sigma: 0.015,
            daily_volume: 1e7,
            gamma: 0.05,
            eta: 0.1,
            lambda: 1e-7,
            spread: 0.0002,
        }
    }

    /// Create parameters for an illiquid asset
    pub fn illiquid() -> Self {
        Self {
            sigma: 0.04,
            daily_volume: 1e5,
            gamma: 0.3,
            eta: 0.5,
            lambda: 1e-5,
            spread: 0.002,
        }
    }

    /// Create parameters for crypto
    pub fn crypto() -> Self {
        Self {
            sigma: 0.04,        // Higher volatility
            daily_volume: 1e8,  // High volume for major pairs
            gamma: 0.08,
            eta: 0.15,
            lambda: 1e-6,
            spread: 0.0003,
        }
    }
}

/// Market impact model trait
pub trait MarketImpactModel: Send + Sync {
    /// Calculate the impact of executing a given quantity
    fn calculate_impact(&self, quantity: f64, time_horizon: f64) -> ImpactComponent;

    /// Calculate the optimal execution trajectory
    fn optimal_trajectory(&self, total_quantity: f64, num_periods: usize) -> Vec<f64>;

    /// Calculate expected execution cost
    fn expected_cost(&self, quantity: f64, time_horizon: f64) -> f64;

    /// Calculate execution risk (variance)
    fn execution_risk(&self, quantity: f64, time_horizon: f64) -> f64;
}

/// Almgren-Chriss optimal execution model
#[derive(Debug, Clone)]
pub struct AlmgrenChrissModel {
    params: AlmgrenChrissParams,
}

impl AlmgrenChrissModel {
    /// Create a new Almgren-Chriss model
    pub fn new(params: AlmgrenChrissParams) -> Self {
        Self { params }
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self::new(AlmgrenChrissParams::default())
    }

    /// Get the model parameters
    pub fn params(&self) -> &AlmgrenChrissParams {
        &self.params
    }

    /// Calculate the urgency parameter (kappa)
    fn kappa(&self, time_horizon: f64) -> f64 {
        let lambda = self.params.lambda;
        let sigma = self.params.sigma;
        let eta = self.params.eta;

        ((lambda * sigma.powi(2)) / eta).sqrt() / time_horizon.sqrt()
    }

    /// Calculate half-life of the trading trajectory
    pub fn half_life(&self, time_horizon: f64) -> f64 {
        let kappa = self.kappa(time_horizon);
        (2.0_f64.ln()) / kappa
    }
}

impl MarketImpactModel for AlmgrenChrissModel {
    fn calculate_impact(&self, quantity: f64, time_horizon: f64) -> ImpactComponent {
        // Fraction of daily volume
        let fraction = quantity / self.params.daily_volume;

        // Trading rate
        let rate = fraction / time_horizon;

        // Permanent impact: g(x) = gamma * x
        let permanent = self.params.gamma * fraction;

        // Temporary impact: h(v) = eta * v + spread/2
        let temporary = self.params.eta * rate + self.params.spread / 2.0;

        ImpactComponent::new(permanent, temporary)
    }

    fn optimal_trajectory(&self, total_quantity: f64, num_periods: usize) -> Vec<f64> {
        if num_periods == 0 {
            return vec![];
        }

        let time_horizon = num_periods as f64;
        let kappa = self.kappa(time_horizon);

        let mut trajectory = Vec::with_capacity(num_periods);
        let mut remaining = total_quantity;

        for t in 0..num_periods {
            let time = t as f64;
            let time_remaining = time_horizon - time;

            // Optimal trading rate from Almgren-Chriss
            let sinh_kt = (kappa * time_remaining).sinh();
            let sinh_kT = (kappa * time_horizon).sinh();

            let trade_fraction = if sinh_kT > 0.0 {
                sinh_kt / sinh_kT
            } else {
                1.0 / num_periods as f64
            };

            let trade_quantity = if t == num_periods - 1 {
                remaining // Execute all remaining on last period
            } else {
                let qty = total_quantity * (1.0 - trade_fraction) - (total_quantity - remaining);
                qty.max(0.0).min(remaining)
            };

            trajectory.push(trade_quantity);
            remaining -= trade_quantity;
        }

        trajectory
    }

    fn expected_cost(&self, quantity: f64, time_horizon: f64) -> f64 {
        let impact = self.calculate_impact(quantity, time_horizon);

        // Total expected cost = permanent + temporary impact
        quantity * (impact.permanent + impact.temporary)
    }

    fn execution_risk(&self, quantity: f64, time_horizon: f64) -> f64 {
        // Timing risk from Almgren-Chriss
        let sigma = self.params.sigma;
        let fraction = quantity / self.params.daily_volume;

        // Variance proportional to sigma^2 * X^2 * T
        sigma.powi(2) * fraction.powi(2) * time_horizon
    }
}

/// Square-root impact model (simpler alternative)
#[derive(Debug, Clone)]
pub struct SquareRootModel {
    /// Impact coefficient
    pub coefficient: f64,
    /// Daily volume
    pub daily_volume: f64,
    /// Daily volatility
    pub volatility: f64,
}

impl Default for SquareRootModel {
    fn default() -> Self {
        Self {
            coefficient: 0.5,
            daily_volume: 1e6,
            volatility: 0.02,
        }
    }
}

impl MarketImpactModel for SquareRootModel {
    fn calculate_impact(&self, quantity: f64, _time_horizon: f64) -> ImpactComponent {
        let fraction = quantity / self.daily_volume;

        // Square-root impact model: I = sigma * c * sqrt(X/V)
        let impact = self.volatility * self.coefficient * fraction.sqrt();

        // Split roughly equally between permanent and temporary
        ImpactComponent::new(impact * 0.5, impact * 0.5)
    }

    fn optimal_trajectory(&self, total_quantity: f64, num_periods: usize) -> Vec<f64> {
        // For square-root model, uniform execution is near-optimal
        if num_periods == 0 {
            return vec![];
        }

        let slice = total_quantity / num_periods as f64;
        vec![slice; num_periods]
    }

    fn expected_cost(&self, quantity: f64, time_horizon: f64) -> f64 {
        let impact = self.calculate_impact(quantity, time_horizon);
        quantity * impact.total
    }

    fn execution_risk(&self, quantity: f64, time_horizon: f64) -> f64 {
        let fraction = quantity / self.daily_volume;
        self.volatility.powi(2) * fraction.powi(2) * time_horizon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_almgren_chriss_impact() {
        let model = AlmgrenChrissModel::default_params();
        let impact = model.calculate_impact(10000.0, 1.0);

        assert!(impact.permanent > 0.0);
        assert!(impact.temporary > 0.0);
        assert!(impact.total > 0.0);
    }

    #[test]
    fn test_optimal_trajectory() {
        let model = AlmgrenChrissModel::default_params();
        let trajectory = model.optimal_trajectory(1000.0, 10);

        assert_eq!(trajectory.len(), 10);

        // Total should equal original quantity
        let total: f64 = trajectory.iter().sum();
        assert!((total - 1000.0).abs() < 1.0);

        // All slices should be non-negative
        for qty in &trajectory {
            assert!(*qty >= 0.0);
        }
    }

    #[test]
    fn test_expected_cost_increases_with_quantity() {
        let model = AlmgrenChrissModel::default_params();

        let cost_small = model.expected_cost(100.0, 1.0);
        let cost_large = model.expected_cost(1000.0, 1.0);

        assert!(cost_large > cost_small);
    }

    #[test]
    fn test_execution_risk_decreases_with_time() {
        let model = AlmgrenChrissModel::default_params();

        let risk_fast = model.execution_risk(1000.0, 0.1);
        let risk_slow = model.execution_risk(1000.0, 1.0);

        // Slower execution has more timing risk
        assert!(risk_slow > risk_fast);
    }

    #[test]
    fn test_square_root_model() {
        let model = SquareRootModel::default();
        let impact = model.calculate_impact(10000.0, 1.0);

        assert!(impact.total > 0.0);

        let trajectory = model.optimal_trajectory(1000.0, 5);
        assert_eq!(trajectory.len(), 5);

        // Should be uniform
        let first = trajectory[0];
        for qty in &trajectory {
            assert!((qty - first).abs() < 0.01);
        }
    }

    #[test]
    fn test_params_presets() {
        let liquid = AlmgrenChrissParams::liquid();
        let illiquid = AlmgrenChrissParams::illiquid();

        // Liquid should have lower impact coefficients
        assert!(liquid.gamma < illiquid.gamma);
        assert!(liquid.eta < illiquid.eta);
        assert!(liquid.spread < illiquid.spread);
    }
}
