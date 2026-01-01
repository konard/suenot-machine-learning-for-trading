//! Learning Rules Module
//!
//! Implements spike-timing-dependent plasticity (STDP) and other learning rules.

/// STDP configuration
#[derive(Debug, Clone, Copy)]
pub struct STDPConfig {
    /// Potentiation amplitude (pre before post)
    pub a_plus: f64,
    /// Depression amplitude (post before pre)
    pub a_minus: f64,
    /// Potentiation time constant (ms)
    pub tau_plus: f64,
    /// Depression time constant (ms)
    pub tau_minus: f64,
    /// Eligibility trace decay
    pub trace_decay: f64,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            trace_decay: 0.99,
        }
    }
}

impl STDPConfig {
    /// Create symmetric STDP
    pub fn symmetric(amplitude: f64, tau: f64) -> Self {
        Self {
            a_plus: amplitude,
            a_minus: amplitude,
            tau_plus: tau,
            tau_minus: tau,
            ..Default::default()
        }
    }

    /// Create asymmetric STDP with stronger depression
    pub fn asymmetric_depression(a_plus: f64, depression_factor: f64, tau: f64) -> Self {
        Self {
            a_plus,
            a_minus: a_plus * depression_factor,
            tau_plus: tau,
            tau_minus: tau,
            ..Default::default()
        }
    }
}

/// Learning rule trait
pub trait LearningRule: Send + Sync {
    /// Calculate weight change given spike timing
    fn weight_change(&self, dt: f64) -> f64;

    /// Update eligibility trace
    fn update_trace(&mut self, stdp_value: f64);

    /// Get current eligibility trace
    fn get_trace(&self) -> f64;

    /// Apply reward signal
    fn apply_reward(&self, reward: f64, trace: f64, learning_rate: f64) -> f64;
}

/// Standard STDP learning rule
#[derive(Debug, Clone)]
pub struct StandardSTDP {
    config: STDPConfig,
    eligibility_trace: f64,
}

impl StandardSTDP {
    pub fn new(config: STDPConfig) -> Self {
        Self {
            config,
            eligibility_trace: 0.0,
        }
    }
}

impl Default for StandardSTDP {
    fn default() -> Self {
        Self::new(STDPConfig::default())
    }
}

impl LearningRule for StandardSTDP {
    fn weight_change(&self, dt: f64) -> f64 {
        if dt > 0.0 {
            // Pre before post -> potentiation
            self.config.a_plus * (-dt / self.config.tau_plus).exp()
        } else if dt < 0.0 {
            // Post before pre -> depression
            -self.config.a_minus * (dt / self.config.tau_minus).exp()
        } else {
            0.0
        }
    }

    fn update_trace(&mut self, stdp_value: f64) {
        self.eligibility_trace = self.eligibility_trace * self.config.trace_decay + stdp_value;
    }

    fn get_trace(&self) -> f64 {
        self.eligibility_trace
    }

    fn apply_reward(&self, reward: f64, trace: f64, learning_rate: f64) -> f64 {
        reward * trace * learning_rate
    }
}

/// Reward-modulated STDP (R-STDP) for reinforcement learning
#[derive(Debug, Clone)]
pub struct RewardModulatedSTDP {
    base: StandardSTDP,
    reward_buffer: f64,
    learning_rate: f64,
}

impl RewardModulatedSTDP {
    pub fn new(config: STDPConfig, learning_rate: f64) -> Self {
        Self {
            base: StandardSTDP::new(config),
            reward_buffer: 0.0,
            learning_rate,
        }
    }

    /// Apply reward signal to update weights
    pub fn apply_reward_signal(&mut self, reward: f64) -> f64 {
        self.reward_buffer = reward;
        self.base.apply_reward(reward, self.base.eligibility_trace, self.learning_rate)
    }

    /// Get buffered reward
    pub fn get_reward(&self) -> f64 {
        self.reward_buffer
    }
}

impl LearningRule for RewardModulatedSTDP {
    fn weight_change(&self, dt: f64) -> f64 {
        self.base.weight_change(dt)
    }

    fn update_trace(&mut self, stdp_value: f64) {
        self.base.update_trace(stdp_value);
    }

    fn get_trace(&self) -> f64 {
        self.base.get_trace()
    }

    fn apply_reward(&self, reward: f64, trace: f64, learning_rate: f64) -> f64 {
        self.base.apply_reward(reward, trace, learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_potentiation() {
        let stdp = StandardSTDP::default();
        let dw = stdp.weight_change(5.0);  // pre 5ms before post
        assert!(dw > 0.0, "Pre before post should potentiate");
    }

    #[test]
    fn test_stdp_depression() {
        let stdp = StandardSTDP::default();
        let dw = stdp.weight_change(-5.0);  // post 5ms before pre
        assert!(dw < 0.0, "Post before pre should depress");
    }

    #[test]
    fn test_stdp_timing_dependence() {
        let stdp = StandardSTDP::default();

        // Closer timing should produce larger changes
        let dw_close = stdp.weight_change(1.0).abs();
        let dw_far = stdp.weight_change(50.0).abs();

        assert!(dw_close > dw_far, "Closer timing should have larger effect");
    }

    #[test]
    fn test_eligibility_trace() {
        let mut stdp = StandardSTDP::default();

        stdp.update_trace(1.0);
        let trace1 = stdp.get_trace();

        stdp.update_trace(0.0);
        let trace2 = stdp.get_trace();

        assert!(trace2 < trace1, "Trace should decay");
    }
}
