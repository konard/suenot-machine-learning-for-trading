//! Result types for environment steps and episodes

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::Observation;
use crate::agents::AgentId;

/// Result of a single environment step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Observations for each agent
    pub observations: HashMap<AgentId, Observation>,
    /// Rewards for each agent
    pub rewards: HashMap<AgentId, f64>,
    /// Whether the episode is done
    pub done: bool,
    /// Additional info
    pub info: StepInfo,
}

impl StepResult {
    /// Create a new step result
    pub fn new(
        observations: HashMap<AgentId, Observation>,
        rewards: HashMap<AgentId, f64>,
        done: bool,
    ) -> Self {
        Self {
            observations,
            rewards,
            done,
            info: StepInfo::default(),
        }
    }

    /// Get observation for a specific agent
    pub fn observation(&self, agent_id: AgentId) -> Option<&Observation> {
        self.observations.get(&agent_id)
    }

    /// Get reward for a specific agent
    pub fn reward(&self, agent_id: AgentId) -> f64 {
        self.rewards.get(&agent_id).copied().unwrap_or(0.0)
    }

    /// Get total reward across all agents
    pub fn total_reward(&self) -> f64 {
        self.rewards.values().sum()
    }
}

/// Additional information from a step
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepInfo {
    /// Current market price
    pub price: f64,
    /// Number of trades executed
    pub n_trades: usize,
    /// Total volume traded
    pub volume: f64,
    /// Current spread
    pub spread: f64,
    /// Current step number
    pub step: usize,
}

/// Result of a complete episode
#[derive(Debug, Clone, Default)]
pub struct EpisodeResult {
    /// Final portfolio values for each agent
    pub final_values: HashMap<AgentId, f64>,
    /// Total PnL for each agent
    pub total_pnl: HashMap<AgentId, f64>,
    /// Number of trades per agent
    pub trade_counts: HashMap<AgentId, usize>,
    /// Total rewards per agent
    pub total_rewards: HashMap<AgentId, f64>,
    /// Price history
    pub price_history: Vec<f64>,
    /// Number of steps
    pub n_steps: usize,
    /// Whether episode ended early (e.g., bankruptcy)
    pub terminated_early: bool,
}

impl EpisodeResult {
    /// Create a new episode result
    pub fn new() -> Self {
        Self::default()
    }

    /// Get winner (agent with highest final value)
    pub fn winner(&self) -> Option<AgentId> {
        self.final_values
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(id, _)| *id)
    }

    /// Get loser (agent with lowest final value)
    pub fn loser(&self) -> Option<AgentId> {
        self.final_values
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(id, _)| *id)
    }

    /// Get rankings (sorted by final value, descending)
    pub fn rankings(&self) -> Vec<(AgentId, f64)> {
        let mut rankings: Vec<_> = self.final_values.iter().map(|(&id, &val)| (id, val)).collect();
        rankings.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        rankings
    }

    /// Calculate returns for each agent
    pub fn returns(&self, initial_value: f64) -> HashMap<AgentId, f64> {
        self.final_values
            .iter()
            .map(|(&id, &val)| (id, (val - initial_value) / initial_value))
            .collect()
    }

    /// Calculate Sharpe ratio approximation for each agent
    pub fn sharpe_ratios(&self, risk_free_rate: f64) -> HashMap<AgentId, f64> {
        // Simplified: uses total return and market volatility
        let market_vol = self.calculate_market_volatility();

        self.returns(100_000.0) // Assuming default initial
            .iter()
            .map(|(&id, &ret)| {
                let sharpe = if market_vol > 0.0 {
                    (ret - risk_free_rate) / market_vol
                } else {
                    0.0
                };
                (id, sharpe)
            })
            .collect()
    }

    /// Calculate market volatility from price history
    fn calculate_market_volatility(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.price_history
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Get summary statistics
    pub fn summary(&self) -> EpisodeSummary {
        let values: Vec<f64> = self.final_values.values().copied().collect();
        let pnls: Vec<f64> = self.total_pnl.values().copied().collect();

        EpisodeSummary {
            n_agents: self.final_values.len(),
            n_steps: self.n_steps,
            mean_final_value: values.iter().sum::<f64>() / values.len().max(1) as f64,
            mean_pnl: pnls.iter().sum::<f64>() / pnls.len().max(1) as f64,
            best_pnl: pnls.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            worst_pnl: pnls.iter().cloned().fold(f64::INFINITY, f64::min),
            total_trades: self.trade_counts.values().sum(),
            market_volatility: self.calculate_market_volatility(),
            final_price: self.price_history.last().copied().unwrap_or(0.0),
        }
    }
}

/// Summary statistics for an episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeSummary {
    pub n_agents: usize,
    pub n_steps: usize,
    pub mean_final_value: f64,
    pub mean_pnl: f64,
    pub best_pnl: f64,
    pub worst_pnl: f64,
    pub total_trades: usize,
    pub market_volatility: f64,
    pub final_price: f64,
}

impl std::fmt::Display for EpisodeSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Episode: {} steps, {} agents | PnL: best={:.2}, worst={:.2}, mean={:.2} | {} trades | Vol: {:.4}",
            self.n_steps,
            self.n_agents,
            self.best_pnl,
            self.worst_pnl,
            self.mean_pnl,
            self.total_trades,
            self.market_volatility
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_result() {
        let mut observations = HashMap::new();
        observations.insert(0, Observation::default());
        observations.insert(1, Observation::default());

        let mut rewards = HashMap::new();
        rewards.insert(0, 100.0);
        rewards.insert(1, -50.0);

        let result = StepResult::new(observations, rewards, false);

        assert_eq!(result.reward(0), 100.0);
        assert_eq!(result.reward(1), -50.0);
        assert_eq!(result.total_reward(), 50.0);
    }

    #[test]
    fn test_episode_result() {
        let mut result = EpisodeResult::new();

        result.final_values.insert(0, 110_000.0);
        result.final_values.insert(1, 90_000.0);
        result.final_values.insert(2, 100_000.0);

        result.total_pnl.insert(0, 10_000.0);
        result.total_pnl.insert(1, -10_000.0);
        result.total_pnl.insert(2, 0.0);

        assert_eq!(result.winner(), Some(0));
        assert_eq!(result.loser(), Some(1));

        let rankings = result.rankings();
        assert_eq!(rankings[0].0, 0); // Best
        assert_eq!(rankings[2].0, 1); // Worst
    }

    #[test]
    fn test_episode_summary() {
        let mut result = EpisodeResult::new();

        result.final_values.insert(0, 110_000.0);
        result.final_values.insert(1, 90_000.0);
        result.total_pnl.insert(0, 10_000.0);
        result.total_pnl.insert(1, -10_000.0);
        result.trade_counts.insert(0, 50);
        result.trade_counts.insert(1, 30);
        result.n_steps = 1000;
        result.price_history = vec![100.0, 101.0, 99.0, 102.0];

        let summary = result.summary();

        assert_eq!(summary.n_agents, 2);
        assert_eq!(summary.n_steps, 1000);
        assert_eq!(summary.best_pnl, 10_000.0);
        assert_eq!(summary.worst_pnl, -10_000.0);
        assert_eq!(summary.total_trades, 80);
    }
}
