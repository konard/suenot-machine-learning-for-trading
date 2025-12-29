//! Independent learners algorithm
//!
//! Each agent learns independently, treating other agents as part of the environment.

use std::collections::HashMap;

use crate::agents::{Agent, AgentId};
use crate::environment::{MultiAgentConfig, MultiAgentEnv};

use super::config::{EvalResult, TrainingConfig, TrainingResult};

/// Independent learners trainer
///
/// Each agent uses its own RL algorithm and learns independently.
/// Other agents are treated as part of the (non-stationary) environment.
pub struct IndependentLearners {
    config: TrainingConfig,
    env_config: MultiAgentConfig,
}

impl IndependentLearners {
    /// Create new independent learners trainer
    pub fn new(config: TrainingConfig, env_config: MultiAgentConfig) -> Self {
        Self { config, env_config }
    }

    /// Train agents independently
    pub fn train(&self, agents: &mut [Box<dyn Agent>]) -> TrainingResult {
        let start_time = std::time::Instant::now();

        let mut env = MultiAgentEnv::new(self.env_config.clone());
        let mut result = TrainingResult::default();
        let mut best_score = f64::NEG_INFINITY;
        let mut episodes_without_improvement = 0;

        for episode in 0..self.config.n_episodes {
            // Run training episode
            let episode_result = env.run_episode(agents, Some(self.config.max_steps));

            // Record mean reward
            let mean_reward: f64 = episode_result.total_rewards.values().sum::<f64>()
                / episode_result.total_rewards.len().max(1) as f64;
            result.history.push(mean_reward);

            // Logging
            if episode % self.config.log_frequency == 0 {
                log::info!(
                    "Episode {}/{}: Mean Reward = {:.2}, Summary: {}",
                    episode,
                    self.config.n_episodes,
                    mean_reward,
                    episode_result.summary()
                );
            }

            // Evaluation
            if episode % self.config.eval_frequency == 0 {
                let eval_result = self.evaluate(agents, episode);
                log::info!("Evaluation: {}", eval_result);

                if eval_result.mean_reward > best_score + self.config.min_delta {
                    best_score = eval_result.mean_reward;
                    result.best_score = best_score;
                    result.best_episode = episode;
                    episodes_without_improvement = 0;
                } else {
                    episodes_without_improvement += 1;
                }

                result.eval_history.push(eval_result);

                // Early stopping
                if self.config.early_stopping
                    && episodes_without_improvement >= self.config.patience
                {
                    log::info!(
                        "Early stopping at episode {} (no improvement for {} evaluations)",
                        episode,
                        self.config.patience
                    );
                    result.converged = true;
                    break;
                }
            }
        }

        result.training_time = start_time.elapsed().as_secs_f64();
        result
    }

    /// Evaluate agents
    fn evaluate(&self, agents: &mut [Box<dyn Agent>], episode: usize) -> EvalResult {
        let mut env = MultiAgentEnv::new(self.env_config.clone());
        let mut rewards = Vec::new();
        let mut pnls = Vec::new();
        let mut wins = 0;
        let mut total = 0;

        for _ in 0..self.config.n_eval_episodes {
            // Reset agents for evaluation
            for agent in agents.iter_mut() {
                agent.reset();
            }

            let result = env.run_episode(agents, Some(self.config.max_steps));

            let mean_reward: f64 = result.total_rewards.values().sum::<f64>()
                / result.total_rewards.len().max(1) as f64;
            rewards.push(mean_reward);

            let mean_pnl: f64 =
                result.total_pnl.values().sum::<f64>() / result.total_pnl.len().max(1) as f64;
            pnls.push(mean_pnl);

            // Count wins (positive PnL)
            for &pnl in result.total_pnl.values() {
                if pnl > 0.0 {
                    wins += 1;
                }
                total += 1;
            }
        }

        let mean_reward = rewards.iter().sum::<f64>() / rewards.len().max(1) as f64;
        let std_reward = {
            let variance = rewards
                .iter()
                .map(|r| (r - mean_reward).powi(2))
                .sum::<f64>()
                / rewards.len().max(1) as f64;
            variance.sqrt()
        };

        EvalResult {
            episode,
            mean_reward,
            std_reward,
            mean_pnl: pnls.iter().sum::<f64>() / pnls.len().max(1) as f64,
            win_rate: wins as f64 / total.max(1) as f64,
        }
    }

    /// Run tournament between agents
    pub fn tournament(
        &self,
        agents: &mut [Box<dyn Agent>],
        n_rounds: usize,
    ) -> HashMap<AgentId, TournamentStats> {
        let mut stats: HashMap<AgentId, TournamentStats> = HashMap::new();
        for i in 0..agents.len() {
            stats.insert(i, TournamentStats::default());
        }

        let mut env = MultiAgentEnv::new(self.env_config.clone());

        for _ in 0..n_rounds {
            let result = env.run_episode(agents, Some(self.config.max_steps));

            // Update stats
            for (&agent_id, &pnl) in &result.total_pnl {
                let agent_stats = stats.get_mut(&agent_id).unwrap();
                agent_stats.games += 1;
                agent_stats.total_pnl += pnl;

                if pnl > 0.0 {
                    agent_stats.wins += 1;
                } else if pnl < 0.0 {
                    agent_stats.losses += 1;
                } else {
                    agent_stats.draws += 1;
                }
            }

            // Determine rankings for this round
            let mut rankings: Vec<_> = result.total_pnl.iter().collect();
            rankings.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

            for (rank, (&agent_id, _)) in rankings.iter().enumerate() {
                let agent_stats = stats.get_mut(&agent_id).unwrap();
                if rank == 0 {
                    agent_stats.first_places += 1;
                } else if rank == rankings.len() - 1 {
                    agent_stats.last_places += 1;
                }
            }
        }

        stats
    }
}

/// Tournament statistics for an agent
#[derive(Debug, Clone, Default)]
pub struct TournamentStats {
    pub games: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub total_pnl: f64,
    pub first_places: usize,
    pub last_places: usize,
}

impl TournamentStats {
    pub fn win_rate(&self) -> f64 {
        if self.games > 0 {
            self.wins as f64 / self.games as f64
        } else {
            0.0
        }
    }

    pub fn avg_pnl(&self) -> f64 {
        if self.games > 0 {
            self.total_pnl / self.games as f64
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for TournamentStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Games: {}, Wins: {} ({:.1}%), AvgPnL: {:.2}, 1st: {}, Last: {}",
            self.games,
            self.wins,
            self.win_rate() * 100.0,
            self.avg_pnl(),
            self.first_places,
            self.last_places
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::{NoiseTrader, TrendFollowingAgent};

    #[test]
    fn test_independent_learners() {
        let config = TrainingConfig::quick();
        let env_config = MultiAgentConfig::default().with_agents(2);

        let trainer = IndependentLearners::new(config, env_config);

        let mut agents: Vec<Box<dyn Agent>> = vec![
            Box::new(NoiseTrader::default()),
            Box::new(TrendFollowingAgent::default()),
        ];

        let result = trainer.train(&mut agents);

        assert!(!result.history.is_empty());
        println!(
            "Training completed in {:.2}s, best score: {:.2}",
            result.training_time, result.best_score
        );
    }

    #[test]
    fn test_tournament() {
        let config = TrainingConfig::quick();
        let env_config = MultiAgentConfig::default().with_agents(3);

        let trainer = IndependentLearners::new(config, env_config);

        let mut agents: Vec<Box<dyn Agent>> = vec![
            Box::new(NoiseTrader::default()),
            Box::new(TrendFollowingAgent::default()),
            Box::new(NoiseTrader::bullish()),
        ];

        let stats = trainer.tournament(&mut agents, 10);

        assert_eq!(stats.len(), 3);

        for (id, s) in &stats {
            println!("Agent {}: {}", id, s);
        }
    }
}
