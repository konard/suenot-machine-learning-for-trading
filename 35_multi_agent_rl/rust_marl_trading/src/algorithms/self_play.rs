//! Self-play training algorithm
//!
//! Trains an agent by playing against historical versions of itself.

use rand::Rng;

use crate::agents::Agent;
use crate::environment::{MultiAgentConfig, MultiAgentEnv};

use super::config::{EvalResult, TrainingConfig, TrainingResult};

/// Self-play trainer
///
/// The main agent plays against a pool of its previous versions.
/// This creates an automatic curriculum and ensures robustness.
pub struct SelfPlayTrainer {
    config: TrainingConfig,
    env_config: MultiAgentConfig,
    /// Number of opponents in the pool
    pool_size: usize,
    /// How often to add current agent to pool
    snapshot_frequency: usize,
    /// Temperature for opponent selection (higher = more uniform)
    selection_temperature: f64,
}

impl SelfPlayTrainer {
    /// Create new self-play trainer
    pub fn new(config: TrainingConfig, env_config: MultiAgentConfig) -> Self {
        Self {
            config,
            env_config,
            pool_size: 5,
            snapshot_frequency: 50,
            selection_temperature: 1.0,
        }
    }

    /// Set pool size
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    /// Set snapshot frequency
    pub fn with_snapshot_frequency(mut self, freq: usize) -> Self {
        self.snapshot_frequency = freq;
        self
    }

    /// Train using self-play
    pub fn train(&self, agent: &mut Box<dyn Agent>) -> TrainingResult {
        let start_time = std::time::Instant::now();

        // Initialize opponent pool with copies of initial agent
        let mut opponent_pool: Vec<Box<dyn Agent>> = Vec::new();
        let mut win_rates: Vec<f64> = Vec::new();

        for _ in 0..self.pool_size {
            opponent_pool.push(agent.clone_box());
            win_rates.push(0.5);
        }

        // Create two-player environment
        let two_player_config = self.env_config.clone().with_agents(2);
        let mut env = MultiAgentEnv::new(two_player_config);

        let mut result = TrainingResult::default();
        let mut best_score = f64::NEG_INFINITY;

        for episode in 0..self.config.n_episodes {
            // Select opponent based on difficulty (prioritize harder opponents)
            let opponent_idx = self.select_opponent(&win_rates);
            let opponent = &mut opponent_pool[opponent_idx];

            // Create agents array for environment
            let mut agents: Vec<Box<dyn Agent>> = vec![agent.clone_box(), opponent.clone_box()];

            // Set IDs
            agents[0].set_id(0);
            agents[1].set_id(1);

            // Run episode
            let episode_result = env.run_episode(&mut agents, Some(self.config.max_steps));

            // Get rewards
            let main_pnl = episode_result.total_pnl.get(&0).copied().unwrap_or(0.0);
            let opponent_pnl = episode_result.total_pnl.get(&1).copied().unwrap_or(0.0);

            let main_won = main_pnl > opponent_pnl;

            // Update win rate for this opponent
            win_rates[opponent_idx] = 0.95 * win_rates[opponent_idx] + 0.05 * (main_won as i32 as f64);

            // Record history
            result.history.push(main_pnl);

            // Learning happens during run_episode, but we need to update the main agent
            // In a real implementation, you'd have the agent learn from the experience

            // Logging
            if episode % self.config.log_frequency == 0 {
                log::info!(
                    "Episode {}/{}: PnL = {:.2}, vs Opponent #{} (WR: {:.1}%)",
                    episode,
                    self.config.n_episodes,
                    main_pnl,
                    opponent_idx,
                    win_rates[opponent_idx] * 100.0
                );
            }

            // Snapshot current agent to pool
            if episode % self.snapshot_frequency == 0 && episode > 0 {
                self.update_pool(agent, &mut opponent_pool, &mut win_rates);
            }

            // Evaluation
            if episode % self.config.eval_frequency == 0 {
                let eval_result = self.evaluate(agent, &opponent_pool, episode);
                log::info!("Evaluation: {}", eval_result);

                if eval_result.mean_reward > best_score {
                    best_score = eval_result.mean_reward;
                    result.best_score = best_score;
                    result.best_episode = episode;
                }

                result.eval_history.push(eval_result);
            }
        }

        result.training_time = start_time.elapsed().as_secs_f64();
        result
    }

    /// Select opponent using softmax over difficulties
    fn select_opponent(&self, win_rates: &[f64]) -> usize {
        let mut rng = rand::thread_rng();

        // Difficulty = 1 - win_rate (harder opponents have lower win rate against them)
        let difficulties: Vec<f64> = win_rates.iter().map(|wr| 1.0 - wr).collect();

        // Softmax
        let max_diff = difficulties.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_diffs: Vec<f64> = difficulties
            .iter()
            .map(|d| ((d - max_diff) / self.selection_temperature).exp())
            .collect();
        let sum_exp: f64 = exp_diffs.iter().sum();
        let probs: Vec<f64> = exp_diffs.iter().map(|e| e / sum_exp).collect();

        // Sample
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        probs.len() - 1
    }

    /// Update opponent pool with current agent
    fn update_pool(
        &self,
        agent: &Box<dyn Agent>,
        pool: &mut Vec<Box<dyn Agent>>,
        win_rates: &mut Vec<f64>,
    ) {
        // Replace the opponent we have the highest win rate against
        let worst_idx = win_rates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        pool[worst_idx] = agent.clone_box();
        win_rates[worst_idx] = 0.5; // Reset win rate
    }

    /// Evaluate against all opponents in pool
    fn evaluate(
        &self,
        agent: &Box<dyn Agent>,
        pool: &[Box<dyn Agent>],
        episode: usize,
    ) -> EvalResult {
        let two_player_config = self.env_config.clone().with_agents(2);
        let mut env = MultiAgentEnv::new(two_player_config);

        let mut total_pnl = 0.0;
        let mut wins = 0;
        let mut games = 0;

        // Play against each opponent
        for opponent in pool {
            for _ in 0..2 {
                // Play 2 games against each
                let mut agents: Vec<Box<dyn Agent>> =
                    vec![agent.clone_box(), opponent.clone_box()];
                agents[0].set_id(0);
                agents[1].set_id(1);

                let result = env.run_episode(&mut agents, Some(self.config.max_steps));

                let main_pnl = result.total_pnl.get(&0).copied().unwrap_or(0.0);
                let opp_pnl = result.total_pnl.get(&1).copied().unwrap_or(0.0);

                total_pnl += main_pnl;
                games += 1;

                if main_pnl > opp_pnl {
                    wins += 1;
                }
            }
        }

        EvalResult {
            episode,
            mean_reward: total_pnl / games.max(1) as f64,
            std_reward: 0.0, // TODO: compute properly
            mean_pnl: total_pnl / games.max(1) as f64,
            win_rate: wins as f64 / games.max(1) as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::RLAgent;

    #[test]
    fn test_self_play_trainer() {
        let config = TrainingConfig::quick().with_episodes(50);
        let env_config = MultiAgentConfig::default();

        let trainer = SelfPlayTrainer::new(config, env_config)
            .with_pool_size(3)
            .with_snapshot_frequency(10);

        let mut agent: Box<dyn Agent> = Box::new(RLAgent::new(8, 32));

        let result = trainer.train(&mut agent);

        assert!(!result.history.is_empty());
        println!(
            "Self-play training completed in {:.2}s",
            result.training_time
        );
    }

    #[test]
    fn test_opponent_selection() {
        let config = TrainingConfig::quick();
        let env_config = MultiAgentConfig::default();
        let trainer = SelfPlayTrainer::new(config, env_config);

        let win_rates = vec![0.8, 0.5, 0.3, 0.6];

        // Run selection many times
        let mut selections = vec![0; 4];
        for _ in 0..1000 {
            let idx = trainer.select_opponent(&win_rates);
            selections[idx] += 1;
        }

        // Should select harder opponents (lower win rate) more often
        println!("Selection frequencies: {:?}", selections);
        assert!(selections[2] > selections[0]); // 0.3 win rate should be selected more than 0.8
    }
}
