//! Training configuration

use serde::{Deserialize, Serialize};

/// Configuration for MARL training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training episodes
    pub n_episodes: usize,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Evaluation frequency (episodes)
    pub eval_frequency: usize,
    /// Number of evaluation episodes
    pub n_eval_episodes: usize,
    /// Whether to use early stopping
    pub early_stopping: bool,
    /// Patience for early stopping
    pub patience: usize,
    /// Minimum improvement for early stopping
    pub min_delta: f64,
    /// Whether to save checkpoints
    pub save_checkpoints: bool,
    /// Checkpoint directory
    pub checkpoint_dir: String,
    /// Logging frequency (episodes)
    pub log_frequency: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            n_episodes: 1000,
            max_steps: 1000,
            eval_frequency: 100,
            n_eval_episodes: 10,
            early_stopping: true,
            patience: 50,
            min_delta: 0.01,
            save_checkpoints: false,
            checkpoint_dir: "checkpoints".to_string(),
            log_frequency: 10,
            seed: None,
        }
    }
}

impl TrainingConfig {
    /// Quick training config (fewer episodes, for testing)
    pub fn quick() -> Self {
        Self {
            n_episodes: 100,
            max_steps: 100,
            eval_frequency: 20,
            n_eval_episodes: 5,
            ..Default::default()
        }
    }

    /// Long training config
    pub fn long() -> Self {
        Self {
            n_episodes: 10000,
            max_steps: 2000,
            eval_frequency: 500,
            n_eval_episodes: 50,
            ..Default::default()
        }
    }

    /// Set number of episodes
    pub fn with_episodes(mut self, n: usize) -> Self {
        self.n_episodes = n;
        self
    }

    /// Set max steps per episode
    pub fn with_max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    /// Enable checkpointing
    pub fn with_checkpoints(mut self, dir: impl Into<String>) -> Self {
        self.save_checkpoints = true;
        self.checkpoint_dir = dir.into();
        self
    }

    /// Set seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Training result
#[derive(Debug, Clone, Default)]
pub struct TrainingResult {
    /// Training history (episode -> mean reward)
    pub history: Vec<f64>,
    /// Evaluation history
    pub eval_history: Vec<EvalResult>,
    /// Best evaluation score
    pub best_score: f64,
    /// Episode of best score
    pub best_episode: usize,
    /// Whether training converged
    pub converged: bool,
    /// Total training time (seconds)
    pub training_time: f64,
}

/// Result of an evaluation run
#[derive(Debug, Clone, Default)]
pub struct EvalResult {
    pub episode: usize,
    pub mean_reward: f64,
    pub std_reward: f64,
    pub mean_pnl: f64,
    pub win_rate: f64,
}

impl std::fmt::Display for EvalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Episode {}: Reward={:.2}±{:.2}, PnL={:.2}, WinRate={:.1}%",
            self.episode,
            self.mean_reward,
            self.std_reward,
            self.mean_pnl,
            self.win_rate * 100.0
        )
    }
}
