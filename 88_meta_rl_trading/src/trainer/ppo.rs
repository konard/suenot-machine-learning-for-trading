//! PPO meta-trainer for Meta-RL agents.
//!
//! Trains the agent across a distribution of trading environments,
//! enabling it to learn a general learning algorithm encoded in its
//! GRU hidden state.

use crate::agent::meta_rl::MetaRLAgent;
use crate::env::trading_env::TradingEnvironment;

/// Experience from a single step
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub value: f64,
    pub done: bool,
    pub prev_action: usize,
    pub prev_reward: f64,
    pub prev_done: bool,
}

/// PPO Meta-Trainer
#[derive(Debug)]
pub struct PPOTrainer {
    agent: MetaRLAgent,
    learning_rate: f64,
    gamma: f64,
    gae_lambda: f64,
    num_episodes_per_trial: usize,
    gradient_epsilon: f64,
}

impl PPOTrainer {
    /// Create a new PPO trainer
    pub fn new(
        agent: MetaRLAgent,
        learning_rate: f64,
        gamma: f64,
        gae_lambda: f64,
        num_episodes_per_trial: usize,
    ) -> Self {
        Self {
            agent,
            learning_rate,
            gamma,
            gae_lambda,
            num_episodes_per_trial,
            gradient_epsilon: 1e-4,
        }
    }

    /// Collect a trial (multiple episodes) in one environment
    pub fn collect_trial(&mut self, env: &mut TradingEnvironment) -> Vec<Experience> {
        let mut experiences = Vec::new();
        self.agent.reset_hidden();

        let mut prev_action = 0usize;
        let mut prev_reward = 0.0;
        let mut prev_done = false;

        for _episode in 0..self.num_episodes_per_trial {
            let mut state = env.reset();
            let mut done = false;

            while !done {
                let (action, value) = self.agent.get_action(
                    &state, prev_action, prev_reward, prev_done,
                );

                let result = env.step(action);

                experiences.push(Experience {
                    state: state.clone(),
                    action,
                    reward: result.reward,
                    value,
                    done: result.done,
                    prev_action,
                    prev_reward,
                    prev_done,
                });

                prev_action = action;
                prev_reward = result.reward;
                prev_done = result.done;
                state = result.state;
                done = result.done;
            }
        }

        experiences
    }

    /// Compute GAE advantages and returns
    fn compute_gae(&self, experiences: &[Experience]) -> (Vec<f64>, Vec<f64>) {
        let n = experiences.len();
        let mut advantages = vec![0.0; n];
        let mut last_gae = 0.0;

        for t in (0..n).rev() {
            let next_value = if t == n - 1 {
                0.0
            } else {
                experiences[t + 1].value
            };
            let done_mask = if experiences[t].done { 0.0 } else { 1.0 };

            let delta = experiences[t].reward
                + self.gamma * next_value * done_mask
                - experiences[t].value;

            last_gae = delta + self.gamma * self.gae_lambda * done_mask * last_gae;
            advantages[t] = last_gae;
        }

        let returns: Vec<f64> = advantages.iter()
            .zip(experiences.iter())
            .map(|(a, e)| a + e.value)
            .collect();

        // Normalize advantages
        let mean = advantages.iter().sum::<f64>() / n as f64;
        let std = (advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        if std > 1e-8 {
            for a in advantages.iter_mut() {
                *a = (*a - mean) / (std + 1e-8);
            }
        }

        (advantages, returns)
    }

    /// Compute numerical policy gradient and update agent
    fn update(&mut self, experiences: &[Experience], advantages: &[f64]) -> f64 {
        // Compute policy gradient numerically
        let params = self.agent.get_parameters();
        let mut gradients = vec![0.0; params.len()];

        // Compute expected advantage under current policy (baseline)
        let baseline_score: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;

        // Numerical gradient approximation using REINFORCE-style update
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            params_plus[i] += self.gradient_epsilon;

            let mut agent_plus = self.agent.clone_agent();
            agent_plus.set_parameters(&params_plus);
            agent_plus.reset_hidden();

            let score_plus = self.evaluate_policy(&mut agent_plus, experiences, advantages);

            let mut params_minus = params.clone();
            params_minus[i] -= self.gradient_epsilon;

            let mut agent_minus = self.agent.clone_agent();
            agent_minus.set_parameters(&params_minus);
            agent_minus.reset_hidden();

            let score_minus = self.evaluate_policy(&mut agent_minus, experiences, advantages);

            gradients[i] = (score_plus - score_minus) / (2.0 * self.gradient_epsilon);
        }

        // SGD step (maximize, so add gradient)
        self.agent.sgd_step(&gradients.iter().map(|g| -g).collect::<Vec<_>>(), self.learning_rate);

        baseline_score
    }

    /// Evaluate policy on experience batch
    fn evaluate_policy(
        &self,
        agent: &mut MetaRLAgent,
        experiences: &[Experience],
        advantages: &[f64],
    ) -> f64 {
        let mut score = 0.0;

        for (exp, adv) in experiences.iter().zip(advantages.iter()) {
            let (probs, _) = agent.forward(
                &exp.state, exp.prev_action, exp.prev_reward, exp.prev_done,
            );

            let prob = probs[exp.action].max(1e-10);
            score += prob.ln() * adv;
        }

        score / experiences.len() as f64
    }

    /// One step of meta-training across multiple environments
    pub fn meta_train_step(&mut self, environments: &mut [TradingEnvironment]) -> f64 {
        let mut total_loss = 0.0;

        for env in environments.iter_mut() {
            let experiences = self.collect_trial(env);
            let (advantages, _returns) = self.compute_gae(&experiences);
            let loss = self.update(&experiences, &advantages);
            total_loss += loss;
        }

        total_loss / environments.len() as f64
    }

    /// Get reference to the agent
    pub fn agent(&self) -> &MetaRLAgent {
        &self.agent
    }

    /// Get mutable reference to the agent
    pub fn agent_mut(&mut self) -> &mut MetaRLAgent {
        &mut self.agent
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub epoch: usize,
    pub avg_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_env() -> TradingEnvironment {
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64) * 0.05).collect();
        let features: Vec<Vec<f64>> = (0..200).map(|_| vec![0.1, 0.2, 0.3]).collect();
        TradingEnvironment::new(prices, features, 10000.0, 0.001, 50)
    }

    #[test]
    fn test_trainer_creation() {
        let agent = MetaRLAgent::new(6, 3, 16);
        let _trainer = PPOTrainer::new(agent, 0.001, 0.99, 0.95, 2);
    }

    #[test]
    fn test_collect_trial() {
        let agent = MetaRLAgent::new(6, 3, 8);
        let mut trainer = PPOTrainer::new(agent, 0.001, 0.99, 0.95, 1);
        let mut env = create_test_env();

        let experiences = trainer.collect_trial(&mut env);
        assert!(!experiences.is_empty());
    }

    #[test]
    fn test_compute_gae() {
        let agent = MetaRLAgent::new(6, 3, 8);
        let mut trainer = PPOTrainer::new(agent, 0.001, 0.99, 0.95, 1);
        let mut env = create_test_env();

        let experiences = trainer.collect_trial(&mut env);
        let (advantages, returns) = trainer.compute_gae(&experiences);

        assert_eq!(advantages.len(), experiences.len());
        assert_eq!(returns.len(), experiences.len());
    }
}
