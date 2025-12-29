//! Population-based training
//!
//! Evolves a population of agents with different hyperparameters.

use rand::Rng;
use std::collections::HashMap;

use crate::agents::{Agent, AgentId, RLAgent};
use crate::environment::{MultiAgentConfig, MultiAgentEnv};

use super::config::{TrainingConfig, TrainingResult};

/// Population member with agent and metadata
#[derive(Clone)]
pub struct PopulationMember {
    /// The agent
    pub agent: Box<dyn Agent>,
    /// Hyperparameters
    pub hyperparams: Hyperparameters,
    /// Fitness score
    pub fitness: f64,
    /// Generation born
    pub generation: usize,
}

/// Hyperparameters for an agent
#[derive(Debug, Clone, Default)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub hidden_size: usize,
}

impl Hyperparameters {
    /// Generate random hyperparameters
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();

        Self {
            learning_rate: 10_f64.powf(rng.gen_range(-5.0..-2.0)),
            gamma: rng.gen_range(0.9..0.999),
            epsilon: rng.gen_range(0.1..1.0),
            hidden_size: *[32, 64, 128, 256].choose(&mut rng).unwrap(),
        }
    }

    /// Mutate hyperparameters
    pub fn mutate(&self) -> Self {
        let mut rng = rand::thread_rng();
        let mut new = self.clone();

        // Each hyperparameter has a chance to mutate
        if rng.gen::<f64>() < 0.3 {
            new.learning_rate *= 10_f64.powf(rng.gen_range(-0.5..0.5));
            new.learning_rate = new.learning_rate.clamp(1e-6, 1e-1);
        }

        if rng.gen::<f64>() < 0.3 {
            new.gamma += rng.gen_range(-0.02..0.02);
            new.gamma = new.gamma.clamp(0.9, 0.999);
        }

        if rng.gen::<f64>() < 0.3 {
            new.epsilon *= rng.gen_range(0.8..1.2);
            new.epsilon = new.epsilon.clamp(0.01, 1.0);
        }

        if rng.gen::<f64>() < 0.2 {
            new.hidden_size = *[32, 64, 128, 256]
                .choose(&mut rng)
                .unwrap();
        }

        new
    }
}

trait Choose<T> {
    fn choose(&mut self, options: &[T]) -> Option<&T>;
}

impl<T> Choose<T> for rand::rngs::ThreadRng {
    fn choose(&mut self, options: &[T]) -> Option<&T> {
        if options.is_empty() {
            None
        } else {
            let idx = self.gen_range(0..options.len());
            Some(&options[idx])
        }
    }
}

/// Population-based trainer
pub struct PopulationTrainer {
    config: TrainingConfig,
    env_config: MultiAgentConfig,
    /// Population size
    population_size: usize,
    /// Number of generations
    n_generations: usize,
    /// Fraction of population to keep
    elite_fraction: f64,
    /// Games per evaluation
    eval_games: usize,
}

impl PopulationTrainer {
    /// Create new population trainer
    pub fn new(config: TrainingConfig, env_config: MultiAgentConfig) -> Self {
        Self {
            config,
            env_config,
            population_size: 20,
            n_generations: 50,
            elite_fraction: 0.25,
            eval_games: 10,
        }
    }

    /// Set population size
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set number of generations
    pub fn with_generations(mut self, n: usize) -> Self {
        self.n_generations = n;
        self
    }

    /// Initialize random population
    fn initialize_population(&self) -> Vec<PopulationMember> {
        (0..self.population_size)
            .map(|_| {
                let hp = Hyperparameters::random();
                let agent = Box::new(RLAgent::new(8, hp.hidden_size));

                PopulationMember {
                    agent,
                    hyperparams: hp,
                    fitness: 0.0,
                    generation: 0,
                }
            })
            .collect()
    }

    /// Evaluate population fitness through tournament
    fn evaluate_population(&self, population: &mut [PopulationMember]) {
        let mut env = MultiAgentEnv::new(self.env_config.clone());
        let n_agents = self.env_config.n_agents.min(population.len());

        // Reset all fitness scores
        for member in population.iter_mut() {
            member.fitness = 0.0;
        }

        // Run tournament games
        for _ in 0..self.eval_games {
            // Randomly select agents to compete
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..population.len()).collect();

            for _ in 0..population.len() {
                let i = rng.gen_range(0..indices.len());
                let j = rng.gen_range(0..indices.len());
                indices.swap(i, j);
            }

            let selected: Vec<usize> = indices.into_iter().take(n_agents).collect();

            // Create agents array
            let mut agents: Vec<Box<dyn Agent>> = selected
                .iter()
                .map(|&i| population[i].agent.clone_box())
                .collect();

            for (i, agent) in agents.iter_mut().enumerate() {
                agent.set_id(i);
            }

            // Run game
            let result = env.run_episode(&mut agents, Some(self.config.max_steps));

            // Update fitness based on PnL
            for (i, &pop_idx) in selected.iter().enumerate() {
                let pnl = result.total_pnl.get(&i).copied().unwrap_or(0.0);
                population[pop_idx].fitness += pnl;
            }
        }

        // Average fitness
        for member in population.iter_mut() {
            member.fitness /= self.eval_games as f64;
        }
    }

    /// Select and reproduce top performers
    fn evolve_population(&self, population: &mut Vec<PopulationMember>, generation: usize) {
        // Sort by fitness (descending)
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let n_elite = (self.population_size as f64 * self.elite_fraction) as usize;
        let n_elite = n_elite.max(1);

        // Keep elite
        let elite: Vec<PopulationMember> = population[..n_elite].to_vec();

        // Generate new population
        let mut new_population = elite.clone();
        let mut rng = rand::thread_rng();

        while new_population.len() < self.population_size {
            // Select parent from elite
            let parent_idx = rng.gen_range(0..elite.len());
            let parent = &elite[parent_idx];

            // Create child with mutated hyperparameters
            let child_hp = parent.hyperparams.mutate();
            let child_agent = Box::new(RLAgent::new(8, child_hp.hidden_size));

            new_population.push(PopulationMember {
                agent: child_agent,
                hyperparams: child_hp,
                fitness: 0.0,
                generation,
            });
        }

        *population = new_population;
    }

    /// Train using population-based approach
    pub fn train(&self) -> (TrainingResult, PopulationMember) {
        let start_time = std::time::Instant::now();

        let mut population = self.initialize_population();
        let mut result = TrainingResult::default();
        let mut best_member: Option<PopulationMember> = None;
        let mut best_fitness = f64::NEG_INFINITY;

        for gen in 0..self.n_generations {
            // Evaluate all members
            self.evaluate_population(&mut population);

            // Find best in this generation
            let gen_best = population
                .iter()
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .unwrap();

            log::info!(
                "Generation {}/{}: Best fitness = {:.2}, HP: lr={:.2e}, gamma={:.3}",
                gen,
                self.n_generations,
                gen_best.fitness,
                gen_best.hyperparams.learning_rate,
                gen_best.hyperparams.gamma
            );

            result.history.push(gen_best.fitness);

            if gen_best.fitness > best_fitness {
                best_fitness = gen_best.fitness;
                best_member = Some(gen_best.clone());
                result.best_score = best_fitness;
                result.best_episode = gen;
            }

            // Evolve
            self.evolve_population(&mut population, gen + 1);
        }

        result.training_time = start_time.elapsed().as_secs_f64();

        let best = best_member.unwrap_or_else(|| population[0].clone());
        (result, best)
    }

    /// Get population statistics
    pub fn population_stats(population: &[PopulationMember]) -> PopulationStats {
        let fitness_values: Vec<f64> = population.iter().map(|m| m.fitness).collect();

        let mean_fitness = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let variance = fitness_values
            .iter()
            .map(|f| (f - mean_fitness).powi(2))
            .sum::<f64>()
            / fitness_values.len() as f64;
        let std_fitness = variance.sqrt();

        let best_fitness = fitness_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let worst_fitness = fitness_values.iter().cloned().fold(f64::INFINITY, f64::min);

        PopulationStats {
            size: population.len(),
            mean_fitness,
            std_fitness,
            best_fitness,
            worst_fitness,
        }
    }
}

/// Statistics about the population
#[derive(Debug, Clone)]
pub struct PopulationStats {
    pub size: usize,
    pub mean_fitness: f64,
    pub std_fitness: f64,
    pub best_fitness: f64,
    pub worst_fitness: f64,
}

impl std::fmt::Display for PopulationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Population(n={}): fitness={:.2}±{:.2} [best={:.2}, worst={:.2}]",
            self.size, self.mean_fitness, self.std_fitness, self.best_fitness, self.worst_fitness
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparameters_random() {
        let hp = Hyperparameters::random();

        assert!(hp.learning_rate > 0.0);
        assert!(hp.gamma > 0.0 && hp.gamma < 1.0);
        assert!(hp.hidden_size > 0);
    }

    #[test]
    fn test_hyperparameters_mutate() {
        let hp = Hyperparameters {
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon: 0.5,
            hidden_size: 64,
        };

        let mutated = hp.mutate();

        // At least some values should be different (most of the time)
        println!("Original: {:?}", hp);
        println!("Mutated: {:?}", mutated);
    }

    #[test]
    fn test_population_trainer() {
        let config = TrainingConfig::quick();
        let env_config = MultiAgentConfig::default().with_agents(4);

        let trainer = PopulationTrainer::new(config, env_config)
            .with_population_size(10)
            .with_generations(5);

        let (result, best) = trainer.train();

        println!(
            "Training completed in {:.2}s, best fitness: {:.2}",
            result.training_time, result.best_score
        );
        println!("Best hyperparams: {:?}", best.hyperparams);
    }
}
