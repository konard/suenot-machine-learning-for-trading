//! Evolutionary Neural Architecture Search for Trading
//!
//! This library implements evolutionary NAS techniques for discovering
//! optimal neural network architectures for trading strategies.

use anyhow::Result;
use ndarray::Array2;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Activation Functions
// ─────────────────────────────────────────────

/// Supported activation functions for network layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU,
    Swish,
    Linear,
}

impl Activation {
    pub const ALL: &'static [Activation] = &[
        Activation::ReLU,
        Activation::Tanh,
        Activation::Sigmoid,
        Activation::LeakyReLU,
        Activation::Swish,
        Activation::Linear,
    ];

    /// Apply the activation function element-wise.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::Swish => x * (1.0 / (1.0 + (-x).exp())),
            Activation::Linear => x,
        }
    }

    /// Pick a random activation.
    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.gen_range(0..Self::ALL.len())]
    }
}

// ─────────────────────────────────────────────
// Genome (Architecture Encoding)
// ─────────────────────────────────────────────

/// A single layer gene within a genome.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerGene {
    pub size: usize,
    pub activation: Activation,
}

/// A genome encoding a neural network architecture as a chromosome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    /// Hidden layer descriptors (input/output dims are external).
    pub layers: Vec<LayerGene>,
    /// Cached single-objective fitness (higher is better).
    pub fitness: Option<f64>,
    /// Generation when this genome was created.
    pub birth_generation: usize,

    // Multi-objective fitness components
    pub accuracy_score: f64,
    pub latency_score: f64,
    pub robustness_score: f64,

    // NSGA-II ranking fields
    pub pareto_rank: usize,
    pub crowding_distance: f64,
}

impl Genome {
    /// Configuration limits.
    pub const MIN_LAYERS: usize = 1;
    pub const MAX_LAYERS: usize = 8;
    pub const MIN_LAYER_SIZE: usize = 4;
    pub const MAX_LAYER_SIZE: usize = 256;

    /// Create a random genome.
    pub fn random(rng: &mut impl Rng, generation: usize) -> Self {
        let num_layers = rng.gen_range(Self::MIN_LAYERS..=3); // start small (complexification)
        let layers = (0..num_layers)
            .map(|_| LayerGene {
                size: rng.gen_range(Self::MIN_LAYER_SIZE..=64),
                activation: Activation::random(rng),
            })
            .collect();
        Genome {
            layers,
            fitness: None,
            birth_generation: generation,
            accuracy_score: 0.0,
            latency_score: 0.0,
            robustness_score: 0.0,
            pareto_rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// Total number of parameters (rough estimate).
    pub fn param_count(&self, input_dim: usize, output_dim: usize) -> usize {
        if self.layers.is_empty() {
            return input_dim * output_dim + output_dim;
        }
        let mut count = 0;
        let mut prev = input_dim;
        for layer in &self.layers {
            count += prev * layer.size + layer.size; // weights + biases
            prev = layer.size;
        }
        count += prev * output_dim + output_dim;
        count
    }

    /// Mutate the genome in-place.
    pub fn mutate(&mut self, rng: &mut impl Rng) {
        let mutation_type = rng.gen_range(0..4);
        match mutation_type {
            0 => self.mutate_add_layer(rng),
            1 => self.mutate_remove_layer(rng),
            2 => self.mutate_change_size(rng),
            3 => self.mutate_change_activation(rng),
            _ => unreachable!(),
        }
        self.fitness = None; // invalidate cached fitness
    }

    fn mutate_add_layer(&mut self, rng: &mut impl Rng) {
        if self.layers.len() >= Self::MAX_LAYERS {
            return;
        }
        let pos = rng.gen_range(0..=self.layers.len());
        let gene = LayerGene {
            size: rng.gen_range(Self::MIN_LAYER_SIZE..=64),
            activation: Activation::random(rng),
        };
        self.layers.insert(pos, gene);
    }

    fn mutate_remove_layer(&mut self, rng: &mut impl Rng) {
        if self.layers.len() <= Self::MIN_LAYERS {
            return;
        }
        let idx = rng.gen_range(0..self.layers.len());
        self.layers.remove(idx);
    }

    fn mutate_change_size(&mut self, rng: &mut impl Rng) {
        if self.layers.is_empty() {
            return;
        }
        let idx = rng.gen_range(0..self.layers.len());
        let delta: i32 = rng.gen_range(-16..=16);
        let new_size = (self.layers[idx].size as i32 + delta)
            .max(Self::MIN_LAYER_SIZE as i32)
            .min(Self::MAX_LAYER_SIZE as i32) as usize;
        self.layers[idx].size = new_size;
    }

    fn mutate_change_activation(&mut self, rng: &mut impl Rng) {
        if self.layers.is_empty() {
            return;
        }
        let idx = rng.gen_range(0..self.layers.len());
        self.layers[idx].activation = Activation::random(rng);
    }

    /// Compute the architecture "fingerprint" for diversity measurement.
    pub fn fingerprint(&self) -> Vec<u64> {
        self.layers
            .iter()
            .map(|l| {
                let act_id = Activation::ALL.iter().position(|a| *a == l.activation).unwrap() as u64;
                (l.size as u64) * 10 + act_id
            })
            .collect()
    }
}

// ─────────────────────────────────────────────
// Crossover
// ─────────────────────────────────────────────

/// Crossover strategy.
#[derive(Debug, Clone, Copy)]
pub enum CrossoverStrategy {
    SinglePoint,
    Uniform,
}

/// Perform crossover between two parent genomes.
pub fn crossover(
    parent_a: &Genome,
    parent_b: &Genome,
    strategy: CrossoverStrategy,
    generation: usize,
    rng: &mut impl Rng,
) -> Genome {
    let min_len = parent_a.layers.len().min(parent_b.layers.len());
    let _max_len = parent_a.layers.len().max(parent_b.layers.len());

    if min_len == 0 {
        // If one parent has no layers, take from the other
        let base = if parent_a.layers.is_empty() {
            parent_b
        } else {
            parent_a
        };
        return Genome {
            layers: base.layers.clone(),
            fitness: None,
            birth_generation: generation,
            accuracy_score: 0.0,
            latency_score: 0.0,
            robustness_score: 0.0,
            pareto_rank: 0,
            crowding_distance: 0.0,
        };
    }

    let layers = match strategy {
        CrossoverStrategy::SinglePoint => {
            let crossover_point = rng.gen_range(1..=min_len);
            let mut result: Vec<LayerGene> =
                parent_a.layers[..crossover_point].to_vec();
            if crossover_point < parent_b.layers.len() {
                result.extend_from_slice(&parent_b.layers[crossover_point..]);
            }
            result
        }
        CrossoverStrategy::Uniform => {
            let len = if rng.gen_bool(0.5) {
                parent_a.layers.len()
            } else {
                parent_b.layers.len()
            };
            (0..len)
                .map(|i| {
                    let from_a = i < parent_a.layers.len();
                    let from_b = i < parent_b.layers.len();
                    match (from_a, from_b) {
                        (true, true) => {
                            if rng.gen_bool(0.5) {
                                parent_a.layers[i].clone()
                            } else {
                                parent_b.layers[i].clone()
                            }
                        }
                        (true, false) => parent_a.layers[i].clone(),
                        (false, true) => parent_b.layers[i].clone(),
                        (false, false) => unreachable!(),
                    }
                })
                .collect()
        }
    };

    // Enforce limits
    let layers = if layers.len() > Genome::MAX_LAYERS {
        layers[..Genome::MAX_LAYERS].to_vec()
    } else if layers.is_empty() {
        vec![LayerGene {
            size: 16,
            activation: Activation::ReLU,
        }]
    } else {
        layers
    };

    Genome {
        layers,
        fitness: None,
        birth_generation: generation,
        accuracy_score: 0.0,
        latency_score: 0.0,
        robustness_score: 0.0,
        pareto_rank: 0,
        crowding_distance: 0.0,
    }
}

// ─────────────────────────────────────────────
// Simple Neural Network Forward Pass
// ─────────────────────────────────────────────

/// A minimal neural network built from a genome, with random weights.
/// Used for fitness evaluation in the evolutionary loop.
pub struct SimpleNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Vec<f64>>,
    activations: Vec<Activation>,
}

impl SimpleNetwork {
    /// Build a network from a genome with random weights.
    pub fn from_genome(
        genome: &Genome,
        input_dim: usize,
        output_dim: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut activations = Vec::new();
        let mut prev_dim = input_dim;

        for layer in &genome.layers {
            let w = Array2::from_shape_fn((prev_dim, layer.size), |_| {
                rng.gen_range(-1.0..1.0) * (2.0 / prev_dim as f64).sqrt()
            });
            let b = vec![0.0; layer.size];
            weights.push(w);
            biases.push(b);
            activations.push(layer.activation);
            prev_dim = layer.size;
        }

        // Output layer (linear activation)
        let w = Array2::from_shape_fn((prev_dim, output_dim), |_| {
            rng.gen_range(-1.0..1.0) * (2.0 / prev_dim as f64).sqrt()
        });
        let b = vec![0.0; output_dim];
        weights.push(w);
        biases.push(b);
        activations.push(Activation::Linear);

        SimpleNetwork {
            weights,
            biases,
            activations,
        }
    }

    /// Forward pass for a single sample.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();

        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let rows = w.nrows();
            let cols = w.ncols();
            let mut next = vec![0.0; cols];
            for j in 0..cols {
                let mut sum = b[j];
                for k in 0..rows.min(current.len()) {
                    sum += current[k] * w[[k, j]];
                }
                next[j] = self.activations[i].apply(sum);
            }
            current = next;
        }

        current
    }
}

// ─────────────────────────────────────────────
// Fitness Evaluation
// ─────────────────────────────────────────────

/// Market data features and labels for fitness evaluation.
#[derive(Debug, Clone)]
pub struct TradingData {
    /// Each row is a feature vector.
    pub features: Vec<Vec<f64>>,
    /// Target: +1.0 for up, -1.0 for down.
    pub labels: Vec<f64>,
}

impl TradingData {
    /// Split data into train/validation/test by ratio.
    pub fn split(&self, train_ratio: f64, val_ratio: f64) -> (TradingData, TradingData, TradingData) {
        let n = self.features.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = (n as f64 * (train_ratio + val_ratio)) as usize;

        let train = TradingData {
            features: self.features[..train_end].to_vec(),
            labels: self.labels[..train_end].to_vec(),
        };
        let val = TradingData {
            features: self.features[train_end..val_end].to_vec(),
            labels: self.labels[train_end..val_end].to_vec(),
        };
        let test = TradingData {
            features: self.features[val_end..].to_vec(),
            labels: self.labels[val_end..].to_vec(),
        };
        (train, val, test)
    }
}

/// Evaluate a genome's fitness on trading data.
/// Returns (accuracy, latency_score, robustness_score).
pub fn evaluate_genome(
    genome: &Genome,
    _train_data: &TradingData,
    val_data: &TradingData,
    input_dim: usize,
    rng: &mut impl Rng,
) -> (f64, f64, f64) {
    let output_dim = 1;

    // Build network and evaluate on validation data
    // We evaluate multiple times with different random weights to assess robustness.
    let n_evals = 3;
    let mut accuracies = Vec::new();

    for _ in 0..n_evals {
        let net = SimpleNetwork::from_genome(genome, input_dim, output_dim, rng);

        let mut correct = 0usize;
        let total = val_data.features.len();

        for (features, label) in val_data.features.iter().zip(val_data.labels.iter()) {
            let output = net.forward(features);
            let prediction = if output[0] > 0.0 { 1.0 } else { -1.0 };
            if prediction * label > 0.0 {
                correct += 1;
            }
        }

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.5
        };
        accuracies.push(accuracy);
    }

    // Accuracy: mean across evaluations
    let mean_accuracy = accuracies.iter().sum::<f64>() / accuracies.len() as f64;

    // Latency score: inversely proportional to parameter count (normalized)
    let params = genome.param_count(input_dim, output_dim) as f64;
    let latency_score = 1.0 / (1.0 + params / 1000.0);

    // Robustness: 1 - std_dev of accuracies (consistency)
    let variance = if accuracies.len() > 1 {
        let mean = mean_accuracy;
        accuracies.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (accuracies.len() - 1) as f64
    } else {
        0.0
    };
    let robustness = 1.0 / (1.0 + variance.sqrt() * 10.0);

    (mean_accuracy, latency_score, robustness)
}

// ─────────────────────────────────────────────
// NSGA-II Inspired Pareto Ranking
// ─────────────────────────────────────────────

/// Check if genome `a` dominates genome `b` across all objectives.
pub fn dominates(a: &Genome, b: &Genome) -> bool {
    let a_scores = [a.accuracy_score, a.latency_score, a.robustness_score];
    let b_scores = [b.accuracy_score, b.latency_score, b.robustness_score];

    let mut all_geq = true;
    let mut any_gt = false;

    for (sa, sb) in a_scores.iter().zip(b_scores.iter()) {
        if sa < sb {
            all_geq = false;
            break;
        }
        if sa > sb {
            any_gt = true;
        }
    }

    all_geq && any_gt
}

/// Assign Pareto ranks to the population (modifies in place).
pub fn assign_pareto_ranks(population: &mut [Genome]) {
    let n = population.len();
    let mut ranks = vec![0usize; n];
    let mut assigned = vec![false; n];
    let mut current_rank = 1;

    loop {
        let mut front = Vec::new();
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let mut is_dominated = false;
            for j in 0..n {
                if i == j || assigned[j] {
                    continue;
                }
                if dominates(&population[j], &population[i]) {
                    is_dominated = true;
                    break;
                }
            }
            if !is_dominated {
                front.push(i);
            }
        }

        if front.is_empty() {
            break;
        }

        for &idx in &front {
            ranks[idx] = current_rank;
            assigned[idx] = true;
        }

        // Compute crowding distance within this front
        compute_crowding_distance(population, &front);

        current_rank += 1;
    }

    for (i, rank) in ranks.iter().enumerate() {
        population[i].pareto_rank = *rank;
    }
}

fn compute_crowding_distance(population: &mut [Genome], front: &[usize]) {
    let m = 3; // number of objectives
    let n = front.len();

    for &idx in front {
        population[idx].crowding_distance = 0.0;
    }

    if n <= 2 {
        for &idx in front {
            population[idx].crowding_distance = f64::INFINITY;
        }
        return;
    }

    for obj in 0..m {
        let mut indices: Vec<usize> = front.to_vec();
        indices.sort_by(|&a, &b| {
            let va = get_objective(&population[a], obj);
            let vb = get_objective(&population[b], obj);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_min = get_objective(&population[indices[0]], obj);
        let f_max = get_objective(&population[indices[n - 1]], obj);
        let range = f_max - f_min;

        population[indices[0]].crowding_distance = f64::INFINITY;
        population[indices[n - 1]].crowding_distance = f64::INFINITY;

        if range > 1e-12 {
            for k in 1..n - 1 {
                let next = get_objective(&population[indices[k + 1]], obj);
                let prev = get_objective(&population[indices[k - 1]], obj);
                population[indices[k]].crowding_distance += (next - prev) / range;
            }
        }
    }
}

fn get_objective(genome: &Genome, index: usize) -> f64 {
    match index {
        0 => genome.accuracy_score,
        1 => genome.latency_score,
        2 => genome.robustness_score,
        _ => 0.0,
    }
}

// ─────────────────────────────────────────────
// Generation Statistics
// ─────────────────────────────────────────────

/// Statistics for one generation.
#[derive(Debug, Clone, Serialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub worst_fitness: f64,
    pub diversity: f64,
    pub best_architecture: String,
    pub population_size: usize,
}

/// Compute pairwise Hamming-like diversity of the population.
pub fn compute_diversity(population: &[Genome]) -> f64 {
    let n = population.len();
    if n < 2 {
        return 0.0;
    }

    let fingerprints: Vec<Vec<u64>> = population.iter().map(|g| g.fingerprint()).collect();
    let mut total_dist = 0.0;
    let mut pairs = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let max_len = fingerprints[i].len().max(fingerprints[j].len());
            let min_len = fingerprints[i].len().min(fingerprints[j].len());
            let mut dist = (max_len - min_len) as f64; // length difference
            for k in 0..min_len {
                if fingerprints[i][k] != fingerprints[j][k] {
                    dist += 1.0;
                }
            }
            total_dist += dist;
            pairs += 1;
        }
    }

    total_dist / pairs as f64
}

fn format_architecture(genome: &Genome) -> String {
    genome
        .layers
        .iter()
        .map(|l| format!("{}{:?}", l.size, l.activation))
        .collect::<Vec<_>>()
        .join(" -> ")
}

// ─────────────────────────────────────────────
// Evolution Engine
// ─────────────────────────────────────────────

/// Configuration for the evolutionary search.
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub num_generations: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub crossover_strategy: CrossoverStrategy,
    pub use_aging: bool,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        EvolutionConfig {
            population_size: 20,
            num_generations: 50,
            tournament_size: 3,
            mutation_rate: 0.3,
            crossover_rate: 0.7,
            crossover_strategy: CrossoverStrategy::Uniform,
            use_aging: true,
            input_dim: 5,
            output_dim: 1,
        }
    }
}

/// The main evolution engine that orchestrates NAS.
pub struct EvolutionEngine {
    pub config: EvolutionConfig,
    pub population: Vec<Genome>,
    pub generation: usize,
    pub history: Vec<GenerationStats>,
    rng: StdRng,
}

impl EvolutionEngine {
    pub fn new(config: EvolutionConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let population = (0..config.population_size)
            .map(|_| Genome::random(&mut rng, 0))
            .collect();

        EvolutionEngine {
            config,
            population,
            generation: 0,
            history: Vec::new(),
            rng,
        }
    }

    /// Initialize the population with random genomes.
    pub fn initialize_population(&mut self) {
        self.population = (0..self.config.population_size)
            .map(|_| Genome::random(&mut self.rng, 0))
            .collect();
        self.generation = 0;
        self.history.clear();
    }

    /// Tournament selection: pick the best out of k random individuals.
    pub fn tournament_select(&mut self) -> Genome {
        let mut best: Option<&Genome> = None;

        for _ in 0..self.config.tournament_size {
            let idx = self.rng.gen_range(0..self.population.len());
            let candidate = &self.population[idx];
            let is_better = match best {
                None => true,
                Some(current_best) => {
                    let cf = current_best.fitness.unwrap_or(f64::NEG_INFINITY);
                    let nf = candidate.fitness.unwrap_or(f64::NEG_INFINITY);
                    nf > cf
                }
            };
            if is_better {
                best = Some(candidate);
            }
        }

        best.unwrap().clone()
    }

    /// Evaluate all genomes in the population.
    pub fn evaluate_population(&mut self, train_data: &TradingData, val_data: &TradingData) {
        let input_dim = self.config.input_dim;

        for genome in &mut self.population {
            if genome.fitness.is_some() {
                continue; // skip already evaluated
            }
            let mut rng = StdRng::seed_from_u64(self.rng.gen());
            let (accuracy, latency, robustness) =
                evaluate_genome(genome, train_data, val_data, input_dim, &mut rng);
            genome.accuracy_score = accuracy;
            genome.latency_score = latency;
            genome.robustness_score = robustness;
            // Composite fitness: weighted sum
            genome.fitness = Some(0.6 * accuracy + 0.2 * latency + 0.2 * robustness);
        }

        // Assign Pareto ranks
        assign_pareto_ranks(&mut self.population);
    }

    /// Run one generation of evolution.
    pub fn step(&mut self, train_data: &TradingData, val_data: &TradingData) {
        self.evaluate_population(train_data, val_data);

        // Record stats
        let stats = self.compute_stats();
        self.history.push(stats);

        if self.config.use_aging {
            self.aging_evolution_step(train_data, val_data);
        } else {
            self.generational_step(train_data, val_data);
        }

        self.generation += 1;
    }

    /// Aging evolution: add one offspring, remove oldest.
    fn aging_evolution_step(&mut self, train_data: &TradingData, val_data: &TradingData) {
        // Select parent via tournament
        let parent = self.tournament_select();

        // Create offspring by mutation (aging evolution typically uses mutation only)
        let mut offspring = parent.clone();
        offspring.birth_generation = self.generation + 1;
        offspring.fitness = None;

        if self.rng.gen_bool(self.config.mutation_rate) {
            offspring.mutate(&mut self.rng);
        }

        // Optionally crossover with another parent
        if self.rng.gen_bool(self.config.crossover_rate * 0.3) {
            let parent_b = self.tournament_select();
            offspring = crossover(
                &offspring,
                &parent_b,
                self.config.crossover_strategy,
                self.generation + 1,
                &mut self.rng,
            );
        }

        // Evaluate offspring
        let input_dim = self.config.input_dim;
        let mut rng = StdRng::seed_from_u64(self.rng.gen());
        let (accuracy, latency, robustness) =
            evaluate_genome(&offspring, train_data, val_data, input_dim, &mut rng);
        offspring.accuracy_score = accuracy;
        offspring.latency_score = latency;
        offspring.robustness_score = robustness;
        offspring.fitness = Some(0.6 * accuracy + 0.2 * latency + 0.2 * robustness);

        // Add offspring
        self.population.push(offspring);

        // Remove oldest individual
        if self.population.len() > self.config.population_size {
            let oldest_idx = self
                .population
                .iter()
                .enumerate()
                .min_by_key(|(_, g)| g.birth_generation)
                .map(|(i, _)| i)
                .unwrap();
            self.population.remove(oldest_idx);
        }
    }

    /// Standard generational replacement.
    fn generational_step(&mut self, train_data: &TradingData, val_data: &TradingData) {
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Elitism: keep the best individual
        if let Some(best) = self
            .population
            .iter()
            .max_by(|a, b| {
                a.fitness
                    .unwrap_or(f64::NEG_INFINITY)
                    .partial_cmp(&b.fitness.unwrap_or(f64::NEG_INFINITY))
                    .unwrap()
            })
            .cloned()
        {
            new_population.push(best);
        }

        while new_population.len() < self.config.population_size {
            let parent_a = self.tournament_select();

            let offspring = if self.rng.gen_bool(self.config.crossover_rate) {
                let parent_b = self.tournament_select();
                crossover(
                    &parent_a,
                    &parent_b,
                    self.config.crossover_strategy,
                    self.generation + 1,
                    &mut self.rng,
                )
            } else {
                let mut child = parent_a.clone();
                child.birth_generation = self.generation + 1;
                child.fitness = None;
                child
            };

            let mut offspring = offspring;
            if self.rng.gen_bool(self.config.mutation_rate) {
                offspring.mutate(&mut self.rng);
            }

            new_population.push(offspring);
        }

        self.population = new_population;

        // Evaluate new population
        self.evaluate_population(train_data, val_data);
    }

    /// Run the full evolutionary loop.
    pub fn run(&mut self, train_data: &TradingData, val_data: &TradingData) {
        for _ in 0..self.config.num_generations {
            self.step(train_data, val_data);
        }
    }

    /// Get the best genome in the current population.
    pub fn best_genome(&self) -> Option<&Genome> {
        self.population
            .iter()
            .max_by(|a, b| {
                a.fitness
                    .unwrap_or(f64::NEG_INFINITY)
                    .partial_cmp(&b.fitness.unwrap_or(f64::NEG_INFINITY))
                    .unwrap()
            })
    }

    /// Get top-k genomes.
    pub fn top_k(&self, k: usize) -> Vec<&Genome> {
        let mut sorted: Vec<&Genome> = self.population.iter().collect();
        sorted.sort_by(|a, b| {
            b.fitness
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(&a.fitness.unwrap_or(f64::NEG_INFINITY))
                .unwrap()
        });
        sorted.truncate(k);
        sorted
    }

    /// Compute generation statistics.
    fn compute_stats(&self) -> GenerationStats {
        let fitnesses: Vec<f64> = self
            .population
            .iter()
            .filter_map(|g| g.fitness)
            .collect();

        let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean = if fitnesses.is_empty() {
            0.0
        } else {
            fitnesses.iter().sum::<f64>() / fitnesses.len() as f64
        };

        let diversity = compute_diversity(&self.population);

        let best_genome = self.best_genome();
        let best_arch = best_genome
            .map(|g| format_architecture(g))
            .unwrap_or_else(|| "N/A".to_string());

        GenerationStats {
            generation: self.generation,
            best_fitness: best,
            mean_fitness: mean,
            worst_fitness: worst,
            diversity,
            best_architecture: best_arch,
            population_size: self.population.len(),
        }
    }

    /// Print convergence summary.
    pub fn print_convergence(&self) {
        println!("\n=== Convergence Analysis ===");
        for stats in &self.history {
            println!(
                "Gen {:3} | Best: {:.4} | Mean: {:.4} | Diversity: {:.2} | Arch: {}",
                stats.generation,
                stats.best_fitness,
                stats.mean_fitness,
                stats.diversity,
                stats.best_architecture
            );
        }
    }
}

// ─────────────────────────────────────────────
// Bybit API Integration
// ─────────────────────────────────────────────

/// A single OHLCV kline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures.
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Client for fetching data from Bybit V5 API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl BybitClient {
    pub fn new() -> Self {
        BybitClient {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = self.client.get(&url).send()?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                Some(Kline {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns newest first; reverse for chronological order
        klines.reverse();

        Ok(klines)
    }
}

/// Convert klines into trading features and labels.
pub fn klines_to_trading_data(klines: &[Kline], lookback: usize) -> TradingData {
    let n = klines.len();
    if n < lookback + 2 {
        return TradingData {
            features: vec![],
            labels: vec![],
        };
    }

    // Compute log returns
    let returns: Vec<f64> = klines
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for i in lookback..returns.len() - 1 {
        let mut feature_vec = Vec::new();

        // Feature 1: recent returns
        let recent_return = returns[i];
        feature_vec.push(recent_return);

        // Feature 2: rolling mean return
        let mean_ret: f64 =
            returns[i - lookback..i].iter().sum::<f64>() / lookback as f64;
        feature_vec.push(mean_ret);

        // Feature 3: rolling volatility
        let variance: f64 = returns[i - lookback..i]
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / lookback as f64;
        feature_vec.push(variance.sqrt());

        // Feature 4: price relative to moving average
        let ma: f64 = klines[i - lookback..i]
            .iter()
            .map(|k| k.close)
            .sum::<f64>()
            / lookback as f64;
        feature_vec.push(klines[i].close / ma - 1.0);

        // Feature 5: volume ratio
        let avg_volume: f64 = klines[i - lookback..i]
            .iter()
            .map(|k| k.volume)
            .sum::<f64>()
            / lookback as f64;
        let vol_ratio = if avg_volume > 0.0 {
            klines[i].volume / avg_volume
        } else {
            1.0
        };
        feature_vec.push(vol_ratio);

        features.push(feature_vec);

        // Label: next period direction
        let label = if returns[i + 1] > 0.0 { 1.0 } else { -1.0 };
        labels.push(label);
    }

    TradingData { features, labels }
}

/// Generate synthetic trading data for testing.
pub fn generate_synthetic_data(n_samples: usize, n_features: usize, seed: u64) -> TradingData {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..n_samples {
        let feat: Vec<f64> = (0..n_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
        // Simple rule: if sum of features > 0 then up, else down (with noise)
        let signal: f64 = feat.iter().sum::<f64>() + rng.gen_range(-0.5..0.5);
        let label = if signal > 0.0 { 1.0 } else { -1.0 };
        features.push(feat);
        labels.push(label);
    }

    TradingData { features, labels }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_apply() {
        assert_eq!(Activation::ReLU.apply(-1.0), 0.0);
        assert_eq!(Activation::ReLU.apply(2.0), 2.0);
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
        assert!((Activation::Tanh.apply(0.0)).abs() < 1e-10);
        assert_eq!(Activation::Linear.apply(3.14), 3.14);
        assert_eq!(Activation::LeakyReLU.apply(-1.0), -0.01);
    }

    #[test]
    fn test_genome_random() {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng, 0);
        assert!(!genome.layers.is_empty());
        assert!(genome.layers.len() <= 3);
        assert!(genome.fitness.is_none());
        assert_eq!(genome.birth_generation, 0);
    }

    #[test]
    fn test_genome_mutation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = Genome::random(&mut rng, 0);
        let original_layers = genome.layers.clone();

        // Apply many mutations to ensure at least one change
        for _ in 0..20 {
            genome.mutate(&mut rng);
        }

        // After many mutations, the genome should have changed
        // (not guaranteed per mutation, but very likely after 20)
        let changed = genome.layers != original_layers;
        assert!(changed, "Expected genome to change after mutations");
    }

    #[test]
    fn test_crossover_single_point() {
        let mut rng = StdRng::seed_from_u64(42);
        let parent_a = Genome::random(&mut rng, 0);
        let parent_b = Genome::random(&mut rng, 0);

        let child = crossover(&parent_a, &parent_b, CrossoverStrategy::SinglePoint, 1, &mut rng);
        assert!(!child.layers.is_empty());
        assert_eq!(child.birth_generation, 1);
        assert!(child.fitness.is_none());
    }

    #[test]
    fn test_crossover_uniform() {
        let mut rng = StdRng::seed_from_u64(42);
        let parent_a = Genome::random(&mut rng, 0);
        let parent_b = Genome::random(&mut rng, 0);

        let child = crossover(&parent_a, &parent_b, CrossoverStrategy::Uniform, 1, &mut rng);
        assert!(!child.layers.is_empty());
    }

    #[test]
    fn test_param_count() {
        let genome = Genome {
            layers: vec![
                LayerGene { size: 16, activation: Activation::ReLU },
                LayerGene { size: 8, activation: Activation::Tanh },
            ],
            fitness: None,
            birth_generation: 0,
            accuracy_score: 0.0,
            latency_score: 0.0,
            robustness_score: 0.0,
            pareto_rank: 0,
            crowding_distance: 0.0,
        };

        // input_dim=5, output_dim=1
        // Layer 1: 5*16 + 16 = 96
        // Layer 2: 16*8 + 8 = 136
        // Output: 8*1 + 1 = 9
        // Total: 241
        assert_eq!(genome.param_count(5, 1), 241);
    }

    #[test]
    fn test_simple_network_forward() {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng, 0);
        let net = SimpleNetwork::from_genome(&genome, 5, 1, &mut rng);

        let input = vec![0.1, -0.2, 0.3, 0.0, -0.1];
        let output = net.forward(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_fitness_evaluation() {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng, 0);

        let data = generate_synthetic_data(100, 5, 42);
        let (train, val, _test) = data.split(0.6, 0.2);

        let (accuracy, latency, robustness) =
            evaluate_genome(&genome, &train, &val, 5, &mut rng);

        assert!(accuracy >= 0.0 && accuracy <= 1.0);
        assert!(latency > 0.0 && latency <= 1.0);
        assert!(robustness > 0.0 && robustness <= 1.0);
    }

    #[test]
    fn test_dominates() {
        let mut a = Genome::random(&mut StdRng::seed_from_u64(1), 0);
        let mut b = Genome::random(&mut StdRng::seed_from_u64(2), 0);

        a.accuracy_score = 0.8;
        a.latency_score = 0.9;
        a.robustness_score = 0.7;

        b.accuracy_score = 0.7;
        b.latency_score = 0.8;
        b.robustness_score = 0.6;

        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_pareto_ranking() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut population: Vec<Genome> = (0..10)
            .map(|_| {
                let mut g = Genome::random(&mut rng, 0);
                g.accuracy_score = rng.gen();
                g.latency_score = rng.gen();
                g.robustness_score = rng.gen();
                g
            })
            .collect();

        assign_pareto_ranks(&mut population);

        // All should have rank >= 1
        for g in &population {
            assert!(g.pareto_rank >= 1);
        }

        // At least one should be rank 1 (non-dominated front)
        assert!(population.iter().any(|g| g.pareto_rank == 1));
    }

    #[test]
    fn test_diversity() {
        let mut rng = StdRng::seed_from_u64(42);

        // Identical population: diversity = 0
        let genome = Genome::random(&mut rng, 0);
        let identical_pop = vec![genome.clone(), genome.clone(), genome.clone()];
        assert_eq!(compute_diversity(&identical_pop), 0.0);

        // Diverse population should have positive diversity
        let diverse_pop: Vec<Genome> = (0..5).map(|_| Genome::random(&mut rng, 0)).collect();
        assert!(compute_diversity(&diverse_pop) > 0.0);
    }

    #[test]
    fn test_evolution_engine() {
        let config = EvolutionConfig {
            population_size: 10,
            num_generations: 5,
            tournament_size: 3,
            mutation_rate: 0.3,
            crossover_rate: 0.5,
            crossover_strategy: CrossoverStrategy::Uniform,
            use_aging: true,
            input_dim: 5,
            output_dim: 1,
        };

        let mut engine = EvolutionEngine::new(config, 42);
        let data = generate_synthetic_data(100, 5, 42);
        let (train, val, _test) = data.split(0.6, 0.2);

        engine.run(&train, &val);

        assert_eq!(engine.history.len(), 5);
        assert!(engine.best_genome().is_some());

        let best = engine.best_genome().unwrap();
        assert!(best.fitness.is_some());
    }

    #[test]
    fn test_generational_evolution() {
        let config = EvolutionConfig {
            population_size: 10,
            num_generations: 3,
            tournament_size: 2,
            mutation_rate: 0.2,
            crossover_rate: 0.7,
            crossover_strategy: CrossoverStrategy::SinglePoint,
            use_aging: false,
            input_dim: 5,
            output_dim: 1,
        };

        let mut engine = EvolutionEngine::new(config, 99);
        let data = generate_synthetic_data(80, 5, 99);
        let (train, val, _test) = data.split(0.6, 0.2);

        engine.run(&train, &val);

        assert_eq!(engine.history.len(), 3);
        assert_eq!(engine.population.len(), 10);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let data = generate_synthetic_data(50, 5, 42);
        assert_eq!(data.features.len(), 50);
        assert_eq!(data.labels.len(), 50);
        assert_eq!(data.features[0].len(), 5);
    }

    #[test]
    fn test_data_split() {
        let data = generate_synthetic_data(100, 5, 42);
        let (train, val, test) = data.split(0.6, 0.2);

        assert_eq!(train.features.len(), 60);
        assert_eq!(val.features.len(), 20);
        assert_eq!(test.features.len(), 20);
    }

    #[test]
    fn test_klines_to_trading_data() {
        // Create synthetic klines
        let klines: Vec<Kline> = (0..50)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 5.0;
                Kline {
                    timestamp: 1000 + i * 60,
                    open: price - 0.1,
                    high: price + 0.5,
                    low: price - 0.5,
                    close: price,
                    volume: 1000.0 + i as f64 * 10.0,
                }
            })
            .collect();

        let data = klines_to_trading_data(&klines, 10);
        assert!(!data.features.is_empty());
        assert_eq!(data.features.len(), data.labels.len());
        assert_eq!(data.features[0].len(), 5); // 5 features
    }

    #[test]
    fn test_top_k() {
        let config = EvolutionConfig {
            population_size: 10,
            num_generations: 3,
            ..Default::default()
        };

        let mut engine = EvolutionEngine::new(config, 42);
        let data = generate_synthetic_data(100, 5, 42);
        let (train, val, _test) = data.split(0.6, 0.2);

        engine.run(&train, &val);

        let top3 = engine.top_k(3);
        assert_eq!(top3.len(), 3);

        // Verify sorted descending by fitness
        for i in 0..top3.len() - 1 {
            assert!(
                top3[i].fitness.unwrap_or(f64::NEG_INFINITY)
                    >= top3[i + 1].fitness.unwrap_or(f64::NEG_INFINITY)
            );
        }
    }
}
