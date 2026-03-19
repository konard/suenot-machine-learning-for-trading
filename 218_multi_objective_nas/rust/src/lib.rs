//! Multi-Objective Neural Architecture Search (NAS) for Trading
//!
//! Implements NSGA-II based multi-objective optimization to find
//! Pareto-optimal neural network architectures that balance
//! accuracy, latency, and model size for trading applications.

use anyhow::{Context, Result};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Bybit API types
// ---------------------------------------------------------------------------

/// Raw JSON response from the Bybit V5 kline endpoint.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// A single OHLCV candlestick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from the Bybit V5 public API (blocking).
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp: BybitResponse = reqwest::blocking::get(&url)
        .context("HTTP request to Bybit failed")?
        .json()
        .context("Failed to parse Bybit JSON response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error {}: {}", resp.ret_code, resp.ret_msg);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Compute simple log-returns from closing prices.
pub fn compute_returns(candles: &[Candle]) -> Vec<f64> {
    candles
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Compute rolling volatility (standard deviation of returns).
pub fn compute_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return vec![];
    }
    (0..=returns.len() - window)
        .map(|i| {
            let slice = &returns[i..i + window];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
            var.sqrt()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Architecture representation
// ---------------------------------------------------------------------------

/// Activation function choices in the search space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    SiLU,
}

impl Activation {
    pub const ALL: [Activation; 5] = [
        Activation::ReLU,
        Activation::GELU,
        Activation::Tanh,
        Activation::Sigmoid,
        Activation::SiLU,
    ];

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.gen_range(0..Self::ALL.len())]
    }
}

/// A candidate neural-network architecture in the NAS search space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NasArchitecture {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub attention_heads: usize,
    pub dropout: f64,
    pub activation: Activation,
    pub use_skip_connections: bool,
    pub use_batch_norm: bool,
}

impl NasArchitecture {
    /// Sample a random architecture.
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            num_layers: rng.gen_range(1..=8),
            hidden_dim: *[32, 64, 128, 256, 512].choose(rng).unwrap(),
            attention_heads: *[0, 1, 2, 4, 8].choose(rng).unwrap(),
            dropout: (rng.gen_range(0..=5) as f64) * 0.1,
            activation: Activation::random(rng),
            use_skip_connections: rng.gen_bool(0.5),
            use_batch_norm: rng.gen_bool(0.5),
        }
    }

    /// Estimate the number of parameters (proxy for model size).
    pub fn estimated_params(&self) -> usize {
        let input_dim = 64; // assumed feature size
        let mut params = input_dim * self.hidden_dim; // first layer
        params += (self.num_layers.saturating_sub(1)) * self.hidden_dim * self.hidden_dim;
        if self.attention_heads > 0 {
            // Q, K, V projections per attention head per layer
            params += self.num_layers * self.attention_heads * self.hidden_dim * 3;
        }
        if self.use_batch_norm {
            params += self.num_layers * self.hidden_dim * 2; // gamma, beta
        }
        params += self.hidden_dim; // output layer
        params
    }

    /// Estimate inference latency in microseconds (simplified model).
    pub fn estimated_latency_us(&self) -> f64 {
        let base = 10.0; // base overhead µs
        let per_layer = 5.0 * (self.hidden_dim as f64 / 64.0);
        let attention_cost = if self.attention_heads > 0 {
            self.attention_heads as f64 * self.hidden_dim as f64 * 0.02
        } else {
            0.0
        };
        let bn_cost = if self.use_batch_norm { 2.0 * self.num_layers as f64 } else { 0.0 };
        base + self.num_layers as f64 * (per_layer + attention_cost) + bn_cost
    }

    /// Crossover: uniform gene-level crossover of two parents.
    pub fn crossover(a: &Self, b: &Self, rng: &mut impl Rng) -> Self {
        Self {
            num_layers: if rng.gen_bool(0.5) { a.num_layers } else { b.num_layers },
            hidden_dim: if rng.gen_bool(0.5) { a.hidden_dim } else { b.hidden_dim },
            attention_heads: if rng.gen_bool(0.5) { a.attention_heads } else { b.attention_heads },
            dropout: if rng.gen_bool(0.5) { a.dropout } else { b.dropout },
            activation: if rng.gen_bool(0.5) { a.activation } else { b.activation },
            use_skip_connections: if rng.gen_bool(0.5) {
                a.use_skip_connections
            } else {
                b.use_skip_connections
            },
            use_batch_norm: if rng.gen_bool(0.5) { a.use_batch_norm } else { b.use_batch_norm },
        }
    }

    /// Mutate the architecture with a given per-gene probability.
    pub fn mutate(&mut self, mutation_rate: f64, rng: &mut impl Rng) {
        if rng.gen_bool(mutation_rate) {
            self.num_layers = rng.gen_range(1..=8);
        }
        if rng.gen_bool(mutation_rate) {
            self.hidden_dim = *[32, 64, 128, 256, 512].choose(rng).unwrap();
        }
        if rng.gen_bool(mutation_rate) {
            self.attention_heads = *[0, 1, 2, 4, 8].choose(rng).unwrap();
        }
        if rng.gen_bool(mutation_rate) {
            self.dropout = (rng.gen_range(0..=5) as f64) * 0.1;
        }
        if rng.gen_bool(mutation_rate) {
            self.activation = Activation::random(rng);
        }
        if rng.gen_bool(mutation_rate) {
            self.use_skip_connections = rng.gen_bool(0.5);
        }
        if rng.gen_bool(mutation_rate) {
            self.use_batch_norm = rng.gen_bool(0.5);
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-objective fitness
// ---------------------------------------------------------------------------

/// A vector of objective values (lower is better for all objectives).
pub type ObjectiveValues = Vec<f64>;

/// An individual in the evolutionary population.
#[derive(Debug, Clone)]
pub struct Individual {
    pub architecture: NasArchitecture,
    pub objectives: ObjectiveValues,
    pub rank: usize,
    pub crowding_distance: f64,
}

/// Evaluate an architecture on the three trading objectives.
///
/// Objectives (all minimised):
///   0: negative accuracy (we negate so lower = better)
///   1: inference latency (µs)
///   2: model size (number of parameters, normalised)
///
/// The `volatility` parameter from market data modulates the accuracy
/// estimate to simulate market-data-dependent performance.
pub fn evaluate_objectives(arch: &NasArchitecture, volatility: f64) -> ObjectiveValues {
    // Simulated accuracy: deeper/wider + attention -> more accurate, but noisy
    let depth_bonus = (arch.num_layers as f64).sqrt() * 0.08;
    let width_bonus = (arch.hidden_dim as f64).log2() * 0.03;
    let attn_bonus = if arch.attention_heads > 0 {
        (arch.attention_heads as f64).sqrt() * 0.04
    } else {
        0.0
    };
    let skip_bonus = if arch.use_skip_connections { 0.02 } else { 0.0 };
    let dropout_penalty = arch.dropout * 0.05;
    let vol_adjustment = volatility * 0.5; // high vol hurts accuracy
    let accuracy = (0.5 + depth_bonus + width_bonus + attn_bonus + skip_bonus
        - dropout_penalty
        - vol_adjustment)
        .clamp(0.3, 0.95);

    let latency = arch.estimated_latency_us();
    let model_size = arch.estimated_params() as f64 / 1_000_000.0; // in millions

    vec![-accuracy, latency, model_size] // negate accuracy for minimisation
}

// ---------------------------------------------------------------------------
// Pareto dominance
// ---------------------------------------------------------------------------

/// Returns true if `a` Pareto-dominates `b` (all objectives minimised).
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut dominated_in_any = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false; // a is worse in this objective
        }
        if ai < bi {
            dominated_in_any = true;
        }
    }
    dominated_in_any
}

// ---------------------------------------------------------------------------
// Non-dominated sorting (NSGA-II fast non-dominated sort)
// ---------------------------------------------------------------------------

/// Partition `individuals` into successive non-dominated fronts.
/// Returns a vector of fronts, each front being a vector of indices
/// into the input slice.
pub fn non_dominated_sort(individuals: &[Individual]) -> Vec<Vec<usize>> {
    let n = individuals.len();
    let mut domination_count = vec![0usize; n]; // how many dominate me
    let mut dominated_set: Vec<Vec<usize>> = vec![vec![]; n]; // whom do I dominate
    let mut fronts: Vec<Vec<usize>> = vec![];

    // Pairwise dominance comparison
    for i in 0..n {
        for j in (i + 1)..n {
            let oi = &individuals[i].objectives;
            let oj = &individuals[j].objectives;
            if dominates(oi, oj) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if dominates(oj, oi) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    // First front: those dominated by nobody
    let mut front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();
    while !front.is_empty() {
        let mut next_front = vec![];
        for &i in &front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(front);
        front = next_front;
    }

    fronts
}

// ---------------------------------------------------------------------------
// Crowding distance
// ---------------------------------------------------------------------------

/// Compute crowding distance for a set of individuals identified by `indices`.
/// Updates `crowding_distance` in-place on the `individuals` slice.
pub fn crowding_distance(individuals: &mut [Individual], indices: &[usize]) {
    let n = indices.len();
    if n == 0 {
        return;
    }
    for &idx in indices {
        individuals[idx].crowding_distance = 0.0;
    }
    if n <= 2 {
        for &idx in indices {
            individuals[idx].crowding_distance = f64::INFINITY;
        }
        return;
    }

    let num_objectives = individuals[indices[0]].objectives.len();
    for m in 0..num_objectives {
        // Sort indices by objective m
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            individuals[a].objectives[m]
                .partial_cmp(&individuals[b].objectives[m])
                .unwrap_or(Ordering::Equal)
        });

        // Boundary solutions get infinite distance
        individuals[sorted[0]].crowding_distance = f64::INFINITY;
        individuals[sorted[n - 1]].crowding_distance = f64::INFINITY;

        let f_min = individuals[sorted[0]].objectives[m];
        let f_max = individuals[sorted[n - 1]].objectives[m];
        let range = f_max - f_min;
        if range < 1e-12 {
            continue;
        }

        for k in 1..n - 1 {
            let diff = individuals[sorted[k + 1]].objectives[m]
                - individuals[sorted[k - 1]].objectives[m];
            individuals[sorted[k]].crowding_distance += diff / range;
        }
    }
}

// ---------------------------------------------------------------------------
// Hypervolume indicator (2-D exact, general approximate)
// ---------------------------------------------------------------------------

/// Compute the exact 2-D hypervolume indicator.
/// `points` should contain 2-D objective vectors (minimisation).
/// `reference` is the reference (anti-optimal) point.
pub fn hypervolume_2d(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    debug_assert!(reference.len() == 2);
    let mut pts: Vec<[f64; 2]> = points
        .iter()
        .filter(|p| p[0] < reference[0] && p[1] < reference[1])
        .map(|p| [p[0], p[1]])
        .collect();
    if pts.is_empty() {
        return 0.0;
    }
    // Sort by first objective ascending
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(Ordering::Equal));

    let mut hv = 0.0;
    let mut y_upper = reference[1];
    for pt in &pts {
        if pt[1] < y_upper {
            hv += (reference[0] - pt[0]) * (y_upper - pt[1]);
            y_upper = pt[1];
        }
    }
    hv
}

/// Approximate hypervolume for arbitrary dimensions using Monte-Carlo sampling.
pub fn hypervolume_mc(
    points: &[Vec<f64>],
    reference: &[f64],
    ideal: &[f64],
    samples: usize,
    rng: &mut impl Rng,
) -> f64 {
    if points.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let dim = reference.len();
    let box_volume: f64 = reference
        .iter()
        .zip(ideal.iter())
        .map(|(r, id)| (r - id).abs())
        .product();

    let mut dominated_count = 0usize;
    for _ in 0..samples {
        let sample: Vec<f64> = (0..dim)
            .map(|d| {
                let lo = ideal[d].min(reference[d]);
                let hi = ideal[d].max(reference[d]);
                rng.gen_range(lo..hi)
            })
            .collect();
        // Check if sample is dominated by any point
        let is_dominated = points.iter().any(|p| dominates(p, &sample));
        if is_dominated {
            dominated_count += 1;
        }
    }
    box_volume * (dominated_count as f64 / samples as f64)
}

// ---------------------------------------------------------------------------
// Selection / evolutionary operators
// ---------------------------------------------------------------------------

/// Binary tournament selection based on rank then crowding distance.
fn tournament_select<'a>(pop: &'a [Individual], rng: &mut impl Rng) -> &'a Individual {
    let a = rng.gen_range(0..pop.len());
    let b = rng.gen_range(0..pop.len());
    let ia = &pop[a];
    let ib = &pop[b];
    if ia.rank < ib.rank {
        ia
    } else if ib.rank < ia.rank {
        ib
    } else if ia.crowding_distance > ib.crowding_distance {
        ia
    } else {
        ib
    }
}

// ---------------------------------------------------------------------------
// NSGA-II search configuration and main loop
// ---------------------------------------------------------------------------

/// Configuration for the multi-objective NAS search.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub population_size: usize,
    pub num_generations: usize,
    pub mutation_rate: f64,
    pub volatility: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            num_generations: 50,
            mutation_rate: 0.2,
            volatility: 0.02,
        }
    }
}

/// Data point for Pareto front visualisation.
#[derive(Debug, Clone, Serialize)]
pub struct ParetoPoint {
    pub architecture: NasArchitecture,
    pub objectives: ObjectiveValues,
}

/// Result returned by the multi-objective search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub pareto_front: Vec<ParetoPoint>,
    pub hypervolume: f64,
    pub generations_run: usize,
}

/// Run the NSGA-II based multi-objective NAS.
pub fn multi_objective_search(config: &SearchConfig) -> SearchResult {
    multi_objective_search_with_seed(config, 42)
}

/// Run the NSGA-II based multi-objective NAS with a specific RNG seed.
pub fn multi_objective_search_with_seed(config: &SearchConfig, seed: u64) -> SearchResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let pop_size = config.population_size;

    // --- Initialise population ---
    let mut population: Vec<Individual> = (0..pop_size)
        .map(|_| {
            let arch = NasArchitecture::random(&mut rng);
            let obj = evaluate_objectives(&arch, config.volatility);
            Individual {
                architecture: arch,
                objectives: obj,
                rank: 0,
                crowding_distance: 0.0,
            }
        })
        .collect();

    // --- Main generational loop ---
    for _gen in 0..config.num_generations {
        // Create offspring
        let mut offspring: Vec<Individual> = Vec::with_capacity(pop_size);
        while offspring.len() < pop_size {
            let p1 = tournament_select(&population, &mut rng);
            let p2 = tournament_select(&population, &mut rng);
            let mut child_arch =
                NasArchitecture::crossover(&p1.architecture, &p2.architecture, &mut rng);
            child_arch.mutate(config.mutation_rate, &mut rng);
            let obj = evaluate_objectives(&child_arch, config.volatility);
            offspring.push(Individual {
                architecture: child_arch,
                objectives: obj,
                rank: 0,
                crowding_distance: 0.0,
            });
        }

        // Combine parent + offspring
        let mut combined = population;
        combined.extend(offspring);

        // Non-dominated sorting
        let fronts = non_dominated_sort(&combined);

        // Assign ranks
        for (rank, front) in fronts.iter().enumerate() {
            for &idx in front {
                combined[idx].rank = rank;
            }
        }

        // Build next population
        let mut next_pop: Vec<Individual> = Vec::with_capacity(pop_size);
        for front_indices in &fronts {
            if next_pop.len() + front_indices.len() <= pop_size {
                // Include entire front
                crowding_distance(&mut combined, front_indices);
                for &idx in front_indices {
                    next_pop.push(combined[idx].clone());
                }
            } else {
                // Partial front: select by crowding distance
                crowding_distance(&mut combined, front_indices);
                let mut sorted_front: Vec<usize> = front_indices.clone();
                sorted_front.sort_by(|&a, &b| {
                    combined[b]
                        .crowding_distance
                        .partial_cmp(&combined[a].crowding_distance)
                        .unwrap_or(Ordering::Equal)
                });
                let remaining = pop_size - next_pop.len();
                for &idx in sorted_front.iter().take(remaining) {
                    next_pop.push(combined[idx].clone());
                }
                break;
            }
        }

        population = next_pop;
    }

    // --- Extract final Pareto front ---
    let fronts = non_dominated_sort(&population);
    let pareto_indices = if !fronts.is_empty() { &fronts[0] } else { &vec![] };

    let pareto_front: Vec<ParetoPoint> = pareto_indices
        .iter()
        .map(|&i| ParetoPoint {
            architecture: population[i].architecture.clone(),
            objectives: population[i].objectives.clone(),
        })
        .collect();

    // Compute hypervolume (project to first two objectives for 2-D HV)
    let points_2d: Vec<Vec<f64>> = pareto_front
        .iter()
        .map(|p| vec![p.objectives[0], p.objectives[1]])
        .collect();
    let reference = vec![0.0, 5000.0]; // worst accuracy (negated 0) and 5000µs latency
    let hv = hypervolume_2d(&points_2d, &reference);

    SearchResult {
        pareto_front,
        hypervolume: hv,
        generations_run: config.num_generations,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates_basic() {
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
        assert!(dominates(&[1.0, 2.0], &[1.0, 3.0]));
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0])); // equal -> no domination
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0])); // incomparable
    }

    #[test]
    fn test_dominates_three_objectives() {
        assert!(dominates(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]));
        assert!(!dominates(&[1.0, 2.0, 1.0], &[2.0, 1.0, 2.0]));
    }

    #[test]
    fn test_non_dominated_sort_simple() {
        let individuals = vec![
            Individual {
                architecture: NasArchitecture::random(&mut StdRng::seed_from_u64(0)),
                objectives: vec![1.0, 4.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            Individual {
                architecture: NasArchitecture::random(&mut StdRng::seed_from_u64(1)),
                objectives: vec![2.0, 3.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            Individual {
                architecture: NasArchitecture::random(&mut StdRng::seed_from_u64(2)),
                objectives: vec![3.0, 5.0], // dominated by both above
                rank: 0,
                crowding_distance: 0.0,
            },
        ];

        let fronts = non_dominated_sort(&individuals);
        assert_eq!(fronts.len(), 2);
        // First front has indices 0 and 1 (non-dominated pair)
        assert_eq!(fronts[0].len(), 2);
        assert!(fronts[0].contains(&0));
        assert!(fronts[0].contains(&1));
        // Second front has index 2
        assert_eq!(fronts[1], vec![2]);
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let mut individuals = vec![
            Individual {
                architecture: NasArchitecture::random(&mut StdRng::seed_from_u64(0)),
                objectives: vec![1.0, 5.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            Individual {
                architecture: NasArchitecture::random(&mut StdRng::seed_from_u64(1)),
                objectives: vec![3.0, 3.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            Individual {
                architecture: NasArchitecture::random(&mut StdRng::seed_from_u64(2)),
                objectives: vec![5.0, 1.0],
                rank: 0,
                crowding_distance: 0.0,
            },
        ];
        let indices = vec![0, 1, 2];
        crowding_distance(&mut individuals, &indices);
        // Boundary solutions should get infinity
        assert!(individuals[0].crowding_distance.is_infinite());
        assert!(individuals[2].crowding_distance.is_infinite());
        // Middle solution should get a finite positive value
        assert!(individuals[1].crowding_distance.is_finite());
        assert!(individuals[1].crowding_distance > 0.0);
    }

    #[test]
    fn test_hypervolume_2d_basic() {
        // Single point (1,1) with reference (2,2) -> area = 1*1 = 1
        let pts = vec![vec![1.0, 1.0]];
        let reference = vec![2.0, 2.0];
        let hv = hypervolume_2d(&pts, &reference);
        assert!((hv - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_hypervolume_2d_two_points() {
        // Two points on the front
        let pts = vec![vec![1.0, 3.0], vec![2.0, 1.0]];
        let reference = vec![4.0, 4.0];
        // Area: (4-1)*(4-3) + (4-2)*(3-1) = 3 + 4 = 7
        let hv = hypervolume_2d(&pts, &reference);
        assert!((hv - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_hypervolume_empty() {
        let pts: Vec<Vec<f64>> = vec![];
        let reference = vec![2.0, 2.0];
        assert_eq!(hypervolume_2d(&pts, &reference), 0.0);
    }

    #[test]
    fn test_evaluate_objectives_returns_three() {
        let arch = NasArchitecture::random(&mut StdRng::seed_from_u64(42));
        let obj = evaluate_objectives(&arch, 0.02);
        assert_eq!(obj.len(), 3);
        assert!(obj[0] < 0.0); // negated accuracy
        assert!(obj[1] > 0.0); // latency
        assert!(obj[2] > 0.0); // model size
    }

    #[test]
    fn test_architecture_crossover_and_mutation() {
        let mut rng = StdRng::seed_from_u64(123);
        let a = NasArchitecture::random(&mut rng);
        let b = NasArchitecture::random(&mut rng);
        let mut child = NasArchitecture::crossover(&a, &b, &mut rng);
        child.mutate(1.0, &mut rng); // mutate everything
        // Just check it doesn't panic and produces valid ranges
        assert!(child.num_layers >= 1 && child.num_layers <= 8);
        assert!(child.dropout >= 0.0 && child.dropout <= 0.5);
    }

    #[test]
    fn test_search_produces_pareto_front() {
        let config = SearchConfig {
            population_size: 30,
            num_generations: 10,
            mutation_rate: 0.3,
            volatility: 0.02,
        };
        let result = multi_objective_search(&config);
        assert!(!result.pareto_front.is_empty());
        assert!(result.hypervolume > 0.0);
        assert_eq!(result.generations_run, 10);
    }

    #[test]
    fn test_pareto_front_is_non_dominated() {
        let config = SearchConfig {
            population_size: 40,
            num_generations: 15,
            mutation_rate: 0.2,
            volatility: 0.02,
        };
        let result = multi_objective_search(&config);
        // Every pair on the front should be mutually non-dominating
        for i in 0..result.pareto_front.len() {
            for j in 0..result.pareto_front.len() {
                if i != j {
                    assert!(
                        !dominates(
                            &result.pareto_front[i].objectives,
                            &result.pareto_front[j].objectives
                        ),
                        "Point {} dominates point {} on the Pareto front",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_returns() {
        let candles = vec![
            Candle { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 100.0, volume: 1.0 },
            Candle { timestamp: 1, open: 100.0, high: 110.0, low: 99.0, close: 105.0, volume: 1.0 },
            Candle { timestamp: 2, open: 105.0, high: 107.0, low: 100.0, close: 103.0, volume: 1.0 },
        ];
        let returns = compute_returns(&candles);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-9);
    }

    #[test]
    fn test_compute_volatility() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let vol = compute_volatility(&returns, 3);
        assert_eq!(vol.len(), 3);
        assert!(vol.iter().all(|v| *v > 0.0));
    }
}
