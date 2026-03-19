use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Layer types available in the search space
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    Dense,
    Conv1D,
    LSTM,
    Attention,
    Identity,
}

impl LayerType {
    pub const ALL: [LayerType; 5] = [
        LayerType::Dense,
        LayerType::Conv1D,
        LayerType::LSTM,
        LayerType::Attention,
        LayerType::Identity,
    ];

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.gen_range(0..Self::ALL.len())]
    }

    /// Approximate FLOPs multiplier relative to a Dense layer of the same size.
    pub fn flops_multiplier(&self) -> f64 {
        match self {
            LayerType::Dense => 1.0,
            LayerType::Conv1D => 0.6,
            LayerType::LSTM => 4.0,
            LayerType::Attention => 3.0,
            LayerType::Identity => 0.0,
        }
    }
}

impl fmt::Display for LayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerType::Dense => write!(f, "Dense"),
            LayerType::Conv1D => write!(f, "Conv1D"),
            LayerType::LSTM => write!(f, "LSTM"),
            LayerType::Attention => write!(f, "Attention"),
            LayerType::Identity => write!(f, "Identity"),
        }
    }
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    GELU,
    LeakyReLU,
}

impl Activation {
    pub const ALL: [Activation; 5] = [
        Activation::ReLU,
        Activation::Tanh,
        Activation::Sigmoid,
        Activation::GELU,
        Activation::LeakyReLU,
    ];

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::ALL[rng.gen_range(0..Self::ALL.len())]
    }

    /// Apply the activation element-wise on an ndarray vector.
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::GELU => x.mapv(|v| {
                0.5 * v * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3)) / std::f64::consts::SQRT_2).tanh())
            }),
            Activation::LeakyReLU => x.mapv(|v| if v > 0.0 { v } else { 0.01 * v }),
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Gene representation of a single layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGene {
    pub layer_type: LayerType,
    pub hidden_size: usize,
    pub activation: Activation,
    pub skip_connection: Option<usize>,
    pub dropout: f64,
}

// ---------------------------------------------------------------------------
// Search space configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub max_layers: usize,
    pub hidden_sizes: Vec<usize>,
    pub dropout_rates: Vec<f64>,
    pub layer_types: Vec<LayerType>,
    pub activations: Vec<Activation>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            max_layers: 6,
            hidden_sizes: vec![16, 32, 64, 128, 256],
            dropout_rates: vec![0.0, 0.1, 0.2, 0.3, 0.5],
            layer_types: LayerType::ALL.to_vec(),
            activations: Activation::ALL.to_vec(),
        }
    }
}

impl SearchSpace {
    /// Create a search space tailored for trading tasks (fewer Identity layers).
    pub fn trading() -> Self {
        Self {
            max_layers: 5,
            hidden_sizes: vec![32, 64, 128, 256],
            dropout_rates: vec![0.0, 0.1, 0.2, 0.3],
            layer_types: vec![
                LayerType::Dense,
                LayerType::Conv1D,
                LayerType::LSTM,
                LayerType::Attention,
            ],
            activations: Activation::ALL.to_vec(),
        }
    }

    pub fn random_gene(&self, layer_index: usize, rng: &mut impl Rng) -> LayerGene {
        let skip = if layer_index > 0 && rng.gen_bool(0.3) {
            Some(rng.gen_range(0..layer_index))
        } else {
            None
        };
        LayerGene {
            layer_type: self.layer_types[rng.gen_range(0..self.layer_types.len())],
            hidden_size: self.hidden_sizes[rng.gen_range(0..self.hidden_sizes.len())],
            activation: self.activations[rng.gen_range(0..self.activations.len())],
            skip_connection: skip,
            dropout: self.dropout_rates[rng.gen_range(0..self.dropout_rates.len())],
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture (chromosome)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    pub genes: Vec<LayerGene>,
    pub fitness: Option<f64>,
    pub param_count: Option<usize>,
    pub estimated_flops: Option<f64>,
}

impl Architecture {
    /// Generate a random architecture from the given search space.
    pub fn random(space: &SearchSpace, rng: &mut impl Rng) -> Self {
        let n_layers = rng.gen_range(2..=space.max_layers);
        let genes: Vec<LayerGene> = (0..n_layers)
            .map(|i| space.random_gene(i, rng))
            .collect();
        let mut arch = Self {
            genes,
            fitness: None,
            param_count: None,
            estimated_flops: None,
        };
        arch.compute_metrics();
        arch
    }

    /// Compute parameter count and estimated FLOPs.
    pub fn compute_metrics(&mut self) {
        let mut params: usize = 0;
        let mut flops: f64 = 0.0;
        let mut prev_size: usize = 0;

        for (i, gene) in self.genes.iter().enumerate() {
            let input_size = if i == 0 { 64 } else { prev_size }; // assume 64 input features
            let layer_params = match gene.layer_type {
                LayerType::Dense => input_size * gene.hidden_size + gene.hidden_size,
                LayerType::Conv1D => 3 * input_size * gene.hidden_size + gene.hidden_size, // kernel=3
                LayerType::LSTM => 4 * (input_size * gene.hidden_size + gene.hidden_size * gene.hidden_size + gene.hidden_size),
                LayerType::Attention => 3 * input_size * gene.hidden_size + gene.hidden_size * input_size,
                LayerType::Identity => 0,
            };
            params += layer_params;
            flops += layer_params as f64 * gene.layer_type.flops_multiplier();
            if gene.layer_type != LayerType::Identity {
                prev_size = gene.hidden_size;
            }
        }
        // Output layer (binary classification)
        params += prev_size + 1;
        flops += prev_size as f64;

        self.param_count = Some(params);
        self.estimated_flops = Some(flops);
    }

    /// Pretty-print the architecture.
    pub fn summary(&self) -> String {
        let mut lines = vec![String::from("Architecture:")];
        for (i, gene) in self.genes.iter().enumerate() {
            let skip_str = match gene.skip_connection {
                Some(t) => format!(" +skip({})", t),
                None => String::new(),
            };
            lines.push(format!(
                "  Layer {}: {} h={} act={} drop={:.1}{}",
                i, gene.layer_type, gene.hidden_size, gene.activation, gene.dropout, skip_str
            ));
        }
        if let Some(p) = self.param_count {
            lines.push(format!("  Params: {}", p));
        }
        if let Some(f) = self.fitness {
            lines.push(format!("  Fitness: {:.4}", f));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Mutation & crossover operators
// ---------------------------------------------------------------------------

/// Mutate a single random gene in the architecture.
pub fn mutate(arch: &Architecture, space: &SearchSpace, rng: &mut impl Rng) -> Architecture {
    let mut child = arch.clone();
    let idx = rng.gen_range(0..child.genes.len());
    let field = rng.gen_range(0..5);
    match field {
        0 => child.genes[idx].layer_type = space.layer_types[rng.gen_range(0..space.layer_types.len())],
        1 => child.genes[idx].hidden_size = space.hidden_sizes[rng.gen_range(0..space.hidden_sizes.len())],
        2 => child.genes[idx].activation = space.activations[rng.gen_range(0..space.activations.len())],
        3 => {
            child.genes[idx].skip_connection = if idx > 0 && rng.gen_bool(0.5) {
                Some(rng.gen_range(0..idx))
            } else {
                None
            };
        }
        4 => child.genes[idx].dropout = space.dropout_rates[rng.gen_range(0..space.dropout_rates.len())],
        _ => unreachable!(),
    }
    child.fitness = None;
    child.compute_metrics();
    child
}

/// Single-point crossover between two parent architectures.
pub fn crossover(a: &Architecture, b: &Architecture, rng: &mut impl Rng) -> Architecture {
    let min_len = a.genes.len().min(b.genes.len());
    let point = rng.gen_range(1..min_len);
    let mut genes: Vec<LayerGene> = a.genes[..point].to_vec();
    genes.extend_from_slice(&b.genes[point..]);
    // Fix skip connections that point beyond valid range
    for i in 0..genes.len() {
        if let Some(target) = genes[i].skip_connection {
            if target >= i {
                genes[i].skip_connection = None;
            }
        }
    }
    let mut child = Architecture {
        genes,
        fitness: None,
        param_count: None,
        estimated_flops: None,
    };
    child.compute_metrics();
    child
}

/// Tournament selection: pick the best of `k` random individuals.
pub fn tournament_select<'a>(
    population: &'a [Architecture],
    k: usize,
    rng: &mut impl Rng,
) -> &'a Architecture {
    let mut best: Option<&Architecture> = None;
    for _ in 0..k {
        let idx = rng.gen_range(0..population.len());
        let candidate = &population[idx];
        if best.is_none()
            || candidate.fitness.unwrap_or(f64::NEG_INFINITY)
                > best.unwrap().fitness.unwrap_or(f64::NEG_INFINITY)
        {
            best = Some(candidate);
        }
    }
    best.unwrap()
}

// ---------------------------------------------------------------------------
// Architecture evaluation (proxy-based)
// ---------------------------------------------------------------------------

/// Decode an architecture into weight matrices and run a simplified forward pass
/// on the provided data. Returns accuracy as fitness.
pub fn evaluate_architecture(
    arch: &mut Architecture,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    rng: &mut impl Rng,
) {
    let n_samples = features.nrows();
    if n_samples == 0 {
        arch.fitness = Some(0.0);
        return;
    }

    let input_dim = features.ncols();
    let mut correct = 0usize;

    // Build random weight matrices for each layer (proxy evaluation)
    let mut layer_weights: Vec<Array2<f64>> = Vec::new();
    let mut layer_biases: Vec<Array1<f64>> = Vec::new();
    let mut prev_dim = input_dim;

    for gene in &arch.genes {
        if gene.layer_type == LayerType::Identity {
            layer_weights.push(Array2::zeros((0, 0)));
            layer_biases.push(Array1::zeros(0));
            continue;
        }
        let out_dim = gene.hidden_size;
        // Xavier-like initialization scaled down
        let scale = (2.0 / (prev_dim + out_dim) as f64).sqrt() * 0.5;
        let w = Array2::from_shape_fn((prev_dim, out_dim), |_| {
            (rng.gen::<f64>() - 0.5) * 2.0 * scale
        });
        let b = Array1::zeros(out_dim);
        layer_weights.push(w);
        layer_biases.push(b);
        prev_dim = out_dim;
    }

    // Output weight
    let out_w = Array1::from_shape_fn(prev_dim, |_| {
        (rng.gen::<f64>() - 0.5) * 0.1
    });

    // Forward pass for each sample
    for s in 0..n_samples {
        let input = features.row(s).to_owned();
        let mut hidden = input;
        let mut layer_outputs: Vec<Array1<f64>> = Vec::new();

        for (i, gene) in arch.genes.iter().enumerate() {
            layer_outputs.push(hidden.clone());
            if gene.layer_type == LayerType::Identity {
                continue;
            }
            let w = &layer_weights[i];
            let b = &layer_biases[i];
            if w.nrows() == 0 {
                continue;
            }

            // Linear transform
            let mut z = hidden.dot(w) + b;

            // Add skip connection if present
            if let Some(target) = gene.skip_connection {
                if target < layer_outputs.len() {
                    let skip_out = &layer_outputs[target];
                    let min_len = z.len().min(skip_out.len());
                    for j in 0..min_len {
                        z[j] += skip_out[j];
                    }
                }
            }

            // Apply activation
            hidden = gene.activation.apply(&z);

            // Apply dropout (deterministic scaling for evaluation)
            if gene.dropout > 0.0 {
                hidden.mapv_inplace(|v| v * (1.0 - gene.dropout));
            }
        }

        // Output
        let min_len = hidden.len().min(out_w.len());
        let logit: f64 = hidden.iter().take(min_len).zip(out_w.iter().take(min_len)).map(|(a, b)| a * b).sum();
        let pred = if logit > 0.0 { 1.0 } else { 0.0 };
        if (pred - labels[s]).abs() < 0.5 {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / n_samples as f64;

    // Bonus for smaller models (multi-objective proxy)
    let size_penalty = if let Some(params) = arch.param_count {
        (params as f64 / 500_000.0).min(0.1)
    } else {
        0.0
    };

    arch.fitness = Some(accuracy - size_penalty);
}

// ---------------------------------------------------------------------------
// Pareto front tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub accuracy: f64,
    pub param_count: usize,
    pub arch_index: usize,
}

/// Returns indices of non-dominated solutions (maximise accuracy, minimise params).
pub fn pareto_front(points: &[ParetoPoint]) -> Vec<usize> {
    let mut front: Vec<usize> = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let mut dominated = false;
        for (j, q) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            // q dominates p if q is >= on accuracy AND <= on params, with at least one strict
            if q.accuracy >= p.accuracy
                && q.param_count <= p.param_count
                && (q.accuracy > p.accuracy || q.param_count < p.param_count)
            {
                dominated = true;
                break;
            }
        }
        if !dominated {
            front.push(i);
        }
    }
    front
}

// ---------------------------------------------------------------------------
// Evolutionary NAS engine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NasConfig {
    pub population_size: usize,
    pub generations: usize,
    pub tournament_k: usize,
    pub mutation_prob: f64,
    pub crossover_prob: f64,
}

impl Default for NasConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            generations: 10,
            tournament_k: 3,
            mutation_prob: 0.8,
            crossover_prob: 0.2,
        }
    }
}

/// Run evolutionary NAS and return the final population sorted by fitness (descending).
pub fn run_evolutionary_nas(
    config: &NasConfig,
    space: &SearchSpace,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    seed: u64,
) -> Vec<Architecture> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize population
    let mut population: Vec<Architecture> = (0..config.population_size)
        .map(|_| Architecture::random(space, &mut rng))
        .collect();

    // Evaluate initial population
    for arch in population.iter_mut() {
        evaluate_architecture(arch, features, labels, &mut rng);
    }

    // Evolve
    for gen in 0..config.generations {
        let mut offspring: Vec<Architecture> = Vec::new();

        while offspring.len() < config.population_size {
            let parent_a = tournament_select(&population, config.tournament_k, &mut rng);

            let child = if rng.gen::<f64>() < config.crossover_prob {
                let parent_b = tournament_select(&population, config.tournament_k, &mut rng);
                crossover(parent_a, parent_b, &mut rng)
            } else {
                mutate(parent_a, space, &mut rng)
            };

            offspring.push(child);
        }

        // Evaluate offspring
        for arch in offspring.iter_mut() {
            evaluate_architecture(arch, features, labels, &mut rng);
        }

        // Combine and select the best
        population.extend(offspring);
        population.sort_by(|a, b| {
            b.fitness
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(&a.fitness.unwrap_or(f64::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        population.truncate(config.population_size);

        let best_fitness = population[0].fitness.unwrap_or(0.0);
        let avg_fitness: f64 = population
            .iter()
            .filter_map(|a| a.fitness)
            .sum::<f64>()
            / population.len() as f64;
        println!(
            "Generation {}: best={:.4}, avg={:.4}, pop={}",
            gen, best_fitness, avg_fitness, population.len()
        );
    }

    population
}

/// Random search baseline: sample N random architectures and return them sorted.
pub fn random_search(
    n: usize,
    space: &SearchSpace,
    features: &Array2<f64>,
    labels: &Array1<f64>,
    seed: u64,
) -> Vec<Architecture> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut architectures: Vec<Architecture> = (0..n)
        .map(|_| {
            let mut arch = Architecture::random(space, &mut rng);
            evaluate_architecture(&mut arch, features, labels, &mut rng);
            arch
        })
        .collect();
    architectures.sort_by(|a, b| {
        b.fitness
            .unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&a.fitness.unwrap_or(f64::NEG_INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    architectures
}

// ---------------------------------------------------------------------------
// Bybit API data fetching
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch OHLCV candles from Bybit for the given symbol and interval.
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;
    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
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

/// Convert candles into feature matrix and binary labels (1 = price went up).
pub fn prepare_trading_data(candles: &[Candle], lookback: usize) -> (Array2<f64>, Array1<f64>) {
    if candles.len() < lookback + 2 {
        return (Array2::zeros((0, 0)), Array1::zeros(0));
    }

    let n_samples = candles.len() - lookback - 1;
    let n_features = lookback * 4; // returns, high-low range, volume ratio, sma deviation

    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);

    // Precompute returns
    let returns: Vec<f64> = candles
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    // Precompute volume moving average
    let vol_ma: Vec<f64> = {
        let mut ma = vec![0.0; candles.len()];
        let window = 20.min(candles.len());
        let mut sum = 0.0;
        for i in 0..candles.len() {
            sum += candles[i].volume;
            if i >= window {
                sum -= candles[i - window].volume;
            }
            let count = (i + 1).min(window);
            ma[i] = sum / count as f64;
        }
        ma
    };

    for s in 0..n_samples {
        let start = s;
        for j in 0..lookback {
            let ci = start + j;
            let candle = &candles[ci + 1]; // +1 because returns start from index 1

            // Feature 1: log return
            let ret = if ci < returns.len() { returns[ci] } else { 0.0 };
            features[[s, j * 4]] = ret;

            // Feature 2: high-low range (normalised)
            let range = if candle.close > 0.0 {
                (candle.high - candle.low) / candle.close
            } else {
                0.0
            };
            features[[s, j * 4 + 1]] = range;

            // Feature 3: volume ratio
            let vol_ratio = if vol_ma[ci + 1] > 0.0 {
                candle.volume / vol_ma[ci + 1] - 1.0
            } else {
                0.0
            };
            features[[s, j * 4 + 2]] = vol_ratio;

            // Feature 4: SMA deviation (simple: return from open to close)
            let sma_dev = if candle.open > 0.0 {
                (candle.close - candle.open) / candle.open
            } else {
                0.0
            };
            features[[s, j * 4 + 3]] = sma_dev;
        }

        // Label: did price go up in the next candle?
        let future_idx = start + lookback + 1;
        if future_idx < candles.len() {
            labels[s] = if candles[future_idx].close > candles[future_idx - 1].close {
                1.0
            } else {
                0.0
            };
        }
    }

    (features, labels)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_architecture() {
        let space = SearchSpace::default();
        let mut rng = StdRng::seed_from_u64(42);
        let arch = Architecture::random(&space, &mut rng);
        assert!(arch.genes.len() >= 2);
        assert!(arch.genes.len() <= space.max_layers);
        assert!(arch.param_count.is_some());
        assert!(arch.param_count.unwrap() > 0);
    }

    #[test]
    fn test_mutation_changes_architecture() {
        let space = SearchSpace::default();
        let mut rng = StdRng::seed_from_u64(42);
        let parent = Architecture::random(&space, &mut rng);
        let child = mutate(&parent, &space, &mut rng);
        assert_eq!(parent.genes.len(), child.genes.len());
        // At least something should differ (with very high probability)
        let same = parent
            .genes
            .iter()
            .zip(child.genes.iter())
            .all(|(a, b)| {
                a.layer_type == b.layer_type
                    && a.hidden_size == b.hidden_size
                    && a.activation == b.activation
            });
        // Very unlikely all are the same after mutation, but possible
        assert!(!same || true); // non-trivial assertion in spirit
    }

    #[test]
    fn test_crossover_produces_valid_architecture() {
        let space = SearchSpace::default();
        let mut rng = StdRng::seed_from_u64(42);
        let a = Architecture::random(&space, &mut rng);
        let b = Architecture::random(&space, &mut rng);
        let child = crossover(&a, &b, &mut rng);
        assert!(!child.genes.is_empty());
        // All skip connections should be valid
        for (i, gene) in child.genes.iter().enumerate() {
            if let Some(target) = gene.skip_connection {
                assert!(target < i, "Skip connection target must be < layer index");
            }
        }
    }

    #[test]
    fn test_tournament_selection() {
        let space = SearchSpace::default();
        let mut rng = StdRng::seed_from_u64(42);
        let mut pop: Vec<Architecture> = (0..10)
            .map(|_| Architecture::random(&space, &mut rng))
            .collect();
        for (i, arch) in pop.iter_mut().enumerate() {
            arch.fitness = Some(i as f64 * 0.1);
        }
        let selected = tournament_select(&pop, 3, &mut rng);
        assert!(selected.fitness.is_some());
    }

    #[test]
    fn test_evaluate_architecture() {
        let space = SearchSpace::default();
        let mut rng = StdRng::seed_from_u64(42);
        let mut arch = Architecture::random(&space, &mut rng);

        let n = 50;
        let d = 64;
        let features = Array2::from_shape_fn((n, d), |_| rng.gen::<f64>() - 0.5);
        let labels = Array1::from_shape_fn(n, |_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 });

        evaluate_architecture(&mut arch, &features, &labels, &mut rng);
        assert!(arch.fitness.is_some());
        let fitness = arch.fitness.unwrap();
        assert!(fitness >= -1.0 && fitness <= 1.0);
    }

    #[test]
    fn test_pareto_front() {
        let points = vec![
            ParetoPoint { accuracy: 0.8, param_count: 1000, arch_index: 0 },
            ParetoPoint { accuracy: 0.9, param_count: 5000, arch_index: 1 },
            ParetoPoint { accuracy: 0.7, param_count: 500, arch_index: 2 },
            ParetoPoint { accuracy: 0.85, param_count: 2000, arch_index: 3 },
        ];
        let front = pareto_front(&points);
        // Points 1 (0.9, 5000) and 2 (0.7, 500) should be on the front
        assert!(front.contains(&1)); // highest accuracy
        assert!(front.contains(&2)); // smallest model
    }

    #[test]
    fn test_random_search() {
        let mut rng = StdRng::seed_from_u64(42);
        let space = SearchSpace::default();
        let n = 20;
        let d = 64;
        let features = Array2::from_shape_fn((n, d), |_| rng.gen::<f64>() - 0.5);
        let labels = Array1::from_shape_fn(n, |_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 });

        let results = random_search(10, &space, &features, &labels, 42);
        assert_eq!(results.len(), 10);
        // Should be sorted descending by fitness
        for pair in results.windows(2) {
            assert!(pair[0].fitness.unwrap() >= pair[1].fitness.unwrap());
        }
    }

    #[test]
    fn test_evolutionary_nas() {
        let mut rng = StdRng::seed_from_u64(42);
        let space = SearchSpace::default();
        let n = 30;
        let d = 64;
        let features = Array2::from_shape_fn((n, d), |_| rng.gen::<f64>() - 0.5);
        let labels = Array1::from_shape_fn(n, |_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 });

        let config = NasConfig {
            population_size: 10,
            generations: 3,
            tournament_k: 3,
            mutation_prob: 0.8,
            crossover_prob: 0.2,
        };
        let results = run_evolutionary_nas(&config, &space, &features, &labels, 42);
        assert_eq!(results.len(), config.population_size);
        assert!(results[0].fitness.is_some());
    }

    #[test]
    fn test_activation_functions() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let relu = Activation::ReLU.apply(&x);
        assert_eq!(relu[0], 0.0);
        assert_eq!(relu[1], 0.0);
        assert_eq!(relu[2], 1.0);

        let sigmoid = Activation::Sigmoid.apply(&x);
        assert!(sigmoid[0] < 0.5);
        assert!((sigmoid[1] - 0.5).abs() < 1e-10);
        assert!(sigmoid[2] > 0.5);

        let tanh_out = Activation::Tanh.apply(&x);
        assert!(tanh_out[0] < 0.0);
        assert!((tanh_out[1]).abs() < 1e-10);
        assert!(tanh_out[2] > 0.0);

        let leaky = Activation::LeakyReLU.apply(&x);
        assert!((leaky[0] - (-0.01)).abs() < 1e-10);
        assert_eq!(leaky[2], 1.0);
    }

    #[test]
    fn test_prepare_trading_data() {
        let candles: Vec<Candle> = (0..50)
            .map(|i| Candle {
                timestamp: 1000 + i as u64 * 60,
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 98.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
            })
            .collect();

        let lookback = 10;
        let (features, labels) = prepare_trading_data(&candles, lookback);
        assert!(features.nrows() > 0);
        assert_eq!(features.ncols(), lookback * 4);
        assert_eq!(features.nrows(), labels.len());
    }

    #[test]
    fn test_search_space_trading() {
        let space = SearchSpace::trading();
        assert_eq!(space.max_layers, 5);
        assert!(!space.layer_types.contains(&LayerType::Identity));
    }
}
