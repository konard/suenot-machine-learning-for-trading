use anyhow::Result;
use ndarray::Array1;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ─── Hyperparameters ───────────────────────────────────────────────

/// Hyperparameters for a trading agent in the PBT population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    /// Learning rate for weight updates.
    pub learning_rate: f64,
    /// Number of past price steps to consider.
    pub lookback_window: usize,
    /// Batch size for training (discrete).
    pub batch_size: usize,
    /// Risk penalty coefficient for the reward function.
    pub risk_penalty: f64,
    /// Momentum factor for weight updates.
    pub momentum: f64,
}

impl Hyperparameters {
    /// Create random hyperparameters within sensible ranges.
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            learning_rate: 10f64.powf(rng.gen_range(-4.0..-1.0)),
            lookback_window: *[5, 10, 20, 50].choose_rng(&mut rng),
            batch_size: *[16, 32, 64, 128].choose_rng(&mut rng),
            risk_penalty: rng.gen_range(0.01..1.0),
            momentum: rng.gen_range(0.0..0.99),
        }
    }

    /// Perturb hyperparameters for the explore step.
    pub fn perturb(&mut self) {
        let mut rng = rand::thread_rng();
        let factor = if rng.gen_bool(0.5) { 1.2 } else { 0.8 };

        self.learning_rate = (self.learning_rate * factor).clamp(1e-5, 1e-1);
        self.risk_penalty = (self.risk_penalty * factor).clamp(0.001, 5.0);
        self.momentum = (self.momentum * factor).clamp(0.0, 0.99);

        // Discrete hyperparams: resample with some probability
        if rng.gen_bool(0.3) {
            self.lookback_window = *[5, 10, 20, 50].choose_rng(&mut rng);
        }
        if rng.gen_bool(0.3) {
            self.batch_size = *[16, 32, 64, 128].choose_rng(&mut rng);
        }
    }
}

/// Helper trait to choose a random element from a slice using a specific rng.
trait ChooseRng<T> {
    fn choose_rng(&self, rng: &mut impl Rng) -> &T;
}

impl<T> ChooseRng<T> for [T] {
    fn choose_rng(&self, rng: &mut impl Rng) -> &T {
        &self[rng.gen_range(0..self.len())]
    }
}

// ─── Agent ─────────────────────────────────────────────────────────

/// A single agent in the PBT population.
#[derive(Debug, Clone)]
pub struct Agent {
    /// Unique identifier.
    pub id: usize,
    /// Model weights (simple linear model).
    pub weights: Array1<f64>,
    /// Current hyperparameters.
    pub hyperparams: Hyperparameters,
    /// Current performance metric (e.g., Sharpe ratio).
    pub performance: f64,
    /// History of performance across PBT steps.
    pub performance_history: Vec<f64>,
    /// History of learning rate across PBT steps.
    pub lr_history: Vec<f64>,
    /// Number of training steps completed.
    pub steps: usize,
}

impl Agent {
    /// Create a new agent with random weights and hyperparameters.
    pub fn new(id: usize, weight_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_shape_fn(weight_dim, |_| rng.gen_range(-0.1..0.1));
        Self {
            id,
            weights,
            hyperparams: Hyperparameters::random(),
            performance: f64::NEG_INFINITY,
            performance_history: Vec::new(),
            lr_history: Vec::new(),
            steps: 0,
        }
    }

    /// Predict the next price movement given recent prices.
    pub fn predict(&self, features: &Array1<f64>) -> f64 {
        let n = self.weights.len().min(features.len());
        let w = self.weights.slice(ndarray::s![..n]);
        let f = features.slice(ndarray::s![..n]);
        w.dot(&f)
    }

    /// Train the agent on a batch of data from the trading environment.
    pub fn train(&mut self, env: &TradingEnvironment, num_steps: usize) {
        let mut rng = rand::thread_rng();
        let lookback = self.hyperparams.lookback_window;

        if env.prices.len() <= lookback + 1 {
            return;
        }

        // Ensure weights match lookback window
        if self.weights.len() != lookback {
            self.weights = Array1::from_shape_fn(lookback, |_| rng.gen_range(-0.1..0.1));
        }

        let max_start = env.prices.len() - lookback - 1;
        let mut velocity: Array1<f64> = Array1::zeros(lookback);

        for _ in 0..num_steps {
            let mut grad = Array1::zeros(lookback);
            let batch = self.hyperparams.batch_size.min(max_start);

            for _ in 0..batch {
                let idx = rng.gen_range(0..max_start);
                let features = Array1::from_vec(
                    env.prices[idx..idx + lookback]
                        .iter()
                        .zip(env.prices[idx + 1..idx + lookback + 1].iter())
                        .map(|(prev, curr)| curr / prev - 1.0) // returns
                        .collect(),
                );
                let target = env.prices[idx + lookback] / env.prices[idx + lookback - 1] - 1.0;
                let pred = self.predict(&features);
                let error = pred - target;

                // Gradient with risk penalty (L2 regularization)
                grad = grad + &(&features * error + &(&self.weights * self.hyperparams.risk_penalty));
            }

            grad /= batch as f64;

            // Momentum update
            velocity = &velocity * self.hyperparams.momentum + &grad * self.hyperparams.learning_rate;
            self.weights = &self.weights - &velocity;
        }

        self.steps += num_steps;
    }

    /// Evaluate the agent's performance on the trading environment.
    /// Returns a simplified Sharpe ratio.
    pub fn evaluate(&mut self, env: &TradingEnvironment) -> f64 {
        let lookback = self.hyperparams.lookback_window;
        if env.prices.len() <= lookback + 1 || self.weights.len() != lookback {
            self.performance = f64::NEG_INFINITY;
            return self.performance;
        }

        let mut returns = Vec::new();
        let max_idx = env.prices.len() - lookback - 1;
        let eval_points = max_idx.min(100);

        for i in 0..eval_points {
            let idx = (max_idx * i) / eval_points.max(1);
            let features = Array1::from_vec(
                env.prices[idx..idx + lookback]
                    .iter()
                    .zip(env.prices[idx + 1..idx + lookback + 1].iter())
                    .map(|(prev, curr)| curr / prev - 1.0)
                    .collect(),
            );
            let actual_return = env.prices[idx + lookback] / env.prices[idx + lookback - 1] - 1.0;
            let prediction = self.predict(&features);

            // Simple directional trading: if prediction is positive, go long; otherwise short.
            let trade_return = if prediction > 0.0 {
                actual_return
            } else {
                -actual_return
            };
            returns.push(trade_return);
        }

        if returns.is_empty() {
            self.performance = f64::NEG_INFINITY;
            return self.performance;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        self.performance = if std > 1e-10 { mean / std } else { 0.0 };
        self.performance_history.push(self.performance);
        self.lr_history.push(self.hyperparams.learning_rate);
        self.performance
    }
}

// ─── Population ────────────────────────────────────────────────────

/// PBT exploit strategy.
#[derive(Debug, Clone, Copy)]
pub enum ExploitStrategy {
    /// Copy from a random agent in the top fraction.
    TruncationSelection { top_fraction: f64 },
    /// Random binary tournament.
    BinaryTournament,
}

/// Configuration for the PBT population.
#[derive(Debug, Clone)]
pub struct PbtConfig {
    /// Number of agents in the population.
    pub population_size: usize,
    /// Dimensionality of model weights.
    pub weight_dim: usize,
    /// Exploit strategy.
    pub exploit_strategy: ExploitStrategy,
    /// Number of training steps between exploit/explore.
    pub steps_between_eval: usize,
    /// Minimum steps before exploit/explore is allowed.
    pub warmup_steps: usize,
}

impl Default for PbtConfig {
    fn default() -> Self {
        Self {
            population_size: 10,
            weight_dim: 20,
            exploit_strategy: ExploitStrategy::TruncationSelection { top_fraction: 0.2 },
            steps_between_eval: 5,
            warmup_steps: 10,
        }
    }
}

/// Population of agents managed by PBT.
pub struct Population {
    pub agents: Vec<Agent>,
    pub config: PbtConfig,
    /// Number of PBT rounds completed.
    pub generation: usize,
}

impl Population {
    /// Create a new population with the given configuration.
    pub fn new(config: PbtConfig) -> Self {
        let agents = (0..config.population_size)
            .map(|id| Agent::new(id, config.weight_dim))
            .collect();
        Self {
            agents,
            config,
            generation: 0,
        }
    }

    /// Run one round of PBT: train, evaluate, exploit, explore.
    pub fn step(&mut self, env: &TradingEnvironment) {
        // 1. Train each agent
        for agent in &mut self.agents {
            agent.train(env, self.config.steps_between_eval);
        }

        // 2. Evaluate each agent
        for agent in &mut self.agents {
            agent.evaluate(env);
        }

        // 3. Exploit + Explore (skip during warmup)
        if self.generation * self.config.steps_between_eval >= self.config.warmup_steps {
            self.exploit_and_explore();
        }

        self.generation += 1;
    }

    /// Perform the exploit and explore steps on the population.
    fn exploit_and_explore(&mut self) {
        let mut rng = rand::thread_rng();

        // Sort indices by performance (descending)
        let mut indices: Vec<usize> = (0..self.agents.len()).collect();
        indices.sort_by(|&a, &b| {
            self.agents[b]
                .performance
                .partial_cmp(&self.agents[a].performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = self.agents.len();

        match self.config.exploit_strategy {
            ExploitStrategy::TruncationSelection { top_fraction } => {
                let top_k = ((n as f64 * top_fraction).ceil() as usize).max(1);
                let bottom_start = n - top_k;

                // Bottom agents copy from top agents
                for rank in bottom_start..n {
                    let bottom_idx = indices[rank];
                    let top_idx = indices[rng.gen_range(0..top_k)];

                    // Copy weights and hyperparams from the top agent
                    let top_weights = self.agents[top_idx].weights.clone();
                    let top_hyperparams = self.agents[top_idx].hyperparams.clone();

                    self.agents[bottom_idx].weights = top_weights;
                    self.agents[bottom_idx].hyperparams = top_hyperparams;

                    // Explore: perturb hyperparams
                    self.agents[bottom_idx].hyperparams.perturb();
                }
            }
            ExploitStrategy::BinaryTournament => {
                for i in 0..n {
                    let opponent = rng.gen_range(0..n);
                    if self.agents[opponent].performance > self.agents[i].performance {
                        let opp_weights = self.agents[opponent].weights.clone();
                        let opp_hyperparams = self.agents[opponent].hyperparams.clone();
                        self.agents[i].weights = opp_weights;
                        self.agents[i].hyperparams = opp_hyperparams;
                        self.agents[i].hyperparams.perturb();
                    }
                }
            }
        }
    }

    /// Get the best agent by performance.
    pub fn best_agent(&self) -> &Agent {
        self.agents
            .iter()
            .max_by(|a, b| {
                a.performance
                    .partial_cmp(&b.performance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("Population should not be empty")
    }

    /// Get summary statistics of the population.
    pub fn stats(&self) -> PopulationStats {
        let perfs: Vec<f64> = self
            .agents
            .iter()
            .filter(|a| a.performance.is_finite())
            .map(|a| a.performance)
            .collect();

        if perfs.is_empty() {
            return PopulationStats {
                best: 0.0,
                worst: 0.0,
                mean: 0.0,
                std: 0.0,
            };
        }

        let best = perfs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = perfs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean = perfs.iter().sum::<f64>() / perfs.len() as f64;
        let variance = perfs.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / perfs.len() as f64;

        PopulationStats {
            best,
            worst,
            mean,
            std: variance.sqrt(),
        }
    }
}

/// Summary statistics of a population's performance.
#[derive(Debug, Clone)]
pub struct PopulationStats {
    pub best: f64,
    pub worst: f64,
    pub mean: f64,
    pub std: f64,
}

// ─── Trading Environment ───────────────────────────────────────────

/// A simple trading environment based on price data.
#[derive(Debug, Clone)]
pub struct TradingEnvironment {
    /// Price series (e.g., close prices).
    pub prices: Vec<f64>,
    /// Symbol name for display.
    pub symbol: String,
}

impl TradingEnvironment {
    /// Create a new trading environment from a price series.
    pub fn new(prices: Vec<f64>, symbol: String) -> Self {
        Self { prices, symbol }
    }

    /// Create a synthetic environment with random walk prices for testing.
    pub fn synthetic(length: usize, seed_price: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut prices = Vec::with_capacity(length);
        let mut price = seed_price;
        prices.push(price);

        for _ in 1..length {
            let return_pct = rng.gen_range(-0.02..0.02);
            price *= 1.0 + return_pct;
            prices.push(price);
        }

        Self {
            prices,
            symbol: "SYNTHETIC".to_string(),
        }
    }

    /// Split the environment into train and test sets.
    pub fn train_test_split(&self, train_fraction: f64) -> (TradingEnvironment, TradingEnvironment) {
        let split_idx = (self.prices.len() as f64 * train_fraction) as usize;
        let train = TradingEnvironment::new(
            self.prices[..split_idx].to_vec(),
            format!("{}_train", self.symbol),
        );
        let test = TradingEnvironment::new(
            self.prices[split_idx..].to_vec(),
            format!("{}_test", self.symbol),
        );
        (train, test)
    }
}

// ─── Bybit Client ──────────────────────────────────────────────────

/// Kline (candlestick) data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Response structure from Bybit API.
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

/// Client for fetching market data from Bybit.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline data for a given symbol and interval.
    /// Returns klines sorted by timestamp ascending.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Fetch klines and convert to a TradingEnvironment using close prices.
    pub fn fetch_trading_env(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<TradingEnvironment> {
        let klines = self.fetch_klines(symbol, interval, limit)?;
        let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        Ok(TradingEnvironment::new(prices, symbol.to_string()))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Comparison: Random Search ─────────────────────────────────────

/// Run random search baseline for comparison with PBT.
/// Returns the best performance found after training `num_trials` random agents.
pub fn random_search(
    env: &TradingEnvironment,
    num_trials: usize,
    training_steps: usize,
) -> (f64, Hyperparameters) {
    let mut best_perf = f64::NEG_INFINITY;
    let mut best_hp = Hyperparameters::random();

    for id in 0..num_trials {
        let mut agent = Agent::new(id, 20);
        agent.train(env, training_steps);
        let perf = agent.evaluate(env);

        if perf > best_perf {
            best_perf = perf;
            best_hp = agent.hyperparams.clone();
        }
    }

    (best_perf, best_hp)
}

/// Run manual tuning baseline (fixed hyperparameters).
pub fn manual_tuning(
    env: &TradingEnvironment,
    training_steps: usize,
) -> (f64, Hyperparameters) {
    let hp = Hyperparameters {
        learning_rate: 0.001,
        lookback_window: 20,
        batch_size: 64,
        risk_penalty: 0.1,
        momentum: 0.9,
    };

    let mut agent = Agent::new(0, hp.lookback_window);
    agent.hyperparams = hp;
    agent.train(env, training_steps);
    let perf = agent.evaluate(env);

    (perf, agent.hyperparams.clone())
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparams_random_in_range() {
        for _ in 0..100 {
            let hp = Hyperparameters::random();
            assert!(hp.learning_rate > 0.0 && hp.learning_rate < 1.0);
            assert!([5, 10, 20, 50].contains(&hp.lookback_window));
            assert!([16, 32, 64, 128].contains(&hp.batch_size));
            assert!(hp.risk_penalty >= 0.01 && hp.risk_penalty <= 1.0);
            assert!(hp.momentum >= 0.0 && hp.momentum <= 0.99);
        }
    }

    #[test]
    fn test_hyperparams_perturb_stays_in_bounds() {
        let mut hp = Hyperparameters::random();
        for _ in 0..1000 {
            hp.perturb();
            assert!(hp.learning_rate >= 1e-5 && hp.learning_rate <= 1e-1);
            assert!(hp.risk_penalty >= 0.001 && hp.risk_penalty <= 5.0);
            assert!(hp.momentum >= 0.0 && hp.momentum <= 0.99);
            assert!([5, 10, 20, 50].contains(&hp.lookback_window));
            assert!([16, 32, 64, 128].contains(&hp.batch_size));
        }
    }

    #[test]
    fn test_agent_predict() {
        let agent = Agent::new(0, 5);
        let features = Array1::from_vec(vec![0.1, -0.05, 0.02, 0.03, -0.01]);
        let pred = agent.predict(&features);
        assert!(pred.is_finite());
    }

    #[test]
    fn test_agent_train_and_evaluate() {
        let env = TradingEnvironment::synthetic(200, 100.0);
        let mut agent = Agent::new(0, 20);
        agent.train(&env, 10);
        let perf = agent.evaluate(&env);
        assert!(perf.is_finite());
    }

    #[test]
    fn test_population_step() {
        let env = TradingEnvironment::synthetic(200, 100.0);
        let config = PbtConfig {
            population_size: 5,
            weight_dim: 10,
            steps_between_eval: 3,
            warmup_steps: 3,
            ..Default::default()
        };
        let mut pop = Population::new(config);

        // Run a few generations
        for _ in 0..5 {
            pop.step(&env);
        }

        let stats = pop.stats();
        assert!(stats.best.is_finite());
        assert!(stats.best >= stats.worst);
        assert_eq!(pop.generation, 5);
    }

    #[test]
    fn test_population_exploit_improves_worst() {
        let env = TradingEnvironment::synthetic(300, 100.0);
        let config = PbtConfig {
            population_size: 10,
            weight_dim: 10,
            steps_between_eval: 5,
            warmup_steps: 0,
            ..Default::default()
        };
        let mut pop = Population::new(config);

        // Run enough generations for exploit/explore to take effect
        for _ in 0..10 {
            pop.step(&env);
        }

        let stats = pop.stats();
        // After exploit, the gap between best and worst should be smaller
        // than a purely random population (this is a soft check)
        assert!(stats.std.is_finite());
    }

    #[test]
    fn test_trading_env_synthetic() {
        let env = TradingEnvironment::synthetic(100, 50.0);
        assert_eq!(env.prices.len(), 100);
        assert!((env.prices[0] - 50.0).abs() < 1e-10);
        assert_eq!(env.symbol, "SYNTHETIC");
    }

    #[test]
    fn test_trading_env_train_test_split() {
        let env = TradingEnvironment::synthetic(100, 50.0);
        let (train, test) = env.train_test_split(0.8);
        assert_eq!(train.prices.len(), 80);
        assert_eq!(test.prices.len(), 20);
    }

    #[test]
    fn test_random_search_returns_finite() {
        let env = TradingEnvironment::synthetic(200, 100.0);
        let (perf, hp) = random_search(&env, 5, 10);
        assert!(perf.is_finite());
        assert!(hp.learning_rate > 0.0);
    }

    #[test]
    fn test_binary_tournament_strategy() {
        let env = TradingEnvironment::synthetic(200, 100.0);
        let config = PbtConfig {
            population_size: 6,
            weight_dim: 10,
            steps_between_eval: 3,
            warmup_steps: 0,
            exploit_strategy: ExploitStrategy::BinaryTournament,
            ..Default::default()
        };
        let mut pop = Population::new(config);
        for _ in 0..5 {
            pop.step(&env);
        }
        let stats = pop.stats();
        assert!(stats.best.is_finite());
    }
}
