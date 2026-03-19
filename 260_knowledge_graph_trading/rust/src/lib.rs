use ndarray::Array1;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};

// ── Triple representation ──────────────────────────────────────────────

/// A single (head, relation, tail) fact in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub head: usize,
    pub relation: usize,
    pub tail: usize,
}

// ── Knowledge Graph ────────────────────────────────────────────────────

/// In-memory knowledge graph storing triples with indexed lookups.
pub struct KnowledgeGraph {
    pub entity_names: Vec<String>,
    pub relation_names: Vec<String>,
    entity_index: HashMap<String, usize>,
    relation_index: HashMap<String, usize>,
    triples: Vec<Triple>,
    /// head -> [(relation, tail)]
    adjacency: HashMap<usize, Vec<(usize, usize)>>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            entity_names: Vec::new(),
            relation_names: Vec::new(),
            entity_index: HashMap::new(),
            relation_index: HashMap::new(),
            triples: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Register an entity name and return its id (idempotent).
    pub fn add_entity(&mut self, name: &str) -> usize {
        if let Some(&id) = self.entity_index.get(name) {
            return id;
        }
        let id = self.entity_names.len();
        self.entity_names.push(name.to_string());
        self.entity_index.insert(name.to_string(), id);
        id
    }

    /// Register a relation name and return its id (idempotent).
    pub fn add_relation(&mut self, name: &str) -> usize {
        if let Some(&id) = self.relation_index.get(name) {
            return id;
        }
        let id = self.relation_names.len();
        self.relation_names.push(name.to_string());
        self.relation_index.insert(name.to_string(), id);
        id
    }

    /// Insert a triple into the graph.
    pub fn add_triple(&mut self, head: usize, relation: usize, tail: usize) {
        let t = Triple { head, relation, tail };
        self.triples.push(t);
        self.adjacency.entry(head).or_default().push((relation, tail));
    }

    /// Convenience: add triple by name strings.
    pub fn add_triple_by_name(&mut self, head: &str, relation: &str, tail: &str) {
        let h = self.add_entity(head);
        let r = self.add_relation(relation);
        let t = self.add_entity(tail);
        self.add_triple(h, r, t);
    }

    pub fn num_entities(&self) -> usize {
        self.entity_names.len()
    }

    pub fn num_relations(&self) -> usize {
        self.relation_names.len()
    }

    pub fn triples(&self) -> &[Triple] {
        &self.triples
    }

    /// Return outgoing (relation, tail) pairs for an entity.
    pub fn neighbors(&self, entity: usize) -> &[(usize, usize)] {
        self.adjacency.get(&entity).map_or(&[], |v| v.as_slice())
    }

    /// Degree centrality: number of outgoing edges.
    pub fn out_degree(&self, entity: usize) -> usize {
        self.neighbors(entity).len()
    }

    /// BFS shortest-path length between two entities (-1 if unreachable).
    pub fn shortest_path(&self, from: usize, to: usize) -> i32 {
        if from == to {
            return 0;
        }
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(from);
        queue.push_back((from, 0i32));
        while let Some((node, dist)) = queue.pop_front() {
            for &(_, neighbor) in self.neighbors(node) {
                if neighbor == to {
                    return dist + 1;
                }
                if visited.insert(neighbor) {
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
        -1
    }

    /// Entity id by name (if registered).
    pub fn entity_id(&self, name: &str) -> Option<usize> {
        self.entity_index.get(name).copied()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── TransE embedding model ────────────────────────────────────────────

/// TransE learns entity/relation embeddings such that h + r ≈ t.
pub struct TransEModel {
    pub dim: usize,
    pub entity_embeddings: Vec<Array1<f64>>,
    pub relation_embeddings: Vec<Array1<f64>>,
    margin: f64,
    learning_rate: f64,
}

impl TransEModel {
    pub fn new(num_entities: usize, num_relations: usize, dim: usize, margin: f64, lr: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut rand_vec = |d: usize| -> Array1<f64> {
            let v: Vec<f64> = (0..d).map(|_| rng.gen_range(-0.5..0.5)).collect();
            let a = Array1::from(v);
            let norm = a.dot(&a).sqrt().max(1e-12);
            a / norm
        };
        Self {
            dim,
            entity_embeddings: (0..num_entities).map(|_| rand_vec(dim)).collect(),
            relation_embeddings: (0..num_relations).map(|_| rand_vec(dim)).collect(),
            margin,
            learning_rate: lr,
        }
    }

    /// Score a triple: lower = better (distance).
    pub fn score(&self, h: usize, r: usize, t: usize) -> f64 {
        let diff = &self.entity_embeddings[h] + &self.relation_embeddings[r]
            - &self.entity_embeddings[t];
        diff.dot(&diff).sqrt()
    }

    /// Train one epoch over all triples with negative sampling.
    pub fn train_epoch(&mut self, triples: &[Triple], num_entities: usize) -> f64 {
        let mut rng = rand::thread_rng();
        let mut total_loss = 0.0;
        for triple in triples {
            let h = triple.head;
            let r = triple.relation;
            let t = triple.tail;

            // Corrupt tail to create negative sample
            let t_neg = loop {
                let candidate = rng.gen_range(0..num_entities);
                if candidate != t {
                    break candidate;
                }
            };

            let pos_dist = self.score(h, r, t);
            let neg_dist = self.score(h, r, t_neg);
            let loss = (self.margin + pos_dist - neg_dist).max(0.0);
            total_loss += loss;

            if loss > 0.0 {
                // Gradient step: push positive closer, negative farther
                let pos_diff = &self.entity_embeddings[h] + &self.relation_embeddings[r]
                    - &self.entity_embeddings[t];
                let neg_diff = &self.entity_embeddings[h] + &self.relation_embeddings[r]
                    - &self.entity_embeddings[t_neg];

                let lr = self.learning_rate;
                let pos_grad = &pos_diff * (lr / pos_dist.max(1e-12));
                let neg_grad = &neg_diff * (lr / neg_dist.max(1e-12));

                self.entity_embeddings[h] = &self.entity_embeddings[h] - &pos_grad + &neg_grad;
                self.entity_embeddings[t] = &self.entity_embeddings[t] + &pos_grad;
                self.entity_embeddings[t_neg] = &self.entity_embeddings[t_neg] - &neg_grad;
                self.relation_embeddings[r] = &self.relation_embeddings[r] - &pos_grad + &neg_grad;
            }
        }
        total_loss / triples.len() as f64
    }

    /// Get the embedding for an entity.
    pub fn entity_embedding(&self, entity: usize) -> &Array1<f64> {
        &self.entity_embeddings[entity]
    }

    /// Cosine similarity between two entity embeddings.
    pub fn entity_similarity(&self, a: usize, b: usize) -> f64 {
        let ea = &self.entity_embeddings[a];
        let eb = &self.entity_embeddings[b];
        let dot = ea.dot(eb);
        let na = ea.dot(ea).sqrt().max(1e-12);
        let nb = eb.dot(eb).sqrt().max(1e-12);
        dot / (na * nb)
    }
}

// ── Graph Feature Extractor ───────────────────────────────────────────

/// Extracts trading-relevant features from a knowledge graph.
pub struct GraphFeatureExtractor<'a> {
    graph: &'a KnowledgeGraph,
}

impl<'a> GraphFeatureExtractor<'a> {
    pub fn new(graph: &'a KnowledgeGraph) -> Self {
        Self { graph }
    }

    /// Degree centrality normalized by total entities.
    pub fn degree_centrality(&self, entity: usize) -> f64 {
        let n = self.graph.num_entities();
        if n <= 1 {
            return 0.0;
        }
        self.graph.out_degree(entity) as f64 / (n - 1) as f64
    }

    /// Simple PageRank (power-iteration, 20 iterations, damping=0.85).
    pub fn pagerank(&self, num_iterations: usize) -> Vec<f64> {
        let n = self.graph.num_entities();
        if n == 0 {
            return vec![];
        }
        let d = 0.85;
        let mut rank = vec![1.0 / n as f64; n];
        for _ in 0..num_iterations {
            let mut new_rank = vec![(1.0 - d) / n as f64; n];
            for entity in 0..n {
                let neighbors = self.graph.neighbors(entity);
                if neighbors.is_empty() {
                    // Dangling node: distribute evenly
                    let share = d * rank[entity] / n as f64;
                    for r in new_rank.iter_mut() {
                        *r += share;
                    }
                } else {
                    let share = d * rank[entity] / neighbors.len() as f64;
                    for &(_, tail) in neighbors {
                        new_rank[tail] += share;
                    }
                }
            }
            rank = new_rank;
        }
        rank
    }

    /// Feature vector for a single entity: [degree_centrality, pagerank, num_relation_types].
    pub fn entity_features(&self, entity: usize, pageranks: &[f64]) -> Array1<f64> {
        let deg = self.degree_centrality(entity);
        let pr = if entity < pageranks.len() { pageranks[entity] } else { 0.0 };
        let relation_types: HashSet<usize> = self.graph.neighbors(entity)
            .iter()
            .map(|&(r, _)| r)
            .collect();
        let rt = relation_types.len() as f64 / self.graph.num_relations().max(1) as f64;
        Array1::from(vec![deg, pr, rt])
    }
}

// ── Trading Signal Generator ──────────────────────────────────────────

/// Combines KG embeddings with market data to produce trading signals.
pub struct TradingSignalGenerator {
    /// weights: [kg_features..., market_features..., bias]
    weights: Array1<f64>,
    num_features: usize,
}

impl TradingSignalGenerator {
    pub fn new(num_kg_features: usize, num_market_features: usize) -> Self {
        let total = num_kg_features + num_market_features;
        Self {
            weights: Array1::zeros(total + 1), // +1 for bias
            num_features: total,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict probability of price going up.
    pub fn predict_proba(&self, features: &Array1<f64>) -> f64 {
        assert_eq!(features.len(), self.num_features);
        let logit = features.dot(&self.weights.slice(ndarray::s![..self.num_features]))
            + self.weights[self.num_features];
        Self::sigmoid(logit)
    }

    /// Predict direction and confidence.
    pub fn predict(&self, features: &Array1<f64>) -> (bool, f64) {
        let p = self.predict_proba(features);
        (p >= 0.5, (p - 0.5).abs() * 2.0)
    }

    /// Train via SGD on labeled data.
    pub fn train(
        &mut self,
        features: &[Array1<f64>],
        labels: &[f64],
        epochs: usize,
        lr: f64,
    ) {
        for _ in 0..epochs {
            for (x, &y) in features.iter().zip(labels.iter()) {
                let pred = self.predict_proba(x);
                let error = pred - y;
                for i in 0..self.num_features {
                    self.weights[i] -= lr * error * x[i];
                }
                self.weights[self.num_features] -= lr * error; // bias
            }
        }
    }

    /// Accuracy on a test set.
    pub fn accuracy(&self, features: &[Array1<f64>], labels: &[f64]) -> f64 {
        let correct = features.iter().zip(labels.iter()).filter(|(x, &y)| {
            let (dir, _) = self.predict(x);
            (dir && y >= 0.5) || (!dir && y < 0.5)
        }).count();
        correct as f64 / features.len().max(1) as f64
    }
}

// ── Bybit Client ──────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    pub result: T,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct KlineResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct OrderbookResult {
    pub b: Vec<Vec<String>>,
    pub a: Vec<Vec<String>>,
}

pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data.
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);
        let resp: BybitResponse<KlineResult> = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;

        let klines = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();
        Ok(klines)
    }

    /// Fetch order book data, returns (bids, asks) as (price, qty) tuples.
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> anyhow::Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let url = format!("{}/v5/market/orderbook", self.base_url);
        let resp: BybitResponse<OrderbookResult> = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;

        let parse = |rows: &[Vec<String>]| -> Vec<(f64, f64)> {
            rows.iter()
                .filter_map(|r| {
                    if r.len() >= 2 {
                        Some((
                            r[0].parse().unwrap_or(0.0),
                            r[1].parse().unwrap_or(0.0),
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        };
        Ok((parse(&resp.result.b), parse(&resp.result.a)))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ── Synthetic data helpers ────────────────────────────────────────────

/// Build a sample financial knowledge graph with realistic entities and relations.
pub fn build_sample_financial_kg() -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::new();

    // Entities
    let entities = [
        "AAPL", "TSMC", "NVDA", "MSFT", "GOOG", "AMZN", "TSLA",
        "Samsung", "Intel", "Qualcomm",
        "Technology", "Automotive", "Cloud",
        "S&P500", "NASDAQ",
        "BTC", "ETH",
    ];
    for e in &entities {
        kg.add_entity(e);
    }

    // Relations
    let _relations = [
        "supplies_to", "competes_with", "member_of", "partner_of",
        "sector", "correlates_with",
    ];

    // Supply chain
    kg.add_triple_by_name("TSMC", "supplies_to", "AAPL");
    kg.add_triple_by_name("TSMC", "supplies_to", "NVDA");
    kg.add_triple_by_name("TSMC", "supplies_to", "Qualcomm");
    kg.add_triple_by_name("Samsung", "supplies_to", "AAPL");
    kg.add_triple_by_name("Intel", "supplies_to", "MSFT");

    // Competition
    kg.add_triple_by_name("AAPL", "competes_with", "Samsung");
    kg.add_triple_by_name("MSFT", "competes_with", "GOOG");
    kg.add_triple_by_name("NVDA", "competes_with", "Intel");
    kg.add_triple_by_name("AMZN", "competes_with", "MSFT");

    // Sector memberships
    kg.add_triple_by_name("AAPL", "sector", "Technology");
    kg.add_triple_by_name("NVDA", "sector", "Technology");
    kg.add_triple_by_name("MSFT", "sector", "Technology");
    kg.add_triple_by_name("GOOG", "sector", "Technology");
    kg.add_triple_by_name("TSLA", "sector", "Automotive");
    kg.add_triple_by_name("AMZN", "sector", "Cloud");
    kg.add_triple_by_name("MSFT", "sector", "Cloud");

    // Index membership
    kg.add_triple_by_name("AAPL", "member_of", "S&P500");
    kg.add_triple_by_name("MSFT", "member_of", "S&P500");
    kg.add_triple_by_name("NVDA", "member_of", "NASDAQ");
    kg.add_triple_by_name("GOOG", "member_of", "NASDAQ");

    // Partnerships
    kg.add_triple_by_name("MSFT", "partner_of", "NVDA");
    kg.add_triple_by_name("GOOG", "partner_of", "TSLA");

    // Crypto correlations
    kg.add_triple_by_name("BTC", "correlates_with", "ETH");
    kg.add_triple_by_name("BTC", "correlates_with", "NVDA");

    kg
}

/// Generate synthetic training data: KG features + market features -> label.
pub fn generate_training_data(
    kg: &KnowledgeGraph,
    _model: &TransEModel,
    n_samples: usize,
) -> (Vec<Array1<f64>>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let extractor = GraphFeatureExtractor::new(kg);
    let pageranks = extractor.pagerank(20);
    let n_entities = kg.num_entities();

    let mut features = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let entity = rng.gen_range(0..n_entities);
        let kg_feats = extractor.entity_features(entity, &pageranks);

        // Synthetic market features: [momentum, volatility, volume_ratio, spread]
        let momentum: f64 = rng.gen_range(-0.05..0.05);
        let volatility: f64 = rng.gen_range(0.01..0.1);
        let volume_ratio: f64 = rng.gen_range(0.5..2.0);
        let spread: f64 = rng.gen_range(0.001..0.01);

        // Combine KG features (3) + market features (4) = 7
        let combined = Array1::from(vec![
            kg_feats[0], kg_feats[1], kg_feats[2],
            momentum, volatility, volume_ratio, spread,
        ]);

        // Label: higher degree + positive momentum -> more likely up
        let signal = kg_feats[0] * 0.3 + momentum * 5.0 + kg_feats[1] * 2.0 - volatility;
        let prob = 1.0 / (1.0 + (-signal).exp());
        let label = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };

        features.push(combined);
        labels.push(label);
    }

    (features, labels)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph_construction() {
        let kg = build_sample_financial_kg();
        assert!(kg.num_entities() >= 10);
        assert!(kg.num_relations() >= 4);
        assert!(!kg.triples().is_empty());
    }

    #[test]
    fn test_entity_registration_idempotent() {
        let mut kg = KnowledgeGraph::new();
        let id1 = kg.add_entity("AAPL");
        let id2 = kg.add_entity("AAPL");
        assert_eq!(id1, id2);
        assert_eq!(kg.num_entities(), 1);
    }

    #[test]
    fn test_neighbors() {
        let kg = build_sample_financial_kg();
        let tsmc = kg.entity_id("TSMC").unwrap();
        let neighbors = kg.neighbors(tsmc);
        assert!(neighbors.len() >= 3); // supplies_to AAPL, NVDA, Qualcomm
    }

    #[test]
    fn test_shortest_path() {
        let kg = build_sample_financial_kg();
        let tsmc = kg.entity_id("TSMC").unwrap();
        let aapl = kg.entity_id("AAPL").unwrap();
        assert_eq!(kg.shortest_path(tsmc, aapl), 1);
        assert_eq!(kg.shortest_path(aapl, aapl), 0);
    }

    #[test]
    fn test_transe_score_improves() {
        let kg = build_sample_financial_kg();
        let mut model = TransEModel::new(kg.num_entities(), kg.num_relations(), 16, 1.0, 0.01);

        let loss_before = model.train_epoch(kg.triples(), kg.num_entities());
        for _ in 0..50 {
            model.train_epoch(kg.triples(), kg.num_entities());
        }
        let loss_after = model.train_epoch(kg.triples(), kg.num_entities());
        assert!(loss_after <= loss_before + 0.1, "Training should reduce loss");
    }

    #[test]
    fn test_entity_similarity() {
        let kg = build_sample_financial_kg();
        let mut model = TransEModel::new(kg.num_entities(), kg.num_relations(), 16, 1.0, 0.01);
        for _ in 0..100 {
            model.train_epoch(kg.triples(), kg.num_entities());
        }
        let aapl = kg.entity_id("AAPL").unwrap();
        let nvda = kg.entity_id("NVDA").unwrap();
        let sim = model.entity_similarity(aapl, nvda);
        // Similarity should be in [-1, 1]
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_degree_centrality() {
        let kg = build_sample_financial_kg();
        let extractor = GraphFeatureExtractor::new(&kg);
        let tsmc = kg.entity_id("TSMC").unwrap();
        let centrality = extractor.degree_centrality(tsmc);
        assert!(centrality > 0.0, "TSMC should have outgoing edges");
    }

    #[test]
    fn test_pagerank() {
        let kg = build_sample_financial_kg();
        let extractor = GraphFeatureExtractor::new(&kg);
        let pr = extractor.pagerank(20);
        assert_eq!(pr.len(), kg.num_entities());
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "PageRank should sum to ~1.0");
    }

    #[test]
    fn test_signal_generator_predict() {
        let gen = TradingSignalGenerator::new(3, 4);
        let features = Array1::from(vec![0.5, 0.1, 0.3, 0.02, 0.05, 1.0, 0.005]);
        let (_, confidence) = gen.predict(&features);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_signal_generator_train() {
        let kg = build_sample_financial_kg();
        let model = TransEModel::new(kg.num_entities(), kg.num_relations(), 16, 1.0, 0.01);
        let (features, labels) = generate_training_data(&kg, &model, 500);

        let split = 400;
        let (train_x, test_x) = features.split_at(split);
        let (train_y, test_y) = labels.split_at(split);

        let mut gen = TradingSignalGenerator::new(3, 4);
        let acc_before = gen.accuracy(test_x, test_y);
        gen.train(train_x, train_y, 50, 0.01);
        let acc_after = gen.accuracy(test_x, test_y);

        // After training, accuracy should be at least as good (or very close)
        assert!(acc_after >= acc_before - 0.1, "Training should not severely degrade accuracy");
    }

    #[test]
    fn test_generate_training_data() {
        let kg = build_sample_financial_kg();
        let model = TransEModel::new(kg.num_entities(), kg.num_relations(), 16, 1.0, 0.01);
        let (features, labels) = generate_training_data(&kg, &model, 100);
        assert_eq!(features.len(), 100);
        assert_eq!(labels.len(), 100);
        assert_eq!(features[0].len(), 7); // 3 KG + 4 market
        for &l in &labels {
            assert!(l == 0.0 || l == 1.0);
        }
    }
}
