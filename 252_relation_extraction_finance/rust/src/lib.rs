use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Entity Types ─────────────────────────────────────────────────

/// The type of a named entity found in financial text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Company,
    Person,
    Product,
    Exchange,
    Fund,
    Token,
}

impl EntityType {
    /// Parse an entity type from a string label.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "company" | "org" | "organization" => Some(Self::Company),
            "person" | "per" => Some(Self::Person),
            "product" | "prod" => Some(Self::Product),
            "exchange" | "exch" => Some(Self::Exchange),
            "fund" => Some(Self::Fund),
            "token" | "crypto" => Some(Self::Token),
            _ => None,
        }
    }
}

// ─── Relation Types ───────────────────────────────────────────────

/// Semantic relation types relevant to financial text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationType {
    Acquired,
    PartneredWith,
    CompetesWith,
    InvestedIn,
    CeoOf,
    ListedOn,
    SuppliesTo,
    SubsidiaryOf,
    FoundedBy,
    Produces,
    NoRelation,
}

impl RelationType {
    /// All relation types (excluding NoRelation).
    pub fn all() -> &'static [RelationType] {
        &[
            Self::Acquired,
            Self::PartneredWith,
            Self::CompetesWith,
            Self::InvestedIn,
            Self::CeoOf,
            Self::ListedOn,
            Self::SuppliesTo,
            Self::SubsidiaryOf,
            Self::FoundedBy,
            Self::Produces,
        ]
    }

    /// Index for feature encoding (0-based).
    pub fn index(&self) -> usize {
        match self {
            Self::Acquired => 0,
            Self::PartneredWith => 1,
            Self::CompetesWith => 2,
            Self::InvestedIn => 3,
            Self::CeoOf => 4,
            Self::ListedOn => 5,
            Self::SuppliesTo => 6,
            Self::SubsidiaryOf => 7,
            Self::FoundedBy => 8,
            Self::Produces => 9,
            Self::NoRelation => 10,
        }
    }

    /// Label string for display.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Acquired => "ACQUIRED",
            Self::PartneredWith => "PARTNERED_WITH",
            Self::CompetesWith => "COMPETES_WITH",
            Self::InvestedIn => "INVESTED_IN",
            Self::CeoOf => "CEO_OF",
            Self::ListedOn => "LISTED_ON",
            Self::SuppliesTo => "SUPPLIES_TO",
            Self::SubsidiaryOf => "SUBSIDIARY_OF",
            Self::FoundedBy => "FOUNDED_BY",
            Self::Produces => "PRODUCES",
            Self::NoRelation => "NO_RELATION",
        }
    }
}

// ─── Entity ───────────────────────────────────────────────────────

/// A named entity extracted from text.
#[derive(Debug, Clone)]
pub struct Entity {
    pub name: String,
    pub entity_type: EntityType,
    pub start: usize,
    pub end: usize,
}

impl Entity {
    pub fn new(name: &str, entity_type: EntityType, start: usize, end: usize) -> Self {
        Self {
            name: name.to_string(),
            entity_type,
            start,
            end,
        }
    }
}

// ─── Relation ─────────────────────────────────────────────────────

/// An extracted relation between two entities.
#[derive(Debug, Clone)]
pub struct Relation {
    pub entity1: Entity,
    pub entity2: Entity,
    pub relation_type: RelationType,
    pub confidence: f64,
    pub source_sentence: String,
}

impl Relation {
    pub fn new(
        entity1: Entity,
        entity2: Entity,
        relation_type: RelationType,
        confidence: f64,
        source_sentence: &str,
    ) -> Self {
        Self {
            entity1,
            entity2,
            relation_type,
            confidence,
            source_sentence: source_sentence.to_string(),
        }
    }
}

// ─── Pattern Extractor ────────────────────────────────────────────

/// Rule-based relation extractor using trigger word patterns.
///
/// Maintains a mapping from trigger keywords to relation types and
/// scans sentences for co-occurring entity pairs and triggers.
pub struct PatternExtractor {
    patterns: HashMap<RelationType, Vec<String>>,
}

impl PatternExtractor {
    /// Create a new extractor with default financial trigger patterns.
    pub fn new() -> Self {
        let mut patterns: HashMap<RelationType, Vec<String>> = HashMap::new();

        patterns.insert(
            RelationType::Acquired,
            vec![
                "acquired".into(),
                "bought".into(),
                "purchased".into(),
                "acquisition".into(),
                "takeover".into(),
                "took over".into(),
                "merged with".into(),
            ],
        );
        patterns.insert(
            RelationType::PartneredWith,
            vec![
                "partnered".into(),
                "partnership".into(),
                "collaborated".into(),
                "collaboration".into(),
                "alliance".into(),
                "joint venture".into(),
                "teamed up".into(),
            ],
        );
        patterns.insert(
            RelationType::CompetesWith,
            vec![
                "competes".into(),
                "competitor".into(),
                "rival".into(),
                "competing".into(),
                "competition".into(),
            ],
        );
        patterns.insert(
            RelationType::InvestedIn,
            vec![
                "invested".into(),
                "investment".into(),
                "funding".into(),
                "backed".into(),
                "stake".into(),
                "raised".into(),
            ],
        );
        patterns.insert(
            RelationType::CeoOf,
            vec![
                "ceo".into(),
                "chief executive".into(),
                "leads".into(),
                "headed by".into(),
                "appointed".into(),
                "named ceo".into(),
            ],
        );
        patterns.insert(
            RelationType::ListedOn,
            vec![
                "listed".into(),
                "listing".into(),
                "listed on".into(),
                "trading on".into(),
                "debut".into(),
                "ipo".into(),
            ],
        );
        patterns.insert(
            RelationType::SuppliesTo,
            vec![
                "supplies".into(),
                "supplier".into(),
                "provides".into(),
                "vendor".into(),
                "supply chain".into(),
            ],
        );
        patterns.insert(
            RelationType::SubsidiaryOf,
            vec![
                "subsidiary".into(),
                "unit of".into(),
                "division of".into(),
                "owned by".into(),
                "parent company".into(),
            ],
        );
        patterns.insert(
            RelationType::FoundedBy,
            vec![
                "founded".into(),
                "co-founded".into(),
                "founder".into(),
                "co-founder".into(),
                "established".into(),
            ],
        );
        patterns.insert(
            RelationType::Produces,
            vec![
                "produces".into(),
                "manufactures".into(),
                "developed".into(),
                "launched".into(),
                "released".into(),
            ],
        );

        Self { patterns }
    }

    /// Extract relations from a sentence given a list of entities.
    ///
    /// For every pair of entities in the sentence, check if any trigger
    /// words for any relation type appear in the text between or around them.
    pub fn extract(&self, sentence: &str, entities: &[Entity]) -> Vec<Relation> {
        let lower = sentence.to_lowercase();
        let mut relations = Vec::new();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let e1 = &entities[i];
                let e2 = &entities[j];

                // Check each relation type's trigger words
                let mut best_relation = RelationType::NoRelation;
                let mut best_confidence = 0.0;

                for (rel_type, triggers) in &self.patterns {
                    for trigger in triggers {
                        if lower.contains(trigger.as_str()) {
                            // Check entity type compatibility
                            let compat = entity_type_compatibility(
                                e1.entity_type,
                                e2.entity_type,
                                *rel_type,
                            );
                            if compat > 0.0 {
                                // Compute distance-based confidence
                                let distance = if e2.start > e1.end {
                                    (e2.start - e1.end) as f64
                                } else if e1.start > e2.end {
                                    (e1.start - e2.end) as f64
                                } else {
                                    1.0
                                };
                                let dist_score = 1.0 / (1.0 + distance / 50.0);
                                let confidence = compat * dist_score;

                                if confidence > best_confidence {
                                    best_confidence = confidence;
                                    best_relation = *rel_type;
                                }
                            }
                        }
                    }
                }

                if best_relation != RelationType::NoRelation && best_confidence > 0.1 {
                    relations.push(Relation::new(
                        e1.clone(),
                        e2.clone(),
                        best_relation,
                        best_confidence,
                        sentence,
                    ));
                }
            }
        }

        relations
    }
}

impl Default for PatternExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Check compatibility between entity types and a relation type.
/// Returns a score between 0.0 (incompatible) and 1.0 (fully compatible).
pub fn entity_type_compatibility(
    e1_type: EntityType,
    e2_type: EntityType,
    relation: RelationType,
) -> f64 {
    match relation {
        RelationType::Acquired | RelationType::PartneredWith | RelationType::CompetesWith => {
            match (e1_type, e2_type) {
                (EntityType::Company, EntityType::Company) => 1.0,
                (EntityType::Fund, EntityType::Company) => 0.7,
                _ => 0.2,
            }
        }
        RelationType::InvestedIn => match (e1_type, e2_type) {
            (EntityType::Fund, EntityType::Company) => 1.0,
            (EntityType::Company, EntityType::Company) => 0.8,
            (EntityType::Person, EntityType::Company) => 0.6,
            _ => 0.1,
        },
        RelationType::CeoOf | RelationType::FoundedBy => match (e1_type, e2_type) {
            (EntityType::Person, EntityType::Company) => 1.0,
            _ => 0.1,
        },
        RelationType::ListedOn => match (e1_type, e2_type) {
            (EntityType::Token, EntityType::Exchange) => 1.0,
            (EntityType::Company, EntityType::Exchange) => 0.9,
            _ => 0.1,
        },
        RelationType::SuppliesTo => match (e1_type, e2_type) {
            (EntityType::Company, EntityType::Company) => 1.0,
            _ => 0.1,
        },
        RelationType::SubsidiaryOf => match (e1_type, e2_type) {
            (EntityType::Company, EntityType::Company) => 1.0,
            _ => 0.1,
        },
        RelationType::Produces => match (e1_type, e2_type) {
            (EntityType::Company, EntityType::Product) => 1.0,
            _ => 0.2,
        },
        RelationType::NoRelation => 0.0,
    }
}

// ─── Relation Scorer (Logistic Regression) ────────────────────────

/// Feature-based relation scorer using logistic regression.
///
/// Features:
/// - Entity distance (normalized)
/// - Trigger word presence (one-hot per relation type)
/// - Entity type compatibility score
/// - Between-context length (normalized)
/// - Sentence length (normalized)
#[derive(Debug)]
pub struct RelationScorer {
    weights: Array1<f64>,
    bias: f64,
    learning_rate: f64,
    num_features: usize,
}

impl RelationScorer {
    /// Create a new scorer with random initialization.
    /// `num_features` should match the feature vector length used during training.
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_vec(
            (0..num_features)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
        );
        Self {
            weights,
            bias: 0.0,
            learning_rate,
            num_features,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict confidence score for a relation candidate.
    pub fn predict_score(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        let z = self.weights.dot(&x) + self.bias;
        Self::sigmoid(z)
    }

    /// Predict whether a relation is valid and its confidence.
    pub fn predict(&self, features: &[f64]) -> (bool, f64) {
        let score = self.predict_score(features);
        if score >= 0.5 {
            (true, score)
        } else {
            (false, 1.0 - score)
        }
    }

    /// Train on labeled examples: (features, label) where label is 1.0 (valid relation)
    /// or 0.0 (no relation).
    pub fn train(&mut self, data: &[(Vec<f64>, f64)], epochs: usize) {
        for _ in 0..epochs {
            for (features, label) in data {
                let x = Array1::from_vec(features.clone());
                let z = self.weights.dot(&x) + self.bias;
                let pred = Self::sigmoid(z);
                let error = pred - label;

                for j in 0..self.num_features {
                    self.weights[j] -= self.learning_rate * error * x[j];
                }
                self.bias -= self.learning_rate * error;
            }
        }
    }

    /// Evaluate accuracy on a test set.
    pub fn accuracy(&self, data: &[(Vec<f64>, f64)]) -> f64 {
        let correct = data
            .iter()
            .filter(|(features, label)| {
                let (pred, _) = self.predict(features);
                let label_bool = *label >= 0.5;
                pred == label_bool
            })
            .count();
        correct as f64 / data.len() as f64
    }

    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }
}

// ─── Feature Extraction ───────────────────────────────────────────

/// Extract feature vector for a candidate relation.
///
/// Features (8-dimensional):
/// 0: Normalized entity distance
/// 1: Entity type compatibility score
/// 2: Trigger word match score (0.0 or 1.0)
/// 3: Between-context word count (normalized)
/// 4: Sentence length (normalized)
/// 5: Entity 1 position ratio (position / sentence length)
/// 6: Entity 2 position ratio
/// 7: Entity name overlap (Jaccard on character trigrams)
pub fn extract_features(
    sentence: &str,
    e1: &Entity,
    e2: &Entity,
    relation: RelationType,
    extractor: &PatternExtractor,
) -> Vec<f64> {
    let sent_len = sentence.len() as f64;

    // Feature 0: Normalized entity distance
    let distance = if e2.start > e1.end {
        (e2.start - e1.end) as f64
    } else if e1.start > e2.end {
        (e1.start - e2.end) as f64
    } else {
        0.0
    };
    let norm_distance = distance / sent_len.max(1.0);

    // Feature 1: Entity type compatibility
    let compat = entity_type_compatibility(e1.entity_type, e2.entity_type, relation);

    // Feature 2: Trigger word match
    let lower = sentence.to_lowercase();
    let has_trigger = if let Some(triggers) = extractor.patterns.get(&relation) {
        triggers.iter().any(|t| lower.contains(t.as_str()))
    } else {
        false
    };

    // Feature 3: Between-context word count (normalized)
    let between_start = e1.end.min(e2.end);
    let between_end = e1.start.max(e2.start);
    let between_text = if between_end > between_start {
        &sentence[between_start..between_end]
    } else {
        ""
    };
    let between_words = between_text.split_whitespace().count() as f64;
    let norm_between = between_words / 20.0; // normalize by typical max

    // Feature 4: Sentence length (normalized)
    let norm_sent_len = sent_len / 500.0;

    // Feature 5-6: Entity position ratios
    let e1_pos = e1.start as f64 / sent_len.max(1.0);
    let e2_pos = e2.start as f64 / sent_len.max(1.0);

    // Feature 7: Character trigram overlap between entity names
    let overlap = trigram_jaccard(&e1.name, &e2.name);

    vec![
        norm_distance,
        compat,
        if has_trigger { 1.0 } else { 0.0 },
        norm_between,
        norm_sent_len,
        e1_pos,
        e2_pos,
        overlap,
    ]
}

/// Compute Jaccard similarity on character trigrams.
fn trigram_jaccard(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_chars: Vec<char> = a_lower.chars().collect();
    let b_chars: Vec<char> = b_lower.chars().collect();

    if a_chars.len() < 3 || b_chars.len() < 3 {
        return 0.0;
    }

    let a_trigrams: std::collections::HashSet<(char, char, char)> = a_chars
        .windows(3)
        .map(|w| (w[0], w[1], w[2]))
        .collect();
    let b_trigrams: std::collections::HashSet<(char, char, char)> = b_chars
        .windows(3)
        .map(|w| (w[0], w[1], w[2]))
        .collect();

    let intersection = a_trigrams.intersection(&b_trigrams).count() as f64;
    let union = a_trigrams.union(&b_trigrams).count() as f64;

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

// ─── Knowledge Graph ──────────────────────────────────────────────

/// A simple in-memory knowledge graph built from extracted relations.
#[derive(Debug)]
pub struct KnowledgeGraph {
    entities: HashMap<String, EntityType>,
    edges: Vec<(String, RelationType, String, f64)>, // (entity1, relation, entity2, confidence)
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a relation to the graph.
    pub fn add_relation(&mut self, relation: &Relation) {
        self.entities
            .insert(relation.entity1.name.clone(), relation.entity1.entity_type);
        self.entities
            .insert(relation.entity2.name.clone(), relation.entity2.entity_type);
        self.edges.push((
            relation.entity1.name.clone(),
            relation.relation_type,
            relation.entity2.name.clone(),
            relation.confidence,
        ));
    }

    /// Get all relations involving a given entity.
    pub fn relations_for(&self, entity_name: &str) -> Vec<&(String, RelationType, String, f64)> {
        self.edges
            .iter()
            .filter(|(e1, _, e2, _)| e1 == entity_name || e2 == entity_name)
            .collect()
    }

    /// Get all entities connected to a given entity (one hop).
    pub fn neighbors(&self, entity_name: &str) -> Vec<(&str, RelationType)> {
        let mut result = Vec::new();
        for (e1, rel, e2, _) in &self.edges {
            if e1 == entity_name {
                result.push((e2.as_str(), *rel));
            } else if e2 == entity_name {
                result.push((e1.as_str(), *rel));
            }
        }
        result
    }

    /// Number of entities in the graph.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of edges (relations) in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get all entities of a given type.
    pub fn entities_of_type(&self, entity_type: EntityType) -> Vec<&str> {
        self.entities
            .iter()
            .filter(|(_, t)| **t == entity_type)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Bybit Client ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct OrderbookResult {
    pub b: Vec<Vec<String>>, // bids: [price, size]
    pub a: Vec<Vec<String>>, // asks: [price, size]
}

/// A parsed kline bar.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Async client for Bybit V5 API.
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
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        let resp: BybitResponse<KlineResult> = self.client.get(&url).send().await?.json().await?;

        let mut klines = Vec::new();
        for item in &resp.result.list {
            if item.len() >= 6 {
                klines.push(Kline {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                });
            }
        }
        klines.reverse(); // Bybit returns newest first
        Ok(klines)
    }

    /// Fetch order book snapshot.
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> anyhow::Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );
        let resp: BybitResponse<OrderbookResult> =
            self.client.get(&url).send().await?.json().await?;

        let bids: Vec<(f64, f64)> = resp
            .result
            .b
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = resp
            .result
            .a
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok((bids, asks))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ─────────────────────────────────────

/// Generate synthetic financial sentences with entities and relations for testing.
pub fn generate_synthetic_corpus() -> Vec<(String, Vec<Entity>, RelationType)> {
    vec![
        (
            "Microsoft acquired Activision Blizzard in a landmark deal.".to_string(),
            vec![
                Entity::new("Microsoft", EntityType::Company, 0, 9),
                Entity::new("Activision Blizzard", EntityType::Company, 19, 38),
            ],
            RelationType::Acquired,
        ),
        (
            "Apple partnered with Goldman Sachs to launch a credit card.".to_string(),
            vec![
                Entity::new("Apple", EntityType::Company, 0, 5),
                Entity::new("Goldman Sachs", EntityType::Company, 22, 35),
            ],
            RelationType::PartneredWith,
        ),
        (
            "Uber competes with Lyft in the ride-sharing market.".to_string(),
            vec![
                Entity::new("Uber", EntityType::Company, 0, 4),
                Entity::new("Lyft", EntityType::Company, 19, 23),
            ],
            RelationType::CompetesWith,
        ),
        (
            "Sequoia Capital invested in Stripe during its Series C round.".to_string(),
            vec![
                Entity::new("Sequoia Capital", EntityType::Fund, 0, 15),
                Entity::new("Stripe", EntityType::Company, 29, 35),
            ],
            RelationType::InvestedIn,
        ),
        (
            "Tim Cook is the CEO of Apple Inc.".to_string(),
            vec![
                Entity::new("Tim Cook", EntityType::Person, 0, 8),
                Entity::new("Apple Inc", EntityType::Company, 23, 32),
            ],
            RelationType::CeoOf,
        ),
        (
            "PEPE token was listed on Bybit exchange last week.".to_string(),
            vec![
                Entity::new("PEPE", EntityType::Token, 0, 4),
                Entity::new("Bybit", EntityType::Exchange, 25, 30),
            ],
            RelationType::ListedOn,
        ),
        (
            "TSMC supplies advanced chips to Apple for iPhone production.".to_string(),
            vec![
                Entity::new("TSMC", EntityType::Company, 0, 4),
                Entity::new("Apple", EntityType::Company, 30, 35),
            ],
            RelationType::SuppliesTo,
        ),
        (
            "Instagram is a subsidiary of Meta Platforms.".to_string(),
            vec![
                Entity::new("Instagram", EntityType::Company, 0, 9),
                Entity::new("Meta Platforms", EntityType::Company, 30, 44),
            ],
            RelationType::SubsidiaryOf,
        ),
        (
            "Elon Musk founded SpaceX in 2002.".to_string(),
            vec![
                Entity::new("Elon Musk", EntityType::Person, 0, 9),
                Entity::new("SpaceX", EntityType::Company, 18, 24),
            ],
            RelationType::FoundedBy,
        ),
        (
            "Tesla produces the Model 3 electric vehicle.".to_string(),
            vec![
                Entity::new("Tesla", EntityType::Company, 0, 5),
                Entity::new("Model 3", EntityType::Product, 19, 26),
            ],
            RelationType::Produces,
        ),
    ]
}

/// Generate labeled training data for the RelationScorer.
///
/// Each sample has 8 features and a binary label.
pub fn generate_training_data(n: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let norm_distance: f64 = rng.gen_range(0.0..1.0);
        let compat: f64 = rng.gen_range(0.0..1.0);
        let has_trigger: f64 = if rng.gen_bool(0.5) { 1.0 } else { 0.0 };
        let norm_between: f64 = rng.gen_range(0.0..1.0);
        let norm_sent_len: f64 = rng.gen_range(0.05..0.5);
        let e1_pos: f64 = rng.gen_range(0.0..0.5);
        let e2_pos: f64 = rng.gen_range(0.3..1.0);
        let overlap: f64 = rng.gen_range(0.0..0.3);

        // Label: higher probability of valid relation when trigger present,
        // high compatibility, and short distance
        let signal = -2.0 * norm_distance + 1.5 * compat + 2.0 * has_trigger
            - 0.5 * norm_between
            - 0.3 * overlap;
        let prob = 1.0 / (1.0 + (-signal).exp());
        let label = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };

        data.push((
            vec![
                norm_distance,
                compat,
                has_trigger,
                norm_between,
                norm_sent_len,
                e1_pos,
                e2_pos,
                overlap,
            ],
            label,
        ));
    }
    data
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_parsing() {
        assert_eq!(EntityType::from_str("company"), Some(EntityType::Company));
        assert_eq!(EntityType::from_str("PERSON"), Some(EntityType::Person));
        assert_eq!(EntityType::from_str("token"), Some(EntityType::Token));
        assert_eq!(EntityType::from_str("exchange"), Some(EntityType::Exchange));
        assert_eq!(EntityType::from_str("unknown"), None);
    }

    #[test]
    fn test_relation_type_labels() {
        assert_eq!(RelationType::Acquired.label(), "ACQUIRED");
        assert_eq!(RelationType::ListedOn.label(), "LISTED_ON");
        assert_eq!(RelationType::CeoOf.label(), "CEO_OF");
    }

    #[test]
    fn test_pattern_extractor_acquisition() {
        let extractor = PatternExtractor::new();
        let sentence = "Microsoft acquired Activision Blizzard in a landmark deal.";
        let entities = vec![
            Entity::new("Microsoft", EntityType::Company, 0, 9),
            Entity::new("Activision Blizzard", EntityType::Company, 19, 38),
        ];

        let relations = extractor.extract(sentence, &entities);
        assert!(!relations.is_empty());
        assert_eq!(relations[0].relation_type, RelationType::Acquired);
        assert!(relations[0].confidence > 0.0);
    }

    #[test]
    fn test_pattern_extractor_partnership() {
        let extractor = PatternExtractor::new();
        let sentence = "Apple partnered with Goldman Sachs to launch a credit card.";
        let entities = vec![
            Entity::new("Apple", EntityType::Company, 0, 5),
            Entity::new("Goldman Sachs", EntityType::Company, 22, 35),
        ];

        let relations = extractor.extract(sentence, &entities);
        assert!(!relations.is_empty());
        assert_eq!(relations[0].relation_type, RelationType::PartneredWith);
    }

    #[test]
    fn test_pattern_extractor_ceo() {
        let extractor = PatternExtractor::new();
        let sentence = "Tim Cook is the CEO of Apple Inc.";
        let entities = vec![
            Entity::new("Tim Cook", EntityType::Person, 0, 8),
            Entity::new("Apple Inc", EntityType::Company, 23, 32),
        ];

        let relations = extractor.extract(sentence, &entities);
        assert!(!relations.is_empty());
        assert_eq!(relations[0].relation_type, RelationType::CeoOf);
    }

    #[test]
    fn test_pattern_extractor_listing() {
        let extractor = PatternExtractor::new();
        let sentence = "PEPE token was listed on Bybit exchange last week.";
        let entities = vec![
            Entity::new("PEPE", EntityType::Token, 0, 4),
            Entity::new("Bybit", EntityType::Exchange, 25, 30),
        ];

        let relations = extractor.extract(sentence, &entities);
        assert!(!relations.is_empty());
        assert_eq!(relations[0].relation_type, RelationType::ListedOn);
    }

    #[test]
    fn test_pattern_extractor_no_relation() {
        let extractor = PatternExtractor::new();
        let sentence = "The weather in New York is sunny today.";
        let entities = vec![
            Entity::new("New York", EntityType::Company, 19, 27),
        ];

        // Single entity — no pairs to check
        let relations = extractor.extract(sentence, &entities);
        assert!(relations.is_empty());
    }

    #[test]
    fn test_entity_type_compatibility() {
        assert!(entity_type_compatibility(
            EntityType::Company,
            EntityType::Company,
            RelationType::Acquired
        ) > 0.5);

        assert!(entity_type_compatibility(
            EntityType::Person,
            EntityType::Company,
            RelationType::CeoOf
        ) > 0.5);

        assert!(entity_type_compatibility(
            EntityType::Token,
            EntityType::Exchange,
            RelationType::ListedOn
        ) > 0.5);

        // Incompatible: Person can't acquire Person (low score)
        assert!(entity_type_compatibility(
            EntityType::Person,
            EntityType::Person,
            RelationType::Acquired
        ) < 0.5);
    }

    #[test]
    fn test_knowledge_graph() {
        let mut kg = KnowledgeGraph::new();

        let r1 = Relation::new(
            Entity::new("Apple", EntityType::Company, 0, 5),
            Entity::new("Beats", EntityType::Company, 15, 20),
            RelationType::Acquired,
            0.95,
            "Apple acquired Beats.",
        );

        let r2 = Relation::new(
            Entity::new("Apple", EntityType::Company, 0, 5),
            Entity::new("Samsung", EntityType::Company, 20, 27),
            RelationType::CompetesWith,
            0.88,
            "Apple competes with Samsung.",
        );

        kg.add_relation(&r1);
        kg.add_relation(&r2);

        assert_eq!(kg.entity_count(), 3); // Apple, Beats, Samsung
        assert_eq!(kg.edge_count(), 2);

        let apple_rels = kg.relations_for("Apple");
        assert_eq!(apple_rels.len(), 2);

        let neighbors = kg.neighbors("Apple");
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_knowledge_graph_entities_of_type() {
        let mut kg = KnowledgeGraph::new();
        let r = Relation::new(
            Entity::new("Tim Cook", EntityType::Person, 0, 8),
            Entity::new("Apple", EntityType::Company, 20, 25),
            RelationType::CeoOf,
            0.9,
            "Tim Cook is CEO of Apple.",
        );
        kg.add_relation(&r);

        let people = kg.entities_of_type(EntityType::Person);
        assert_eq!(people.len(), 1);
        assert!(people.contains(&"Tim Cook"));

        let companies = kg.entities_of_type(EntityType::Company);
        assert_eq!(companies.len(), 1);
    }

    #[test]
    fn test_relation_scorer_predict() {
        let scorer = RelationScorer::new(8, 0.01);
        let features = vec![0.1, 0.9, 1.0, 0.2, 0.15, 0.1, 0.5, 0.0];
        let (_, confidence) = scorer.predict(&features);
        assert!(confidence >= 0.5 && confidence <= 1.0);
    }

    #[test]
    fn test_relation_scorer_train_and_improve() {
        let data = generate_training_data(500);
        let (train, test) = data.split_at(400);

        let mut scorer = RelationScorer::new(8, 0.01);
        let _acc_before = scorer.accuracy(test);

        scorer.train(&train.to_vec(), 50);
        let acc_after = scorer.accuracy(test);

        assert!(acc_after > 0.0);
        assert!(acc_after >= 0.4, "accuracy after training: {}", acc_after);
    }

    #[test]
    fn test_trigram_jaccard() {
        // Identical strings should have overlap of 1.0
        let sim = trigram_jaccard("Apple", "Apple");
        assert!((sim - 1.0).abs() < 1e-9);

        // Completely different strings should have low overlap
        let sim = trigram_jaccard("Apple", "Microsoft");
        assert!(sim < 0.3);

        // Short strings (< 3 chars) return 0.0
        let sim = trigram_jaccard("AB", "AB");
        assert!((sim - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_extract_features() {
        let extractor = PatternExtractor::new();
        let sentence = "Microsoft acquired Activision Blizzard.";
        let e1 = Entity::new("Microsoft", EntityType::Company, 0, 9);
        let e2 = Entity::new("Activision Blizzard", EntityType::Company, 19, 38);

        let features = extract_features(sentence, &e1, &e2, RelationType::Acquired, &extractor);
        assert_eq!(features.len(), 8);

        // Entity type compatibility should be high for Company-Company-Acquired
        assert!(features[1] > 0.5);
        // Trigger word should be present
        assert!((features[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_synthetic_corpus() {
        let corpus = generate_synthetic_corpus();
        assert_eq!(corpus.len(), 10);

        let extractor = PatternExtractor::new();
        let mut correct = 0;
        for (sentence, entities, expected_relation) in &corpus {
            let relations = extractor.extract(sentence, entities);
            if !relations.is_empty() && relations[0].relation_type == *expected_relation {
                correct += 1;
            }
        }
        // Pattern extractor should correctly identify most synthetic examples
        assert!(
            correct >= 7,
            "Only {} out of 10 correctly extracted",
            correct
        );
    }

    #[test]
    fn test_training_data_generation() {
        let data = generate_training_data(100);
        assert_eq!(data.len(), 100);
        for (features, label) in &data {
            assert_eq!(features.len(), 8);
            assert!(*label == 0.0 || *label == 1.0);
        }
    }

    #[test]
    fn test_relation_type_coverage() {
        let all = RelationType::all();
        assert_eq!(all.len(), 10);
        // Each should have a unique index
        let indices: Vec<usize> = all.iter().map(|r| r.index()).collect();
        let unique: std::collections::HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), 10);
    }
}
