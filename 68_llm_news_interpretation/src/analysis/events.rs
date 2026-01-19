//! Event classification for news content

use serde::{Deserialize, Serialize};

/// Types of events that can be classified from news
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventType {
    /// Regulatory actions, laws, compliance
    Regulatory,
    /// Business partnerships and collaborations
    Partnership,
    /// Technical updates and upgrades
    Technical,
    /// Security incidents, hacks, exploits
    Security,
    /// Market movements and trading activity
    Market,
    /// Adoption by institutions or corporations
    Adoption,
    /// Macroeconomic events
    Macro,
    /// Market sentiment and social trends
    Sentiment,
    /// Cannot determine event type
    Unknown,
}

impl EventType {
    /// Parse event type from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "regulatory" | "regulation" | "legal" => EventType::Regulatory,
            "partnership" | "collaboration" | "integration" => EventType::Partnership,
            "technical" | "upgrade" | "update" | "development" => EventType::Technical,
            "security" | "hack" | "exploit" | "vulnerability" => EventType::Security,
            "market" | "price" | "trading" => EventType::Market,
            "adoption" | "institutional" | "corporate" => EventType::Adoption,
            "macro" | "economic" | "global" => EventType::Macro,
            "sentiment" | "social" | "trend" => EventType::Sentiment,
            _ => EventType::Unknown,
        }
    }

    /// Get typical market impact for this event type
    pub fn typical_impact(&self) -> MarketImpact {
        match self {
            EventType::Regulatory => MarketImpact::High,
            EventType::Partnership => MarketImpact::Medium,
            EventType::Technical => MarketImpact::Medium,
            EventType::Security => MarketImpact::High,
            EventType::Market => MarketImpact::Medium,
            EventType::Adoption => MarketImpact::High,
            EventType::Macro => MarketImpact::High,
            EventType::Sentiment => MarketImpact::Low,
            EventType::Unknown => MarketImpact::Low,
        }
    }

    /// Get typical time horizon for market impact
    pub fn time_horizon(&self) -> TimeHorizon {
        match self {
            EventType::Regulatory => TimeHorizon::LongTerm,
            EventType::Partnership => TimeHorizon::MediumTerm,
            EventType::Technical => TimeHorizon::MediumTerm,
            EventType::Security => TimeHorizon::ShortTerm,
            EventType::Market => TimeHorizon::ShortTerm,
            EventType::Adoption => TimeHorizon::LongTerm,
            EventType::Macro => TimeHorizon::LongTerm,
            EventType::Sentiment => TimeHorizon::ShortTerm,
            EventType::Unknown => TimeHorizon::Unknown,
        }
    }

    /// Get event type name
    pub fn name(&self) -> &'static str {
        match self {
            EventType::Regulatory => "Regulatory",
            EventType::Partnership => "Partnership",
            EventType::Technical => "Technical",
            EventType::Security => "Security",
            EventType::Market => "Market",
            EventType::Adoption => "Adoption",
            EventType::Macro => "Macro",
            EventType::Sentiment => "Sentiment",
            EventType::Unknown => "Unknown",
        }
    }
}

impl Default for EventType {
    fn default() -> Self {
        EventType::Unknown
    }
}

/// Expected market impact level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MarketImpact {
    /// Low impact, minimal price movement expected
    Low,
    /// Medium impact, moderate price movement possible
    Medium,
    /// High impact, significant price movement likely
    High,
}

impl MarketImpact {
    /// Get numeric multiplier for impact
    pub fn multiplier(&self) -> f64 {
        match self {
            MarketImpact::Low => 0.5,
            MarketImpact::Medium => 1.0,
            MarketImpact::High => 2.0,
        }
    }
}

/// Time horizon for expected impact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeHorizon {
    /// Immediate impact (minutes to hours)
    ShortTerm,
    /// Medium-term impact (days to weeks)
    MediumTerm,
    /// Long-term impact (weeks to months)
    LongTerm,
    /// Unknown time horizon
    Unknown,
}

/// Event classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedEvent {
    /// Primary event type
    pub event_type: EventType,
    /// Secondary event types if applicable
    pub secondary_types: Vec<EventType>,
    /// Confidence in classification
    pub confidence: f64,
    /// Expected market impact
    pub impact: MarketImpact,
    /// Time horizon for impact
    pub time_horizon: TimeHorizon,
}

/// Event classifier using rule-based and LLM approaches
#[derive(Debug, Default)]
pub struct EventClassifier {
    /// Keyword mappings for each event type
    keyword_map: Vec<(EventType, Vec<String>)>,
}

impl EventClassifier {
    /// Create a new event classifier
    pub fn new() -> Self {
        Self {
            keyword_map: Self::default_keyword_map(),
        }
    }

    /// Classify event type from text using rules
    pub fn classify(&self, text: &str) -> ClassifiedEvent {
        let text_lower = text.to_lowercase();
        let mut scores: Vec<(EventType, f64)> = Vec::new();

        for (event_type, keywords) in &self.keyword_map {
            let mut score = 0.0;
            for keyword in keywords {
                if text_lower.contains(keyword) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                scores.push((*event_type, score));
            }
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (event_type, primary_score) = scores.first().copied().unwrap_or((EventType::Unknown, 0.0));

        let secondary_types: Vec<EventType> = scores
            .iter()
            .skip(1)
            .take(2)
            .map(|(t, _)| *t)
            .collect();

        // Calculate confidence based on score distribution
        let total_score: f64 = scores.iter().map(|(_, s)| s).sum();
        let confidence = if total_score > 0.0 {
            (primary_score / total_score * 0.8).min(0.95)
        } else {
            0.1
        };

        ClassifiedEvent {
            event_type,
            secondary_types,
            confidence,
            impact: event_type.typical_impact(),
            time_horizon: event_type.time_horizon(),
        }
    }

    /// Get default keyword mappings
    fn default_keyword_map() -> Vec<(EventType, Vec<String>)> {
        vec![
            (
                EventType::Regulatory,
                vec![
                    "sec".to_string(),
                    "cftc".to_string(),
                    "regulation".to_string(),
                    "regulatory".to_string(),
                    "law".to_string(),
                    "legal".to_string(),
                    "compliance".to_string(),
                    "ban".to_string(),
                    "approve".to_string(),
                    "license".to_string(),
                    "lawsuit".to_string(),
                    "court".to_string(),
                ],
            ),
            (
                EventType::Partnership,
                vec![
                    "partnership".to_string(),
                    "partner".to_string(),
                    "collaboration".to_string(),
                    "integrate".to_string(),
                    "integration".to_string(),
                    "alliance".to_string(),
                    "joint venture".to_string(),
                ],
            ),
            (
                EventType::Technical,
                vec![
                    "upgrade".to_string(),
                    "update".to_string(),
                    "fork".to_string(),
                    "mainnet".to_string(),
                    "testnet".to_string(),
                    "protocol".to_string(),
                    "development".to_string(),
                    "release".to_string(),
                    "version".to_string(),
                    "improvement".to_string(),
                ],
            ),
            (
                EventType::Security,
                vec![
                    "hack".to_string(),
                    "exploit".to_string(),
                    "vulnerability".to_string(),
                    "breach".to_string(),
                    "attack".to_string(),
                    "stolen".to_string(),
                    "drain".to_string(),
                    "security".to_string(),
                    "bug".to_string(),
                ],
            ),
            (
                EventType::Market,
                vec![
                    "price".to_string(),
                    "trading".to_string(),
                    "volume".to_string(),
                    "liquidation".to_string(),
                    "futures".to_string(),
                    "options".to_string(),
                    "spot".to_string(),
                    "whale".to_string(),
                ],
            ),
            (
                EventType::Adoption,
                vec![
                    "adoption".to_string(),
                    "institutional".to_string(),
                    "corporate".to_string(),
                    "accepts".to_string(),
                    "accepts bitcoin".to_string(),
                    "treasury".to_string(),
                    "etf".to_string(),
                ],
            ),
            (
                EventType::Macro,
                vec![
                    "fed".to_string(),
                    "interest rate".to_string(),
                    "inflation".to_string(),
                    "gdp".to_string(),
                    "economy".to_string(),
                    "recession".to_string(),
                    "stimulus".to_string(),
                ],
            ),
            (
                EventType::Sentiment,
                vec![
                    "sentiment".to_string(),
                    "fear".to_string(),
                    "greed".to_string(),
                    "fomo".to_string(),
                    "fud".to_string(),
                    "trending".to_string(),
                    "viral".to_string(),
                ],
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_from_str() {
        assert_eq!(EventType::from_str("regulatory"), EventType::Regulatory);
        assert_eq!(EventType::from_str("HACK"), EventType::Security);
        assert_eq!(EventType::from_str("unknown_type"), EventType::Unknown);
    }

    #[test]
    fn test_classify_regulatory() {
        let classifier = EventClassifier::new();
        let result = classifier.classify("SEC approves Bitcoin ETF application");

        assert_eq!(result.event_type, EventType::Regulatory);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_classify_security() {
        let classifier = EventClassifier::new();
        let result = classifier.classify("Major DeFi protocol hacked, $50M stolen");

        assert_eq!(result.event_type, EventType::Security);
        assert_eq!(result.impact, MarketImpact::High);
    }

    #[test]
    fn test_market_impact_multiplier() {
        assert_eq!(MarketImpact::Low.multiplier(), 0.5);
        assert_eq!(MarketImpact::Medium.multiplier(), 1.0);
        assert_eq!(MarketImpact::High.multiplier(), 2.0);
    }
}
