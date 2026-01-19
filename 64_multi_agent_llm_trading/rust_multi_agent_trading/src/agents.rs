//! Agent definitions and implementations.
//!
//! This module provides the core agent trait and various specialized agents
//! for multi-agent trading systems.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

use crate::data::MarketData;
use crate::error::{Result, TradingError};

/// Trading signal types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display, EnumString)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum Signal {
    StrongBuy,
    Buy,
    Neutral,
    Sell,
    StrongSell,
}

impl Signal {
    /// Convert signal to numeric value for aggregation.
    pub fn to_numeric(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Neutral => 0.0,
            Signal::Sell => -0.5,
            Signal::StrongSell => -1.0,
        }
    }

    /// Create signal from numeric score.
    pub fn from_score(score: f64) -> Self {
        if score > 0.5 {
            Signal::StrongBuy
        } else if score > 0.2 {
            Signal::Buy
        } else if score < -0.5 {
            Signal::StrongSell
        } else if score < -0.2 {
            Signal::Sell
        } else {
            Signal::Neutral
        }
    }
}

/// Container for agent analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    pub agent_name: String,
    pub agent_type: String,
    pub symbol: String,
    pub signal: Signal,
    pub confidence: f64,
    pub reasoning: String,
    pub metrics: HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
}

impl Analysis {
    /// Create a new analysis result.
    pub fn new(
        agent_name: &str,
        agent_type: &str,
        symbol: &str,
        signal: Signal,
        confidence: f64,
        reasoning: &str,
    ) -> Self {
        Self {
            agent_name: agent_name.to_string(),
            agent_type: agent_type.to_string(),
            symbol: symbol.to_string(),
            signal,
            confidence: confidence.clamp(0.0, 1.0),
            reasoning: reasoning.to_string(),
            metrics: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    /// Add a metric to the analysis.
    pub fn with_metric(mut self, key: &str, value: f64) -> Self {
        self.metrics.insert(key.to_string(), value);
        self
    }

    /// Add multiple metrics.
    pub fn with_metrics(mut self, metrics: HashMap<String, f64>) -> Self {
        self.metrics.extend(metrics);
        self
    }
}

/// Context for agent analysis.
#[derive(Debug, Clone, Default)]
pub struct AnalysisContext {
    pub fundamentals: Option<HashMap<String, f64>>,
    pub sentiment: Option<HashMap<String, f64>>,
    pub news: Option<Vec<String>>,
    pub previous_analyses: Option<Vec<Analysis>>,
    pub custom: HashMap<String, String>,
}

impl AnalysisContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_fundamentals(mut self, fundamentals: HashMap<String, f64>) -> Self {
        self.fundamentals = Some(fundamentals);
        self
    }

    pub fn with_sentiment(mut self, sentiment: HashMap<String, f64>) -> Self {
        self.sentiment = Some(sentiment);
        self
    }

    pub fn with_news(mut self, news: Vec<String>) -> Self {
        self.news = Some(news);
        self
    }
}

/// Trait for all trading agents.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Get the agent's name.
    fn name(&self) -> &str;

    /// Get the agent's type.
    fn agent_type(&self) -> &str;

    /// Perform analysis on market data.
    async fn analyze(
        &self,
        symbol: &str,
        data: &MarketData,
        context: Option<&AnalysisContext>,
    ) -> Result<Analysis>;
}

/// Technical analysis agent.
pub struct TechnicalAgent {
    name: String,
}

impl TechnicalAgent {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl Agent for TechnicalAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> &str {
        "technical"
    }

    async fn analyze(
        &self,
        symbol: &str,
        data: &MarketData,
        _context: Option<&AnalysisContext>,
    ) -> Result<Analysis> {
        if data.len() < 50 {
            return Err(TradingError::InsufficientData {
                required: 50,
                actual: data.len(),
            });
        }

        let closes = data.close_prices();
        let current_price = *closes.last().unwrap();

        // Calculate indicators
        let sma_20 = data.sma(20);
        let sma_50 = data.sma(50);
        let rsi = data.rsi(14);

        let current_sma_20 = *sma_20.last().unwrap_or(&current_price);
        let current_sma_50 = *sma_50.last().unwrap_or(&current_price);
        let current_rsi = *rsi.last().unwrap_or(&50.0);

        // Scoring
        let mut score = 0;
        let mut reasons = Vec::new();

        // RSI analysis
        if current_rsi < 30.0 {
            score += 2;
            reasons.push(format!("RSI oversold ({:.1})", current_rsi));
        } else if current_rsi > 70.0 {
            score -= 2;
            reasons.push(format!("RSI overbought ({:.1})", current_rsi));
        }

        // MA analysis
        if current_price > current_sma_20 && current_sma_20 > current_sma_50 {
            score += 2;
            reasons.push("Price above rising MAs (bullish trend)".to_string());
        } else if current_price < current_sma_20 && current_sma_20 < current_sma_50 {
            score -= 2;
            reasons.push("Price below falling MAs (bearish trend)".to_string());
        }

        // Determine signal
        let signal = if score >= 3 {
            Signal::StrongBuy
        } else if score >= 1 {
            Signal::Buy
        } else if score <= -3 {
            Signal::StrongSell
        } else if score <= -1 {
            Signal::Sell
        } else {
            Signal::Neutral
        };

        let confidence = (score.abs() as f64 / 5.0 + 0.4).min(0.95);

        let mut metrics = HashMap::new();
        metrics.insert("rsi".to_string(), current_rsi);
        metrics.insert("sma_20".to_string(), current_sma_20);
        metrics.insert("sma_50".to_string(), current_sma_50);
        metrics.insert("score".to_string(), score as f64);

        Ok(Analysis::new(
            &self.name,
            "technical",
            symbol,
            signal,
            confidence,
            &reasons.join("; "),
        )
        .with_metrics(metrics))
    }
}

/// Bull researcher agent (optimistic).
pub struct BullAgent {
    name: String,
}

impl BullAgent {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl Agent for BullAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> &str {
        "bull_researcher"
    }

    async fn analyze(
        &self,
        symbol: &str,
        data: &MarketData,
        _context: Option<&AnalysisContext>,
    ) -> Result<Analysis> {
        let closes = data.close_prices();
        let mut confidence_boost = 0.0;
        let mut reasons = Vec::new();

        // Look for bullish signals
        if closes.len() >= 30 {
            let returns_30d = closes.last().unwrap() / closes[closes.len() - 30] - 1.0;

            if returns_30d > 0.0 {
                reasons.push(format!(
                    "Price up {:.1}% over 30 days - positive momentum",
                    returns_30d * 100.0
                ));
                confidence_boost += 0.1;
            } else {
                reasons.push("Recent pullback provides attractive entry point".to_string());
            }
        }

        // Always find something positive
        reasons.push("Long-term fundamentals remain intact".to_string());

        let signal = if confidence_boost > 0.1 {
            Signal::StrongBuy
        } else {
            Signal::Buy
        };

        let confidence = (0.7 + confidence_boost).min(0.95);

        Ok(Analysis::new(
            &self.name,
            "bull_researcher",
            symbol,
            signal,
            confidence,
            &reasons.join(" | "),
        ))
    }
}

/// Bear researcher agent (pessimistic).
pub struct BearAgent {
    name: String,
}

impl BearAgent {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl Agent for BearAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> &str {
        "bear_researcher"
    }

    async fn analyze(
        &self,
        symbol: &str,
        data: &MarketData,
        _context: Option<&AnalysisContext>,
    ) -> Result<Analysis> {
        let closes = data.close_prices();
        let mut confidence_boost = 0.0;
        let mut reasons = Vec::new();

        // Look for bearish signals
        if closes.len() >= 30 {
            let returns_30d = closes.last().unwrap() / closes[closes.len() - 30] - 1.0;

            if returns_30d > 0.2 {
                reasons.push(format!(
                    "Up {:.1}% in 30 days - potentially overextended",
                    returns_30d * 100.0
                ));
                confidence_boost += 0.1;
            }
        }

        // Volatility risk
        let volatility = data.volatility(30);
        if let Some(&vol) = volatility.last() {
            if vol > 0.3 {
                reasons.push(format!(
                    "High volatility ({:.0}% annualized) indicates risk",
                    vol * 100.0
                ));
                confidence_boost += 0.05;
            }
        }

        // Always find something negative
        reasons.push("Macro environment remains uncertain".to_string());

        let signal = if confidence_boost > 0.1 {
            Signal::StrongSell
        } else {
            Signal::Sell
        };

        let confidence = (0.7 + confidence_boost).min(0.95);

        Ok(Analysis::new(
            &self.name,
            "bear_researcher",
            symbol,
            signal,
            confidence,
            &reasons.join(" | "),
        ))
    }
}

/// Risk manager agent.
pub struct RiskManagerAgent {
    name: String,
    max_position_pct: f64,
    max_drawdown: f64,
}

impl RiskManagerAgent {
    pub fn new(name: &str, max_position_pct: f64, max_drawdown: f64) -> Self {
        Self {
            name: name.to_string(),
            max_position_pct,
            max_drawdown,
        }
    }
}

#[async_trait]
impl Agent for RiskManagerAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> &str {
        "risk_manager"
    }

    async fn analyze(
        &self,
        symbol: &str,
        data: &MarketData,
        _context: Option<&AnalysisContext>,
    ) -> Result<Analysis> {
        let volatility = data.volatility(30);
        let current_vol = volatility.last().copied().unwrap_or(0.2);
        let max_dd = data.max_drawdown();

        let mut risk_score = 0;
        let mut reasons = Vec::new();

        // Volatility assessment
        if current_vol > 0.5 {
            reasons.push(format!(
                "Very high volatility ({:.0}%) - reduce position size",
                current_vol * 100.0
            ));
            risk_score += 2;
        } else if current_vol > 0.3 {
            reasons.push(format!("High volatility ({:.0}%) - caution advised", current_vol * 100.0));
            risk_score += 1;
        } else {
            reasons.push(format!("Acceptable volatility ({:.0}%)", current_vol * 100.0));
        }

        // Drawdown assessment
        if max_dd > 0.3 {
            reasons.push(format!(
                "History of large drawdowns ({:.0}%)",
                max_dd * 100.0
            ));
            risk_score += 1;
        }

        // Position size recommendation
        let recommended_size = if risk_score >= 3 {
            self.max_position_pct * 0.25
        } else if risk_score >= 2 {
            self.max_position_pct * 0.5
        } else if risk_score >= 1 {
            self.max_position_pct * 0.75
        } else {
            self.max_position_pct
        };

        reasons.push(format!(
            "Recommended position size: {:.1}% of portfolio",
            recommended_size * 100.0
        ));

        let signal = match risk_score {
            0 => Signal::Buy,
            1 => Signal::Neutral,
            2 => Signal::Sell,
            _ => Signal::StrongSell,
        };

        let mut metrics = HashMap::new();
        metrics.insert("volatility".to_string(), current_vol);
        metrics.insert("max_drawdown".to_string(), max_dd);
        metrics.insert("risk_score".to_string(), risk_score as f64);
        metrics.insert("recommended_position_pct".to_string(), recommended_size);

        Ok(Analysis::new(
            &self.name,
            "risk_manager",
            symbol,
            signal,
            0.85,
            &reasons.join("; "),
        )
        .with_metrics(metrics))
    }
}

/// Trader agent that aggregates other agents' analyses.
pub struct TraderAgent {
    name: String,
    weights: HashMap<String, f64>,
}

impl TraderAgent {
    pub fn new(name: &str) -> Self {
        let mut weights = HashMap::new();
        weights.insert("technical".to_string(), 0.25);
        weights.insert("fundamental".to_string(), 0.20);
        weights.insert("sentiment".to_string(), 0.15);
        weights.insert("news".to_string(), 0.15);
        weights.insert("bull_researcher".to_string(), 0.10);
        weights.insert("bear_researcher".to_string(), 0.10);
        weights.insert("risk_manager".to_string(), 0.05);

        Self {
            name: name.to_string(),
            weights,
        }
    }

    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.weights = weights;
        self
    }

    /// Aggregate multiple analyses into a final decision.
    pub fn aggregate(&self, symbol: &str, analyses: &[Analysis]) -> Analysis {
        if analyses.is_empty() {
            return Analysis::new(
                &self.name,
                "trader",
                symbol,
                Signal::Neutral,
                0.3,
                "No analyses available for aggregation",
            );
        }

        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;

        for analysis in analyses {
            let weight = self.weights.get(&analysis.agent_type).copied().unwrap_or(0.1);
            let contribution = analysis.signal.to_numeric() * weight * analysis.confidence;
            weighted_score += contribution;
            total_weight += weight;
        }

        let normalized_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        };

        let signal = Signal::from_score(normalized_score);

        let bullish_count = analyses
            .iter()
            .filter(|a| matches!(a.signal, Signal::Buy | Signal::StrongBuy))
            .count();
        let bearish_count = analyses
            .iter()
            .filter(|a| matches!(a.signal, Signal::Sell | Signal::StrongSell))
            .count();

        let reasoning = format!(
            "Weighted score: {:.2}; Bullish agents: {}; Bearish agents: {}",
            normalized_score, bullish_count, bearish_count
        );

        let confidence = 0.5 + normalized_score.abs() * 0.4;

        let mut metrics = HashMap::new();
        metrics.insert("weighted_score".to_string(), normalized_score);
        metrics.insert("bullish_count".to_string(), bullish_count as f64);
        metrics.insert("bearish_count".to_string(), bearish_count as f64);

        Analysis::new(&self.name, "trader", symbol, signal, confidence, &reasoning)
            .with_metrics(metrics)
    }
}

#[async_trait]
impl Agent for TraderAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> &str {
        "trader"
    }

    async fn analyze(
        &self,
        symbol: &str,
        _data: &MarketData,
        context: Option<&AnalysisContext>,
    ) -> Result<Analysis> {
        let analyses = context
            .and_then(|c| c.previous_analyses.as_ref())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);

        Ok(self.aggregate(symbol, analyses))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::create_mock_data;

    #[tokio::test]
    async fn test_technical_agent() {
        let agent = TechnicalAgent::new("Test-Tech");
        let data = create_mock_data("TEST", 100, 100.0);

        let result = agent.analyze("TEST", &data, None).await;
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.agent_type, "technical");
        assert!((0.0..=1.0).contains(&analysis.confidence));
    }

    #[tokio::test]
    async fn test_bull_agent() {
        let agent = BullAgent::new("Test-Bull");
        let data = create_mock_data("TEST", 100, 100.0);

        let result = agent.analyze("TEST", &data, None).await;
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(matches!(analysis.signal, Signal::Buy | Signal::StrongBuy));
    }

    #[tokio::test]
    async fn test_bear_agent() {
        let agent = BearAgent::new("Test-Bear");
        let data = create_mock_data("TEST", 100, 100.0);

        let result = agent.analyze("TEST", &data, None).await;
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(matches!(analysis.signal, Signal::Sell | Signal::StrongSell));
    }

    #[test]
    fn test_signal_conversion() {
        assert_eq!(Signal::StrongBuy.to_numeric(), 1.0);
        assert_eq!(Signal::Neutral.to_numeric(), 0.0);
        assert_eq!(Signal::StrongSell.to_numeric(), -1.0);

        assert!(matches!(Signal::from_score(0.8), Signal::StrongBuy));
        assert!(matches!(Signal::from_score(0.0), Signal::Neutral));
        assert!(matches!(Signal::from_score(-0.8), Signal::StrongSell));
    }
}
