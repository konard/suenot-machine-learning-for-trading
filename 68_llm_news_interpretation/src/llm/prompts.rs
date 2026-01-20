//! Prompt templates for LLM news analysis

use crate::news::NewsItem;

/// Prompt builder for news analysis
#[derive(Debug, Clone)]
pub struct PromptBuilder {
    /// System context for the LLM
    system_context: String,
    /// Target assets to focus on
    target_assets: Vec<String>,
    /// Analysis depth level
    depth: AnalysisDepth,
}

/// Analysis depth level
#[derive(Debug, Clone, Copy, Default)]
pub enum AnalysisDepth {
    /// Quick sentiment only
    Quick,
    /// Standard analysis
    #[default]
    Standard,
    /// Deep analysis with reasoning
    Deep,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self {
            system_context: DEFAULT_SYSTEM_CONTEXT.to_string(),
            target_assets: vec![],
            depth: AnalysisDepth::Standard,
        }
    }
}

impl PromptBuilder {
    /// Create a new prompt builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set custom system context
    pub fn with_system_context(mut self, context: &str) -> Self {
        self.system_context = context.to_string();
        self
    }

    /// Set target assets to focus on
    pub fn with_target_assets(mut self, assets: Vec<String>) -> Self {
        self.target_assets = assets;
        self
    }

    /// Set analysis depth
    pub fn with_depth(mut self, depth: AnalysisDepth) -> Self {
        self.depth = depth;
        self
    }

    /// Build analysis prompt for a single news item
    pub fn build_single_analysis(&self, news: &NewsItem) -> String {
        let asset_focus = if self.target_assets.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nFocus especially on impact to: {}",
                self.target_assets.join(", ")
            )
        };

        let depth_instructions = match self.depth {
            AnalysisDepth::Quick => QUICK_ANALYSIS_INSTRUCTIONS,
            AnalysisDepth::Standard => STANDARD_ANALYSIS_INSTRUCTIONS,
            AnalysisDepth::Deep => DEEP_ANALYSIS_INSTRUCTIONS,
        };

        format!(
            r#"Analyze the following cryptocurrency/financial news article:

TITLE: {}
SOURCE: {:?}
PUBLISHED: {}

CONTENT:
{}

{}{}

Respond with a JSON object containing your analysis."#,
            news.title,
            news.source,
            news.published_at,
            news.content,
            depth_instructions,
            asset_focus
        )
    }

    /// Build batch analysis prompt for multiple news items
    pub fn build_batch_analysis(&self, news_items: &[NewsItem]) -> String {
        let news_list: String = news_items
            .iter()
            .enumerate()
            .map(|(i, n)| {
                format!(
                    "{}. [{}] {} - {}",
                    i + 1,
                    n.source.name(),
                    n.title,
                    n.content.chars().take(200).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        format!(
            r#"Analyze the following batch of {} cryptocurrency/financial news items and provide an aggregate market sentiment assessment:

NEWS ITEMS:
{}

For each significant news item, assess:
1. Individual sentiment (-1.0 to +1.0)
2. Market impact (low/medium/high)
3. Affected assets

Then provide:
- Overall market sentiment
- Key themes
- Recommended portfolio actions

Respond with a JSON object."#,
            news_items.len(),
            news_list
        )
    }

    /// Build event classification prompt
    pub fn build_event_classification(&self, text: &str) -> String {
        format!(
            r#"Classify the following news text into one of these event categories:

CATEGORIES:
- regulatory: Government/regulatory actions, laws, compliance
- partnership: Business partnerships, collaborations, integrations
- technical: Technical updates, upgrades, network changes
- security: Hacks, exploits, vulnerabilities, security incidents
- market: Market movements, trading activity, price action
- adoption: Corporate/institutional adoption, new users
- macro: Macroeconomic events, global financial news
- sentiment: Market sentiment, social trends, influencer activity
- unknown: Cannot determine

TEXT:
{}

Respond with JSON: {{"event_type": "category", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"#,
            text
        )
    }

    /// Build entity extraction prompt
    pub fn build_entity_extraction(&self, text: &str) -> String {
        format!(
            r#"Extract all relevant entities from the following cryptocurrency/financial news text:

TEXT:
{}

Extract:
1. Cryptocurrencies/tokens mentioned (with symbols if known)
2. Companies/organizations mentioned
3. People mentioned (with roles if known)
4. Monetary amounts
5. Dates/timeframes
6. Locations/jurisdictions

Respond with JSON: {{"cryptocurrencies": [], "organizations": [], "people": [], "amounts": [], "dates": [], "locations": []}}"#,
            text
        )
    }

    /// Get the system context
    pub fn system_context(&self) -> &str {
        &self.system_context
    }
}

/// Default system context for financial analysis
const DEFAULT_SYSTEM_CONTEXT: &str = r#"You are an expert financial analyst specializing in cryptocurrency markets. Your role is to:

1. Analyze news articles and social media posts for trading-relevant information
2. Assess market sentiment on a scale from -1.0 (extremely bearish) to +1.0 (extremely bullish)
3. Identify affected assets and potential market impact
4. Provide actionable insights for trading decisions

Guidelines:
- Be objective and data-driven
- Consider both immediate and longer-term implications
- Account for source reliability
- Identify potential market manipulation or fake news
- Always provide confidence scores for your assessments

Output format: Always respond with valid JSON."#;

/// Quick analysis instructions
const QUICK_ANALYSIS_INSTRUCTIONS: &str = r#"Provide a quick sentiment analysis:

Required JSON fields:
{
  "sentiment": <-1.0 to 1.0>,
  "confidence": <0.0 to 1.0>,
  "affected_assets": [<list of ticker symbols>]
}"#;

/// Standard analysis instructions
const STANDARD_ANALYSIS_INSTRUCTIONS: &str = r#"Provide a comprehensive analysis:

Required JSON fields:
{
  "sentiment": <-1.0 to 1.0>,
  "confidence": <0.0 to 1.0>,
  "event_type": <"regulatory"|"partnership"|"technical"|"security"|"market"|"adoption"|"macro"|"sentiment"|"unknown">,
  "affected_assets": [<list of ticker symbols>],
  "entities": [<key entities mentioned>],
  "summary": "<one sentence summary>",
  "market_impact": <"low"|"medium"|"high">,
  "time_horizon": <"immediate"|"short_term"|"long_term">
}"#;

/// Deep analysis instructions
const DEEP_ANALYSIS_INSTRUCTIONS: &str = r#"Provide an in-depth analysis with reasoning:

Required JSON fields:
{
  "sentiment": <-1.0 to 1.0>,
  "confidence": <0.0 to 1.0>,
  "event_type": <"regulatory"|"partnership"|"technical"|"security"|"market"|"adoption"|"macro"|"sentiment"|"unknown">,
  "affected_assets": [<list of ticker symbols>],
  "entities": [<key entities mentioned>],
  "summary": "<one sentence summary>",
  "market_impact": <"low"|"medium"|"high">,
  "time_horizon": <"immediate"|"short_term"|"long_term">,
  "reasoning": "<detailed explanation of your analysis>",
  "risks": [<potential risks or caveats>],
  "opportunities": [<potential trading opportunities>],
  "related_events": [<similar historical events if applicable>],
  "recommended_action": <"strong_buy"|"buy"|"hold"|"sell"|"strong_sell">
}"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::news::NewsSource;
    use chrono::Utc;

    fn create_test_news() -> NewsItem {
        NewsItem {
            id: "test-1".to_string(),
            title: "Bitcoin ETF Approved by SEC".to_string(),
            content: "The SEC has finally approved the first spot Bitcoin ETF...".to_string(),
            source: NewsSource::CoinDesk,
            url: Some("https://example.com/news".to_string()),
            published_at: Utc::now(),
            symbols: vec!["BTC".to_string()],
            relevance_score: 0.9,
        }
    }

    #[test]
    fn test_build_single_analysis() {
        let builder = PromptBuilder::new();
        let news = create_test_news();
        let prompt = builder.build_single_analysis(&news);

        assert!(prompt.contains("Bitcoin ETF Approved"));
        assert!(prompt.contains("SEC has finally approved"));
    }

    #[test]
    fn test_with_target_assets() {
        let builder = PromptBuilder::new()
            .with_target_assets(vec!["BTC".to_string(), "ETH".to_string()]);
        let news = create_test_news();
        let prompt = builder.build_single_analysis(&news);

        assert!(prompt.contains("BTC"));
        assert!(prompt.contains("ETH"));
    }

    #[test]
    fn test_build_event_classification() {
        let builder = PromptBuilder::new();
        let prompt = builder.build_event_classification("SEC approves Bitcoin ETF");

        assert!(prompt.contains("regulatory"));
        assert!(prompt.contains("event_type"));
    }

    #[test]
    fn test_build_entity_extraction() {
        let builder = PromptBuilder::new();
        let prompt = builder.build_entity_extraction("Binance CEO announces new partnership");

        assert!(prompt.contains("organizations"));
        assert!(prompt.contains("people"));
    }
}
