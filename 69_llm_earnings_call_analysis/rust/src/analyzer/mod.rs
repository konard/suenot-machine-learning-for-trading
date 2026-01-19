//! Earnings call analysis modules
//!
//! This module provides tools for parsing and analyzing earnings call transcripts.

pub mod parser;
pub mod sentiment;

pub use parser::{TranscriptParser, TranscriptSegment, SpeakerRole};
pub use sentiment::{SentimentAnalyzer, SentimentResult};

/// Main earnings analyzer combining parsing and sentiment analysis
pub struct EarningsAnalyzer {
    parser: TranscriptParser,
    sentiment_analyzer: SentimentAnalyzer,
}

impl EarningsAnalyzer {
    /// Create a new earnings analyzer
    pub fn new() -> Self {
        Self {
            parser: TranscriptParser::new(),
            sentiment_analyzer: SentimentAnalyzer::new(),
        }
    }

    /// Analyze an earnings call transcript
    pub fn analyze(&self, transcript: &str) -> EarningsAnalysis {
        // Parse transcript into segments
        let segments = self.parser.parse(transcript);

        // Separate segments by section
        let prepared_remarks: Vec<_> = segments.iter()
            .filter(|s| s.section == "prepared_remarks")
            .collect();

        let qa_segments: Vec<_> = segments.iter()
            .filter(|s| s.section == "qa")
            .collect();

        let management_segments: Vec<_> = segments.iter()
            .filter(|s| matches!(s.role, SpeakerRole::CEO | SpeakerRole::CFO))
            .collect();

        // Analyze sentiment for each section
        let prepared_text: String = prepared_remarks.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let qa_text: String = qa_segments.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let management_text: String = management_segments.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let prepared_sentiment = self.sentiment_analyzer.analyze(&prepared_text);
        let qa_sentiment = self.sentiment_analyzer.analyze(&qa_text);
        let management_sentiment = self.sentiment_analyzer.analyze(&management_text);

        // Calculate overall sentiment (weighted)
        let overall_net = if !prepared_text.is_empty() && !qa_text.is_empty() {
            0.6 * prepared_sentiment.score + 0.4 * qa_sentiment.score
        } else if !prepared_text.is_empty() {
            prepared_sentiment.score
        } else if !qa_text.is_empty() {
            qa_sentiment.score
        } else {
            0.0
        };

        // Calculate confidence level
        let confidence_level = management_sentiment.confidence;

        EarningsAnalysis {
            overall_sentiment: SentimentScore {
                net_sentiment: overall_net,
                positive_score: prepared_sentiment.positive_score,
                negative_score: prepared_sentiment.negative_score,
                hedging_score: prepared_sentiment.hedging_score,
            },
            confidence: ConfidenceScore {
                overall: confidence_level,
                hedging_count: prepared_sentiment.hedging_words.len(),
                confidence_count: prepared_sentiment.confidence_words.len(),
            },
            guidance: GuidanceAssessment {
                direction: self.detect_guidance(&prepared_text),
                has_quantitative: self.has_quantitative_guidance(&prepared_text),
            },
            qa_quality: QAQuality {
                responsiveness: qa_sentiment.confidence,
                transparency: if qa_sentiment.hedging_score < 0.05 {
                    "high".to_string()
                } else if qa_sentiment.hedging_score < 0.10 {
                    "medium".to_string()
                } else {
                    "low".to_string()
                },
            },
            key_themes: prepared_sentiment.positive_phrases,
            risk_factors: prepared_sentiment.negative_phrases,
            segments_analyzed: segments.len(),
            prepared_remarks_count: prepared_remarks.len(),
            qa_segments_count: qa_segments.len(),
        }
    }

    /// Detect guidance direction from text
    fn detect_guidance(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        if text_lower.contains("raising guidance") ||
           text_lower.contains("raise our guidance") ||
           text_lower.contains("raising our guidance") ||
           text_lower.contains("increasing our outlook") ||
           text_lower.contains("raise guidance") {
            "raised".to_string()
        } else if text_lower.contains("lowering guidance") ||
                  text_lower.contains("lower our guidance") ||
                  text_lower.contains("lowering our guidance") ||
                  text_lower.contains("reducing our outlook") ||
                  text_lower.contains("lower guidance") {
            "lowered".to_string()
        } else if text_lower.contains("maintaining guidance") ||
                  text_lower.contains("reaffirming our guidance") ||
                  text_lower.contains("maintain our guidance") {
            "maintained".to_string()
        } else {
            "not_provided".to_string()
        }
    }

    /// Check if text contains quantitative guidance
    fn has_quantitative_guidance(&self, text: &str) -> bool {
        // Simple check for numbers followed by guidance-related words
        let has_numbers = text.chars().any(|c| c.is_ascii_digit());
        let has_guidance_words = text.to_lowercase().contains("guidance") ||
                                  text.to_lowercase().contains("outlook") ||
                                  text.to_lowercase().contains("expect") ||
                                  text.to_lowercase().contains("forecast");
        has_numbers && has_guidance_words
    }
}

impl Default for EarningsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete earnings analysis result
#[derive(Debug, Clone)]
pub struct EarningsAnalysis {
    pub overall_sentiment: SentimentScore,
    pub confidence: ConfidenceScore,
    pub guidance: GuidanceAssessment,
    pub qa_quality: QAQuality,
    pub key_themes: Vec<String>,
    pub risk_factors: Vec<String>,
    pub segments_analyzed: usize,
    pub prepared_remarks_count: usize,
    pub qa_segments_count: usize,
}

/// Sentiment score with breakdown
#[derive(Debug, Clone)]
pub struct SentimentScore {
    pub net_sentiment: f64,
    pub positive_score: f64,
    pub negative_score: f64,
    pub hedging_score: f64,
}

/// Confidence score with details
#[derive(Debug, Clone)]
pub struct ConfidenceScore {
    pub overall: f64,
    pub hedging_count: usize,
    pub confidence_count: usize,
}

/// Guidance assessment
#[derive(Debug, Clone)]
pub struct GuidanceAssessment {
    pub direction: String,
    pub has_quantitative: bool,
}

/// Q&A quality assessment
#[derive(Debug, Clone)]
pub struct QAQuality {
    pub responsiveness: f64,
    pub transparency: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_earnings_analyzer() {
        let analyzer = EarningsAnalyzer::new();

        let transcript = r#"
John Smith - CEO:
Good morning. We delivered strong results this quarter with revenue
exceeding expectations. We're confident about our growth trajectory
and are raising guidance for the full year.

Question-and-Answer Session

Analyst:
What about competitive pressures?

John Smith - CEO:
We continue to see strong demand and market share gains.
        "#;

        let analysis = analyzer.analyze(transcript);

        assert!(analysis.overall_sentiment.net_sentiment > 0.0);
        assert!(analysis.segments_analyzed > 0);
    }

    #[test]
    fn test_guidance_detection() {
        let analyzer = EarningsAnalyzer::new();

        assert_eq!(analyzer.detect_guidance("We are raising guidance"), "raised");
        assert_eq!(analyzer.detect_guidance("We are lowering guidance"), "lowered");
        assert_eq!(analyzer.detect_guidance("We are maintaining guidance"), "maintained");
        assert_eq!(analyzer.detect_guidance("No guidance mentioned"), "not_provided");
    }

    #[test]
    fn test_bearish_analysis() {
        let analyzer = EarningsAnalyzer::new();

        let transcript = r#"
Jane Doe - CFO:
We faced significant challenges this quarter.
Revenue declined due to headwinds and uncertainty.
We are lowering guidance given the weak outlook.
        "#;

        let analysis = analyzer.analyze(transcript);

        // Should have negative sentiment
        assert!(analysis.overall_sentiment.negative_score > 0.0);
        assert_eq!(analysis.guidance.direction, "lowered");
    }
}
