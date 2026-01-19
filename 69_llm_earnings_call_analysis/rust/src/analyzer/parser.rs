//! Transcript parsing module
//!
//! Parses earnings call transcripts into structured segments.

use regex::Regex;

/// Role of a speaker in the earnings call
#[derive(Debug, Clone, PartialEq)]
pub enum SpeakerRole {
    CEO,
    CFO,
    Analyst,
    Operator,
    Other,
}

/// A segment of the transcript
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub speaker: String,
    pub role: SpeakerRole,
    pub text: String,
    pub section: String,
}

/// Parser for earnings call transcripts
pub struct TranscriptParser {
    speaker_patterns: Vec<Regex>,
    ceo_keywords: Vec<&'static str>,
    cfo_keywords: Vec<&'static str>,
    analyst_keywords: Vec<&'static str>,
}

impl TranscriptParser {
    /// Create a new transcript parser
    pub fn new() -> Self {
        Self {
            speaker_patterns: vec![
                Regex::new(r"^([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–]\s*(.+)$").unwrap(),
                Regex::new(r"^([A-Z][a-z]+ [A-Z][a-z]+):").unwrap(),
                Regex::new(r"^\[([A-Z][a-z]+ [A-Z][a-z]+)\]").unwrap(),
            ],
            ceo_keywords: vec!["ceo", "chief executive", "president"],
            cfo_keywords: vec!["cfo", "chief financial", "finance"],
            analyst_keywords: vec!["analyst", "research", "capital", "securities", "bank"],
        }
    }

    /// Parse a transcript into segments
    pub fn parse(&self, transcript: &str) -> Vec<TranscriptSegment> {
        let mut segments = Vec::new();
        let mut current_speaker: Option<String> = None;
        let mut current_role = SpeakerRole::Other;
        let mut current_text = Vec::new();
        let mut current_section = self.detect_initial_section(transcript);

        for line in transcript.lines() {
            // Check for Q&A section marker
            if self.is_qa_start(line) {
                // Save current segment
                if let Some(speaker) = &current_speaker {
                    if !current_text.is_empty() {
                        segments.push(TranscriptSegment {
                            speaker: speaker.clone(),
                            role: current_role.clone(),
                            text: current_text.join(" "),
                            section: current_section.clone(),
                        });
                    }
                }
                current_section = "qa".to_string();
                current_text.clear();
                continue;
            }

            // Check for new speaker
            if let Some((speaker, role_hint)) = self.extract_speaker(line) {
                // Save previous segment
                if let Some(prev_speaker) = &current_speaker {
                    if !current_text.is_empty() {
                        segments.push(TranscriptSegment {
                            speaker: prev_speaker.clone(),
                            role: current_role.clone(),
                            text: current_text.join(" "),
                            section: current_section.clone(),
                        });
                    }
                }

                current_speaker = Some(speaker.clone());
                current_role = self.identify_role(&speaker, &role_hint);
                current_text = vec![self.clean_speaker_line(line)];
            } else {
                // Continue current segment
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    current_text.push(trimmed.to_string());
                }
            }
        }

        // Add final segment
        if let Some(speaker) = current_speaker {
            if !current_text.is_empty() {
                segments.push(TranscriptSegment {
                    speaker,
                    role: current_role,
                    text: current_text.join(" "),
                    section: current_section,
                });
            }
        }

        segments
    }

    /// Detect initial section from transcript beginning
    fn detect_initial_section(&self, transcript: &str) -> String {
        let start = &transcript[..transcript.len().min(500)].to_lowercase();
        if start.contains("question") && start.contains("answer") {
            "qa".to_string()
        } else {
            "prepared_remarks".to_string()
        }
    }

    /// Check if line marks start of Q&A session
    fn is_qa_start(&self, line: &str) -> bool {
        let qa_markers = [
            "question-and-answer",
            "question and answer",
            "q&a session",
            "we will now take questions",
            "open the floor for questions",
            "operator instructions",
        ];

        let line_lower = line.to_lowercase();
        qa_markers.iter().any(|marker| line_lower.contains(marker))
    }

    /// Extract speaker name and role hint from line
    fn extract_speaker(&self, line: &str) -> Option<(String, String)> {
        for pattern in &self.speaker_patterns {
            if let Some(caps) = pattern.captures(line) {
                let speaker = caps.get(1)?.as_str().to_string();
                let role_hint = caps.get(2)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                return Some((speaker, role_hint));
            }
        }
        None
    }

    /// Identify speaker role from name and context
    fn identify_role(&self, speaker: &str, role_hint: &str) -> SpeakerRole {
        let combined = format!("{} {}", speaker, role_hint).to_lowercase();

        if self.ceo_keywords.iter().any(|kw| combined.contains(kw)) {
            SpeakerRole::CEO
        } else if self.cfo_keywords.iter().any(|kw| combined.contains(kw)) {
            SpeakerRole::CFO
        } else if self.analyst_keywords.iter().any(|kw| combined.contains(kw)) {
            SpeakerRole::Analyst
        } else if combined.contains("operator") {
            SpeakerRole::Operator
        } else {
            SpeakerRole::Other
        }
    }

    /// Remove speaker identification from line
    fn clean_speaker_line(&self, line: &str) -> String {
        let mut result = line.to_string();
        for pattern in &self.speaker_patterns {
            result = pattern.replace(&result, "").to_string();
        }
        result.trim().to_string()
    }
}

impl Default for TranscriptParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_transcript() {
        let parser = TranscriptParser::new();

        let transcript = r#"
John Smith - CEO:
Welcome to our earnings call. We had a great quarter.

Jane Doe - CFO:
Revenue grew 25% year over year.

Question-and-Answer Session

Analyst - Goldman Sachs:
Can you comment on margins?

John Smith - CEO:
Margins expanded significantly.
        "#;

        let segments = parser.parse(transcript);

        assert!(!segments.is_empty());

        // Check for CEO segment
        let ceo_segments: Vec<_> = segments.iter()
            .filter(|s| s.role == SpeakerRole::CEO)
            .collect();
        assert!(!ceo_segments.is_empty());

        // Check for Q&A section
        let qa_segments: Vec<_> = segments.iter()
            .filter(|s| s.section == "qa")
            .collect();
        assert!(!qa_segments.is_empty());
    }

    #[test]
    fn test_speaker_identification() {
        let parser = TranscriptParser::new();

        let (speaker, hint) = parser.extract_speaker("John Smith - CEO:").unwrap();
        assert_eq!(speaker, "John Smith");

        let role = parser.identify_role(&speaker, &hint);
        assert_eq!(role, SpeakerRole::CEO);
    }
}
