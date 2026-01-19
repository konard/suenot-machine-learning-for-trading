"""
LLM Earnings Call Analyzer

This module provides tools for analyzing earnings call transcripts
using Large Language Models to extract trading signals.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np


class SpeakerRole(Enum):
    """Role of a speaker in the earnings call"""
    CEO = "ceo"
    CFO = "cfo"
    ANALYST = "analyst"
    OPERATOR = "operator"
    OTHER = "other"


class SignalDirection(Enum):
    """Trading signal direction"""
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


@dataclass
class TranscriptSegment:
    """A segment of the earnings call transcript"""
    speaker: str
    role: SpeakerRole
    text: str
    section: str  # 'prepared_remarks' or 'qa'
    timestamp: Optional[str] = None


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    positive_phrases: List[str] = field(default_factory=list)
    negative_phrases: List[str] = field(default_factory=list)
    neutral_phrases: List[str] = field(default_factory=list)


@dataclass
class TradingSignal:
    """Trading signal from earnings analysis"""
    direction: SignalDirection
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    sentiment_score: float
    confidence_level: float
    guidance_direction: str
    qa_quality: float
    reasoning: str


class EarningsTranscriptParser:
    """
    Parser for earnings call transcripts

    Handles various transcript formats and extracts structured data
    """

    # Common patterns for speaker identification
    SPEAKER_PATTERNS = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–]\s*(.+)$',  # "John Smith - CEO"
        r'^([A-Z][a-z]+ [A-Z][a-z]+):',  # "John Smith:"
        r'^\[([A-Z][a-z]+ [A-Z][a-z]+)\]',  # "[John Smith]"
    ]

    CEO_KEYWORDS = ['ceo', 'chief executive', 'president']
    CFO_KEYWORDS = ['cfo', 'chief financial', 'finance']
    ANALYST_KEYWORDS = ['analyst', 'research', 'capital', 'securities', 'bank']

    def __init__(self):
        self.segments: List[TranscriptSegment] = []

    def parse(self, transcript: str) -> List[TranscriptSegment]:
        """
        Parse a transcript into segments

        Args:
            transcript: Raw transcript text

        Returns:
            List of TranscriptSegment objects
        """
        self.segments = []

        # Split into lines
        lines = transcript.split('\n')

        current_speaker = None
        current_role = SpeakerRole.OTHER
        current_text = []
        current_section = self._detect_initial_section(transcript)

        for line in lines:
            # Check for section markers
            if self._is_qa_start(line):
                # Save current segment
                if current_speaker and current_text:
                    self._add_segment(current_speaker, current_role,
                                    ' '.join(current_text), current_section)
                current_section = 'qa'
                current_text = []
                continue

            # Check for new speaker
            speaker_match = self._extract_speaker(line)
            if speaker_match:
                # Save previous segment
                if current_speaker and current_text:
                    self._add_segment(current_speaker, current_role,
                                    ' '.join(current_text), current_section)

                current_speaker, role_hint = speaker_match
                current_role = self._identify_role(current_speaker, role_hint)
                current_text = [self._clean_speaker_line(line)]
            else:
                # Continue current segment
                if line.strip():
                    current_text.append(line.strip())

        # Add final segment
        if current_speaker and current_text:
            self._add_segment(current_speaker, current_role,
                            ' '.join(current_text), current_section)

        return self.segments

    def _detect_initial_section(self, transcript: str) -> str:
        """Detect if transcript starts with prepared remarks or Q&A"""
        lower_text = transcript[:500].lower()
        if 'question' in lower_text and 'answer' in lower_text:
            return 'qa'
        return 'prepared_remarks'

    def _is_qa_start(self, line: str) -> bool:
        """Check if line marks the start of Q&A session"""
        qa_markers = [
            'question-and-answer',
            'question and answer',
            'q&a session',
            'we will now take questions',
            'open the floor for questions',
            'operator instructions'
        ]
        return any(marker in line.lower() for marker in qa_markers)

    def _extract_speaker(self, line: str) -> Optional[Tuple[str, str]]:
        """Extract speaker name and role hint from line"""
        for pattern in self.SPEAKER_PATTERNS:
            match = re.match(pattern, line)
            if match:
                groups = match.groups()
                speaker = groups[0]
                role_hint = groups[1] if len(groups) > 1 else ""
                return (speaker, role_hint)
        return None

    def _identify_role(self, speaker: str, role_hint: str) -> SpeakerRole:
        """Identify speaker role from name and context"""
        combined = f"{speaker} {role_hint}".lower()

        if any(kw in combined for kw in self.CEO_KEYWORDS):
            return SpeakerRole.CEO
        elif any(kw in combined for kw in self.CFO_KEYWORDS):
            return SpeakerRole.CFO
        elif any(kw in combined for kw in self.ANALYST_KEYWORDS):
            return SpeakerRole.ANALYST
        elif 'operator' in combined:
            return SpeakerRole.OPERATOR

        return SpeakerRole.OTHER

    def _clean_speaker_line(self, line: str) -> str:
        """Remove speaker identification from line"""
        for pattern in self.SPEAKER_PATTERNS:
            line = re.sub(pattern, '', line)
        return line.strip()

    def _add_segment(self, speaker: str, role: SpeakerRole,
                    text: str, section: str):
        """Add a segment to the list"""
        if text.strip():
            self.segments.append(TranscriptSegment(
                speaker=speaker,
                role=role,
                text=text,
                section=section
            ))

    def get_prepared_remarks(self) -> List[TranscriptSegment]:
        """Get only prepared remarks segments"""
        return [s for s in self.segments if s.section == 'prepared_remarks']

    def get_qa_segments(self) -> List[TranscriptSegment]:
        """Get only Q&A segments"""
        return [s for s in self.segments if s.section == 'qa']

    def get_management_segments(self) -> List[TranscriptSegment]:
        """Get segments from CEO/CFO"""
        return [s for s in self.segments
                if s.role in [SpeakerRole.CEO, SpeakerRole.CFO]]

    def get_analyst_questions(self) -> List[TranscriptSegment]:
        """Get analyst questions from Q&A"""
        return [s for s in self.segments
                if s.role == SpeakerRole.ANALYST and s.section == 'qa']


class EarningsSentimentAnalyzer:
    """
    Specialized sentiment analyzer for earnings calls

    Combines rule-based financial sentiment lexicon with optional LLM analysis
    """

    # Financial sentiment lexicon
    POSITIVE_TERMS = {
        'exceeded', 'outperformed', 'strong', 'robust', 'growth',
        'momentum', 'confident', 'optimistic', 'accelerating',
        'record', 'beat', 'exceeded expectations', 'raising guidance',
        'expanding margins', 'market share gains', 'solid', 'excellent',
        'outstanding', 'remarkable', 'significant improvement',
        'ahead of plan', 'better than expected', 'tailwinds'
    }

    NEGATIVE_TERMS = {
        'missed', 'underperformed', 'weak', 'challenging', 'headwinds',
        'slowdown', 'concerned', 'cautious', 'decelerating',
        'decline', 'lowering guidance', 'margin pressure',
        'competitive pressure', 'uncertain', 'difficult',
        'disappointing', 'below expectations', 'softness',
        'volatility', 'disruption', 'delay', 'risk'
    }

    HEDGING_TERMS = {
        'may', 'might', 'could', 'possibly', 'potentially',
        'subject to', 'depending on', 'uncertain', 'if',
        'assuming', 'contingent', 'volatile', 'somewhat',
        'relatively', 'approximately', 'around'
    }

    CONFIDENCE_TERMS = {
        'will', 'definitely', 'certainly', 'clearly', 'confident',
        'committed', 'expect', 'believe strongly', 'convinced',
        'absolutely', 'undoubtedly', 'firmly', 'sure'
    }

    def __init__(self, llm_client=None):
        """
        Initialize sentiment analyzer

        Args:
            llm_client: Optional OpenAI client for LLM-based analysis
        """
        self.llm_client = llm_client

    def analyze(self, text: str, use_llm: bool = False) -> SentimentResult:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze
            use_llm: Whether to use LLM for analysis (requires llm_client)

        Returns:
            SentimentResult with scores and phrases
        """
        # Rule-based analysis (always performed)
        rule_result = self._rule_based_sentiment(text)

        if use_llm and self.llm_client:
            # LLM analysis
            llm_result = self._llm_sentiment(text)
            # Combine results (weighted average)
            combined_score = 0.3 * rule_result.score + 0.7 * llm_result.score
            combined_confidence = min(rule_result.confidence, llm_result.confidence)

            return SentimentResult(
                score=combined_score,
                confidence=combined_confidence,
                positive_phrases=llm_result.positive_phrases,
                negative_phrases=llm_result.negative_phrases,
                neutral_phrases=llm_result.neutral_phrases
            )

        return rule_result

    def _rule_based_sentiment(self, text: str) -> SentimentResult:
        """Rule-based sentiment using financial lexicon"""
        text_lower = text.lower()

        positive_count = sum(1 for term in self.POSITIVE_TERMS
                           if term in text_lower)
        negative_count = sum(1 for term in self.NEGATIVE_TERMS
                           if term in text_lower)
        hedging_count = sum(1 for term in self.HEDGING_TERMS
                          if term in text_lower)
        confidence_count = sum(1 for term in self.CONFIDENCE_TERMS
                             if term in text_lower)

        # Calculate sentiment score
        total_sentiment = positive_count + negative_count
        if total_sentiment > 0:
            score = (positive_count - negative_count) / total_sentiment
        else:
            score = 0.0

        # Adjust for hedging
        hedging_penalty = min(hedging_count * 0.1, 0.3)
        confidence_boost = min(confidence_count * 0.1, 0.3)

        score = score - hedging_penalty + confidence_boost
        score = max(-1, min(1, score))  # Clamp to [-1, 1]

        # Calculate confidence based on signal clarity
        confidence = min(total_sentiment / 10, 1.0)

        # Extract found phrases
        positive_phrases = [term for term in self.POSITIVE_TERMS
                          if term in text_lower]
        negative_phrases = [term for term in self.NEGATIVE_TERMS
                          if term in text_lower]

        return SentimentResult(
            score=score,
            confidence=confidence,
            positive_phrases=list(positive_phrases),
            negative_phrases=list(negative_phrases),
            neutral_phrases=[]
        )

    def _llm_sentiment(self, text: str) -> SentimentResult:
        """LLM-based sentiment analysis"""
        prompt = f"""
        Analyze the sentiment of this earnings call excerpt.

        Text: {text[:2000]}

        Respond with JSON only:
        {{
            "sentiment_score": <float -1 to 1>,
            "confidence": <float 0 to 1>,
            "positive_phrases": ["phrase1", "phrase2"],
            "negative_phrases": ["phrase1", "phrase2"],
            "neutral_phrases": ["phrase1", "phrase2"]
        }}
        """

        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return SentimentResult(
            score=result['sentiment_score'],
            confidence=result['confidence'],
            positive_phrases=result.get('positive_phrases', []),
            negative_phrases=result.get('negative_phrases', []),
            neutral_phrases=result.get('neutral_phrases', [])
        )

    def analyze_confidence_level(self, text: str) -> float:
        """
        Analyze management confidence level

        Args:
            text: Text to analyze

        Returns:
            Confidence score between 0 and 1
        """
        text_lower = text.lower()

        hedging = sum(1 for term in self.HEDGING_TERMS if term in text_lower)
        confident = sum(1 for term in self.CONFIDENCE_TERMS if term in text_lower)

        total = hedging + confident
        if total == 0:
            return 0.5  # Neutral

        return confident / total


class EarningsSignalGenerator:
    """
    Generate trading signals from earnings call analysis
    """

    def __init__(self,
                 sentiment_weight: float = 0.35,
                 confidence_weight: float = 0.20,
                 guidance_weight: float = 0.30,
                 qa_weight: float = 0.15,
                 signal_threshold: float = 0.3):
        """
        Initialize signal generator

        Args:
            sentiment_weight: Weight for sentiment score
            confidence_weight: Weight for management confidence
            guidance_weight: Weight for guidance direction
            qa_weight: Weight for Q&A quality
            signal_threshold: Minimum score for non-neutral signal
        """
        self.weights = {
            'sentiment': sentiment_weight,
            'confidence': confidence_weight,
            'guidance': guidance_weight,
            'qa': qa_weight
        }
        self.signal_threshold = signal_threshold

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def generate_signal(self,
                       analysis: Dict,
                       historical_sentiment: Optional[float] = None) -> TradingSignal:
        """
        Generate trading signal from analysis results

        Args:
            analysis: Dictionary with analysis results containing:
                - overall_sentiment: dict with 'score' key
                - management_confidence: dict with 'score' key
                - guidance_assessment: dict with 'direction' key
                - qa_quality: dict with 'score' key
            historical_sentiment: Average sentiment from previous calls

        Returns:
            TradingSignal with direction and strength
        """
        # Extract components
        sentiment = analysis['overall_sentiment']['score']
        confidence = analysis['management_confidence']['score']
        guidance = self._guidance_to_score(analysis['guidance_assessment']['direction'])
        qa_quality = analysis['qa_quality']['score']

        # Adjust for historical baseline
        if historical_sentiment is not None:
            sentiment_delta = sentiment - historical_sentiment
            # Boost signal if sentiment improved/declined significantly
            sentiment = sentiment + 0.5 * sentiment_delta
            sentiment = max(-1, min(1, sentiment))

        # Calculate composite score
        composite = (
            self.weights['sentiment'] * sentiment +
            self.weights['confidence'] * (confidence - 0.5) * 2 +  # Center at 0
            self.weights['guidance'] * guidance +
            self.weights['qa'] * (qa_quality - 0.5) * 2  # Center at 0
        )

        # Determine direction
        if composite > self.signal_threshold:
            direction = SignalDirection.BULLISH
        elif composite < -self.signal_threshold:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        # Calculate strength (how far from threshold)
        if direction == SignalDirection.NEUTRAL:
            strength = abs(composite) / self.signal_threshold
        else:
            strength = min(abs(composite), 1.0)

        # Calculate confidence based on signal clarity
        signal_confidence = self._calculate_confidence(
            sentiment, confidence, qa_quality, analysis
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            sentiment, confidence, guidance, qa_quality, direction
        )

        return TradingSignal(
            direction=direction,
            strength=strength,
            confidence=signal_confidence,
            sentiment_score=sentiment,
            confidence_level=confidence,
            guidance_direction=analysis['guidance_assessment']['direction'],
            qa_quality=qa_quality,
            reasoning=reasoning
        )

    def _guidance_to_score(self, guidance: str) -> float:
        """Convert guidance direction to numeric score"""
        mapping = {
            'raised': 1.0,
            'maintained': 0.0,
            'lowered': -1.0,
            'not_provided': 0.0
        }
        return mapping.get(guidance.lower(), 0.0)

    def _calculate_confidence(self,
                             sentiment: float,
                             confidence: float,
                             qa_quality: float,
                             analysis: Dict) -> float:
        """Calculate confidence in the signal"""
        sentiment_clarity = abs(sentiment)

        # Check for conflicting signals
        signals = [sentiment > 0.3, confidence > 0.6, qa_quality > 0.6]
        signal_agreement = sum(signals) / len(signals)

        confidence_score = (
            0.3 * sentiment_clarity +
            0.3 * confidence +
            0.2 * qa_quality +
            0.2 * signal_agreement
        )

        return min(confidence_score, 1.0)

    def _generate_reasoning(self,
                           sentiment: float,
                           confidence: float,
                           guidance: float,
                           qa_quality: float,
                           direction: SignalDirection) -> str:
        """Generate human-readable reasoning for the signal"""
        reasons = []

        # Sentiment
        if sentiment > 0.3:
            reasons.append("positive overall sentiment")
        elif sentiment < -0.3:
            reasons.append("negative overall sentiment")
        else:
            reasons.append("neutral sentiment")

        # Confidence
        if confidence > 0.7:
            reasons.append("high management confidence")
        elif confidence < 0.3:
            reasons.append("management appears uncertain")

        # Guidance
        if guidance > 0:
            reasons.append("raised guidance")
        elif guidance < 0:
            reasons.append("lowered guidance")

        # Q&A
        if qa_quality > 0.7:
            reasons.append("transparent Q&A responses")
        elif qa_quality < 0.3:
            reasons.append("evasive Q&A responses")

        direction_text = {
            SignalDirection.BULLISH: "Bullish signal",
            SignalDirection.BEARISH: "Bearish signal",
            SignalDirection.NEUTRAL: "Neutral signal"
        }

        return f"{direction_text[direction]} based on: {', '.join(reasons)}."


def analyze_earnings_call_simple(transcript: str) -> Dict:
    """
    Simple interface to analyze an earnings call transcript

    This function provides a quick analysis using rule-based methods
    without requiring an LLM API key.

    Args:
        transcript: The earnings call transcript text

    Returns:
        Dictionary with analysis results
    """
    # Parse transcript
    parser = EarningsTranscriptParser()
    segments = parser.parse(transcript)

    # Analyze sentiment
    analyzer = EarningsSentimentAnalyzer()

    # Analyze prepared remarks
    prepared = parser.get_prepared_remarks()
    prepared_text = ' '.join([s.text for s in prepared])
    prepared_sentiment = analyzer.analyze(prepared_text)

    # Analyze Q&A
    qa = parser.get_qa_segments()
    qa_text = ' '.join([s.text for s in qa])
    qa_sentiment = analyzer.analyze(qa_text)

    # Management confidence
    management = parser.get_management_segments()
    management_text = ' '.join([s.text for s in management])
    confidence_level = analyzer.analyze_confidence_level(management_text)

    # Combine results
    overall_sentiment = 0.6 * prepared_sentiment.score + 0.4 * qa_sentiment.score

    # Build analysis dictionary
    analysis = {
        'overall_sentiment': {
            'score': overall_sentiment,
            'explanation': f"Based on prepared remarks ({prepared_sentiment.score:.2f}) and Q&A ({qa_sentiment.score:.2f})"
        },
        'management_confidence': {
            'score': confidence_level,
            'hedging_examples': list(analyzer.HEDGING_TERMS & set(management_text.lower().split())),
            'confidence_examples': list(analyzer.CONFIDENCE_TERMS & set(management_text.lower().split()))
        },
        'guidance_assessment': {
            'direction': 'not_provided',  # Would need more sophisticated analysis
            'revenue_guidance': None,
            'eps_guidance': None,
            'key_drivers': []
        },
        'qa_quality': {
            'score': min(qa_sentiment.confidence, 0.8),  # Proxy for Q&A quality
            'transparency_level': 'medium',
            'evasive_responses': []
        },
        'key_themes': prepared_sentiment.positive_phrases[:5],
        'risk_factors': prepared_sentiment.negative_phrases[:5],
        'segments_analyzed': len(segments),
        'prepared_remarks_segments': len(prepared),
        'qa_segments': len(qa)
    }

    return analysis


if __name__ == "__main__":
    # Example usage
    sample_transcript = """
    John Smith - CEO:
    Good morning everyone. We're pleased to report another strong quarter.
    Revenue grew 25% year over year, exceeding our expectations.
    We're confident about our growth trajectory and are raising guidance
    for the full year.

    Jane Doe - CFO:
    Our margins expanded significantly. We delivered robust free cash flow
    and our balance sheet remains solid. We expect continued momentum
    in the coming quarters.

    Question-and-Answer Session

    Analyst - Goldman Sachs:
    Can you elaborate on the competitive landscape?

    John Smith - CEO:
    We're seeing strong market share gains. Our differentiated products
    continue to resonate with customers.

    Analyst - Morgan Stanley:
    What about macro headwinds?

    Jane Doe - CFO:
    While there is some uncertainty in the macro environment, our
    business fundamentals remain strong. We're confident in our ability
    to navigate any challenges.
    """

    # Analyze
    results = analyze_earnings_call_simple(sample_transcript)

    print("=== Earnings Call Analysis ===")
    print(f"\nOverall Sentiment: {results['overall_sentiment']['score']:.2f}")
    print(f"Explanation: {results['overall_sentiment']['explanation']}")
    print(f"\nManagement Confidence: {results['management_confidence']['score']:.2f}")
    print(f"\nSegments Analyzed: {results['segments_analyzed']}")
    print(f"  - Prepared Remarks: {results['prepared_remarks_segments']}")
    print(f"  - Q&A: {results['qa_segments']}")
    print(f"\nPositive Themes: {results['key_themes']}")
    print(f"Risk Factors: {results['risk_factors']}")

    # Generate trading signal
    signal_gen = EarningsSignalGenerator()
    signal = signal_gen.generate_signal(results)

    print(f"\n=== Trading Signal ===")
    print(f"Direction: {signal.direction.value}")
    print(f"Strength: {signal.strength:.2f}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reasoning: {signal.reasoning}")
