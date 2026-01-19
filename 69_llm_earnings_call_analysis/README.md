# Chapter 69: LLM Earnings Call Analysis — Extracting Trading Signals from Corporate Communications

## Overview

Earnings calls are quarterly conference calls where public companies discuss financial results with analysts and investors. These calls contain rich information about company performance, management sentiment, forward guidance, and market expectations. Large Language Models (LLMs) can analyze these transcripts to extract actionable trading signals that were previously difficult to quantify.

This chapter explores how to apply LLMs to earnings call analysis for cryptocurrency and stock trading, using advanced NLP techniques to extract sentiment, detect management confidence, identify key themes, and generate probabilistic trading signals.

## Core Concepts

### What is Earnings Call Analysis?

Earnings calls consist of two main parts:
1. **Prepared Remarks**: Management's scripted presentation of financial results
2. **Q&A Session**: Analysts ask questions, management responds spontaneously

```
Earnings Call Structure:
├── Opening (CFO/CEO introduction)
├── Financial Results (scripted, typically positive framing)
├── Forward Guidance (future expectations)
├── Q&A Session (unscripted, more revealing)
│   ├── Analyst questions (probing concerns)
│   └── Management responses (confidence indicators)
└── Closing remarks

LLM Analysis Focus:
├── Sentiment polarity (positive/negative/neutral)
├── Confidence markers (hedging language vs. assertive)
├── Guidance changes (beat/meet/miss expectations)
├── Risk disclosure (new risks mentioned)
└── Tone shifts (comparing to previous calls)
```

### Why LLMs for Earnings Call Analysis?

1. **Contextual Understanding**: LLMs understand financial jargon and context
2. **Nuance Detection**: Capture subtle sentiment shifts human analysts might miss
3. **Scale**: Process hundreds of earnings calls efficiently
4. **Consistency**: Apply uniform analysis across all transcripts
5. **Multi-factor Extraction**: Extract multiple signals simultaneously
6. **Q&A Analysis**: Evaluate management responsiveness and transparency

### Key Linguistic Features

```
Confidence Indicators:
├── Strong: "We are confident", "clearly", "definitely"
├── Moderate: "We believe", "we expect", "likely"
├── Weak: "We hope", "potentially", "might"
└── Hedging: "Subject to", "uncertain", "challenging"

Sentiment Markers:
├── Positive: "Strong performance", "exceeded expectations"
├── Negative: "Headwinds", "challenging environment"
├── Neutral: "In line with", "as expected"
└── Mixed: "Despite challenges, we managed to..."

Forward-Looking Signals:
├── Bullish: "Raising guidance", "accelerating growth"
├── Bearish: "Lowering expectations", "cautious outlook"
└── Neutral: "Maintaining guidance", "stable outlook"
```

## Trading Strategy

**Strategy Overview:** Use LLM to analyze earnings call transcripts and generate trading signals based on:
1. Overall sentiment score
2. Management confidence level
3. Guidance direction (raise/maintain/lower)
4. Q&A transparency score
5. Comparison to previous quarters

### Signal Generation

```
1. Transcript Processing:
   - Parse transcript into sections (prepared remarks, Q&A)
   - Clean and normalize text
   - Identify speakers and their roles

2. LLM Analysis:
   - Extract sentiment scores per section
   - Identify confidence markers
   - Detect guidance changes
   - Analyze Q&A quality

3. Signal Aggregation:
   - Weight signals by importance
   - Compare to analyst consensus
   - Generate final trading signal

4. Risk Assessment:
   - Evaluate uncertainty in analysis
   - Check for conflicting signals
   - Adjust position size accordingly
```

### Entry Signals

- **Long Signal**: High positive sentiment + Strong confidence + Raised guidance + Good Q&A
- **Short Signal**: Negative sentiment + Hedging language + Lowered guidance + Evasive Q&A
- **Hold Signal**: Mixed signals or neutral sentiment

### Risk Management

- **Confidence Threshold**: Only trade when LLM confidence > threshold
- **Signal Strength**: Scale position size with signal strength
- **Earnings Volatility**: Account for post-earnings price movements
- **Stop Loss**: Use historical earnings reaction for stop placement

## Technical Specification

### Mathematical Foundation

#### Sentiment Scoring

```
Sentiment Score Calculation:
├── Section-level sentiment: S_i ∈ [-1, 1]
├── Section weights: w_i (prepared remarks, Q&A, guidance)
├── Aggregate sentiment: S = Σ(w_i × S_i) / Σ(w_i)
│
├── Confidence adjustment:
│   S_adj = S × confidence_factor
│   where confidence_factor ∈ [0.5, 1.5]
│
└── Historical normalization:
    S_normalized = (S_adj - μ_historical) / σ_historical
```

#### Signal Strength

```
Signal Strength Components:
├── Sentiment magnitude: |S_normalized|
├── Guidance change: G ∈ {-1, 0, 1}
├── Confidence level: C ∈ [0, 1]
├── Q&A quality: Q ∈ [0, 1]
│
Signal = w_s × S_normalized + w_g × G + w_c × C + w_q × Q

Position sizing:
Position = base_size × tanh(Signal × scale_factor)
```

### Architecture Diagram

```
                    Earnings Call Transcript
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Transcript Parser        │
            │  ├── Section detection      │
            │  ├── Speaker identification │
            │  ├── Q&A pairing            │
            │  └── Text normalization     │
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │      LLM Analysis Engine    │
            │                             │
            │  ┌───────────────────────┐  │
            │  │ Sentiment Extraction  │  │
            │  │ - Section sentiment   │  │
            │  │ - Entity sentiment    │  │
            │  │ - Temporal changes    │  │
            │  └───────────┬───────────┘  │
            │              │              │
            │  ┌───────────────────────┐  │
            │  │ Confidence Analysis   │  │
            │  │ - Hedging detection   │  │
            │  │ - Certainty markers   │  │
            │  │ - Tone analysis       │  │
            │  └───────────┬───────────┘  │
            │              │              │
            │  ┌───────────────────────┐  │
            │  │ Guidance Extraction   │  │
            │  │ - Revenue guidance    │  │
            │  │ - EPS guidance        │  │
            │  │ - Direction change    │  │
            │  └───────────┬───────────┘  │
            │              │              │
            │  ┌───────────────────────┐  │
            │  │ Q&A Quality Scoring   │  │
            │  │ - Responsiveness      │  │
            │  │ - Transparency        │  │
            │  │ - Evasion detection   │  │
            │  └───────────────────────┘  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │  Sentiment  │ │ Confidence  │ │  Guidance   │
     │   Score     │ │   Level     │ │  Direction  │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌─────────────────────────────┐
            │     Signal Aggregation      │
            │  ├── Weight combination     │
            │  ├── Historical comparison  │
            │  ├── Consensus adjustment   │
            │  └── Confidence scaling     │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │     Trading Decision        │
            │  ├── Signal direction       │
            │  ├── Position sizing        │
            │  ├── Entry timing           │
            │  └── Stop loss placement    │
            └─────────────────────────────┘
```

### Prompt Engineering for Earnings Analysis

```python
import json
from openai import OpenAI

EARNINGS_ANALYSIS_PROMPT = """
You are an expert financial analyst specializing in earnings call analysis.
Analyze the following earnings call transcript and provide a structured assessment.

Transcript:
{transcript}

Provide your analysis in the following JSON format:
{{
    "overall_sentiment": {{
        "score": <float between -1 and 1>,
        "explanation": "<brief explanation>"
    }},
    "management_confidence": {{
        "score": <float between 0 and 1>,
        "hedging_examples": ["<example phrases>"],
        "confidence_examples": ["<example phrases>"]
    }},
    "guidance_assessment": {{
        "direction": "<raised|maintained|lowered|not_provided>",
        "revenue_guidance": "<specific guidance if mentioned>",
        "eps_guidance": "<specific guidance if mentioned>",
        "key_drivers": ["<main factors mentioned>"]
    }},
    "qa_quality": {{
        "score": <float between 0 and 1>,
        "transparency_level": "<high|medium|low>",
        "evasive_responses": ["<questions that were dodged>"]
    }},
    "key_themes": ["<main topics discussed>"],
    "risk_factors": ["<new or emphasized risks>"],
    "trading_signal": {{
        "direction": "<bullish|neutral|bearish>",
        "strength": <float between 0 and 1>,
        "reasoning": "<brief explanation>"
    }}
}}
"""

def analyze_earnings_call(transcript: str, client: OpenAI) -> dict:
    """
    Analyze an earnings call transcript using LLM

    Args:
        transcript: The earnings call transcript text
        client: OpenAI client instance

    Returns:
        Structured analysis dictionary
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an expert financial analyst. Provide precise, actionable analysis."
            },
            {
                "role": "user",
                "content": EARNINGS_ANALYSIS_PROMPT.format(transcript=transcript)
            }
        ],
        temperature=0.1,  # Low temperature for consistent analysis
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

### Transcript Parsing

```python
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class SpeakerRole(Enum):
    CEO = "ceo"
    CFO = "cfo"
    ANALYST = "analyst"
    OPERATOR = "operator"
    OTHER = "other"

@dataclass
class TranscriptSegment:
    """A segment of the earnings call transcript"""
    speaker: str
    role: SpeakerRole
    text: str
    section: str  # 'prepared_remarks' or 'qa'
    timestamp: Optional[str] = None

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
```

### Sentiment Analysis Module

```python
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    positive_phrases: List[str]
    negative_phrases: List[str]
    neutral_phrases: List[str]

class EarningsSentimentAnalyzer:
    """
    Specialized sentiment analyzer for earnings calls

    Combines LLM analysis with rule-based financial sentiment
    """

    # Financial sentiment lexicon (subset)
    POSITIVE_TERMS = {
        'exceeded', 'outperformed', 'strong', 'robust', 'growth',
        'momentum', 'confident', 'optimistic', 'accelerating',
        'record', 'beat', 'exceeded expectations', 'raising guidance',
        'expanding margins', 'market share gains'
    }

    NEGATIVE_TERMS = {
        'missed', 'underperformed', 'weak', 'challenging', 'headwinds',
        'slowdown', 'concerned', 'cautious', 'decelerating',
        'decline', 'lowering guidance', 'margin pressure',
        'competitive pressure', 'uncertain', 'difficult'
    }

    HEDGING_TERMS = {
        'may', 'might', 'could', 'possibly', 'potentially',
        'subject to', 'depending on', 'uncertain', 'if',
        'assuming', 'contingent', 'volatile'
    }

    CONFIDENCE_TERMS = {
        'will', 'definitely', 'certainly', 'clearly', 'confident',
        'committed', 'expect', 'believe strongly', 'convinced'
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def analyze(self, text: str, use_llm: bool = True) -> SentimentResult:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze
            use_llm: Whether to use LLM for analysis

        Returns:
            SentimentResult with scores and phrases
        """
        # Rule-based analysis
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
        words = text_lower.split()

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

        # Extract phrases
        positive_phrases = [term for term in self.POSITIVE_TERMS
                          if term in text_lower]
        negative_phrases = [term for term in self.NEGATIVE_TERMS
                          if term in text_lower]

        return SentimentResult(
            score=score,
            confidence=confidence,
            positive_phrases=positive_phrases,
            negative_phrases=negative_phrases,
            neutral_phrases=[]
        )

    def _llm_sentiment(self, text: str) -> SentimentResult:
        """LLM-based sentiment analysis"""
        prompt = f"""
        Analyze the sentiment of this earnings call excerpt.

        Text: {text[:2000]}  # Truncate for API limits

        Respond with JSON:
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
```

### Trading Signal Generator

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

class SignalDirection(Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"

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
        Generate trading signal from LLM analysis

        Args:
            analysis: Dictionary from LLM analysis
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
        # Higher confidence when:
        # 1. Sentiment is clear (not neutral)
        # 2. Management appears confident
        # 3. Q&A was transparent
        # 4. Few conflicting signals

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
```

### Backtesting Framework

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

@dataclass
class BacktestResult:
    """Results from backtesting earnings strategy"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return_per_trade: float
    num_trades: int
    trades: pd.DataFrame

class EarningsBacktest:
    """
    Backtest LLM earnings call analysis strategy
    """

    def __init__(self,
                 hold_period: int = 5,  # Days to hold after earnings
                 position_size: float = 0.1,  # Fraction of portfolio
                 signal_threshold: float = 0.5):
        """
        Initialize backtest

        Args:
            hold_period: Number of days to hold position
            position_size: Base position size as fraction
            signal_threshold: Minimum signal strength to trade
        """
        self.hold_period = hold_period
        self.position_size = position_size
        self.signal_threshold = signal_threshold

    def run(self,
            signals: List[TradingSignal],
            earnings_dates: List[datetime],
            price_data: pd.DataFrame,
            symbols: List[str]) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            signals: List of trading signals
            earnings_dates: Dates of earnings calls
            price_data: DataFrame with columns for each symbol
            symbols: List of stock symbols

        Returns:
            BacktestResult with performance metrics
        """
        trades = []
        portfolio_value = 1.0
        peak_value = 1.0
        max_drawdown = 0.0

        for i, (signal, date, symbol) in enumerate(zip(signals, earnings_dates, symbols)):
            # Skip weak signals
            if signal.strength < self.signal_threshold:
                continue

            # Skip if signal direction is neutral
            if signal.direction == SignalDirection.NEUTRAL:
                continue

            # Get entry and exit prices
            entry_date = date + timedelta(days=1)  # Enter day after earnings
            exit_date = entry_date + timedelta(days=self.hold_period)

            try:
                entry_price = price_data.loc[entry_date, symbol]
                exit_price = price_data.loc[exit_date, symbol]
            except KeyError:
                continue  # Skip if dates not in data

            # Calculate return
            if signal.direction == SignalDirection.BULLISH:
                trade_return = (exit_price - entry_price) / entry_price
            else:  # BEARISH
                trade_return = (entry_price - exit_price) / entry_price

            # Scale by position size and signal strength
            position = self.position_size * signal.strength * signal.confidence
            portfolio_return = trade_return * position

            portfolio_value *= (1 + portfolio_return)

            # Track drawdown
            peak_value = max(peak_value, portfolio_value)
            drawdown = (peak_value - portfolio_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)

            trades.append({
                'date': date,
                'symbol': symbol,
                'direction': signal.direction.value,
                'signal_strength': signal.strength,
                'confidence': signal.confidence,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'trade_return': trade_return,
                'portfolio_return': portfolio_return,
                'portfolio_value': portfolio_value
            })

        trades_df = pd.DataFrame(trades)

        # Calculate metrics
        if len(trades_df) > 0:
            returns = trades_df['portfolio_return']
            total_return = portfolio_value - 1

            # Annualized Sharpe (assuming ~4 earnings per year per stock)
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252 / self.hold_period)
            else:
                sharpe = 0.0

            # Sortino
            downside = returns[returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = returns.mean() / downside.std() * np.sqrt(252 / self.hold_period)
            else:
                sortino = 0.0

            win_rate = (returns > 0).mean()
            avg_return = returns.mean()
        else:
            total_return = 0
            sharpe = 0
            sortino = 0
            win_rate = 0
            avg_return = 0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_return_per_trade=avg_return,
            num_trades=len(trades_df),
            trades=trades_df
        )

    def calculate_metrics(self, results: BacktestResult) -> Dict:
        """
        Calculate detailed performance metrics

        Args:
            results: BacktestResult from run()

        Returns:
            Dictionary of performance metrics
        """
        trades = results.trades

        if len(trades) == 0:
            return {'error': 'No trades executed'}

        # Direction analysis
        bullish_trades = trades[trades['direction'] == 'bullish']
        bearish_trades = trades[trades['direction'] == 'bearish']

        metrics = {
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'num_trades': results.num_trades,
            'avg_return_per_trade': results.avg_return_per_trade,

            # Directional breakdown
            'bullish_trades': len(bullish_trades),
            'bearish_trades': len(bearish_trades),
            'bullish_win_rate': (bullish_trades['trade_return'] > 0).mean() if len(bullish_trades) > 0 else 0,
            'bearish_win_rate': (bearish_trades['trade_return'] > 0).mean() if len(bearish_trades) > 0 else 0,

            # Signal quality
            'avg_signal_strength': trades['signal_strength'].mean(),
            'avg_confidence': trades['confidence'].mean(),

            # Correlation between signal and return
            'signal_return_correlation': trades['signal_strength'].corr(trades['trade_return'])
        }

        return metrics
```

## Data Requirements

```
Earnings Call Data:
├── Source: Company IR websites, SEC EDGAR, data providers
├── Format: Text transcripts, audio files
├── Frequency: Quarterly (4x per year per company)
├── History: 2+ years recommended for backtesting
│
Required Fields:
├── Company identifier (ticker, CUSIP)
├── Earnings date and time
├── Full transcript text
├── Speaker identification
├── Section markers (prepared/Q&A)
│
Price Data:
├── Source: Yahoo Finance, Polygon.io, Alpha Vantage
├── Frequency: Daily OHLCV
├── Pre/post earnings prices
└── Volume data for liquidity checks

Preprocessing:
├── Text cleaning and normalization
├── Speaker role identification
├── Section segmentation
├── Date alignment with price data
└── Missing data handling
```

## Cryptocurrency Application

### Applying to Crypto Markets

While traditional earnings calls don't exist for cryptocurrencies, similar analyses can be applied to:

1. **Project Updates**: Quarterly/monthly development updates
2. **AMA Sessions**: Ask Me Anything with project teams
3. **Governance Calls**: DAO governance discussions
4. **Partnership Announcements**: Major collaboration announcements
5. **Protocol Upgrade Discussions**: Technical upgrade explanations

### Bybit Integration for Crypto Data

```python
import requests
from typing import List, Dict
from datetime import datetime

class BybitClient:
    """
    Client for fetching Bybit market data
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def get_klines(self,
                   symbol: str,
                   interval: str,
                   limit: int = 200) -> List[Dict]:
        """
        Get candlestick data from Bybit

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1h", "4h", "1d")
            limit: Number of candles to fetch

        Returns:
            List of candlestick dictionaries
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        candles = []
        for item in data['result']['list']:
            candles.append({
                'timestamp': datetime.fromtimestamp(int(item[0]) / 1000),
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5])
            })

        return candles[::-1]  # Reverse to chronological order

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker information"""
        endpoint = f"{self.BASE_URL}/v5/market/tickers"

        params = {
            "category": "linear",
            "symbol": symbol
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        ticker = data['result']['list'][0]
        return {
            'symbol': ticker['symbol'],
            'last_price': float(ticker['lastPrice']),
            'bid': float(ticker['bid1Price']),
            'ask': float(ticker['ask1Price']),
            'volume_24h': float(ticker['volume24h']),
            'price_change_24h': float(ticker['price24hPcnt'])
        }
```

## Key Metrics

- **Sentiment Accuracy**: Correlation between predicted sentiment and price movement
- **Signal Precision**: Percentage of correct directional predictions
- **Sharpe Ratio**: Risk-adjusted return of earnings strategy
- **Information Coefficient**: Correlation between signal strength and returns
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest peak-to-trough decline

## Dependencies

```python
# Core
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0

# LLM APIs
openai>=1.0.0
anthropic>=0.7.0
tiktoken>=0.5.0

# NLP
transformers>=4.30.0
spacy>=3.5.0

# Market Data
yfinance>=0.2.0
ccxt>=4.0.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
```

## Expected Outcomes

1. **Sentiment Extraction**: Accurate extraction of overall tone and specific sentiment markers
2. **Confidence Detection**: Identification of management hedging vs. assertiveness
3. **Guidance Analysis**: Correct classification of guidance direction
4. **Q&A Scoring**: Assessment of management transparency
5. **Trading Signals**: Actionable signals with calibrated confidence
6. **Backtest Results**: Expected Sharpe Ratio 0.8-1.5 on earnings announcements

## References

1. **Large Language Models in Equity Markets: Applications, Techniques, and Insights** (2025)
   - URL: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365

2. **MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents** (2025)
   - URL: https://arxiv.org/abs/2502.00415

3. **Can ChatGPT Forecast Stock Price Movements?** (Lopez-Lira & Tang, 2024)
   - URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788

4. **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models** (Araci, 2019)
   - URL: https://arxiv.org/abs/1908.10063

5. **Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Alpha** (2025)
   - URL: https://arxiv.org/abs/2508.04975

6. **Can Large Language Models Forecast Time Series of Earnings per Share?** (2025)
   - URL: https://www.tandfonline.com/doi/full/10.1080/00128775.2025.2534144

## Rust Implementation

This chapter includes a complete Rust implementation for high-performance earnings call analysis on cryptocurrency data from Bybit. See `rust/` directory.

### Features:
- Transcript parsing and section detection
- Sentiment analysis with financial lexicon
- LLM API integration for advanced analysis
- Trading signal generation
- Backtesting framework with comprehensive metrics
- Real-time data fetching from Bybit API
- Modular and extensible design

## Difficulty Level

Advanced

Requires understanding of: Natural Language Processing, Sentiment Analysis, LLM APIs, Financial Analysis, Trading Systems, Event-driven Strategies
