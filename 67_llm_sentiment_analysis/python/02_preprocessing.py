"""
Chapter 67: LLM Sentiment Analysis for Trading
02_preprocessing.py - Text preprocessing for sentiment analysis

This module demonstrates text preprocessing techniques for preparing
financial text data for LLM sentiment analysis.
"""

import re
import unicodedata
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedText:
    """Represents preprocessed text with metadata."""
    original: str
    cleaned: str
    tokens: List[str]
    symbols_mentioned: List[str]
    word_count: int
    char_count: int


class TextPreprocessor:
    """
    Preprocesses financial text for sentiment analysis.

    Features:
    - Unicode normalization
    - URL and email removal
    - Ticker symbol extraction
    - Noise removal while preserving sentiment-relevant content
    """

    # Common stock ticker patterns
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

    # Crypto symbols
    CRYPTO_SYMBOLS = {
        'BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'AVAX', 'DOT',
        'MATIC', 'LINK', 'UNI', 'ATOM', 'LTC', 'BCH', 'XLM'
    }

    # Common financial abbreviations to preserve
    FINANCIAL_TERMS = {
        'P/E', 'EPS', 'EBITDA', 'YoY', 'QoQ', 'IPO', 'ETF',
        'ATH', 'ATL', 'HODL', 'FUD', 'FOMO', 'DCA', 'RSI', 'MACD'
    }

    def __init__(
        self,
        lowercase: bool = False,  # Keep case for LLMs
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_extra_whitespace: bool = True,
        preserve_sentiment_punctuation: bool = True
    ):
        """
        Initialize the preprocessor.

        Args:
            lowercase: Convert text to lowercase (not recommended for LLMs)
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_extra_whitespace: Collapse multiple whitespace
            preserve_sentiment_punctuation: Keep !, ?, etc.
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_extra_whitespace = remove_extra_whitespace
        self.preserve_sentiment_punctuation = preserve_sentiment_punctuation

    def preprocess(self, text: str) -> ProcessedText:
        """
        Preprocess a single text string.

        Args:
            text: Raw text to preprocess

        Returns:
            ProcessedText object with cleaned text and metadata
        """
        original = text

        # 1. Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # 2. Extract ticker symbols before cleaning
        symbols = self._extract_symbols(text)

        # 3. Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # 4. Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)

        # 5. Handle special characters
        text = self._clean_special_chars(text)

        # 6. Normalize whitespace
        if self.remove_extra_whitespace:
            text = ' '.join(text.split())

        # 7. Optional lowercase
        if self.lowercase:
            text = text.lower()

        # 8. Basic tokenization
        tokens = self._tokenize(text)

        return ProcessedText(
            original=original,
            cleaned=text,
            tokens=tokens,
            symbols_mentioned=symbols,
            word_count=len(tokens),
            char_count=len(text)
        )

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock and crypto symbols from text."""
        symbols = set()

        # Find $TICKER patterns
        ticker_matches = self.TICKER_PATTERN.findall(text)
        symbols.update(ticker_matches)

        # Find crypto symbols
        words = text.upper().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.CRYPTO_SYMBOLS:
                symbols.add(clean_word)

        return sorted(list(symbols))

    def _clean_special_chars(self, text: str) -> str:
        """Clean special characters while preserving sentiment indicators."""
        # Replace multiple punctuation with single
        if self.preserve_sentiment_punctuation:
            text = re.sub(r'([!?.]){2,}', r'\1', text)
        else:
            text = re.sub(r'[^\w\s]', ' ', text)

        # Remove non-ASCII characters except common ones
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.split()

    def preprocess_batch(self, texts: List[str]) -> List[ProcessedText]:
        """
        Preprocess multiple texts.

        Args:
            texts: List of raw texts

        Returns:
            List of ProcessedText objects
        """
        return [self.preprocess(text) for text in texts]


class FinancialTextCleaner:
    """
    Specialized cleaner for financial text with domain knowledge.
    """

    # Words that should not be removed as they carry sentiment
    SENTIMENT_WORDS = {
        'surge', 'plunge', 'crash', 'soar', 'rally', 'dump',
        'bullish', 'bearish', 'breakout', 'breakdown', 'moon',
        'dip', 'peak', 'bottom', 'top', 'resistance', 'support'
    }

    # Negation words that modify sentiment
    NEGATION_WORDS = {'not', 'no', 'never', 'none', "n't", 'neither', 'nor'}

    # Intensifiers that amplify sentiment
    INTENSIFIERS = {
        'very', 'extremely', 'highly', 'significantly', 'massively',
        'strongly', 'absolutely', 'completely', 'totally'
    }

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def clean_for_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Clean text specifically for sentiment analysis.

        Args:
            text: Raw financial text

        Returns:
            Dictionary with cleaned text and extracted features
        """
        # Basic preprocessing
        processed = self.preprocessor.preprocess(text)

        # Extract sentiment-relevant features
        features = {
            'cleaned_text': processed.cleaned,
            'symbols': processed.symbols_mentioned,
            'has_negation': self._has_negation(processed.cleaned),
            'has_intensifier': self._has_intensifier(processed.cleaned),
            'sentiment_words': self._extract_sentiment_words(processed.cleaned),
            'word_count': processed.word_count
        }

        return features

    def _has_negation(self, text: str) -> bool:
        """Check if text contains negation words."""
        words = set(text.lower().split())
        return bool(words & self.NEGATION_WORDS)

    def _has_intensifier(self, text: str) -> bool:
        """Check if text contains intensifier words."""
        words = set(text.lower().split())
        return bool(words & self.INTENSIFIERS)

    def _extract_sentiment_words(self, text: str) -> List[str]:
        """Extract sentiment-carrying words from text."""
        words = text.lower().split()
        return [w for w in words if w in self.SENTIMENT_WORDS]


def prepare_for_llm(
    text: str,
    max_length: int = 512,
    include_context: bool = True
) -> str:
    """
    Prepare text for LLM input with optional context.

    Args:
        text: Raw text to prepare
        max_length: Maximum text length (tokens approximate)
        include_context: Include domain context in prompt

    Returns:
        Formatted text ready for LLM input
    """
    cleaner = FinancialTextCleaner()
    features = cleaner.clean_for_sentiment(text)

    cleaned_text = features['cleaned_text']

    # Truncate if too long (rough approximation: 1 word ≈ 1.3 tokens)
    words = cleaned_text.split()
    max_words = int(max_length / 1.3)
    if len(words) > max_words:
        cleaned_text = ' '.join(words[:max_words]) + '...'

    if include_context and features['symbols']:
        symbols_str = ', '.join(features['symbols'])
        context = f"[Relevant symbols: {symbols_str}]\n"
        return context + cleaned_text

    return cleaned_text


def main():
    """Demonstrate text preprocessing."""

    print("=" * 60)
    print("LLM Sentiment Analysis - Text Preprocessing Demo")
    print("=" * 60)

    # Example financial texts
    texts = [
        "🚀 $BTC is absolutely MOONING! Just broke through $50k resistance! #Bitcoin #ToTheMoon https://crypto.news/btc-rally",
        "Tesla $TSLA misses Q3 delivery targets. Stock down 5% in pre-market trading. Not great news for EV sector.",
        "Breaking: Fed announces rate decision. Markets react cautiously. $SPY $QQQ showing mixed signals.",
        "I'm not saying $ETH will crash, but this pump feels unsustainable. Could see a pullback to $3k support.",
    ]

    preprocessor = TextPreprocessor()
    cleaner = FinancialTextCleaner()

    for i, text in enumerate(texts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Original: {text[:80]}...")

        # Basic preprocessing
        processed = preprocessor.preprocess(text)
        print(f"Cleaned: {processed.cleaned[:80]}...")
        print(f"Symbols: {processed.symbols_mentioned}")

        # Financial-specific cleaning
        features = cleaner.clean_for_sentiment(text)
        print(f"Has negation: {features['has_negation']}")
        print(f"Has intensifier: {features['has_intensifier']}")
        print(f"Sentiment words: {features['sentiment_words']}")

        # Prepare for LLM
        llm_input = prepare_for_llm(text)
        print(f"LLM-ready: {llm_input[:80]}...")


if __name__ == "__main__":
    main()
