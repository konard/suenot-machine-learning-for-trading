"""
Example 02: RAG-based Trading Signal Generation

This example demonstrates how to generate trading signals
using the RAGTradingSignalGenerator with retrieved documents.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from retriever import Document, SimpleRetriever
from signals import RAGTradingSignalGenerator, SignalDirection


def create_sample_documents():
    """Create sample documents with varying sentiments."""
    return [
        # Bullish Tesla documents
        Document.create(
            text="Tesla reported record Q4 deliveries of 484,507 vehicles, "
                 "beating analyst expectations by 15%. Strong growth in China "
                 "and Europe. Multiple analysts upgraded their price targets. "
                 "The company's market share continues to expand.",
            ticker="TSLA",
            source="Reuters",
            date=datetime.now() - timedelta(days=1),
            doc_type="news"
        ),
        Document.create(
            text="Tesla's energy storage business is booming with Megapack "
                 "deployments up 200% year-over-year. The company sees strong "
                 "demand for grid-scale battery solutions. Revenue from energy "
                 "segment exceeded $1.5 billion.",
            ticker="TSLA",
            source="Bloomberg",
            date=datetime.now() - timedelta(days=2),
            doc_type="news"
        ),

        # Bearish Apple documents
        Document.create(
            text="Apple faces significant challenges in China as Huawei regains "
                 "market share. iPhone sales declined 20% in the region last quarter. "
                 "Analysts express concern about Apple's competitive position in "
                 "the world's largest smartphone market.",
            ticker="AAPL",
            source="WSJ",
            date=datetime.now() - timedelta(days=1),
            doc_type="news"
        ),
        Document.create(
            text="Apple's services growth is slowing as regulatory pressure mounts. "
                 "EU fines could reach billions. App Store revenue growth fell to "
                 "single digits for the first time. Analysts downgrade to hold.",
            ticker="AAPL",
            source="Financial Times",
            date=datetime.now() - timedelta(days=2),
            doc_type="news"
        ),

        # Mixed NVIDIA documents
        Document.create(
            text="NVIDIA reported exceptional AI chip demand with data center "
                 "revenue up 200%. However, China export restrictions could "
                 "impact future growth. The company beat estimates but gave "
                 "cautious guidance citing supply constraints.",
            ticker="NVDA",
            source="Reuters",
            date=datetime.now() - timedelta(days=1),
            doc_type="news"
        ),
        Document.create(
            text="NVIDIA faces increasing competition from AMD and custom chips "
                 "developed by major tech companies. Intel is also ramping up "
                 "its AI accelerator offerings. Market share erosion is a concern "
                 "but NVIDIA maintains technology leadership.",
            ticker="NVDA",
            source="Bloomberg",
            date=datetime.now() - timedelta(days=2),
            doc_type="news"
        ),

        # Neutral Microsoft documents
        Document.create(
            text="Microsoft's cloud business continues steady growth with Azure "
                 "revenue up 29%. The company's AI investments are showing promise "
                 "but competition from AWS and Google Cloud remains intense. "
                 "Guidance was in line with expectations.",
            ticker="MSFT",
            source="CNBC",
            date=datetime.now() - timedelta(days=1),
            doc_type="news"
        ),
    ]


def main():
    print("=" * 60)
    print("RAG Trading Signal Generation Demo")
    print("=" * 60)

    # Create documents
    documents = create_sample_documents()
    print(f"\nCreated {len(documents)} sample documents")

    # Initialize retriever and add documents
    retriever = SimpleRetriever()
    retriever.add_documents(documents)
    print(f"Indexed documents")

    # Initialize signal generator
    generator = RAGTradingSignalGenerator(retriever)

    # Generate signals for different tickers
    tickers = ["TSLA", "AAPL", "NVDA", "MSFT"]

    print("\n" + "=" * 60)
    print("TRADING SIGNALS")
    print("=" * 60)

    signals = {}
    for ticker in tickers:
        print(f"\n--- {ticker} ---")

        signal = generator.generate_signal(ticker)
        signals[ticker] = signal

        # Display signal with color-coding hint
        direction_symbol = {
            SignalDirection.LONG: "[BUY]",
            SignalDirection.SHORT: "[SELL]",
            SignalDirection.NEUTRAL: "[HOLD]"
        }

        print(f"Direction:  {direction_symbol[signal.direction]} {signal.direction.value}")
        print(f"Confidence: {signal.confidence:.0%}")
        print(f"Sources:    {', '.join(signal.sources)}")
        print(f"Reasoning:  {signal.reasoning}")

        if signal.metadata:
            print(f"Details:")
            print(f"  - Documents analyzed: {signal.metadata.get('documents_analyzed', 'N/A')}")
            print(f"  - Positive signals: {signal.metadata.get('positive_signals', 'N/A')}")
            print(f"  - Negative signals: {signal.metadata.get('negative_signals', 'N/A')}")
            if 'avg_sentiment' in signal.metadata:
                print(f"  - Avg sentiment: {signal.metadata['avg_sentiment']:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SIGNAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Ticker':<8} {'Signal':<10} {'Confidence':<12} {'Action'}")
    print("-" * 50)

    for ticker, signal in signals.items():
        action = ""
        if signal.direction == SignalDirection.LONG and signal.confidence > 0.6:
            action = "Consider buying"
        elif signal.direction == SignalDirection.SHORT and signal.confidence > 0.6:
            action = "Consider selling"
        elif signal.confidence < 0.5:
            action = "Low confidence - wait"
        else:
            action = "Hold / No action"

        print(f"{ticker:<8} {signal.direction.value:<10} {signal.confidence:>10.0%}   {action}")

    # Batch generation example
    print("\n" + "=" * 60)
    print("BATCH SIGNAL GENERATION")
    print("=" * 60)

    batch_signals = generator.batch_generate(["TSLA", "AAPL", "NVDA"])
    print(f"\nGenerated {len(batch_signals)} signals in batch")

    for ticker, signal in batch_signals.items():
        print(f"  {ticker}: {signal}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
