"""
Example 01: Document Retrieval System Demo

This example demonstrates how to use the FinancialDocumentRetriever
to index and search financial documents.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from retriever import Document, FinancialDocumentRetriever, SimpleRetriever


def create_sample_documents():
    """Create sample financial documents for the demo."""
    return [
        # Tesla documents
        Document.create(
            text="Tesla reported record Q4 deliveries of 484,507 vehicles, "
                 "beating analyst expectations by 15%. The Model Y was the "
                 "best-selling EV globally. Strong demand in China and Europe "
                 "drove the growth. Analysts at Goldman Sachs upgraded their "
                 "price target to $300.",
            ticker="TSLA",
            source="Reuters",
            date=datetime.now() - timedelta(days=1),
            doc_type="news"
        ),
        Document.create(
            text="Tesla faces increasing competition from Chinese EV makers. "
                 "BYD surpassed Tesla in quarterly sales for the first time. "
                 "Margin pressure concerns are rising among analysts as Tesla "
                 "cuts prices to maintain market share.",
            ticker="TSLA",
            source="Bloomberg",
            date=datetime.now() - timedelta(days=2),
            doc_type="news"
        ),
        Document.create(
            text="Tesla's Cybertruck production is ramping up at Gigafactory "
                 "Texas. Initial deliveries have begun, though supply remains "
                 "limited. The company expects to produce 250,000 units in 2024.",
            ticker="TSLA",
            source="Electrek",
            date=datetime.now() - timedelta(days=3),
            doc_type="news"
        ),

        # Apple documents
        Document.create(
            text="Apple announced strong holiday quarter results with iPhone 15 "
                 "sales exceeding expectations. Services revenue hit an all-time "
                 "high of $22.3 billion. The company raised guidance for next quarter.",
            ticker="AAPL",
            source="WSJ",
            date=datetime.now() - timedelta(days=1),
            doc_type="news"
        ),
        Document.create(
            text="Apple faces regulatory challenges in EU over app store fees. "
                 "New Digital Markets Act rules could reduce services revenue by "
                 "5-10% in Europe. Apple is appealing the commission's decisions.",
            ticker="AAPL",
            source="Financial Times",
            date=datetime.now() - timedelta(days=4),
            doc_type="news"
        ),
        Document.create(
            text="Apple Vision Pro pre-orders exceeded internal expectations. "
                 "The $3,499 mixed reality headset sold out within hours. "
                 "Analysts see significant potential in the enterprise market.",
            ticker="AAPL",
            source="CNBC",
            date=datetime.now() - timedelta(days=2),
            doc_type="news"
        ),

        # NVIDIA documents
        Document.create(
            text="NVIDIA's data center revenue surged 200% year-over-year as "
                 "AI chip demand continues to accelerate. The company raised "
                 "guidance for next quarter, expecting $16 billion in revenue. "
                 "CEO Jensen Huang called it a 'new computing era'.",
            ticker="NVDA",
            source="WSJ",
            date=datetime.now() - timedelta(days=1),
            doc_type="news"
        ),
        Document.create(
            text="NVIDIA faces supply constraints for its H100 AI chips. "
                 "Major tech companies are placing orders 6-12 months in advance. "
                 "The company is working with TSMC to increase production capacity.",
            ticker="NVDA",
            source="Reuters",
            date=datetime.now() - timedelta(days=3),
            doc_type="news"
        ),

        # Market-wide documents
        Document.create(
            text="Federal Reserve minutes showed officials remain cautious about "
                 "cutting interest rates too quickly. Markets reacted negatively "
                 "to the hawkish tone. The Fed emphasized data dependency for "
                 "future rate decisions.",
            ticker=None,
            source="Federal Reserve",
            date=datetime.now() - timedelta(days=1),
            doc_type="filing"
        ),
        Document.create(
            text="US inflation data came in below expectations at 3.1% year-over-year. "
                 "Core inflation also showed signs of cooling. Markets rallied on "
                 "hopes for earlier rate cuts from the Federal Reserve.",
            ticker=None,
            source="Bureau of Labor Statistics",
            date=datetime.now() - timedelta(days=2),
            doc_type="filing"
        ),
    ]


def main():
    print("=" * 60)
    print("RAG Document Retrieval Demo")
    print("=" * 60)

    # Create documents
    documents = create_sample_documents()
    print(f"\nCreated {len(documents)} sample documents")

    # Initialize retriever (using SimpleRetriever to avoid dependencies)
    print("\nInitializing retriever...")
    retriever = SimpleRetriever()

    # Add documents
    count = retriever.add_documents(documents)
    print(f"Indexed {count} documents")

    # Demo searches
    searches = [
        {
            "query": "Tesla delivery numbers and sales",
            "ticker": "TSLA",
            "top_k": 3
        },
        {
            "query": "Apple iPhone sales and revenue",
            "ticker": "AAPL",
            "top_k": 3
        },
        {
            "query": "AI chip demand and NVIDIA",
            "ticker": None,
            "top_k": 3
        },
        {
            "query": "Federal Reserve interest rates",
            "ticker": None,
            "top_k": 2
        },
        {
            "query": "Competition and market share concerns",
            "ticker": None,
            "top_k": 3
        },
    ]

    for search in searches:
        print("\n" + "=" * 60)
        print(f"Query: \"{search['query']}\"")
        if search["ticker"]:
            print(f"Ticker filter: {search['ticker']}")
        print("-" * 60)

        results = retriever.search(
            query=search["query"],
            ticker=search["ticker"],
            top_k=search["top_k"]
        )

        if results:
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] Score: {result.score:.3f}")
                print(f"    Source: {result.document.source}")
                print(f"    Ticker: {result.document.ticker or 'N/A'}")
                print(f"    Date: {result.document.date.strftime('%Y-%m-%d')}")
                print(f"    Type: {result.document.doc_type}")
                # Truncate text for display
                text = result.document.text
                if len(text) > 150:
                    text = text[:150] + "..."
                print(f"    Text: {text}")
        else:
            print("No results found")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
