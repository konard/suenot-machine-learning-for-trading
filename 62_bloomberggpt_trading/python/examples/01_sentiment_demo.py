"""
Example 01: Financial Sentiment Analysis Demo

This script demonstrates how to use the financial sentiment analyzer
to analyze sentiment in financial news and documents.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run sentiment analysis demonstration."""
    print("=" * 60)
    print("Financial Sentiment Analysis Demo")
    print("=" * 60)

    # Sample financial texts
    sample_texts = [
        {
            "text": "Apple Inc. reported record quarterly revenue of $123.9 billion, "
                   "beating analyst expectations by a significant margin.",
            "expected": "POSITIVE"
        },
        {
            "text": "Tesla shares plummeted 12% after the company reported disappointing "
                   "delivery numbers and warned of production challenges.",
            "expected": "NEGATIVE"
        },
        {
            "text": "The Federal Reserve announced it would keep interest rates unchanged "
                   "at its latest meeting, in line with market expectations.",
            "expected": "NEUTRAL"
        },
        {
            "text": "Microsoft's Azure cloud revenue surged 29% year-over-year, "
                   "solidifying its position as the second-largest cloud provider.",
            "expected": "POSITIVE"
        },
        {
            "text": "Amazon announced plans to lay off 18,000 employees amid concerns "
                   "about slowing e-commerce growth and economic uncertainty.",
            "expected": "NEGATIVE"
        }
    ]

    try:
        from sentiment import FinancialSentimentAnalyzer

        print("\nInitializing FinancialSentimentAnalyzer...")
        print("(This may take a moment to download the model on first run)\n")

        analyzer = FinancialSentimentAnalyzer(model_name="ProsusAI/finbert")

        print("\nAnalyzing sample financial texts:\n")
        print("-" * 60)

        for i, sample in enumerate(sample_texts, 1):
            text = sample["text"]
            expected = sample["expected"]

            result = analyzer.analyze(text)

            print(f"\n[{i}] Text: {text[:80]}...")
            print(f"    Expected: {expected}")
            print(f"    Predicted: {result.label} (confidence: {result.confidence:.2%})")
            print(f"    Probabilities: P={result.positive_prob:.2%}, "
                  f"N={result.negative_prob:.2%}, Neu={result.neutral_prob:.2%}")

            match = "MATCH" if result.label == expected else "MISMATCH"
            print(f"    Result: {match}")

        # Demonstrate aspect-based sentiment
        print("\n" + "=" * 60)
        print("Aspect-Based Sentiment Analysis")
        print("=" * 60)

        multi_aspect_text = """
        Apple reported strong iPhone sales in China, with revenue up 8% year-over-year.
        However, Mac sales declined significantly due to weak consumer demand.
        CEO Tim Cook expressed optimism about the company's AI initiatives,
        calling them 'the next major growth driver'.
        """

        entities = ["iPhone", "Mac", "AI", "Tim Cook"]

        print(f"\nText:\n{multi_aspect_text.strip()}\n")
        print(f"Entities to analyze: {entities}\n")

        aspects = analyzer.analyze_aspects(multi_aspect_text, entities)

        for entity, sentiment in aspects.items():
            if sentiment.label != "NOT_MENTIONED":
                print(f"  {entity}: {sentiment.label} (confidence: {sentiment.confidence:.2%})")
            else:
                print(f"  {entity}: Not mentioned in text")

    except ImportError as e:
        print(f"\nNote: Could not import sentiment module: {e}")
        print("Running with mock analyzer instead...\n")

        # Mock demonstration
        for i, sample in enumerate(sample_texts, 1):
            text = sample["text"]
            expected = sample["expected"]
            print(f"[{i}] Text: {text[:80]}...")
            print(f"    Expected sentiment: {expected}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
