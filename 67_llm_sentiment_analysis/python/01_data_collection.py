"""
Chapter 67: LLM Sentiment Analysis for Trading
01_data_collection.py - Collecting news and social media data for sentiment analysis

This module demonstrates how to collect financial news and social media data
from various sources for sentiment analysis.
"""

import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article with metadata."""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    symbols: List[str]  # Related stock/crypto symbols


@dataclass
class SocialMediaPost:
    """Represents a social media post."""
    text: str
    platform: str
    author: str
    created_at: datetime
    likes: int
    shares: int
    symbols: List[str]


class NewsCollector:
    """
    Collects financial news from various sources.

    In production, you would integrate with actual APIs:
    - NewsAPI (newsapi.org)
    - Alpha Vantage News
    - Finnhub News
    - CryptoCompare News
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the news collector.

        Args:
            api_key: Optional API key for news services
        """
        self.api_key = api_key
        self._rate_limit_delay = 1.0  # seconds between requests

    def fetch_crypto_news(self, symbol: str, limit: int = 100) -> List[NewsArticle]:
        """
        Fetch cryptocurrency news for a given symbol.

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            limit: Maximum number of articles to fetch

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching crypto news for {symbol}, limit={limit}")

        # In production, this would call actual APIs
        # Here we provide example data structure
        example_articles = [
            NewsArticle(
                title=f"Bitcoin Surges 5% Amid Institutional Adoption",
                content="Major financial institutions are increasingly...",
                source="CryptoNews",
                published_at=datetime.now() - timedelta(hours=2),
                url="https://example.com/news/1",
                symbols=["BTC", "BTCUSDT"]
            ),
            NewsArticle(
                title="Ethereum 2.0 Staking Reaches New Milestone",
                content="The Ethereum network has seen significant growth...",
                source="CoinDesk",
                published_at=datetime.now() - timedelta(hours=5),
                url="https://example.com/news/2",
                symbols=["ETH", "ETHUSDT"]
            ),
        ]

        return example_articles[:limit]

    def fetch_stock_news(self, symbol: str, limit: int = 100) -> List[NewsArticle]:
        """
        Fetch stock market news for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            limit: Maximum number of articles to fetch

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"Fetching stock news for {symbol}, limit={limit}")

        example_articles = [
            NewsArticle(
                title=f"Apple Reports Record Quarterly Revenue",
                content="Apple Inc. announced its quarterly earnings...",
                source="Reuters",
                published_at=datetime.now() - timedelta(hours=1),
                url="https://example.com/news/3",
                symbols=["AAPL"]
            ),
            NewsArticle(
                title="Tesla Delivery Numbers Beat Expectations",
                content="Electric vehicle maker Tesla delivered more...",
                source="Bloomberg",
                published_at=datetime.now() - timedelta(hours=3),
                url="https://example.com/news/4",
                symbols=["TSLA"]
            ),
        ]

        return example_articles[:limit]


class SocialMediaCollector:
    """
    Collects social media posts related to financial markets.

    In production, integrate with:
    - Twitter/X API
    - Reddit API (PRAW)
    - StockTwits API
    """

    def __init__(self):
        self._rate_limit_delay = 2.0

    def fetch_twitter_posts(self, query: str, limit: int = 100) -> List[SocialMediaPost]:
        """
        Fetch Twitter/X posts matching a query.

        Args:
            query: Search query (e.g., '$BTC', '#Bitcoin')
            limit: Maximum number of posts

        Returns:
            List of SocialMediaPost objects
        """
        logger.info(f"Fetching Twitter posts for query: {query}")

        # Example data structure
        example_posts = [
            SocialMediaPost(
                text="$BTC looking bullish! Breaking out of the wedge pattern.",
                platform="twitter",
                author="crypto_trader_123",
                created_at=datetime.now() - timedelta(minutes=30),
                likes=150,
                shares=45,
                symbols=["BTC"]
            ),
            SocialMediaPost(
                text="Just bought more $ETH. This dip is a gift!",
                platform="twitter",
                author="eth_enthusiast",
                created_at=datetime.now() - timedelta(hours=1),
                likes=89,
                shares=12,
                symbols=["ETH"]
            ),
        ]

        return example_posts[:limit]

    def fetch_reddit_posts(self, subreddit: str, limit: int = 100) -> List[SocialMediaPost]:
        """
        Fetch Reddit posts from a subreddit.

        Args:
            subreddit: Subreddit name (e.g., 'wallstreetbets', 'cryptocurrency')
            limit: Maximum number of posts

        Returns:
            List of SocialMediaPost objects
        """
        logger.info(f"Fetching Reddit posts from r/{subreddit}")

        example_posts = [
            SocialMediaPost(
                text="DD: Why I think GME is about to moon. Here's my analysis...",
                platform="reddit",
                author="diamond_hands_ape",
                created_at=datetime.now() - timedelta(hours=2),
                likes=1500,
                shares=300,
                symbols=["GME"]
            ),
            SocialMediaPost(
                text="YOLO'd my savings into TSLA calls. Wish me luck!",
                platform="reddit",
                author="options_gambler",
                created_at=datetime.now() - timedelta(hours=4),
                likes=850,
                shares=150,
                symbols=["TSLA"]
            ),
        ]

        return example_posts[:limit]


class DataAggregator:
    """
    Aggregates data from multiple sources into a unified format.
    """

    def __init__(self):
        self.news_collector = NewsCollector()
        self.social_collector = SocialMediaCollector()

    def collect_all_data(
        self,
        symbols: List[str],
        include_news: bool = True,
        include_social: bool = True,
        news_limit: int = 50,
        social_limit: int = 100
    ) -> dict:
        """
        Collect all available data for given symbols.

        Args:
            symbols: List of symbols to collect data for
            include_news: Whether to include news articles
            include_social: Whether to include social media posts
            news_limit: Maximum news articles per symbol
            social_limit: Maximum social posts per query

        Returns:
            Dictionary with collected data organized by symbol
        """
        result = {}

        for symbol in symbols:
            logger.info(f"Collecting data for {symbol}")

            result[symbol] = {
                "news": [],
                "social_media": [],
                "collected_at": datetime.now().isoformat()
            }

            if include_news:
                # Determine if crypto or stock
                if symbol in ["BTC", "ETH", "SOL", "BTCUSDT", "ETHUSDT"]:
                    articles = self.news_collector.fetch_crypto_news(symbol, news_limit)
                else:
                    articles = self.news_collector.fetch_stock_news(symbol, news_limit)

                result[symbol]["news"] = [
                    {
                        "title": a.title,
                        "content": a.content,
                        "source": a.source,
                        "published_at": a.published_at.isoformat(),
                        "url": a.url
                    }
                    for a in articles
                ]

            if include_social:
                # Fetch from Twitter
                twitter_posts = self.social_collector.fetch_twitter_posts(
                    f"${symbol}", social_limit // 2
                )

                # Fetch from Reddit (if applicable)
                if symbol in ["BTC", "ETH"]:
                    subreddit = "cryptocurrency"
                else:
                    subreddit = "wallstreetbets"

                reddit_posts = self.social_collector.fetch_reddit_posts(
                    subreddit, social_limit // 2
                )

                all_posts = twitter_posts + reddit_posts

                result[symbol]["social_media"] = [
                    {
                        "text": p.text,
                        "platform": p.platform,
                        "author": p.author,
                        "created_at": p.created_at.isoformat(),
                        "engagement": p.likes + p.shares
                    }
                    for p in all_posts
                ]

        return result

    def save_to_json(self, data: dict, filepath: str):
        """Save collected data to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {filepath}")


def main():
    """Main function demonstrating data collection."""

    # Initialize aggregator
    aggregator = DataAggregator()

    # Symbols to analyze
    symbols = ["BTC", "ETH", "AAPL", "TSLA"]

    # Collect data
    print("=" * 60)
    print("LLM Sentiment Analysis - Data Collection Demo")
    print("=" * 60)

    data = aggregator.collect_all_data(
        symbols=symbols,
        include_news=True,
        include_social=True,
        news_limit=10,
        social_limit=20
    )

    # Display summary
    print("\nData Collection Summary:")
    print("-" * 40)
    for symbol, symbol_data in data.items():
        news_count = len(symbol_data["news"])
        social_count = len(symbol_data["social_media"])
        print(f"{symbol}: {news_count} news articles, {social_count} social posts")

    # Save to file
    output_file = "collected_data.json"
    aggregator.save_to_json(data, output_file)

    print(f"\nData saved to {output_file}")
    print("\nExample news article:")
    if data["BTC"]["news"]:
        article = data["BTC"]["news"][0]
        print(f"  Title: {article['title']}")
        print(f"  Source: {article['source']}")
        print(f"  Published: {article['published_at']}")


if __name__ == "__main__":
    main()
