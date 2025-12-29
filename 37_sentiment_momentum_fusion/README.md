# Chapter 37: Sentiment-Momentum Fusion — Social Media Enhanced Momentum

## Overview

Price momentum и sentiment momentum из социальных сетей могут усиливать друг друга. В этой главе мы комбинируем традиционный price momentum с sentiment сигналами из Twitter, Reddit и StockTwits для построения enhanced momentum стратегии.

## Trading Strategy

**Суть стратегии:** Покупаем акции с положительным price momentum И положительным sentiment momentum. Продаем при дивергенции между ценой и sentiment.

**Сигнал на вход:**
- Strong Long: Positive price momentum + positive sentiment momentum
- Long: Positive price momentum, neutral sentiment
- Short: Negative price momentum + negative sentiment momentum
- Exit: Divergence (price up, sentiment down or vice versa)

**Edge:** Sentiment предсказывает будущий momentum; divergence предсказывает reversal

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_collection.ipynb` | Сбор данных из Twitter, Reddit, StockTwits |
| 2 | `02_text_preprocessing.ipynb` | Очистка и подготовка текста |
| 3 | `03_finbert_sentiment.ipynb` | FinBERT для финансового sentiment |
| 4 | `04_sentiment_aggregation.ipynb` | Агрегация sentiment по тикерам |
| 5 | `05_sentiment_momentum.ipynb` | Расчет sentiment momentum |
| 6 | `06_price_momentum.ipynb` | Традиционный price momentum |
| 7 | `07_fusion_features.ipynb` | Комбинация price + sentiment features |
| 8 | `08_divergence_detection.ipynb` | Обнаружение divergence сигналов |
| 9 | `09_ml_model.ipynb` | ML модель для combined signal |
| 10 | `10_backtesting.ipynb` | Backtest с transaction costs |
| 11 | `11_real_time_pipeline.ipynb` | Real-time scoring pipeline |

### Data Sources

```
Twitter/X:
├── Cashtags ($AAPL, $TSLA)
├── Financial influencers
├── Company mentions
└── API: Twitter API v2 / Scrapers

Reddit:
├── r/wallstreetbets
├── r/stocks
├── r/investing
├── r/options
└── API: PRAW (Python Reddit API Wrapper)

StockTwits:
├── Message stream per ticker
├── Bullish/Bearish labels (user-provided)
├── Trending tickers
└── API: StockTwits API

News Headlines:
├── Financial news (Reuters, Bloomberg)
├── Seeking Alpha
├── Yahoo Finance
└── Benzinga
```

### Sentiment Extraction with FinBERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBERTSentiment:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.labels = ['negative', 'neutral', 'positive']

    def predict(self, texts):
        """
        Score sentiment for list of texts
        Returns: scores in [-1, 1] range
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=512, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Convert to continuous score: -1 (negative) to +1 (positive)
        scores = probs[:, 2] - probs[:, 0]  # positive - negative

        return scores.numpy()

    def predict_with_confidence(self, texts):
        """Return sentiment score and confidence"""
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=512, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        scores = probs[:, 2] - probs[:, 0]
        confidence = probs.max(dim=1).values

        return scores.numpy(), confidence.numpy()
```

### Sentiment Aggregation

```python
class SentimentAggregator:
    """
    Aggregate sentiment across multiple sources for each ticker
    """
    def __init__(self, decay_halflife_hours=24):
        self.decay_halflife = decay_halflife_hours

    def aggregate_daily(self, ticker, date, messages):
        """
        Aggregate sentiment for ticker on given date
        """
        if len(messages) == 0:
            return {'sentiment': 0, 'volume': 0, 'confidence': 0}

        # Time decay weighting (more recent = higher weight)
        hours_ago = (date - messages['timestamp']).dt.total_seconds() / 3600
        time_weights = 0.5 ** (hours_ago / self.decay_halflife)

        # Volume weighting (engagement)
        engagement = messages['likes'] + messages['retweets'] + messages['replies']
        engagement_weights = np.log1p(engagement)

        # Combine weights
        weights = time_weights * engagement_weights
        weights = weights / weights.sum()

        # Weighted average sentiment
        weighted_sentiment = (messages['sentiment'] * weights).sum()

        # Confidence based on volume and agreement
        sentiment_std = messages['sentiment'].std()
        confidence = 1 - sentiment_std  # Higher agreement = higher confidence

        return {
            'sentiment': weighted_sentiment,
            'volume': len(messages),
            'confidence': confidence,
            'bullish_ratio': (messages['sentiment'] > 0).mean()
        }
```

### Sentiment Momentum

```python
def calculate_sentiment_momentum(sentiment_series, windows=[1, 5, 20]):
    """
    Calculate sentiment momentum over multiple windows
    """
    momentum = {}

    for window in windows:
        # Level momentum: current vs past average
        sma = sentiment_series.rolling(window).mean()
        momentum[f'sent_mom_{window}d'] = sentiment_series - sma.shift(1)

        # Change momentum: acceleration of sentiment
        momentum[f'sent_change_{window}d'] = sentiment_series.diff(window)

        # Volume-weighted momentum
        if 'volume' in sentiment_series.columns:
            vw_sent = (sentiment_series['sentiment'] * sentiment_series['volume']).rolling(window).sum()
            vw_sent /= sentiment_series['volume'].rolling(window).sum()
            momentum[f'sent_vw_mom_{window}d'] = vw_sent - vw_sent.shift(window)

    return pd.DataFrame(momentum)
```

### Price-Sentiment Fusion

```python
class SentimentMomentumFusion:
    """
    Combine price momentum with sentiment momentum
    """
    def __init__(self):
        self.price_weight = 0.6
        self.sentiment_weight = 0.4

    def calculate_combined_score(self, price_data, sentiment_data):
        """
        Calculate combined momentum score
        """
        # Price momentum (normalized)
        price_mom = price_data['return_20d']
        price_mom_z = (price_mom - price_mom.mean()) / price_mom.std()

        # Sentiment momentum (normalized)
        sent_mom = sentiment_data['sent_mom_5d']
        sent_mom_z = (sent_mom - sent_mom.mean()) / sent_mom.std()

        # Combined score
        combined = self.price_weight * price_mom_z + self.sentiment_weight * sent_mom_z

        return combined

    def detect_divergence(self, price_data, sentiment_data, threshold=1.5):
        """
        Detect divergence between price and sentiment momentum
        """
        price_direction = np.sign(price_data['return_5d'])
        sent_direction = np.sign(sentiment_data['sent_change_5d'])

        # Divergence: price and sentiment moving in opposite directions
        divergence = price_direction != sent_direction

        # Strong divergence: magnitude also significant
        price_magnitude = abs(price_data['return_5d']) > price_data['return_5d'].std()
        sent_magnitude = abs(sentiment_data['sent_change_5d']) > sentiment_data['sent_change_5d'].std()

        strong_divergence = divergence & price_magnitude & sent_magnitude

        return {
            'divergence': divergence,
            'strong_divergence': strong_divergence,
            'type': np.where(price_direction > sent_direction, 'bearish', 'bullish')
        }
```

### ML Model for Signal Generation

```python
class SentimentMomentumModel:
    """
    ML model combining price and sentiment features
    """
    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05
        )

        self.features = [
            # Price features
            'return_5d', 'return_20d', 'return_60d',
            'volatility_20d', 'volume_ratio',

            # Sentiment features
            'sentiment_score', 'sentiment_volume',
            'sent_mom_5d', 'sent_mom_20d',
            'bullish_ratio',

            # Interaction features
            'price_sent_correlation',
            'divergence_flag',
            'sentiment_leads_price'
        ]

    def create_labels(self, returns, forward_days=5, threshold=0.02):
        """
        Create classification labels based on forward returns
        """
        forward_returns = returns.shift(-forward_days)
        labels = np.where(forward_returns > threshold, 1,
                         np.where(forward_returns < -threshold, -1, 0))
        return labels

    def train(self, X, y):
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X.iloc[train_idx], y.iloc[train_idx])
            val_pred = self.model.predict(X.iloc[val_idx])
            print(f"Validation accuracy: {accuracy_score(y.iloc[val_idx], val_pred)}")

    def predict_signal(self, X):
        proba = self.model.predict_proba(X)
        # Return probability of positive class minus negative
        return proba[:, 2] - proba[:, 0]
```

### Key Metrics

- **Sentiment Quality:** Correlation with forward returns, IC
- **Signal Quality:** Hit rate, Average return per signal
- **Strategy:** Sharpe, Max DD, Turnover
- **Comparison:** vs price-only momentum, vs sentiment-only

### Dependencies

```python
transformers>=4.30.0
torch>=2.0.0
tweepy>=4.14.0        # Twitter API
praw>=7.7.0           # Reddit API
pandas>=1.5.0
numpy>=1.23.0
lightgbm>=4.0.0
yfinance>=0.2.0
```

## Expected Outcomes

1. **Data pipeline** для Twitter, Reddit, StockTwits
2. **FinBERT sentiment scoring** для финансовых текстов
3. **Sentiment aggregation** по тикерам и времени
4. **Fusion strategy** комбинирующая price + sentiment
5. **Divergence signals** для reversal prediction
6. **Backtest results:** Improved Sharpe vs price-only momentum

## References

- [FinBERT: A Pretrained Language Model for Financial Communications](https://arxiv.org/abs/1908.10063)
- [Predicting Stock Movements with Social Media](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2756815)
- [Twitter Mood Predicts the Stock Market](https://arxiv.org/abs/1010.3003)
- [Retail Investor Sentiment and Behavior](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3776874)

## Difficulty Level

⭐⭐⭐⭐☆ (Advanced)

Требуется понимание: NLP/Transformers, Social media APIs, Sentiment analysis, Momentum strategies
