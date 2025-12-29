# Chapter 29: Earnings Surprise Prediction — Event-Driven Strategy

## Overview

Earnings announcements — одни из самых важных событий для цен акций. Post-Earnings Announcement Drift (PEAD) — известная аномалия, когда акции продолжают двигаться в направлении surprise после объявления. В этой главе мы используем ML + NLP для предсказания earnings surprise и эксплуатации PEAD.

## Trading Strategy

**Суть стратегии:** Предсказание earnings surprise (beat/miss) до объявления на основе:
- NLP анализа предыдущих earnings calls
- Analyst estimate revisions
- Options implied expectations
- Sector peer performance

**Сигнал на вход:**
- Long: T-1 до earnings при предсказанном beat с high confidence
- Exit: T+5 после earnings (exploit PEAD)

**Edge:** Информационное преимущество из альтернативных данных + PEAD exploitation

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_collection.ipynb` | Сбор earnings dates, estimates, actuals |
| 2 | `02_earnings_calls_nlp.ipynb` | Парсинг и NLP анализ earnings call transcripts |
| 3 | `03_analyst_estimates.ipynb` | Feature engineering из analyst revisions |
| 4 | `04_options_implied.ipynb` | Expected move из options straddle pricing |
| 5 | `05_peer_signals.ipynb` | Sector peers performance как предиктор |
| 6 | `06_feature_engineering.ipynb` | Объединение всех features |
| 7 | `07_model_training.ipynb` | Classification model: beat/meet/miss |
| 8 | `08_pead_analysis.ipynb` | Анализ Post-Earnings Announcement Drift |
| 9 | `09_trading_strategy.ipynb` | Entry/exit rules, position sizing |
| 10 | `10_backtesting.ipynb` | Event-driven backtest |
| 11 | `11_risk_management.ipynb` | Overnight gap risk, position limits |

### Data Requirements

```
Earnings Data:
├── Earnings dates calendar (5+ лет)
├── Analyst estimates (consensus, range)
├── Actual EPS/Revenue
├── Estimate revision history
└── Whisper numbers (если доступны)

Earnings Calls:
├── Transcripts (Seeking Alpha, FactSet)
├── Audio (для sentiment, опционально)
├── Management guidance
└── Q&A section

Market Data:
├── Stock prices (daily + intraday around earnings)
├── Options prices (for implied move)
├── Sector ETF prices
├── VIX around earnings

Alternative:
├── Social media sentiment (StockTwits, Twitter)
├── Web traffic trends (SimilarWeb)
└── Credit card data (если доступно)
```

### Feature Engineering

```python
# Analyst-based features
features = {
    # Estimate revisions
    'revision_30d': (current_est - est_30d_ago) / est_30d_ago,
    'revision_trend': np.sign(revisions).sum(),
    'estimate_dispersion': std(analyst_estimates) / mean(analyst_estimates),
    'num_analysts': count(analysts),

    # Historical patterns
    'beat_streak': consecutive_beats,
    'avg_surprise_4q': mean(last_4_surprises),
    'guidance_vs_consensus': guidance / consensus - 1,

    # Options-implied
    'implied_move': straddle_price / stock_price,
    'iv_percentile': current_iv / historical_iv,
    'put_call_ratio': put_volume / call_volume,

    # Sector signals
    'sector_earnings_trend': mean(sector_peer_surprises),
    'sector_momentum_20d': sector_etf_return_20d,

    # NLP from prior earnings call
    'sentiment_score': sentiment(transcript),
    'uncertainty_words': count_uncertain_phrases,
    'forward_guidance_sentiment': sentiment(guidance_section),
    'q_a_sentiment_delta': qa_sentiment - prepared_sentiment
}
```

### NLP Pipeline for Earnings Calls

```python
# Transcript processing
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# FinBERT for financial sentiment
finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

# Key sections to analyze:
sections = [
    'prepared_remarks',    # Management presentation
    'guidance',            # Forward-looking statements
    'q_and_a',            # Analyst questions & answers
]

# Features extracted:
nlp_features = {
    'overall_sentiment': float,      # -1 to +1
    'uncertainty_ratio': float,      # Uncertain phrases / total
    'forward_looking_ratio': float,  # Future tense usage
    'quantitative_density': float,   # Numbers per sentence
    'management_confidence': float,  # Hedge words analysis
}
```

### Model Architecture

```
Input Features (50+):
├── Analyst estimates (10 features)
├── Historical patterns (8 features)
├── Options-implied (6 features)
├── Sector signals (5 features)
├── NLP features (15 features)
└── Technical (6 features)

Model Options:
├── LightGBM (baseline, interpretable)
├── XGBoost with early stopping
├── Neural network (for NLP integration)
└── Ensemble of above

Output:
├── P(Beat) — probability of beating estimates
├── P(Miss) — probability of missing
├── Expected surprise magnitude
```

### PEAD Exploitation

```
Documented PEAD patterns:
├── Day 0 (announcement): 50% of move
├── Day 1-5: 25% of drift
├── Day 6-20: 15% of drift
└── Day 21-60: 10% of drift

Strategy timing:
├── Entry: T-1 (day before earnings)
├── Hold through announcement
├── Exit: T+3 to T+5 (capture PEAD)
└── Stop-loss: -5% from entry
```

### Risk Management

```
Position Sizing:
├── Max 2% portfolio per earnings play
├── Max 5 concurrent earnings positions
├── Reduce size for high IV stocks

Risk Factors:
├── Overnight gap risk (can't stop-loss)
├── IV crush after earnings
├── Binary outcome nature
└── Guidance more important than EPS
```

### Key Metrics

- **Prediction:** Accuracy, Precision/Recall for beats, AUC-ROC
- **Strategy:** Win rate, Avg win/loss, Profit factor, Sharpe
- **Events:** Trades per quarter, Avg holding period

### Dependencies

```python
transformers>=4.30.0  # FinBERT
torch>=2.0.0
lightgbm>=4.0.0
pandas>=1.5.0
numpy>=1.23.0
yfinance>=0.2.0
beautifulsoup4>=4.12.0  # Scraping transcripts
```

## Expected Outcomes

1. **Earnings calendar database** с estimates и actuals
2. **NLP pipeline** для анализа earnings call transcripts
3. **Feature set (50+ features)** из multiple data sources
4. **Classification model** с accuracy > 55% (random = 33% для 3 classes)
5. **Event-driven backtesting framework**
6. **Strategy results** с positive expectancy после costs

## References

- [Post-Earnings-Announcement Drift: The Role of Revenue Surprises](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=224560)
- [Earnings Announcement Premium and Trading Volume](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2696573)
- [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
- [Lazy Prices: Evidence from Quarterly Earnings Announcements](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1658471)

## Difficulty Level

⭐⭐⭐⭐☆ (Advanced)

Требуется понимание: NLP/Transformers, Event studies, Options basics, Corporate finance
