# Глава 29: Предсказание сюрпризов прибыли — Событийная торговая стратегия

## Обзор

Объявления о прибыли (earnings announcements) — одни из самых важных событий для цен акций. Post-Earnings Announcement Drift (PEAD) — известная рыночная аномалия, при которой акции продолжают двигаться в направлении сюрприза после объявления результатов. В этой главе мы используем машинное обучение и обработку естественного языка (NLP) для предсказания сюрпризов прибыли и эксплуатации PEAD.

### Что такое Earnings Surprise?

**Earnings Surprise (сюрприз прибыли)** — это разница между фактической прибылью на акцию (EPS) и консенсус-прогнозом аналитиков:

```
Surprise = (Actual EPS - Expected EPS) / |Expected EPS|
```

- **Positive Surprise (Beat):** Компания превысила ожидания аналитиков
- **Negative Surprise (Miss):** Компания не оправдала ожиданий
- **Meet:** Прибыль совпала с прогнозом

### Почему это работает?

Эффективный рынок должен моментально учитывать всю информацию в ценах. Однако исследования показывают:

1. **Недореакция рынка:** Инвесторы не сразу полностью оценивают значимость сюрприза
2. **PEAD:** Цены продолжают двигаться в направлении сюрприза до 60 дней после объявления
3. **Информационное преимущество:** Альтернативные данные могут предсказать сюрприз до объявления

---

## Содержание

1. [Торговая стратегия](#торговая-стратегия)
2. [Техническая спецификация](#техническая-спецификация)
   - [Ноутбуки для создания](#ноутбуки-для-создания)
   - [Требования к данным](#требования-к-данным)
3. [Конструирование признаков](#конструирование-признаков)
   - [Признаки на основе оценок аналитиков](#признаки-на-основе-оценок-аналитиков)
   - [Признаки из опционов](#признаки-из-опционов)
   - [Секторные сигналы](#секторные-сигналы)
   - [NLP-признаки](#nlp-признаки)
4. [NLP-пайплайн для earnings calls](#nlp-пайплайн-для-earnings-calls)
5. [Архитектура модели](#архитектура-модели)
6. [Эксплуатация PEAD](#эксплуатация-pead)
7. [Управление рисками](#управление-рисками)
8. [Ключевые метрики](#ключевые-метрики)
9. [Зависимости](#зависимости)
10. [Ожидаемые результаты](#ожидаемые-результаты)
11. [Ссылки](#ссылки)

---

## Торговая стратегия

### Суть стратегии

Предсказание earnings surprise (beat/miss) до официального объявления на основе:

- **NLP-анализа** предыдущих earnings calls (телеконференций по результатам)
- **Revisions аналитиков** — изменений прогнозов за последние 30-90 дней
- **Options implied expectations** — ожидаемого движения из цен опционов
- **Sector peer performance** — результатов компаний того же сектора

### Сигнал на вход

| Направление | Условие | Время входа |
|-------------|---------|-------------|
| **Long** | Предсказанный beat с высокой confidence (>70%) | T-1 (за день до earnings) |
| **Short** | Предсказанный miss с высокой confidence (>70%) | T-1 |
| **Exit** | Фиксация прибыли/убытка | T+3 до T+5 после earnings |

### Edge (преимущество)

1. **Информационное преимущество** из альтернативных данных
2. **PEAD exploitation** — использование систематической недореакции рынка
3. **Мультифакторный подход** — комбинация 50+ признаков

---

## Техническая спецификация

### Ноутбуки для создания

| # | Ноутбук | Описание |
|---|---------|----------|
| 1 | `01_data_collection.ipynb` | Сбор earnings dates, estimates, actuals |
| 2 | `02_earnings_calls_nlp.ipynb` | Парсинг и NLP-анализ earnings call transcripts |
| 3 | `03_analyst_estimates.ipynb` | Feature engineering из analyst revisions |
| 4 | `04_options_implied.ipynb` | Expected move из options straddle pricing |
| 5 | `05_peer_signals.ipynb` | Sector peers performance как предиктор |
| 6 | `06_feature_engineering.ipynb` | Объединение всех признаков |
| 7 | `07_model_training.ipynb` | Классификационная модель: beat/meet/miss |
| 8 | `08_pead_analysis.ipynb` | Анализ Post-Earnings Announcement Drift |
| 9 | `09_trading_strategy.ipynb` | Правила входа/выхода, размер позиции |
| 10 | `10_backtesting.ipynb` | Event-driven бэктест |
| 11 | `11_risk_management.ipynb` | Overnight gap risk, лимиты позиций |

### Требования к данным

```
Данные о прибыли (Earnings Data):
├── Earnings dates calendar (5+ лет)
├── Analyst estimates (consensus, range)
├── Actual EPS/Revenue
├── Estimate revision history
└── Whisper numbers (неофициальные ожидания)

Earnings Calls:
├── Transcripts (Seeking Alpha, FactSet)
├── Audio (для анализа тональности, опционально)
├── Management guidance
└── Q&A section

Рыночные данные:
├── Stock prices (daily + intraday around earnings)
├── Options prices (for implied move)
├── Sector ETF prices
└── VIX around earnings

Альтернативные данные:
├── Social media sentiment (StockTwits, Twitter/X)
├── Web traffic trends (SimilarWeb)
└── Credit card data (если доступно)
```

---

## Конструирование признаков

### Признаки на основе оценок аналитиков

```python
analyst_features = {
    # Ревизии оценок
    'revision_30d': (current_est - est_30d_ago) / est_30d_ago,
    'revision_60d': (current_est - est_60d_ago) / est_60d_ago,
    'revision_trend': np.sign(revisions).sum(),  # Направление изменений

    # Разброс оценок
    'estimate_dispersion': std(analyst_estimates) / mean(analyst_estimates),
    'estimate_range': (high_est - low_est) / mean_est,
    'num_analysts': count(analysts),

    # Исторические паттерны
    'beat_streak': consecutive_beats,  # Серия beat
    'avg_surprise_4q': mean(last_4_surprises),
    'surprise_volatility': std(last_8_surprises),

    # Guidance
    'guidance_vs_consensus': guidance / consensus - 1,
    'guidance_raised': int(current_guidance > prior_guidance),
}
```

### Признаки из опционов

```python
options_features = {
    # Ожидаемое движение
    'implied_move': straddle_price / stock_price,  # ATM straddle
    'iv_percentile': current_iv / iv_52w_range,
    'iv_term_structure': front_iv / back_iv,

    # Put/Call соотношения
    'put_call_ratio': put_volume / call_volume,
    'put_call_oi_ratio': put_oi / call_oi,

    # Skew (перекос)
    'iv_skew': put_25d_iv - call_25d_iv,
    'skew_change_5d': current_skew - skew_5d_ago,
}
```

### Секторные сигналы

```python
sector_features = {
    # Результаты peers
    'sector_earnings_trend': mean(sector_peer_surprises),
    'sector_beat_rate': peers_beat / peers_reported,

    # Momentum
    'sector_momentum_20d': sector_etf_return_20d,
    'relative_strength': stock_return_20d - sector_return_20d,

    # Корреляция
    'sector_correlation': corr(stock, sector_etf, 60d),
}
```

### NLP-признаки

```python
nlp_features = {
    # Sentiment предыдущего earnings call
    'overall_sentiment': finbert_sentiment(transcript),
    'prepared_sentiment': sentiment(prepared_remarks),
    'qa_sentiment': sentiment(qa_section),
    'sentiment_delta': qa_sentiment - prepared_sentiment,

    # Uncertainty analysis
    'uncertainty_ratio': uncertain_phrases / total_phrases,
    'hedge_word_count': count(['may', 'might', 'could', 'possibly']),

    # Forward-looking statements
    'forward_looking_ratio': future_tense_count / total_sentences,
    'guidance_confidence': confidence_score(guidance_section),

    # Quantitative density
    'numeric_density': numbers_per_sentence,
    'specific_targets': count(specific_numbers_mentioned),
}
```

---

## NLP-пайплайн для Earnings Calls

### Обработка транскриптов

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загрузка FinBERT для финансового sentiment
model_name = 'ProsusAI/finbert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_transcript(transcript: str) -> dict:
    """Анализ earnings call transcript."""

    # Разделение на секции
    sections = {
        'prepared_remarks': extract_prepared_remarks(transcript),
        'guidance': extract_guidance(transcript),
        'q_and_a': extract_qa(transcript),
    }

    features = {}

    for section_name, text in sections.items():
        # Sentiment анализ
        inputs = tokenizer(text, return_tensors='pt',
                          truncation=True, max_length=512)
        outputs = model(**inputs)
        sentiment = torch.softmax(outputs.logits, dim=1)

        features[f'{section_name}_positive'] = sentiment[0][0].item()
        features[f'{section_name}_negative'] = sentiment[0][1].item()
        features[f'{section_name}_neutral'] = sentiment[0][2].item()

        # Uncertainty words
        uncertainty_words = ['may', 'might', 'could', 'possibly',
                            'uncertain', 'unclear', 'risk']
        features[f'{section_name}_uncertainty'] = sum(
            text.lower().count(w) for w in uncertainty_words
        ) / len(text.split())

    return features
```

### Ключевые секции для анализа

| Секция | Описание | Важность |
|--------|----------|----------|
| **Prepared Remarks** | Подготовленная презентация менеджмента | Высокая |
| **Guidance** | Прогнозы на следующий квартал/год | Критическая |
| **Q&A** | Ответы на вопросы аналитиков | Очень высокая |

### Сигнальные слова и фразы

```python
# Позитивные индикаторы
positive_signals = [
    'exceeded expectations', 'record results', 'strong momentum',
    'raising guidance', 'ahead of schedule', 'outperformed',
]

# Негативные индикаторы
negative_signals = [
    'challenging environment', 'headwinds', 'below expectations',
    'lowering guidance', 'restructuring', 'difficult quarter',
]

# Неопределённость
uncertainty_signals = [
    'visibility is limited', 'uncertain outlook', 'volatile conditions',
    'cautiously optimistic', 'depends on', 'too early to tell',
]
```

---

## Архитектура модели

### Входные признаки

```
Всего признаков: 50+
├── Analyst estimates (10 признаков)
├── Historical patterns (8 признаков)
├── Options-implied (6 признаков)
├── Sector signals (5 признаков)
├── NLP features (15 признаков)
└── Technical (6 признаков)
```

### Модели-кандидаты

| Модель | Преимущества | Недостатки |
|--------|--------------|------------|
| **LightGBM** | Быстрый, интерпретируемый, хорошо работает с табличными данными | Не использует структуру текста |
| **XGBoost** | Точный, early stopping | Требует тюнинга |
| **Neural Network** | Может интегрировать NLP embeddings | Требует больше данных |
| **Ensemble** | Максимальная точность | Сложнее интерпретировать |

### Целевая переменная

```python
# Трёхклассовая классификация
def get_target(actual_eps, expected_eps, threshold=0.02):
    surprise = (actual_eps - expected_eps) / abs(expected_eps)
    if surprise > threshold:
        return 'beat'  # 0
    elif surprise < -threshold:
        return 'miss'  # 1
    else:
        return 'meet'  # 2

# Бинарная классификация (упрощённая)
def get_binary_target(actual_eps, expected_eps):
    return int(actual_eps > expected_eps)  # 1 = beat, 0 = miss/meet
```

### Обучение модели

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Time-series cross-validation (важно для финансовых данных!)
tscv = TimeSeriesSplit(n_splits=5)

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': -1,
}

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50)],
    )
```

---

## Эксплуатация PEAD

### Документированные паттерны PEAD

Post-Earnings Announcement Drift — систематическое отклонение цен после объявления:

| Период | Доля общего движения | Стратегия |
|--------|---------------------|-----------|
| Day 0 (объявление) | ~50% | Gap at open |
| Day 1-5 | ~25% | **Основное окно входа** |
| Day 6-20 | ~15% | Продолжение тренда |
| Day 21-60 | ~10% | Затухание эффекта |

### Тайминг стратегии

```python
# Оптимальные точки входа и выхода
trading_rules = {
    'entry': {
        'timing': 'T-1 close',  # Покупка за день до earnings
        'condition': 'prediction_confidence > 0.70',
    },
    'exit': {
        'take_profit': 'T+3 to T+5',  # Выход на 3-5 день после earnings
        'stop_loss': -0.05,  # -5% от цены входа
    },
    'position_size': {
        'base': 0.02,  # 2% портфеля на сделку
        'confidence_adjusted': 'base * confidence_score',
    }
}
```

### Пример торговой логики

```python
def should_trade(prediction: dict, market_conditions: dict) -> dict:
    """Определение торгового сигнала."""

    signal = {'action': None, 'size': 0}

    # Проверка confidence
    if prediction['beat_prob'] > 0.70:
        signal['action'] = 'long'
        signal['size'] = 0.02 * prediction['beat_prob']
    elif prediction['miss_prob'] > 0.70:
        signal['action'] = 'short'
        signal['size'] = 0.02 * prediction['miss_prob']

    # Корректировка на implied volatility
    if market_conditions['iv_percentile'] > 0.80:
        signal['size'] *= 0.5  # Уменьшаем размер при высокой IV

    return signal
```

---

## Управление рисками

### Sizing позиций

```
Правила размера позиции:
├── Max 2% портфеля на одну earnings-сделку
├── Max 5 одновременных earnings-позиций (10% портфеля)
├── Уменьшение размера при высокой IV (>80 percentile)
└── Увеличение размера при высокой confidence (>85%)
```

### Основные риски

| Риск | Описание | Митигация |
|------|----------|-----------|
| **Overnight Gap** | Невозможность использовать stop-loss при gap | Ограничение размера позиции |
| **IV Crush** | Падение implied volatility после earnings | Не использовать опционы, только акции |
| **Binary Outcome** | Результат практически бинарный | Диверсификация по множеству earnings |
| **Guidance > EPS** | Рынок реагирует на guidance, не только EPS | Включить guidance в модель |
| **After-hours Trading** | Основное движение в after-hours | Учитывать pre-market для входа |

### Правила выхода

```python
exit_rules = {
    'take_profit': {
        'target': '+10%',
        'trailing_stop': '50% of max profit',
    },
    'stop_loss': {
        'hard_stop': '-5%',
        'time_stop': 'T+5 EOD',  # Выход не позже 5 дней
    },
    'special_cases': {
        'guidance_cut': 'Exit immediately if guidance lowered',
        'unusual_volume': 'Review if volume > 3x average',
    }
}
```

---

## Ключевые метрики

### Метрики предсказания

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **Accuracy** | Доля правильных предсказаний | >55% (random = 33%) |
| **Precision (Beat)** | Точность предсказаний beat | >60% |
| **Recall (Beat)** | Полнота выявления beat | >50% |
| **AUC-ROC** | Качество разделения классов | >0.65 |

### Метрики стратегии

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **Win Rate** | Доля прибыльных сделок | >55% |
| **Avg Win/Loss** | Соотношение средней прибыли к убытку | >1.5 |
| **Profit Factor** | Gross profit / Gross loss | >1.5 |
| **Sharpe Ratio** | Risk-adjusted return | >1.0 |
| **Max Drawdown** | Максимальная просадка | <15% |

### Метрики событий

| Метрика | Описание |
|---------|----------|
| **Trades per Quarter** | Количество сделок (цель: 50-100) |
| **Avg Holding Period** | Средний период удержания (цель: 3-5 дней) |
| **Sector Distribution** | Распределение по секторам |
| **Time Distribution** | Распределение по месяцам (earnings season) |

---

## Зависимости

### Python-библиотеки

```python
# NLP и Transformers
transformers>=4.30.0  # FinBERT и другие модели
torch>=2.0.0         # Backend для transformers
sentencepiece>=0.1.99

# Machine Learning
lightgbm>=4.0.0      # Основная модель
xgboost>=2.0.0       # Альтернативная модель
scikit-learn>=1.3.0  # Preprocessing, metrics

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Data Sources
yfinance>=0.2.0      # Цены акций
beautifulsoup4>=4.12.0  # Парсинг transcripts
requests>=2.31.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
```

### Источники данных

| Данные | Бесплатные источники | Платные источники |
|--------|---------------------|-------------------|
| Earnings Dates | Yahoo Finance, SEC | FactSet, Bloomberg |
| Estimates | Yahoo Finance (ограничено) | FactSet, Refinitiv |
| Transcripts | Seeking Alpha (ограничено) | FactSet, AlphaSense |
| Options Data | Yahoo Finance | CBOE, OptionMetrics |

---

## Ожидаемые результаты

После завершения этой главы вы получите:

1. **База данных earnings calendar** с estimates и actuals за 5+ лет
2. **NLP-пайплайн** для анализа earnings call transcripts
3. **Набор признаков (50+)** из множества источников данных
4. **Классификационная модель** с accuracy > 55% (vs random 33%)
5. **Event-driven backtesting framework** для earnings-стратегий
6. **Торговая стратегия** с положительным ожиданием после учёта costs

### Примерные результаты бэктеста

```
Период: 2018-2023 (5 лет)
Количество сделок: 850
Win Rate: 58.2%
Average Win: +4.2%
Average Loss: -2.8%
Profit Factor: 1.75
Annualized Return: 18.5%
Sharpe Ratio: 1.35
Max Drawdown: -12.3%
```

---

## Ссылки

### Академические работы

1. **PEAD Original Research:**
   - [Post-Earnings-Announcement Drift: The Role of Revenue Surprises](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=224560) — Bernard & Thomas

2. **Earnings Announcement Premium:**
   - [Earnings Announcement Premium and Trading Volume](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2696573) — Frazzini & Lamont

3. **NLP в финансах:**
   - [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)

4. **Lazy Prices:**
   - [Lazy Prices: Evidence from Quarterly Earnings Announcements](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1658471) — Cohen, Lou, Malloy

### Практические ресурсы

- [Seeking Alpha Earnings Calendar](https://seekingalpha.com/earnings/earnings-calendar)
- [Yahoo Finance Earnings](https://finance.yahoo.com/calendar/earnings)
- [EDGAR SEC Filings](https://www.sec.gov/cgi-bin/browse-edgar)

---

## Уровень сложности

⭐⭐⭐⭐☆ (Продвинутый)

### Требуемые знания

- **NLP/Transformers:** Понимание BERT-подобных моделей, fine-tuning
- **Event Studies:** Методология анализа событий, abnormal returns
- **Options Basics:** Implied volatility, straddles, Greeks
- **Corporate Finance:** Earnings, EPS, guidance, analyst coverage

### Время на изучение

- Теория: 4-6 часов
- Практика (ноутбуки): 15-20 часов
- Полная реализация стратегии: 30-40 часов

---

## Адаптация для криптовалют

Концепции earnings surprise можно адаптировать для крипто-рынка:

| Традиционные акции | Криптовалюты |
|-------------------|--------------|
| Earnings announcement | Protocol updates, hard forks |
| Analyst estimates | Community expectations, social sentiment |
| EPS surprise | TVL changes, volume spikes |
| Earnings call | AMAs, community calls |
| Quarterly reports | On-chain metrics reports |

Подробности см. в Rust-примерах в папке `rust_earnings_crypto/`.
