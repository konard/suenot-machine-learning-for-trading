# Глава 33: Обнаружение Аномалий — Стратегия Хеджирования Хвостовых Рисков

## Обзор

Аномальные рыночные условия часто предшествуют кризисам или резким движениям. Anomaly detection позволяет идентифицировать "необычные" состояния рынка до того, как они станут очевидными. В этой главе мы строим систему раннего предупреждения и автоматического хеджирования.

## Торговая Стратегия

**Суть стратегии:** Мониторинг multivariate anomaly score. При превышении порога — автоматическая покупка защитных инструментов (VIX calls, put spreads, treasuries).

**Сигнал на хедж:**
- Anomaly score > 95-й перцентиль → Лёгкий хедж (5% портфеля)
- Anomaly score > 99-й перцентиль → Тяжёлый хедж (15% портфеля)

**Инструменты:** VIX calls, SPY puts, TLT longs, Gold

## Техническая Спецификация

### Ноутбуки для создания

| # | Ноутбук | Описание |
|---|---------|----------|
| 1 | `01_market_features.ipynb` | Feature engineering для состояния рынка |
| 2 | `02_historical_crises.ipynb` | Анализ исторических кризисов как ground truth |
| 3 | `03_isolation_forest.ipynb` | Isolation Forest для обнаружения аномалий |
| 4 | `04_one_class_svm.ipynb` | Подход One-Class SVM |
| 5 | `05_autoencoder_anomaly.ipynb` | Ошибка реконструкции автоэнкодера |
| 6 | `06_mahalanobis_distance.ipynb` | Статистический подход |
| 7 | `07_ensemble_detector.ipynb` | Комбинация методов |
| 8 | `08_threshold_calibration.ipynb` | Калибровка порогов на исторических кризисах |
| 9 | `09_hedging_instruments.ipynb` | Выбор и ценообразование защитных инструментов |
| 10 | `10_hedging_strategy.ipynb` | Автоматическая логика хеджирования |
| 11 | `11_backtesting.ipynb` | Бэктест с анализом затрат-выгод |
| 12 | `12_real_time_monitoring.ipynb` | Дашборд для мониторинга |

### Требования к данным

```
Рыночные индикаторы:
├── VIX и структура VIX (VIX, VIX3M, VIX6M)
├── Кредитные спреды (HY-IG, TED spread, LIBOR-OIS)
├── Фондовые индексы (SPY, секторные ETF)
├── Облигационные рынки (TLT, LQD, HYG)
├── Валютный стресс (USD index, JPY, CHF)
├── Сигналы сырьевых товаров (Gold, Oil)
└── Межрыночные корреляции

Исторические даты кризисов (Ground Truth):
├── 2008 Финансовый кризис
├── 2010 Flash Crash
├── 2011 Европейский долговой кризис
├── 2015 Девальвация юаня
├── 2018 Volmageddon
├── 2020 COVID Crash
└── 2022 Шок ставок
```

### Feature Engineering

```python
features = {
    # Волатильность
    'vix_level': VIX,
    'vix_percentile': rolling_percentile(VIX, 252),
    'vix_term_structure': (VIX3M - VIX) / VIX,  # Контанго/Бэквордация
    'realized_vs_implied': realized_vol_20d / VIX,
    'vol_of_vol': rolling_std(VIX, 20),

    # Кредит
    'hy_spread': HYG_yield - treasury_yield,
    'hy_spread_change_5d': hy_spread.pct_change(5),
    'ted_spread': LIBOR_3m - treasury_3m,
    'credit_momentum': LQD.pct_change(20),

    # Акции
    'spy_drawdown': SPY / SPY.rolling(252).max() - 1,
    'spy_momentum_20d': SPY.pct_change(20),
    'breadth': pct_stocks_above_200ma,
    'sector_dispersion': std(sector_returns),

    # Межрыночные
    'stock_bond_corr': rolling_corr(SPY, TLT, 60),
    'gold_correlation': rolling_corr(SPY, GLD, 60),
    'currency_stress': USD_index_change_20d,

    # Внутренняя структура рынка
    'put_call_ratio': equity_put_volume / equity_call_volume,
    'margin_debt_change': margin_debt.pct_change(63),
    'fund_flows': equity_fund_flows_4w
}
```

### Модели обнаружения аномалий

```python
# 1. Isolation Forest
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Ожидаемые 5% аномалий
    random_state=42
)
anomaly_score_if = -iso_forest.decision_function(features)

# 2. One-Class SVM
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.05
)
anomaly_score_svm = -oc_svm.decision_function(features)

# 3. Автоэнкодер
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Anomaly score = ошибка реконструкции
reconstruction_error = torch.mean((x - autoencoder(x))**2, dim=1)

# 4. Расстояние Махаланобиса
from scipy.spatial.distance import mahalanobis

def mahalanobis_score(x, mean, cov_inv):
    return mahalanobis(x, mean, cov_inv)
```

### Ансамблевый скор аномалии

```python
def ensemble_anomaly_score(features, models, weights=None):
    """
    Объединение нескольких детекторов аномалий
    """
    if weights is None:
        weights = [1/len(models)] * len(models)

    scores = []
    for model in models:
        # Нормализация скоров к [0, 1]
        raw_score = model.score(features)
        normalized = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
        scores.append(normalized)

    # Взвешенное среднее
    ensemble_score = sum(w * s for w, s in zip(weights, scores))

    return ensemble_score
```

### Стратегия хеджирования

```python
def hedging_decision(anomaly_score, thresholds, portfolio_value):
    """
    Определение размера хеджа на основе anomaly score
    """
    if anomaly_score > thresholds['extreme']:  # 99-й перцентиль
        hedge_pct = 0.15
        instruments = {
            'VIX_calls': 0.05,
            'SPY_puts': 0.05,
            'TLT': 0.03,
            'GLD': 0.02
        }
    elif anomaly_score > thresholds['high']:  # 95-й перцентиль
        hedge_pct = 0.05
        instruments = {
            'VIX_calls': 0.02,
            'SPY_puts': 0.02,
            'TLT': 0.01
        }
    else:
        hedge_pct = 0
        instruments = {}

    return {k: v * portfolio_value for k, v in instruments.items()}
```

### Анализ затрат и выгод

```
Затраты на хеджирование:
├── VIX calls: ~3-5% decay в месяц при контанго
├── SPY puts: временной распад + спред
├── Альтернативные издержки: деньги на хедж не инвестированы

Выгоды:
├── Снижение просадки во время кризисов
├── Быстрее восстановление (меньше потери капитала)
├── Психологическое преимущество (оставаться в рынке)

Оптимизация:
├── Минимизировать: E[hedge_cost] - λ * E[tail_loss_reduction]
├── Бэктест на множестве кризисов
└── Найти оптимальную калибровку порогов
```

### Ключевые метрики

- **Обнаружение:** True Positive Rate, False Positive Rate, AUC
- **Тайминг:** Дни предупреждения до кризиса, Ложные тревоги в год
- **Портфель:** Max Drawdown с хеджем и без, Время восстановления
- **Стоимость:** Годовые затраты на хедж, Коэффициент эффективности хеджа

### Зависимости

```python
scikit-learn>=1.2.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
yfinance>=0.2.0
```

## Ожидаемые результаты

1. **Набор фичей** из 20+ индикаторов рыночного стресса
2. **Множество детекторов аномалий** (IF, OCSVM, AE, Махаланобис)
3. **Ансамблевая модель** с калиброванными порогами
4. **Стратегия хеджирования** с автоматическим исполнением
5. **Результаты:** 30-50% снижение просадки при < 3% годовых затрат

## Ссылки

- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [A Survey of Anomaly Detection Techniques](https://link.springer.com/article/10.1023/A:1006412129858)
- [Tail Risk Hedging](https://www.amazon.com/Tail-Risk-Hedging-Portfolio-Management/dp/0071791752)
- [Predicting Stock Market Crashes](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2635170)

## Уровень сложности

⭐⭐⭐⭐☆ (Продвинутый)

Требуется понимание: Обнаружение аномалий, Ценообразование опционов, Управление рисками, Анализ кризисов
