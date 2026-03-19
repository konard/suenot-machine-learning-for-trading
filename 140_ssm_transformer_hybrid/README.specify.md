# Chapter 140: SSM-Transformer Hybrid

## Описание

Гибридные архитектуры SSM (Mamba) и Transformer для алгоритмического трейдинга. Объединяет линейную по времени обработку SSM для дальних зависимостей с механизмом внимания Transformer для локальных паттернов.

## Техническое задание

### Цели
1. Изучить теоретические основы гибридных SSM-Transformer архитектур (Jamba, Griffin, Zamba)
2. Реализовать полную модель на Python с PyTorch
3. Создать высокопроизводительную версию на Rust для production
4. Протестировать на данных фондового рынка и криптовалют (Bybit)
5. Провести бэктестинг торговой стратегии с мультизадачными предсказаниями

### Ключевые компоненты
- SSM блок (Mamba-стиль): селективное сканирование с входо-зависимыми параметрами
- Transformer блок: multi-head causal self-attention + FFN
- Гибридная модель: чередование SSM и Transformer слоёв
- Мультизадачные выходы: направление, волатильность, амплитуда доходности
- Взвешивание потерь на основе неопределённости (Kendall et al.)
- Генерация торговых сигналов с обратной волатильностью позиционирования
- Полный фреймворк бэктестинга

### Метрики
- Sharpe Ratio / Sortino Ratio для стратегий
- Maximum Drawdown
- Win Rate / Profit Factor / Calmar Ratio
- Сравнение с LSTM, Transformer, Mamba baselines

## Научные работы

1. **Jamba: A Hybrid Transformer-Mamba Language Model**
   - URL: https://arxiv.org/abs/2403.19887
   - Год: 2024
2. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
   - URL: https://arxiv.org/abs/2312.00752
   - Год: 2023
3. **Griffin: Mixing Gated Linear Recurrences with Local Attention**
   - URL: https://arxiv.org/abs/2402.19427
   - Год: 2024
4. **Transformers are SSMs (Mamba-2)**
   - URL: https://arxiv.org/abs/2405.21060
   - Год: 2024

## Данные
- **Bybit API** (v5): криптовалюты (BTCUSDT, ETHUSDT) — часовые свечи
- **Yahoo Finance** (yfinance): акции (AAPL, SPY) — дневные данные
- Синтетические данные для тестирования (fallback)

## Реализация

### Python
- PyTorch (SSM блок, Transformer блок, гибридная модель)
- NumPy, Pandas (feature engineering)
- requests (Bybit API)
- scikit-learn (метрики)

### Rust
- tokio (async runtime для API)
- reqwest (HTTP клиент для Bybit)
- serde/serde_json (JSON)
- rand/statrs (генерация данных, статистика)
- chrono (временные метки)
- tracing (логирование)

## Структура
```
140_ssm_transformer_hybrid/
├── README.md                        # Основное содержание (English)
├── README.ru.md                     # Перевод (Russian)
├── readme.simple.md                 # Упрощённое объяснение (English)
├── readme.simple.ru.md              # Упрощённое объяснение (Russian)
├── README.specify.md                # Техническое задание
├── Cargo.toml                       # Rust проект
├── Cargo.lock                       # Rust зависимости
├── docs/
│   └── ru/
│       └── theory.md                # Глубокая математическая теория
├── python/
│   ├── __init__.py
│   ├── ssm_transformer_model.py     # Модель (SSMBlock, TransformerBlock, SSMTransformerHybrid)
│   ├── data_loader.py               # Загрузка данных (Bybit, yfinance, synthetic)
│   ├── backtest.py                  # Фреймворк бэктестинга
│   └── requirements.txt             # Python зависимости
├── src/
│   ├── lib.rs                       # Корневой модуль библиотеки
│   ├── model/
│   │   ├── mod.rs
│   │   ├── ssm_block.rs             # SSM блок (selective scan)
│   │   ├── transformer_block.rs     # Transformer блок (attention + FFN)
│   │   └── hybrid.rs                # Гибридная модель
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs                 # Bybit API клиент
│   │   └── features.rs              # Генерация признаков
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── signals.rs               # Торговые сигналы
│   │   └── strategy.rs              # Торговая стратегия
│   └── backtest/
│       ├── mod.rs
│       └── engine.rs                # Движок бэктестинга
└── examples/
    ├── basic_hybrid.rs              # Базовый пример
    ├── bybit_trading.rs             # Торговля с данными Bybit
    └── backtest_strategy.rs         # Полный бэктест
```
