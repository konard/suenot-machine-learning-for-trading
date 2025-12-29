# Crypto Embeddings - Rust

Библиотека векторных представлений слов для анализа криптовалютной торговли с интеграцией биржи Bybit.

## Обзор

Эта Rust библиотека предоставляет инструменты для:
- Получения данных о криптовалютах в реальном времени с биржи Bybit
- Предобработки текстов, связанных с торговлей
- Обучения эмбеддингов в стиле Word2Vec
- Анализа тональности криптовалютных текстов
- Анализа сходства документов

## Структура проекта

```
rust_crypto_embeddings/
├── Cargo.toml           # Зависимости проекта
├── src/
│   ├── lib.rs           # Точка входа библиотеки
│   ├── main.rs          # CLI приложение
│   ├── api/             # Клиент API Bybit
│   │   └── mod.rs
│   ├── embeddings/      # Реализация Word2Vec
│   │   └── mod.rs
│   ├── preprocessing/   # Токенизация текста
│   │   └── mod.rs
│   ├── analysis/        # Анализ тональности и сходства
│   │   └── mod.rs
│   └── utils/           # Общие утилиты
│       └── mod.rs
├── examples/
│   ├── fetch_trades.rs      # Использование API Bybit
│   ├── train_embeddings.rs  # Обучение векторов слов
│   └── analyze_sentiment.rs # Анализ тональности
└── data/
    └── sample_corpus.txt    # Примеры данных для обучения
```

## Установка

Добавьте в ваш `Cargo.toml`:

```toml
[dependencies]
crypto_embeddings = { path = "./rust_crypto_embeddings" }
```

Или клонируйте и соберите:

```bash
cd rust_crypto_embeddings
cargo build --release
```

## Быстрый старт

### 1. Получение рыночных данных

```rust
use crypto_embeddings::BybitClient;

#[tokio::main]
async fn main() {
    let client = BybitClient::new();

    // Получить тикер
    let ticker = client.get_ticker("BTCUSDT").await.unwrap();
    println!("Цена BTC: ${}", ticker.last_price);

    // Получить последние сделки
    let trades = client.get_recent_trades("BTCUSDT", 100).await.unwrap();
    println!("Получено {} сделок", trades.len());
}
```

### 2. Обучение эмбеддингов

```rust
use crypto_embeddings::{Word2Vec, Tokenizer};

fn main() {
    let texts = vec![
        "BTC bullish breakout momentum",
        "ETH bearish support breakdown",
    ];

    // Токенизация
    let tokenizer = Tokenizer::new();
    let sentences: Vec<Vec<String>> = texts
        .iter()
        .map(|t| tokenizer.tokenize(t))
        .collect();

    // Обучение
    let mut model = Word2Vec::new(100, 5, 2);
    model.build_vocab(&sentences);
    model.train(&sentences, 5).unwrap();

    // Поиск похожих слов
    let similar = model.most_similar("bullish", 5).unwrap();
    for (word, score) in similar {
        println!("{}: {:.4}", word, score);
    }
}
```

### 3. Анализ тональности

```rust
use crypto_embeddings::analysis::SentimentAnalyzer;
use crypto_embeddings::Word2Vec;

fn main() {
    let model = Word2Vec::new(10, 2, 1);
    let analyzer = SentimentAnalyzer::new(model);

    let result = analyzer.analyze("BTC pumping to the moon!");
    println!("Тональность: {:?}, Оценка: {}", result.label, result.score);
}
```

## Использование CLI

```bash
# Получить сделки
cargo run -- fetch -s BTCUSDT -l 100 -o trades.csv

# Обучить эмбеддинги
cargo run -- train -i corpus.txt -o model.vec -d 100 -w 5

# Найти похожие слова
cargo run -- similar -m model.vec -w bullish -n 10

# Аналогии слов
cargo run -- analogy -m model.vec --positive btc bullish --negative eth
```

## Запуск примеров

```bash
# Получить сделки с Bybit
cargo run --example fetch_trades

# Обучить эмбеддинги на примерах
cargo run --example train_embeddings

# Анализ тональности
cargo run --example analyze_sentiment
```

## Описание модулей

### Модуль API (`src/api/`)

Клиент Bybit V5 API поддерживает:
- Последние сделки (`get_recent_trades`)
- Стакан заявок (`get_orderbook`)
- Свечи (`get_klines`)
- Информация о тикере (`get_ticker`)
- Список торговых пар (`get_symbols`)

### Модуль предобработки (`src/preprocessing/`)

Утилиты обработки текста:
- `Tokenizer` - Токенизация с удалением стоп-слов
- `CryptoVocab` - Определение крипто-специфичной лексики
- Генерация N-грамм и определение фраз

### Модуль эмбеддингов (`src/embeddings/`)

Реализация Word2Vec:
- Архитектура Skip-gram с отрицательным сэмплированием
- Построение словаря с фильтрацией по частоте
- Сохранение/загрузка модели (текстовый формат word2vec)
- Поиск похожих слов и аналогии

### Модуль анализа (`src/analysis/`)

Инструменты анализа:
- `SentimentAnalyzer` - Анализ тональности на основе лексикона и эмбеддингов
- `SimilarityAnalyzer` - Сходство документов через усреднённые эмбеддинги
- `TrendAnalyzer` - Определение тем из корпуса текстов

## Параметры Word2Vec

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| dim | 100 | Размерность эмбеддинга |
| window | 5 | Размер контекстного окна |
| min_count | 5 | Минимальная частота слова |
| learning_rate | 0.025 | Начальная скорость обучения |
| negative_samples | 5 | Количество негативных примеров |

## Тестирование

```bash
cargo test
```

## Лицензия

MIT License
