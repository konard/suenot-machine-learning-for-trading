# Rust NLP для криптотрейдинга (Bybit)

Модульная библиотека на Rust для анализа текстовых данных криптовалютного рынка с использованием Bybit API.

## 📦 Структура проекта

```
rust_nlp_crypto/
├── Cargo.toml              # Зависимости и настройки проекта
├── README.md               # Документация
├── src/
│   ├── main.rs             # Точка входа CLI
│   ├── lib.rs              # Экспорт модулей библиотеки
│   ├── api/                # Модуль работы с Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент
│   │   ├── announcements.rs # Новости и анонсы
│   │   └── market.rs       # Рыночные данные
│   ├── nlp/                # Модуль NLP
│   │   ├── mod.rs
│   │   ├── tokenizer.rs    # Токенизация
│   │   ├── preprocessor.rs # Предобработка текста
│   │   └── vectorizer.rs   # Векторизация (Bag of Words, TF-IDF)
│   ├── sentiment/          # Модуль анализа настроений
│   │   ├── mod.rs
│   │   ├── analyzer.rs     # Анализатор настроений
│   │   ├── lexicon.rs      # Словарь настроений
│   │   └── naive_bayes.rs  # Классификатор Наивный Байес
│   ├── signals/            # Торговые сигналы
│   │   ├── mod.rs
│   │   └── generator.rs    # Генератор сигналов
│   └── models/             # Модели данных
│       ├── mod.rs
│       └── types.rs        # Типы данных
└── examples/               # Примеры использования
    ├── fetch_announcements.rs
    ├── sentiment_analysis.rs
    └── trading_signals.rs
```

## 🚀 Быстрый старт

### Установка

```bash
cd rust_nlp_crypto
cargo build --release
```

### Запуск примеров

```bash
# Получение анонсов Bybit
cargo run --example fetch_announcements

# Анализ настроений
cargo run --example sentiment_analysis

# Генерация торговых сигналов
cargo run --example trading_signals
```

### Использование CLI

```bash
# Анализ настроений последних новостей
cargo run -- analyze --source announcements --limit 10

# Генерация сигналов
cargo run -- signals --symbol BTCUSDT --timeframe 1h
```

## 📚 Модули

### 1. API (`src/api/`)

Работа с публичным API Bybit:
- Получение анонсов и новостей
- Получение рыночных данных (цены, объёмы)
- Асинхронные запросы с обработкой ошибок

### 2. NLP (`src/nlp/`)

Обработка естественного языка:
- **Tokenizer**: разбиение текста на токены
- **Preprocessor**: очистка, нормализация, стемминг
- **Vectorizer**: Bag of Words, TF-IDF

### 3. Sentiment (`src/sentiment/`)

Анализ настроений:
- **Lexicon**: словарь с оценками слов
- **Analyzer**: определение полярности текста
- **NaiveBayes**: классификатор для категоризации

### 4. Signals (`src/signals/`)

Генерация торговых сигналов:
- Агрегация настроений по временным периодам
- Комбинирование с рыночными данными
- Выдача рекомендаций (Buy/Sell/Hold)

## 🔧 Примеры кода

### Анализ настроения текста

```rust
use rust_nlp_crypto::sentiment::SentimentAnalyzer;

let analyzer = SentimentAnalyzer::new();
let text = "Bitcoin is showing strong bullish momentum!";
let result = analyzer.analyze(text);

println!("Sentiment: {:?}", result.polarity); // Positive
println!("Score: {:.2}", result.score);       // 0.75
```

### Получение новостей Bybit

```rust
use rust_nlp_crypto::api::BybitClient;

#[tokio::main]
async fn main() {
    let client = BybitClient::new();
    let announcements = client.get_announcements(10).await.unwrap();

    for ann in announcements {
        println!("{}: {}", ann.publish_time, ann.title);
    }
}
```

### Генерация торговых сигналов

```rust
use rust_nlp_crypto::signals::SignalGenerator;

let generator = SignalGenerator::new();
let signals = generator.generate("BTCUSDT", &news_items).await;

for signal in signals {
    println!("Signal: {:?}, Confidence: {:.2}%",
             signal.action, signal.confidence * 100.0);
}
```

## 📊 Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Main                            │
├─────────────────────────────────────────────────────────────┤
│                     Signal Generator                         │
├──────────────────┬──────────────────┬───────────────────────┤
│   Bybit API      │      NLP         │     Sentiment         │
│   - Announcements│   - Tokenizer    │   - Analyzer          │
│   - Market Data  │   - Preprocessor │   - Lexicon           │
│                  │   - Vectorizer   │   - Naive Bayes       │
├──────────────────┴──────────────────┴───────────────────────┤
│                      Models / Types                          │
└─────────────────────────────────────────────────────────────┘
```

## ⚠️ Важно

- Этот код предназначен для **образовательных целей**
- Не используйте для реальной торговли без тщательного тестирования
- API Bybit имеет лимиты запросов — учитывайте это
- Анализ настроений — лишь один из факторов для принятия решений

## 📖 Связь с главой

Этот Rust-проект демонстрирует концепции из главы 14:
- Токенизация и предобработка текста
- Создание матрицы документ-терм (Bag of Words)
- TF-IDF взвешивание
- Наивный Байесовский классификатор
- Анализ настроений для трейдинга

Только вместо Python мы используем Rust для высокой производительности!
