# Тематическое моделирование на Rust для криптовалютных рынков

Этот проект предоставляет модульную реализацию алгоритмов тематического моделирования (LSI и LDA) на Rust для анализа данных криптовалютного рынка с биржи Bybit.

## Возможности

- **Интеграция с Bybit API**: Получение анонсов, рыночных тикеров и исторических данных
- **Предобработка текста**: Токенизация, удаление стоп-слов, TF-IDF векторизация
- **LSI (Латентное семантическое индексирование)**: Извлечение тем на основе SVD
- **LDA (Латентное размещение Дирихле)**: Вероятностное тематическое моделирование с выборкой Гиббса
- **Метрики оценки**: Перплексия, когерентность, разнообразие тем

## Структура проекта

```
rust_examples/
├── Cargo.toml                 # Зависимости проекта
├── README.md                  # Документация (английский)
├── README.ru.md               # Этот файл
├── src/
│   ├── lib.rs                 # Точка входа библиотеки
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs           # Клиент Bybit API
│   ├── preprocessing/
│   │   ├── mod.rs
│   │   ├── tokenizer.rs       # Токенизация текста
│   │   └── vectorizer.rs      # TF-IDF и count векторизация
│   ├── models/
│   │   ├── mod.rs
│   │   ├── lsi.rs             # Реализация LSI
│   │   └── lda.rs             # Реализация LDA
│   ├── utils/
│   │   ├── mod.rs
│   │   ├── io.rs              # Загрузка/сохранение данных
│   │   └── evaluation.rs      # Метрики оценки моделей
│   └── bin/
│       ├── fetch_data.rs      # Пример получения данных
│       ├── lsi_example.rs     # Демонстрация LSI
│       ├── lda_example.rs     # Демонстрация LDA
│       └── analyze_market.rs  # Полный пайплайн анализа
└── data/                      # Сохранённые датасеты (создаются при запуске)
```

## Установка

### Требования

- Rust 1.70+ (установка с https://rustup.rs)
- OpenBLAS (для операций линейной алгебры)

### Ubuntu/Debian
```bash
sudo apt-get install libopenblas-dev
```

### macOS
```bash
brew install openblas
```

### Сборка
```bash
cd rust_examples
cargo build --release
```

## Использование

### 1. Получение данных с Bybit

```bash
cargo run --release --bin fetch_data
```

Это:
- Подключится к публичному API Bybit
- Загрузит последние анонсы
- Получит текущие рыночные тикеры
- Сохранит данные для анализа

### 2. Запуск примера LSI

```bash
cargo run --release --bin lsi_example
```

Вывод включает:
- Обнаруженные темы с ключевыми словами
- Распределение документов по темам
- Анализ сходства документов

### 3. Запуск примера LDA

```bash
cargo run --release --bin lda_example
```

Вывод включает:
- Распределения тем
- Показатели перплексии и когерентности
- Категоризацию документов

### 4. Полный анализ рынка

```bash
cargo run --release --bin analyze_market
```

Объединяет:
- Данные рынка в реальном времени
- Извлечение тем с помощью LSI и LDA
- Корреляцию символов и тем
- Генерацию торговых инсайтов

## API Справочник

### Клиент Bybit

```rust
use topic_modeling::api::bybit::BybitClient;

let client = BybitClient::new();

// Получить анонсы
let announcements = client.get_announcements("en-US", 50)?;

// Получить рыночные тикеры
let tickers = client.get_tickers("spot", Some("BTCUSDT"))?;

// Получить исторические свечи
let klines = client.get_klines("spot", "BTCUSDT", "60", 100)?;
```

### Предобработка текста

```rust
use topic_modeling::preprocessing::tokenizer::Tokenizer;
use topic_modeling::preprocessing::vectorizer::TfIdfVectorizer;

// Создать токенизатор для крипто-текстов
let tokenizer = Tokenizer::for_crypto().min_length(3);
let tokens = tokenizer.tokenize("Цена Bitcoin растёт...");

// Построить TF-IDF матрицу
let mut vectorizer = TfIdfVectorizer::new()
    .min_df(2)
    .max_df_ratio(0.8)
    .max_features(500);

let matrix = vectorizer.fit_transform(&tokenized_docs);
```

### Модель LSI

```rust
use topic_modeling::models::lsi::LSI;

let mut lsi = LSI::new(5)?;  // 5 тем
lsi.fit(&tfidf_matrix, vocabulary, terms)?;

// Получить обнаруженные темы
let topics = lsi.get_topics(10)?;  // Топ-10 слов на тему

// Найти похожие документы
let similar = lsi.most_similar_documents(0, 5)?;
```

### Модель LDA

```rust
use topic_modeling::models::lda::{LDA, LdaConfig};

let config = LdaConfig::new(5)   // 5 тем
    .alpha(0.1)                  // Априор документ-тема
    .beta(0.01)                  // Априор тема-слово
    .n_iterations(1000)
    .random_seed(42);

let mut lda = LDA::new(config)?;
lda.fit(&count_matrix, vocabulary, terms)?;

// Получить темы с вероятностями
let topics = lda.get_topics(10)?;

// Получить перплексию
let perplexity = lda.perplexity(&count_matrix)?;
```

## Конфигурация

### Гиперпараметры LDA

| Параметр | Описание | Типичные значения |
|----------|----------|-------------------|
| `n_topics` | Количество тем | 5-50 |
| `alpha` | Априор документ-тема | 0.01-0.5 (меньше = разреженнее) |
| `beta` | Априор тема-слово | 0.001-0.1 (меньше = разреженнее) |
| `n_iterations` | Итерации выборки Гиббса | 500-2000 |
| `burn_in` | Итерации для отбрасывания | 50-200 |

### Опции токенизатора

```rust
let tokenizer = Tokenizer::new()
    .min_length(2)           // Минимальная длина токена
    .max_length(50)          // Максимальная длина токена
    .lowercase(true)         // Приводить к нижнему регистру
    .remove_numbers(false);  // Сохранять числа для крипто-контекста
```

## Примеры

### Анализ крипто-анонсов

```rust
use topic_modeling::api::bybit::{BybitClient, MarketDocument};
use topic_modeling::models::lda::{LDA, LdaConfig};

// Получить анонсы
let client = BybitClient::new();
let announcements = client.get_announcements("en-US", 50)?;

// Преобразовать в документы
let docs: Vec<MarketDocument> = announcements
    .iter()
    .map(MarketDocument::from_announcement)
    .collect();

// Предобработка и обучение LDA...
```

### Отслеживание трендов тем

```rust
// Группировка документов по временным периодам
let mut period_docs = HashMap::new();
for doc in documents {
    let period = doc.timestamp / (24 * 3600 * 1000);  // По дням
    period_docs.entry(period).or_insert(Vec::new()).push(doc);
}

// Обучить LDA на каждом периоде и сравнить распределения тем
```

## Советы по производительности

1. **Используйте Release режим**: Всегда запускайте с `--release` для ускорения в 10-50 раз
2. **Ограничивайте словарь**: Используйте `max_features` для ограничения размера словаря
3. **Настраивайте итерации**: Начните с меньшего числа итераций, увеличивайте при необходимости
4. **Параллельная обработка**: Библиотека использует Rayon для параллельных операций

## Тестирование

```bash
# Запустить все тесты
cargo test

# Запустить тесты конкретного модуля
cargo test models::lda

# Запустить с выводом
cargo test -- --nocapture
```

## Ограничения

- LSI использует power iteration SVD (медленнее LAPACK, но без внешних зависимостей)
- Выборка Гиббса в LDA может быть медленной для больших корпусов
- Нет GPU-ускорения

## Лицензия

MIT License - см. файл LICENSE для деталей.

## Ссылки

- [Латентный семантический анализ](https://ru.wikipedia.org/wiki/Латентно-семантический_анализ)
- [Латентное размещение Дирихле](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [Документация Bybit API](https://bybit-exchange.github.io/docs/)
