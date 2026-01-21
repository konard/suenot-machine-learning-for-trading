# Глава 84: Matching Networks для финансов

## Обзор

Matching Networks - это подход мета-обучения для one-shot и few-shot обучения, использующий классификатор на основе внимания (attention) поверх обученных эмбеддингов. В отличие от прототипных сетей, которые вычисляют центроиды классов, Matching Networks используют мягкий подход ближайших соседей с весами внимания для классификации новых примеров. Это делает их особенно мощными для финансовых приложений, где рыночные паттерны могут не образовывать простые кластеры.

## Содержание

1. [Введение](#введение)
2. [Теоретические основы](#теоретические-основы)
3. [Компоненты архитектуры](#компоненты-архитектуры)
4. [Полноконтекстные эмбеддинги (FCE)](#полноконтекстные-эмбеддинги-fce)
5. [Применение на финансовых рынках](#применение-на-финансовых-рынках)
6. [Few-Shot распознавание рыночных паттернов](#few-shot-распознавание-рыночных-паттернов)
7. [Стратегия реализации](#стратегия-реализации)
8. [Интеграция с Bybit](#интеграция-с-bybit)
9. [Управление рисками](#управление-рисками)
10. [Метрики производительности](#метрики-производительности)
11. [Сравнение с прототипными сетями](#сравнение-с-прототипными-сетями)
12. [Ссылки](#ссылки)

---

## Введение

Традиционные подходы машинного обучения для трейдинга требуют большого количества размеченных данных для каждого рыночного паттерна или режима. Однако финансовые рынки представляют уникальные вызовы:

- **Редкость паттернов**: Некоторые торговые паттерны (двойная вершина, голова и плечи) встречаются редко
- **Контекстная чувствительность**: Один и тот же паттерн может иметь разное значение в разных рыночных контекстах
- **Быстрая адаптация**: Необходимость распознавать новые паттерны с небольшим количеством примеров

### Почему Matching Networks для трейдинга?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Проблема классификации на основе внимания             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Прототипные сети:                   Matching Networks:                │
│   ────────────────                    ─────────────────                  │
│   Сравнение с центроидами классов     Сравнение со ВСЕМИ примерами     │
│   Простое усреднение                  Взвешенное голосование           │
│                                                                          │
│   Проблема: Паттерны могут не         Решение: Учимся сходству с       │
│   образовывать кластеры вокруг        отдельными примерами, используя  │
│   единого центроида                   контекстно-зависимые эмбеддинги  │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────┐        │
│   │                                                            │        │
│   │   Прототипные:   [●] ← центроид   Matching:               │        │
│   │                   ▲                 Query → [?]            │        │
│   │                 / | \                       ↓              │        │
│   │               ○   ○   ○              [a₁○ a₂○ a₃○ ...]    │        │
│   │             (support)                веса внимания         │        │
│   │                                                            │        │
│   └────────────────────────────────────────────────────────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Ключевые преимущества

| Аспект | Прототипные сети | Matching Networks |
|--------|------------------|-------------------|
| Метод классификации | Расстояние до центроидов | Внимание по всем примерам |
| Осведомлённость о контексте | Ограниченная | Полная через FCE |
| Эмбеддинги | Фиксированные | Контекстно-зависимые |
| Гибкость паттернов | Предполагает кластерную структуру | Обрабатывает сложные распределения |
| Вычислительная стоимость | Ниже | Выше (вычисление внимания) |
| Интерпретируемость | Высокая (расстояния до прототипов) | Высокая (веса внимания) |

## Теоретические основы

### Фреймворк Matching Networks

Matching Networks обучают нейронную сеть, которая отображает небольшой размеченный support set S и неразмеченный пример x̂ на его метку ŷ, без необходимости дообучения:

$$P(ŷ | x̂, S) = \sum_{i=1}^{k} a(x̂, x_i) y_i$$

где $a(x̂, x_i)$ - механизм внимания, вычисляющий насколько query $x̂$ похож на каждый support пример $x_i$.

### Математическая формулировка

**Ядро внимания**: Внимание вычисляется как softmax по косинусным сходствам:

$$a(x̂, x_i) = \frac{\exp(c(f(x̂), g(x_i)))}{\sum_{j=1}^{k} \exp(c(f(x̂), g(x_j)))}$$

где:
- $f$ - функция эмбеддинга для query
- $g$ - функция эмбеддинга для support примеров
- $c$ - косинусное сходство: $c(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$

### Ключевые компоненты

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Архитектура Matching Networks                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SUPPORT SET S = {(x₁,y₁), (x₂,y₂), ..., (xₖ,yₖ)}                    │
│        ↓                                                                │
│   ┌────────────────────────────────────────┐                           │
│   │  g(xᵢ, S) - Эмбеддинг support          │                           │
│   │  ──────────────────────────────        │                           │
│   │  Использует двунаправленную LSTM       │                           │
│   │  для кодирования каждого примера       │                           │
│   │  с контекстом всего support set        │                           │
│   └────────────────────────────────────────┘                           │
│                                                                         │
│   QUERY x̂                                                              │
│        ↓                                                                │
│   ┌────────────────────────────────────────┐                           │
│   │  f(x̂, S) - Эмбеддинг Query (FCE)       │                           │
│   │  ──────────────────────────────        │                           │
│   │  LSTM с вниманием, который читает      │                           │
│   │  эмбеддинги support set                │                           │
│   └────────────────────────────────────────┘                           │
│                                                                         │
│   КЛАССИФИКАЦИЯ                                                         │
│        ↓                                                                │
│   ┌────────────────────────────────────────┐                           │
│   │  Взвешенное голосование на основе      │                           │
│   │  внимания:                              │                           │
│   │  P(y|x̂,S) = Σᵢ a(x̂,xᵢ)yᵢ             │                           │
│   │                                        │                           │
│   │  где a = softmax(cosine_similarity)    │                           │
│   └────────────────────────────────────────┘                           │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Эпизодическое обучение

Обучение повторяет структуру задач тестового времени:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Процесс эпизодического обучения                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Каждый эпизод симулирует задачу few-shot классификации:             │
│                                                                         │
│   Шаг 1: Выбираем N классов (напр., 5 рыночных паттернов)             │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │ Двойная вершина │ Двойное дно │ Пробой │ Разворот │ Тренд │        │
│   └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Шаг 2: Для каждого класса выбираем K support + Q query примеров     │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Двойная вершина: [s1, s2, s3, s4, s5] | [q1, q2, q3]      │      │
│   │  Двойное дно:     [s1, s2, s3, s4, s5] | [q1, q2, q3]      │      │
│   │  ...                                                        │      │
│   └─────────────────────────────────────────────────────────────┘      │
│        Support Set (5-shot)       Query Set                             │
│                                                                         │
│   Шаг 3: Кодируем support примеры с помощью g(·, S)                   │
│   Шаг 4: Кодируем query примеры с помощью f(·, S)                     │
│   Шаг 5: Вычисляем веса внимания и предсказания                       │
│   Шаг 6: Вычисляем cross-entropy loss и делаем backprop               │
│                                                                         │
│   Ключевой инсайт: Условия обучения и тестирования должны совпадать!   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Компоненты архитектуры

### Функции эмбеддинга

**Базовый эмбеддинг (без FCE)**:

При использовании простых эмбеддингов без Full Context Embeddings, обе функции f и g являются одной нейронной сетью:

```python
# Простой эмбеддинг: f = g = neural_network
embedding = neural_network(x)  # Одинаково для query и support
```

**Полноконтекстные эмбеддинги (FCE)**:

FCE использует контекстно-зависимые эмбеддинги, которые обусловлены всем support set:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Полноконтекстные эмбеддинги (FCE)                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SUPPORT ЭМБЕДДИНГ g(xᵢ, S):                                          │
│   ────────────────────────────                                         │
│   Шаг 1: Вычисляем базовые эмбеддинги g'(xⱼ) для всех xⱼ ∈ S         │
│   Шаг 2: Пропускаем через двунаправленную LSTM все g'(xⱼ)            │
│   Шаг 3: g(xᵢ, S) = h→ᵢ + h←ᵢ + g'(xᵢ)  (прямой + обратный + skip)   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │  g'(x₁) → [LSTM→] → h→₁                                 │          │
│   │  g'(x₂) → [LSTM→] → h→₂                                 │          │
│   │  g'(x₃) → [LSTM→] → h→₃                                 │          │
│   │          ← [LSTM←]                                       │          │
│   │  g(xᵢ,S) = h→ᵢ + h←ᵢ + g'(xᵢ)                          │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
│   QUERY ЭМБЕДДИНГ f(x̂, S):                                             │
│   ────────────────────────                                             │
│   Использует LSTM с вниманием по support эмбеддингам                  │
│                                                                         │
│   for k = 1 to K шагов обработки:                                     │
│       h, c = LSTM(f'(x̂), [h, r], c)                                   │
│       h = h + f'(x̂)              # Skip-соединение                    │
│       r = attention_readout(h, g(S))  # Внимание к support            │
│   f(x̂, S) = h                                                         │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │  f'(x̂) → [LSTM] → h₁ → attention → r₁                  │          │
│   │          → [LSTM] → h₂ → attention → r₂                 │          │
│   │          → ...                                          │          │
│   │          → [LSTM] → hₖ = f(x̂, S)                       │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Механизм чтения с вниманием

Эмбеддинг query использует внимание для сбора информации из support примеров:

```python
def attention_readout(h, g_support):
    """
    Вычисляет взвешенное по вниманию чтение из support эмбеддингов.

    Args:
        h: Текущее скрытое состояние query
        g_support: Support эмбеддинги (n_support, embed_dim)

    Returns:
        r: Вектор чтения (взвешенная сумма support эмбеддингов)
    """
    # Вычисляем веса внимания
    scores = h @ g_support.T  # (embed_dim,) @ (n_support, embed_dim).T
    attention = softmax(scores)  # (n_support,)

    # Вычисляем взвешенную сумму
    r = attention @ g_support  # (embed_dim,)

    return r
```

### Сеть эмбеддинга для трейдинга

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Базовая сеть эмбеддинга для трейдинга               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Вход: Рыночные признаки [OHLCV, индикаторы, стакан, ...]            │
│   Форма: (batch_size, sequence_length, feature_dim)                    │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Блок временных свёрток                                     │      │
│   │  ─────────────────────────                                  │      │
│   │  Conv1D(in=features, out=64, kernel=3) → BatchNorm → ReLU   │      │
│   │  Conv1D(in=64, out=128, kernel=3) → BatchNorm → ReLU        │      │
│   │  Conv1D(in=128, out=128, kernel=3) → BatchNorm → ReLU       │      │
│   │  MaxPool1D(kernel=2)                                        │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            ↓                                            │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Блок двунаправленной LSTM                                  │      │
│   │  ────────────────────────────                               │      │
│   │  BiLSTM(hidden=128, layers=2)                               │      │
│   │  Выход: concatenate(forward, backward) = 256 измерений      │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            ↓                                            │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  Голова проекции                                             │      │
│   │  ───────────────                                            │      │
│   │  Linear(in=256, out=128) → ReLU                             │      │
│   │  Linear(in=128, out=embedding_dim)                          │      │
│   │  L2 нормализация                                            │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            ↓                                            │
│   Выход: g'(x) - Базовый вектор эмбеддинга (embedding_dim,)            │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Полноконтекстные эмбеддинги (FCE)

### Почему FCE важны для трейдинга

Финансовые рынки сильно зависят от контекста. Один и тот же паттерн может иметь разные значения в зависимости от:

- **Рыночного режима**: Бычий vs. медвежий рынок
- **Волатильности**: Высокая vs. низкая волатильность
- **Других активов**: Коррелированные движения
- **Состава support set**: Какие примеры доступны

FCE позволяет сети:
1. Обуславливать support эмбеддинги всем support set
2. Обуславливать query эмбеддинги доступными support примерами
3. Создавать контекстно-зависимые представления

### Кодирование Support Set с BiLSTM

```rust
/// Полноконтекстный эмбеддинг для Support Set
pub struct SupportSetEncoder {
    /// Базовая сеть эмбеддинга
    base_embedding: BaseEmbedding,
    /// Двунаправленная LSTM для контекста
    bi_lstm: BiLSTM,
}

impl SupportSetEncoder {
    /// Кодирование support set с полным контекстом
    pub fn encode(&self, support_set: &[MarketWindow]) -> Vec<Embedding> {
        // Шаг 1: Получаем базовые эмбеддинги для всех support примеров
        let base_embeddings: Vec<Embedding> = support_set
            .iter()
            .map(|x| self.base_embedding.forward(x))
            .collect();

        // Шаг 2: Пропускаем через двунаправленную LSTM
        let forward_states = self.bi_lstm.forward(&base_embeddings);
        let backward_states = self.bi_lstm.backward(&base_embeddings);

        // Шаг 3: Комбинируем со skip-соединением
        base_embeddings
            .iter()
            .enumerate()
            .map(|(i, base)| {
                &forward_states[i] + &backward_states[i] + base
            })
            .collect()
    }
}
```

### Кодирование Query с вниманием

```rust
/// Полноконтекстный эмбеддинг для Query
pub struct QueryEncoder {
    /// Базовая сеть эмбеддинга
    base_embedding: BaseEmbedding,
    /// LSTM для итеративного уточнения
    lstm: LSTM,
    /// Количество шагов обработки
    num_steps: usize,
}

impl QueryEncoder {
    /// Кодирование query с вниманием к support set
    pub fn encode(
        &self,
        query: &MarketWindow,
        support_embeddings: &[Embedding],
    ) -> Embedding {
        // Получаем базовый эмбеддинг query
        let query_base = self.base_embedding.forward(query);

        // Инициализируем состояние LSTM
        let mut h = query_base.clone();
        let mut c = Embedding::zeros(self.lstm.hidden_dim);

        // Итеративное уточнение с вниманием
        for _ in 0..self.num_steps {
            // Конкатенируем h с readout для входа LSTM
            let r = self.attention_readout(&h, support_embeddings);
            let input = concat(&[&query_base, &r]);

            // Шаг LSTM
            (h, c) = self.lstm.step(&input, (&h, &c));

            // Skip-соединение
            h = h + &query_base;
        }

        h
    }

    /// Вычисление взвешенного по вниманию чтения из support эмбеддингов
    fn attention_readout(
        &self,
        h: &Embedding,
        support_embeddings: &[Embedding],
    ) -> Embedding {
        // Вычисляем веса внимания (косинусное сходство)
        let scores: Vec<f32> = support_embeddings
            .iter()
            .map(|s| cosine_similarity(h, s))
            .collect();

        // Softmax
        let attention = softmax(&scores);

        // Взвешенная сумма
        support_embeddings
            .iter()
            .zip(attention.iter())
            .fold(Embedding::zeros(h.dim()), |acc, (s, &a)| {
                acc + s * a
            })
    }
}
```

## Применение на финансовых рынках

### Классификация рыночных паттернов

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Классы рыночных паттернов                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Класс 0: ПРОДОЛЖЕНИЕ_ТРЕНДА                                          │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Характеристики:                                           │        │
│   │  • Цена движется в установленном направлении              │        │
│   │  • Откаты в рамках структуры тренда                       │        │
│   │  • Объём подтверждает движения                            │        │
│   │  • Индикаторы моментума выровнены                         │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Класс 1: РАЗВОРОТ_ТРЕНДА                                             │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Характеристики:                                           │        │
│   │  • Дивергенция в индикаторах моментума                    │        │
│   │  • Всплеск объёма на разворотной свече                    │        │
│   │  • Пробой ключевой поддержки/сопротивления               │        │
│   │  • Изменение структуры рынка                              │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Класс 2: ПРОБОЙ                                                      │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Характеристики:                                           │        │
│   │  • Цена пробивает диапазон консолидации                   │        │
│   │  • Расширение объёма                                       │        │
│   │  • Рост волатильности                                      │        │
│   │  • Сильное направленное движение                          │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Класс 3: ЛОЖНЫЙ_ПРОБОЙ                                               │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Характеристики:                                           │        │
│   │  • Начальный пробой ключевого уровня                      │        │
│   │  • Быстрый разворот обратно в диапазон                   │        │
│   │  • Объём не подтверждает                                  │        │
│   │  • Ловушка для трейдеров                                  │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
│   Класс 4: КОНСОЛИДАЦИЯ                                                │
│   ┌───────────────────────────────────────────────────────────┐        │
│   │  Характеристики:                                           │        │
│   │  • Цена в диапазоне                                       │        │
│   │  • Снижающаяся волатильность                              │        │
│   │  • Сужение объёма                                         │        │
│   │  • Накопление энергии для следующего движения             │        │
│   └───────────────────────────────────────────────────────────┘        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Торговая стратегия на основе распознавания паттернов

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Торговые сигналы на основе паттернов                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Пайплайн детекции паттернов:                                         │
│   ─────────────────────────────                                        │
│   1. Поддерживаем support set размеченных примеров паттернов          │
│   2. Извлекаем признаки текущего рыночного окна                       │
│   3. Кодируем support set с FCE (функция g)                           │
│   4. Кодируем query с вниманием к support (функция f)                 │
│   5. Вычисляем веса внимания ко всем support примерам                 │
│   6. Генерируем предсказание через взвешенное голосование             │
│   7. Исполняем торговый сигнал на основе паттерна и уверенности       │
│                                                                         │
│   Паттерн → Сигнал:                                                    │
│   ┌───────────────────────┬──────────────────────────────────────┐    │
│   │ Паттерн               │ Действие                              │    │
│   ├───────────────────────┼──────────────────────────────────────┤    │
│   │ ПРОДОЛЖЕНИЕ_ТРЕНДА    │ Вход/удержание по направлению тренда │    │
│   │ РАЗВОРОТ_ТРЕНДА       │ Закрытие позиций, рассмотреть разворот│   │
│   │ ПРОБОЙ                │ Вход в направлении пробоя            │    │
│   │ ЛОЖНЫЙ_ПРОБОЙ         │ Торговля против движения, тесный стоп│    │
│   │ КОНСОЛИДАЦИЯ          │ Ожидание, уменьшение размера позиции │    │
│   └───────────────────────┴──────────────────────────────────────┘    │
│                                                                         │
│   Размер позиции на основе уверенности:                                │
│   ──────────────────────────────────────                               │
│   position_size = base_size × max_attention_weight                     │
│                                                                         │
│   Интерпретируемость через внимание:                                   │
│   ───────────────────────────────────                                  │
│   • Наибольшие веса внимания показывают какие support примеры         │
│     наиболее похожи на текущий рынок                                  │
│   • Обеспечивает объяснимые предсказания                              │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Few-Shot распознавание рыночных паттернов

### Генерация эпизодов для обучения

```python
def generate_matching_episode(
    dataset,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 10
):
    """
    Генерация одного обучающего эпизода для matching network.

    Args:
        dataset: Исторические рыночные данные с метками паттернов
        n_way: Количество классов паттернов в эпизоде
        k_shot: Количество support примеров на класс
        n_query: Количество query примеров на класс

    Returns:
        support_set: Список кортежей (features, label)
        query_set: Список кортежей (features, label)
    """
    # Выбираем n_way классов из доступных классов паттернов
    available_classes = dataset.get_pattern_classes()
    sampled_classes = random.sample(available_classes, n_way)

    support_set = []
    query_set = []

    for new_label, original_class in enumerate(sampled_classes):
        # Получаем все примеры для этого паттерна
        class_samples = dataset.get_samples_for_pattern(original_class)

        # Выбираем k_shot + n_query примеров
        sampled_indices = random.sample(
            range(len(class_samples)),
            k_shot + n_query
        )

        # Разделяем на support и query
        for i, idx in enumerate(sampled_indices):
            if i < k_shot:
                support_set.append((class_samples[idx], new_label))
            else:
                query_set.append((class_samples[idx], new_label))

    return support_set, query_set
```

### Прямой проход Matching Network

```python
def matching_network_forward(
    support_set,
    query,
    base_embedding_fn,
    support_encoder,
    query_encoder,
):
    """
    Прямой проход matching network.

    Args:
        support_set: Список кортежей (features, label)
        query: Признаки query
        base_embedding_fn: Базовая функция эмбеддинга g'
        support_encoder: FCE энкодер support
        query_encoder: FCE энкодер query

    Returns:
        class_probabilities: Распределение вероятностей по классам
        attention_weights: Веса внимания к каждому support примеру
    """
    # Разделяем признаки и метки
    support_features = [s[0] for s in support_set]
    support_labels = [s[1] for s in support_set]
    n_classes = len(set(support_labels))

    # Кодируем support set с FCE
    support_embeddings = support_encoder.encode(support_features)

    # Кодируем query с вниманием к support
    query_embedding = query_encoder.encode(query, support_embeddings)

    # Вычисляем веса внимания (косинусное сходство + softmax)
    similarities = [
        cosine_similarity(query_embedding, s_emb)
        for s_emb in support_embeddings
    ]
    attention_weights = softmax(similarities)

    # Взвешенное голосование: P(y|x) = Σ a(x, x_i) * y_i
    class_probabilities = np.zeros(n_classes)
    for i, (weight, label) in enumerate(zip(attention_weights, support_labels)):
        class_probabilities[label] += weight

    return class_probabilities, attention_weights
```

## Стратегия реализации

### Python реализация (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingNetwork(nn.Module):
    """
    Matching Network для few-shot классификации рыночных паттернов.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        lstm_layers: int = 1,
        fce_steps: int = 5,
        use_fce: bool = True,
    ):
        super().__init__()

        self.use_fce = use_fce
        self.fce_steps = fce_steps

        # Базовая сеть эмбеддинга
        self.base_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        if use_fce:
            # Энкодер support set (двунаправленная LSTM)
            self.support_lstm = nn.LSTM(
                embedding_dim,
                embedding_dim,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
            )
            self.support_proj = nn.Linear(embedding_dim * 2, embedding_dim)

            # Энкодер query (LSTM с вниманием)
            self.query_lstm = nn.LSTMCell(
                embedding_dim * 2,  # Эмбеддинг query + readout
                embedding_dim,
            )

    def encode_support(self, support: torch.Tensor) -> torch.Tensor:
        """
        Кодирование support set с FCE.

        Args:
            support: (n_support, input_dim)

        Returns:
            embeddings: (n_support, embedding_dim)
        """
        # Базовые эмбеддинги
        base_emb = self.base_embedding(support)  # (n_support, embedding_dim)

        if not self.use_fce:
            return F.normalize(base_emb, dim=-1)

        # Применяем двунаправленную LSTM
        base_emb_seq = base_emb.unsqueeze(0)  # (1, n_support, embedding_dim)
        lstm_out, _ = self.support_lstm(base_emb_seq)  # (1, n_support, 2*embedding_dim)
        lstm_out = lstm_out.squeeze(0)  # (n_support, 2*embedding_dim)

        # Проекция и skip-соединение
        context_emb = self.support_proj(lstm_out)  # (n_support, embedding_dim)
        full_emb = context_emb + base_emb  # Skip-соединение

        return F.normalize(full_emb, dim=-1)

    def encode_query(
        self,
        query: torch.Tensor,
        support_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Кодирование query с вниманием к support.

        Args:
            query: (batch_size, input_dim)
            support_embeddings: (n_support, embedding_dim)

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        batch_size = query.shape[0]
        embed_dim = support_embeddings.shape[-1]

        # Базовый эмбеддинг
        query_base = self.base_embedding(query)  # (batch_size, embedding_dim)

        if not self.use_fce:
            return F.normalize(query_base, dim=-1)

        # Инициализируем состояние LSTM
        h = query_base
        c = torch.zeros_like(h)

        # Итеративное уточнение с вниманием
        for _ in range(self.fce_steps):
            # Чтение с вниманием
            r = self._attention_readout(h, support_embeddings)  # (batch_size, embedding_dim)

            # Вход LSTM: конкатенация базового query и readout
            lstm_input = torch.cat([query_base, r], dim=-1)  # (batch_size, 2*embedding_dim)

            # Шаг LSTM
            h, c = self.query_lstm(lstm_input, (h, c))

            # Skip-соединение
            h = h + query_base

        return F.normalize(h, dim=-1)

    def _attention_readout(
        self,
        query: torch.Tensor,
        support_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисление взвешенного по вниманию чтения из support.

        Args:
            query: (batch_size, embedding_dim)
            support_embeddings: (n_support, embedding_dim)

        Returns:
            readout: (batch_size, embedding_dim)
        """
        # Косинусное сходство
        query_norm = F.normalize(query, dim=-1)
        support_norm = F.normalize(support_embeddings, dim=-1)

        # (batch_size, n_support)
        similarities = torch.mm(query_norm, support_norm.t())
        attention = F.softmax(similarities, dim=-1)

        # Взвешенная сумма
        readout = torch.mm(attention, support_embeddings)  # (batch_size, embedding_dim)

        return readout

    def forward(
        self,
        support: torch.Tensor,
        support_labels: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход для классификации.

        Args:
            support: (n_support, input_dim)
            support_labels: (n_support,)
            query: (n_query, input_dim)

        Returns:
            class_probs: (n_query, n_classes)
            attention: (n_query, n_support)
        """
        # Кодируем support и query
        support_emb = self.encode_support(support)
        query_emb = self.encode_query(query, support_emb)

        # Вычисляем веса внимания
        similarities = torch.mm(query_emb, support_emb.t())  # (n_query, n_support)
        attention = F.softmax(similarities, dim=-1)

        # Агрегируем по классам
        n_classes = support_labels.max().item() + 1
        class_probs = torch.zeros(query.shape[0], n_classes, device=query.device)

        for i in range(support.shape[0]):
            class_probs[:, support_labels[i]] += attention[:, i]

        return class_probs, attention
```

### Rust реализация

Смотрите директорию `src/` для полной Rust реализации с:
- `network/` - Архитектура matching network
- `data/` - Интеграция Bybit и данных акций
- `strategy/` - Реализация торговой стратегии
- `training/` - Утилиты эпизодического обучения

## Интеграция с Bybit

### Детекция паттернов в реальном времени

```rust
use crate::data::bybit::{BybitClient, KlineData};
use crate::network::MatchingNetwork;
use crate::strategy::PatternStrategy;

/// Детектор паттернов в реальном времени с использованием Matching Networks
pub struct BybitPatternDetector {
    client: BybitClient,
    network: MatchingNetwork,
    support_set: SupportSet,
    strategy: PatternStrategy,
}

impl BybitPatternDetector {
    /// Детекция паттерна в текущем рынке
    pub async fn detect_pattern(&self, symbol: &str) -> Result<PatternDetection> {
        // Получаем последние свечи
        let klines = self.client
            .get_klines(symbol, "15m", 100)
            .await?;

        // Извлекаем признаки
        let features = self.extract_features(&klines);

        // Классифицируем с помощью matching network
        let (class_probs, attention) = self.network.forward(
            &self.support_set,
            &features,
        );

        // Получаем предсказание и уверенность
        let predicted_class = class_probs.argmax();
        let confidence = class_probs[predicted_class];

        // Находим наиболее похожие support примеры
        let similar_examples = self.get_top_attention(attention, 3);

        Ok(PatternDetection {
            pattern: predicted_class.into(),
            confidence,
            similar_examples,
            trading_signal: self.strategy.generate_signal(predicted_class, confidence),
        })
    }
}
```

### Извлечение признаков

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Извлечение признаков для Matching                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Ценовые признаки:                                                    │
│   • Доходности (разные таймфреймы)                                     │
│   • Позиция цены относительно скользящих средних                      │
│   • Паттерн более высоких максимумов / более низких минимумов         │
│   • Близость к поддержке/сопротивлению                                │
│                                                                         │
│   Объёмные признаки:                                                   │
│   • Объём относительно среднего                                        │
│   • Профиль объёма (распределение по ценовым уровням)                 │
│   • Соотношение объёма покупок/продаж (из сделок Bybit)              │
│                                                                         │
│   Признаки волатильности:                                              │
│   • ATR (Average True Range)                                           │
│   • Ширина полос Боллинджера                                          │
│   • Индикатор режима волатильности                                    │
│                                                                         │
│   Признаки моментума:                                                  │
│   • RSI и дивергенция RSI                                             │
│   • MACD и гистограмма                                                │
│   • Стохастический осциллятор                                         │
│                                                                         │
│   Крипто-специфичные признаки:                                        │
│   • Ставка финансирования                                             │
│   • Изменение открытого интереса                                      │
│   • Соотношение Long/Short                                            │
│   • Уровни ликвидаций                                                 │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Управление рисками

### Размер позиции на основе уверенности

```rust
impl PatternStrategy {
    /// Расчёт размера позиции на основе уверенности в паттерне
    pub fn calculate_position_size(
        &self,
        confidence: f32,
        attention_entropy: f32,
        account_balance: f32,
    ) -> f32 {
        // Базовая позиция как процент от счёта
        let base_position = account_balance * self.base_risk_pct;

        // Масштабирование по уверенности (выше уверенность = больше позиция)
        let confidence_factor = if confidence > 0.8 {
            1.0
        } else if confidence > 0.6 {
            0.7
        } else if confidence > 0.4 {
            0.4
        } else {
            0.0  // Не торгуем если уверенность слишком низкая
        };

        // Масштабирование по энтропии внимания (ниже энтропия = более решительно)
        // Высокая энтропия означает что внимание распределено между многими примерами (неуверенность)
        let entropy_factor = (1.0 - attention_entropy / self.max_entropy).max(0.5);

        base_position * confidence_factor * entropy_factor
    }
}
```

### Стоп-лосс и тейк-профит

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Правила управления рисками по паттернам              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ПРОДОЛЖЕНИЕ_ТРЕНДА:                                                   │
│   • Стоп: Ниже последнего свинг-лоу (лонг) / выше свинг-хая (шорт)    │
│   • Тейк-профит: Трейлинг стоп на основе ATR                          │
│   • Риск/Прибыль: минимум 1:2                                          │
│                                                                         │
│   РАЗВОРОТ_ТРЕНДА:                                                      │
│   • Стоп: За уровнем разворота                                         │
│   • Тейк-профит: Предыдущая поддержка/сопротивление                   │
│   • Риск/Прибыль: минимум 1:3 (компенсация низкого винрейта)          │
│                                                                         │
│   ПРОБОЙ:                                                               │
│   • Стоп: Ниже/выше уровня пробоя                                      │
│   • Тейк-профит: Измеренное движение равное диапазону                 │
│   • Риск/Прибыль: минимум 1:2                                          │
│                                                                         │
│   ЛОЖНЫЙ_ПРОБОЙ:                                                        │
│   • Стоп: За хаем/лоу ложного пробоя                                  │
│   • Тейк-профит: Противоположная сторона диапазона                    │
│   • Риск/Прибыль: минимум 1:2                                          │
│                                                                         │
│   КОНСОЛИДАЦИЯ:                                                         │
│   • Действие: Уменьшить размер позиции или оставаться вне рынка       │
│   • Ждать сигнала пробоя                                              │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Метрики производительности

### Метрики классификации

| Метрика | Описание | Цель |
|---------|----------|------|
| Accuracy | Общая доля правильных предсказаний | > 60% |
| F1-Score | Гармоническое среднее precision/recall | > 0.55 |
| Top-2 Accuracy | Правильный класс в топ-2 предсказаниях | > 80% |
| Калибровка уверенности | Предсказанная уверенность соответствует точности | ECE < 0.1 |

### Торговые метрики

| Метрика | Описание | Цель |
|---------|----------|------|
| Sharpe Ratio | Доходность с поправкой на риск | > 1.5 |
| Sortino Ratio | Доходность с поправкой на риск снижения | > 2.0 |
| Maximum Drawdown | Наибольшее снижение от пика до впадины | < 15% |
| Win Rate | Процент прибыльных сделок | > 50% |
| Profit Factor | Валовая прибыль / Валовый убыток | > 1.5 |

### Метрики Few-Shot

| Метрика | Описание | Формула |
|---------|----------|---------|
| 5-shot Accuracy | Точность с 5 примерами на класс | Стандартная метрика |
| 1-shot Accuracy | Точность с 1 примером на класс | Более сложная метрика |
| Энтропия внимания | Насколько распределено внимание | $-\sum a_i \log a_i$ |
| Консистентность прототипов | Стабильность между эпизодами | Дисперсия предсказаний |

## Сравнение с прототипными сетями

### Ключевые различия

```
┌────────────────────────────────────────────────────────────────────────┐
│           Matching Networks vs Прототипные сети                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Механизм классификации:                                               │
│   ───────────────────────                                              │
│   Прототипные: Сравнение с центроидом класса (прототипом)              │
│   Matching:    Взвешенное голосование по ВСЕМ примерам                 │
│                                                                         │
│   Эмбеддинги:                                                          │
│   ───────────                                                          │
│   Прототипные: Фиксированный эмбеддинг f(x)                            │
│   Matching:    Контекстно-зависимый через FCE: f(x, S), g(x, S)        │
│                                                                         │
│   Вычисления:                                                          │
│   ───────────                                                          │
│   Прототипные: O(n_classes) расстояний                                 │
│   Matching:    O(n_support) весов внимания                             │
│                                                                         │
│   Когда использовать каждый:                                           │
│   ──────────────────────────                                           │
│   Прототипные: Паттерны образуют плотные кластеры                      │
│   Matching:    Паттерны более распределены / контекстно-зависимы       │
│                                                                         │
│   Интерпретируемость:                                                  │
│   ──────────────────                                                   │
│   Прототипные: "Это похоже на типичный паттерн X"                      │
│   Matching:    "Это похоже на примеры 3, 7, 12 (внимание)"            │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Когда выбирать Matching Networks

1. **Сложные распределения паттернов**: Когда паттерны не образуют аккуратные кластеры
2. **Важен контекст**: Когда одни признаки означают разное в разных контекстах
3. **Интерпретируемость через примеры**: Когда важно знать какие конкретные примеры похожи
4. **Маленькие support set**: Хорошо работает даже с очень малым числом примеров на класс

## Ссылки

1. **Matching Networks for One Shot Learning**
   - Авторы: Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., Wierstra, D.
   - URL: https://arxiv.org/abs/1606.04080
   - Год: 2016
   - Конференция: NeurIPS 2016

2. **Prototypical Networks for Few-shot Learning**
   - Авторы: Snell, J., Swersky, K., Zemel, R.
   - URL: https://arxiv.org/abs/1703.05175
   - Год: 2017

3. **Meta-Learning for Semi-Supervised Few-Shot Classification**
   - Авторы: Ren, M., et al.
   - URL: https://arxiv.org/abs/1803.00676
   - Год: 2018

4. **Few-Shot Learning: A Survey**
   - Авторы: Wang, Y., et al.
   - URL: https://arxiv.org/abs/1904.05046
   - Год: 2019

---

## Быстрый старт

### Python

```python
from matching_network import MatchingNetwork, MarketPatternClassifier

# Создаём сеть
network = MatchingNetwork(
    input_dim=15,
    hidden_dim=64,
    embedding_dim=64,
    use_fce=True,
    fce_steps=5,
)

# Создаём классификатор
classifier = MarketPatternClassifier(network)

# Обучаем на support set
classifier.fit(support_features, support_labels)

# Классифицируем новые паттерны
predictions, attention = classifier.predict(query_features)

# Интерпретируем предсказания
for i, (pred, att) in enumerate(zip(predictions, attention)):
    print(f"Query {i}: Паттерн {pred}")
    print(f"  Наиболее похожие support примеры: {att.argsort()[-3:][::-1]}")
```

### Rust

```rust
use matching_networks_finance::{
    MatchingNetwork, Config, BybitClient, PatternStrategy
};

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализация сети
    let config = Config::default();
    let network = MatchingNetwork::new(config)?;

    // Загрузка support set
    let support_set = load_pattern_examples("data/support_set.json")?;

    // Подключение к Bybit
    let client = BybitClient::new()?;

    // Детекция паттернов в реальном времени
    let detection = network.detect_pattern(
        &support_set,
        &client.get_klines("BTCUSDT", "15m", 100).await?,
    )?;

    println!("Паттерн: {:?}", detection.pattern);
    println!("Уверенность: {:.2}", detection.confidence);
    println!("Торговый сигнал: {:?}", detection.trading_signal);

    Ok(())
}
```

---

## Структура файлов

```
84_matching_networks_finance/
├── README.md                    # Этот файл (English)
├── README.ru.md                 # Русский перевод
├── readme.simple.md             # Простое объяснение (English)
├── readme.simple.ru.md          # Простое объяснение (Russian)
├── README.specify.md            # Техническая спецификация
├── Cargo.toml                   # Rust зависимости
├── python/
│   ├── matching_network.py      # Основная реализация
│   ├── data_loader.py           # Утилиты загрузки данных
│   ├── trainer.py               # Утилиты обучения
│   └── backtest.py              # Фреймворк бэктестинга
├── src/
│   ├── lib.rs                   # Корень Rust библиотеки
│   ├── network/
│   │   ├── mod.rs
│   │   ├── embedding.rs         # Сети эмбеддинга
│   │   ├── attention.rs         # Механизмы внимания
│   │   └── fce.rs               # Полноконтекстные эмбеддинги
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs             # Интеграция с Bybit API
│   │   ├── stock.rs             # Данные акций (yfinance)
│   │   └── features.rs          # Извлечение признаков
│   ├── strategy/
│   │   ├── mod.rs
│   │   └── pattern_strategy.rs  # Торговая стратегия
│   ├── training/
│   │   ├── mod.rs
│   │   └── episodic.rs          # Эпизодическое обучение
│   └── utils/
│       ├── mod.rs
│       └── metrics.rs           # Метрики производительности
├── examples/
│   ├── basic_matching.rs        # Базовый пример использования
│   ├── pattern_detection.rs     # Пример детекции паттернов
│   └── backtest.rs              # Пример бэктестинга
└── tests/
    └── integration_tests.rs     # Интеграционные тесты
```
