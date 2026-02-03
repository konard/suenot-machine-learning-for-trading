# Глава 119: Rule Extraction для Трейдинга — Извлечение Интерпретируемых Правил из Черного Ящика Моделей

В этой главе мы исследуем техники извлечения правил, которые трансформируют непрозрачные модели машинного обучения типа "черный ящик" в интерпретируемые правила принятия решений для торговых приложений. Хотя нейронные сети и ансамблевые методы демонстрируют превосходную прогностическую способность, их непрозрачность создает проблемы для управления рисками, соответствия регуляторным требованиям и валидации стратегий. Извлечение правил устраняет этот разрыв, дистиллируя сложные модели в человекочитаемые правила типа if-then.

Мы научимся извлекать правила принятия решений из обученных нейронных сетей и моделей градиентного бустинга, оценивать точность и покрытие правил, а также применять эти правила для прозрачных алгоритмических торговых стратегий. Глава охватывает педагогические методы (модель-агностические подходы), декомпозиционные методы (архитектурно-специфичные техники) и гибридные подходы, сочетающие обе парадигмы.

## Содержание

1. [Извлечение правил: от черного ящика к прозрачности](#извлечение-правил-от-черного-ящика-к-прозрачности)
2. [Педагогические методы извлечения правил](#педагогические-методы-извлечения-правил)
   * [Алгоритм TREPAN](#алгоритм-trepan)
   * [Извлечение правил через последовательное покрытие](#извлечение-правил-через-последовательное-покрытие)
3. [Декомпозиционное извлечение правил](#декомпозиционное-извлечение-правил)
   * [Извлечение диаграмм решений из нейронных сетей](#извлечение-диаграмм-решений-из-нейронных-сетей)
   * [Извлечение правил из деревьев решений и ансамблей](#извлечение-правил-из-деревьев-решений-и-ансамблей)
4. [Пример кода: построение пайплайна извлечения правил](#пример-кода-построение-пайплайна-извлечения-правил)
   * [Подготовка данных: акции и криптовалюты](#подготовка-данных-акции-и-криптовалюты)
   * [Обучение моделей черного ящика](#обучение-моделей-черного-ящика)
   * [Извлечение правил из нейронных сетей](#извлечение-правил-из-нейронных-сетей)
   * [Извлечение правил из градиентного бустинга](#извлечение-правил-из-градиентного-бустинга)
5. [Метрики оценки правил](#метрики-оценки-правил)
6. [Пример кода: торговая стратегия на извлеченных правилах](#пример-кода-торговая-стратегия-на-извлеченных-правилах)
   * [Генерация сигналов на основе правил](#генерация-сигналов-на-основе-правил)
   * [Бэктестинг стратегии на правилах](#бэктестинг-стратегии-на-правилах)
7. [Реализация на Rust для продакшена](#реализация-на-rust-для-продакшена)

## Извлечение правил: от черного ящика к прозрачности

Извлечение правил — это процесс получения символического, человекочитаемого знания из обученных моделей машинного обучения. В торговых приложениях это служит нескольким критически важным целям:

- **Соответствие регуляторным требованиям**: финансовые регуляторы все чаще требуют объяснимости решений алгоритмической торговли
- **Управление рисками**: понимание причин предсказаний модели помогает выявить потенциальные точки отказа
- **Валидация стратегии**: эксперты предметной области могут проверить, соответствуют ли извлеченные правила интуиции рынка
- **Отладка**: правила раскрывают, какие паттерны выучила модель, включая ложные корреляции

### Типы извлечения правил

| Метод | Описание | Плюсы | Минусы |
|-------|----------|-------|--------|
| **Педагогический** | Рассматривает модель как черный ящик, учит правила из пар вход-выход | Модель-агностический, простой | Может упустить внутреннюю структуру |
| **Декомпозиционный** | Анализирует архитектуру модели напрямую | Захватывает точное поведение | Архитектурно-специфичный |
| **Эклектический** | Комбинирует педагогический и декомпозиционный | Лучшее из обоих подходов | Более сложный |

## Педагогические методы извлечения правил

Педагогические методы рассматривают обученную модель как оракула и извлекают правила, наблюдая за её поведением вход-выход.

### Алгоритм TREPAN

TREPAN (Trees Paraphrasing Networks) строит дерево решений, которое имитирует поведение нейронной сети:

```python
def trepan_extract(model, X_train, max_depth=10):
    """
    Извлечение правил дерева решений из любой модели черного ящика.

    Args:
        model: Обученная модель черного ящика с методом predict
        X_train: Обучающие признаки
        max_depth: Максимальная глубина дерева

    Returns:
        Обученное дерево решений, аппроксимирующее модель
    """
    from sklearn.tree import DecisionTreeClassifier

    # Получаем предсказания модели как псевдо-метки
    y_pseudo = model.predict(X_train)

    # Обучаем интерпретируемое дерево на псевдо-метках
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train, y_pseudo)

    return tree
```

### Извлечение правил через последовательное покрытие

Последовательное покрытие итеративно извлекает правила, покрывающие подмножества данных:

```python
def sequential_covering(model, X, feature_names, min_coverage=0.05):
    """
    Извлечение правил с помощью алгоритма последовательного покрытия.
    """
    rules = []
    uncovered = np.ones(len(X), dtype=bool)

    while uncovered.sum() / len(X) > min_coverage:
        rule = find_best_rule(model, X[uncovered], feature_names)
        if rule is None:
            break
        rules.append(rule)
        uncovered = uncovered & ~rule.covers(X)

    return rules
```

## Декомпозиционное извлечение правил

Декомпозиционные методы анализируют внутреннюю структуру модели для извлечения правил.

### Извлечение диаграмм решений из нейронных сетей

На основе исследовательской работы "Extracting Rules from Neural Networks as Decision Diagrams" (arXiv:2104.06411) мы можем преобразовать вычисления нейронной сети в бинарные диаграммы решений (BDD):

```python
class NeuralNetworkToRules:
    """
    Извлечение правил из нейронных сетей с использованием
    преобразования в диаграммы решений.
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def extract_layer_rules(self, layer_idx, threshold=0.5):
        """
        Извлечение правил из конкретного слоя путем анализа паттернов активации.
        """
        weights = self.model.layers[layer_idx].get_weights()[0]
        biases = self.model.layers[layer_idx].get_weights()[1]

        rules = []
        for neuron_idx in range(weights.shape[1]):
            w = weights[:, neuron_idx]
            b = biases[neuron_idx]

            # Создаем правило: if sum(w_i * x_i) + b > threshold
            conditions = []
            for feat_idx, weight in enumerate(w):
                if abs(weight) > 0.1:  # Значимый вес
                    conditions.append({
                        'feature': self.feature_names[feat_idx],
                        'weight': weight
                    })

            rules.append({
                'conditions': conditions,
                'bias': b,
                'threshold': threshold
            })

        return rules
```

### Извлечение правил из деревьев решений и ансамблей

Для моделей на основе деревьев правила можно извлечь напрямую из структуры дерева:

```python
def extract_tree_rules(tree, feature_names):
    """
    Извлечение правил if-then из дерева решений.
    """
    tree_ = tree.tree_
    rules = []

    def recurse(node, path):
        if tree_.feature[node] != -2:  # Не лист
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            # Левая ветка: feature <= threshold
            left_path = path + [(feature, '<=', threshold)]
            recurse(tree_.children_left[node], left_path)

            # Правая ветка: feature > threshold
            right_path = path + [(feature, '>', threshold)]
            recurse(tree_.children_right[node], right_path)
        else:
            # Листовой узел — создаем правило
            prediction = tree_.value[node].argmax()
            rules.append({'conditions': path, 'prediction': prediction})

    recurse(0, [])
    return rules
```

## Пример кода: построение пайплайна извлечения правил

### Подготовка данных: акции и криптовалюты

Мы используем два источника данных для всестороннего тестирования:

```python
import yfinance as yf
import pandas as pd
import numpy as np

def prepare_stock_data(ticker='SPY', period='2y'):
    """
    Загрузка и подготовка данных фондового рынка с техническими индикаторами.
    """
    df = yf.download(ticker, period=period)

    # Технические индикаторы
    df['returns'] = df['Close'].pct_change()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['rsi'] = compute_rsi(df['Close'], 14)
    df['volatility'] = df['returns'].rolling(20).std()

    # Таргет: направление следующего дня
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)

    return df.dropna()

def prepare_crypto_data(symbol='BTCUSDT', interval='1h', limit=1000):
    """
    Получение данных криптовалют из Bybit API.
    """
    import requests

    url = "https://api.bybit.com/v5/market/kline"
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # Преобразование типов
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Добавление признаков
    df['returns'] = df['close'].pct_change()
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)

    return df.dropna()
```

### Обучение моделей черного ящика

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

def train_models(X, y):
    """
    Обучение нейронной сети и моделей градиентного бустинга.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Нейронная сеть
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        max_iter=500,
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)

    # Градиентный бустинг
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    return nn_model, gb_model, scaler, X_test, y_test
```

### Извлечение правил из нейронных сетей

```python
def extract_nn_rules(nn_model, X_train, feature_names, max_rules=20):
    """
    Извлечение интерпретируемых правил из нейронной сети
    с использованием TREPAN-подобного подхода.
    """
    from sklearn.tree import DecisionTreeClassifier

    # Генерация псевдо-меток от NN
    y_pseudo = nn_model.predict(X_train)

    # Обучение интерпретируемого дерева
    tree = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        max_leaf_nodes=max_rules
    )
    tree.fit(X_train, y_pseudo)

    # Извлечение правил
    rules = extract_tree_rules(tree, feature_names)

    # Вычисление точности соответствия
    tree_preds = tree.predict(X_train)
    fidelity = (tree_preds == y_pseudo).mean()

    return rules, fidelity, tree
```

### Извлечение правил из градиентного бустинга

```python
def extract_gb_rules(gb_model, feature_names, importance_threshold=0.05):
    """
    Извлечение правил из ансамбля градиентного бустинга.
    """
    all_rules = []

    for tree_idx, tree in enumerate(gb_model.estimators_.ravel()):
        rules = extract_tree_rules(tree, feature_names)

        # Взвешивание правил по важности дерева
        for rule in rules:
            rule['tree_idx'] = tree_idx
            rule['weight'] = 1.0 / len(gb_model.estimators_)

        all_rules.extend(rules)

    # Фильтрация по важности признаков
    importances = gb_model.feature_importances_
    important_features = set(
        feature_names[i] for i, imp in enumerate(importances)
        if imp >= importance_threshold
    )

    filtered_rules = [
        rule for rule in all_rules
        if all(cond[0] in important_features for cond in rule['conditions'])
    ]

    return consolidate_rules(filtered_rules)
```

## Метрики оценки правил

```python
def evaluate_rules(rules, model, X, y, feature_names):
    """
    Оценка извлеченных правил относительно исходной модели и истинных меток.
    """
    metrics = {}

    # Точность соответствия: насколько правила совпадают с предсказаниями модели
    model_preds = model.predict(X)
    rule_preds = apply_rules(rules, X, feature_names)
    metrics['fidelity'] = (model_preds == rule_preds).mean()

    # Точность: насколько правила предсказывают истинные метки
    metrics['accuracy'] = (y == rule_preds).mean()

    # Покрытие: доля примеров, покрытых хотя бы одним правилом
    coverage_mask = np.zeros(len(X), dtype=bool)
    for rule in rules:
        coverage_mask |= rule_covers(rule, X, feature_names)
    metrics['coverage'] = coverage_mask.mean()

    # Сложность: среднее количество условий на правило
    metrics['avg_conditions'] = np.mean([
        len(rule['conditions']) for rule in rules
    ])

    # Количество правил
    metrics['n_rules'] = len(rules)

    return metrics
```

## Пример кода: торговая стратегия на извлеченных правилах

### Генерация сигналов на основе правил

```python
class RuleBasedStrategy:
    """
    Торговая стратегия на извлеченных правилах.
    """

    def __init__(self, rules, feature_names):
        self.rules = rules
        self.feature_names = feature_names

    def generate_signal(self, features):
        """
        Генерация торгового сигнала на основе извлеченных правил.

        Возвращает:
            1 для покупки, -1 для продажи, 0 для удержания
        """
        buy_score = 0
        sell_score = 0

        for rule in self.rules:
            if self._rule_matches(rule, features):
                if rule['prediction'] == 1:  # Бычий
                    buy_score += rule.get('weight', 1.0)
                else:  # Медвежий
                    sell_score += rule.get('weight', 1.0)

        # Генерация сигнала на основе разницы баллов
        score_diff = buy_score - sell_score

        if score_diff > 0.5:
            return 1
        elif score_diff < -0.5:
            return -1
        else:
            return 0

    def _rule_matches(self, rule, features):
        """Проверка выполнения всех условий правила."""
        for feature, operator, threshold in rule['conditions']:
            feat_idx = self.feature_names.index(feature)
            value = features[feat_idx]

            if operator == '<=' and value > threshold:
                return False
            if operator == '>' and value <= threshold:
                return False

        return True

    def explain_signal(self, features):
        """
        Предоставление человекочитаемого объяснения сигнала.
        """
        explanations = []

        for rule in self.rules:
            if self._rule_matches(rule, features):
                conditions_str = ' И '.join([
                    f"{feat} {op} {thresh:.4f}"
                    for feat, op, thresh in rule['conditions']
                ])
                direction = "ПОКУПКА" if rule['prediction'] == 1 else "ПРОДАЖА"
                explanations.append(
                    f"Сработало правило: ЕСЛИ {conditions_str} ТО {direction}"
                )

        return explanations
```

### Бэктестинг стратегии на правилах

```python
def backtest_rule_strategy(rules, X, y, prices, feature_names,
                            initial_capital=100000):
    """
    Бэктестинг торговой стратегии на правилах.
    """
    strategy = RuleBasedStrategy(rules, feature_names)

    capital = initial_capital
    position = 0
    returns = []
    trades = []

    for i in range(len(X) - 1):
        signal = strategy.generate_signal(X[i])
        price = prices[i]
        next_price = prices[i + 1]

        # Исполнение сделок
        if signal == 1 and position <= 0:  # Сигнал на покупку
            position = capital / price
            trades.append({
                'idx': i,
                'type': 'ПОКУПКА',
                'price': price,
                'explanation': strategy.explain_signal(X[i])
            })
        elif signal == -1 and position >= 0:  # Сигнал на продажу
            if position > 0:
                capital = position * price
            position = -capital / price
            trades.append({
                'idx': i,
                'type': 'ПРОДАЖА',
                'price': price,
                'explanation': strategy.explain_signal(X[i])
            })

        # Расчет доходности
        if position > 0:
            ret = (next_price - price) / price
        elif position < 0:
            ret = (price - next_price) / price
        else:
            ret = 0

        returns.append(ret)

    # Расчет метрик
    returns = np.array(returns)
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    max_dd = compute_max_drawdown(returns)
    total_return = (1 + returns).prod() - 1

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'n_trades': len(trades),
        'trades': trades
    }
```

## Реализация на Rust для продакшена

Директория `rust_examples/` содержит высокопроизводительную реализацию на Rust для извлечения и исполнения правил в продакшене:

```rust
// Пример: Извлечение правил из дерева решений на Rust
use ndarray::Array2;

pub struct Rule {
    pub conditions: Vec<Condition>,
    pub prediction: i32,
    pub confidence: f64,
}

pub struct Condition {
    pub feature_idx: usize,
    pub operator: Operator,
    pub threshold: f64,
}

pub enum Operator {
    LessOrEqual,
    GreaterThan,
}

impl Rule {
    pub fn matches(&self, features: &[f64]) -> bool {
        self.conditions.iter().all(|cond| {
            let value = features[cond.feature_idx];
            match cond.operator {
                Operator::LessOrEqual => value <= cond.threshold,
                Operator::GreaterThan => value > cond.threshold,
            }
        })
    }

    pub fn to_string(&self, feature_names: &[String]) -> String {
        let conditions: Vec<String> = self.conditions.iter().map(|c| {
            let op = match c.operator {
                Operator::LessOrEqual => "<=",
                Operator::GreaterThan => ">",
            };
            format!("{} {} {:.4}", feature_names[c.feature_idx], op, c.threshold)
        }).collect();

        let prediction = if self.prediction == 1 { "ПОКУПКА" } else { "ПРОДАЖА" };
        format!("ЕСЛИ {} ТО {}", conditions.join(" И "), prediction)
    }
}
```

## Ключевые метрики

| Метрика | Описание | Цель |
|---------|----------|------|
| **Fidelity (Точность соответствия)** | Согласованность правил с исходной моделью | > 90% |
| **Accuracy (Точность)** | Точность предсказаний правил на тестовых данных | > 55% |
| **Coverage (Покрытие)** | Доля примеров, покрытых правилами | > 95% |
| **Complexity (Сложность)** | Среднее число условий на правило | < 5 |
| **Sharpe Ratio (Коэффициент Шарпа)** | Риск-скорректированная доходность стратегии | > 1.0 |
| **Max Drawdown (Максимальная просадка)** | Наибольшее падение от пика до дна | < 20% |

## Зависимости

### Python
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=2.0.0
yfinance>=0.2.0
requests>=2.28.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

### Rust
См. `rust_examples/Cargo.toml` для полного списка зависимостей.

## Ожидаемые результаты

После завершения этой главы вы сможете:

1. **Извлекать интерпретируемые правила** из нейронных сетей и ансамблевых моделей
2. **Оценивать качество правил** с помощью метрик точности соответствия, покрытия и сложности
3. **Строить прозрачные торговые стратегии** на основе извлеченных правил
4. **Объяснять торговые решения** в человекочитаемом виде
5. **Разворачивать системы на правилах** в продакшене с использованием Rust

## Научные работы

1. **Extracting Rules from Neural Networks as Decision Diagrams**
   - URL: https://arxiv.org/abs/2104.06411
   - Год: 2021
   - Ключевая идея: нейронные сети можно преобразовать в бинарные диаграммы решений

2. **TREPAN: Extracting Tree-Structured Representations of Trained Networks**
   - Авторы: Craven & Shavlik
   - Ключевая идея: деревья решений могут аппроксимировать поведение нейронных сетей

3. **Interpretable Machine Learning: A Guide for Making Black Box Models Explainable**
   - Автор: Christoph Molnar
   - URL: https://christophm.github.io/interpretable-ml-book/

4. **Born Again Trees: From Deep Forests to Interpretable Trees**
   - URL: https://arxiv.org/abs/2003.11132
   - Ключевая идея: знания ансамбля можно дистиллировать в одно дерево

## Уровень сложности

**Средний — Продвинутый**

Предварительные требования:
- Понимание деревьев решений и нейронных сетей
- Знакомство с библиотеками машинного обучения Python
- Базовые знания торговых стратегий
- Опционально: программирование на Rust для продакшен-реализации

## Дисклеймеры

- **Не является финансовой рекомендацией**: Этот материал предназначен только для образовательных целей. Прошлые результаты не гарантируют будущих доходов.
- **Ограничения модели**: Извлеченные правила являются приближениями; они могут не отражать все поведение исходной модели.
- **Рыночный риск**: Все торговые стратегии несут риск финансовых потерь. Всегда используйте надлежащее управление рисками.
- **Качество данных**: Эффективность стратегии зависит от качества данных и рыночных условий.
