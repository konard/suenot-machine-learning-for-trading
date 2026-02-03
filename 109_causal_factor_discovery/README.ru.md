# Глава 109: Обнаружение каузальных факторов

## Обзор

Обнаружение каузальных факторов — это подход машинного обучения, который выходит за рамки корреляционного анализа для выявления факторов, имеющих подлинные каузальные связи с доходностью активов. Традиционные факторные модели (такие как Fama-French) опираются на эмпирически наблюдаемые корреляции, которые могут быть ложными, изменяющимися во времени или искажёнными скрытыми переменными. Методы каузального обнаружения направлены на выявление скрытой каузальной структуры из наблюдательных данных, что позволяет создавать более устойчивые и интерпретируемые торговые стратегии.

Ключевая идея заключается в том, что хотя корреляция может нарушаться при смене режимов, истинные каузальные связи остаются стабильными. В этой главе рассматриваются алгоритмы каузального обнаружения (PC, FCI, GES, NOTEARS), их применение к финансовым временным рядам и практические реализации на Python и Rust для систематической торговли.

## Содержание

1. [Введение в каузальное обнаружение](#введение-в-каузальное-обнаружение)
2. [От корреляции к причинности](#от-корреляции-к-причинности)
3. [Математические основы](#математические-основы)
4. [Алгоритмы каузального обнаружения](#алгоритмы-каузального-обнаружения)
5. [Применение в торговле](#применение-в-торговле)
6. [Реализация на Python](#реализация-на-python)
7. [Реализация на Rust](#реализация-на-rust)
8. [Практические примеры с данными акций и криптовалют](#практические-примеры-с-данными-акций-и-криптовалют)
9. [Фреймворк для бэктестинга](#фреймворк-для-бэктестинга)
10. [Оценка производительности](#оценка-производительности)
11. [Направления развития](#направления-развития)
12. [Список литературы](#список-литературы)

---

## Введение в каузальное обнаружение

### Почему причинность важна в торговле

Рассмотрим классический пример: продажи мороженого и случаи утопления сильно коррелируют. Но употребление мороженого не вызывает утопление — оба явления вызваны общим фактором: жаркой погодой.

На финансовых рынках возникают аналогичные проблемы:
- Две акции могут коррелировать, потому что находятся в одном секторе (конфаундинг)
- Фактор может казаться предиктивным только потому, что коррелирует с истинным драйвером
- Корреляции могут разворачиваться при рыночном стрессе (разрушение корреляций)

Каузальное обнаружение направлено на выявление истинной каузальной структуры:

```
Ложная корреляция:               Каузальная связь:
   Мороженое                       Процентные ставки
      ↑                                  ↓
      |  коррелирует                     |  вызывает
      ↓                                  ↓
   Утопления                       Доходность акций

(Неверно: нет каузальной связи)  (Верно: прямое воздействие)
```

### Структурные каузальные модели (SCM)

Структурная каузальная модель состоит из:
1. **Эндогенные переменные** V = {V₁, V₂, ..., Vₙ}: Наблюдаемые переменные
2. **Экзогенные переменные** U = {U₁, U₂, ..., Uₙ}: Ненаблюдаемый шум/возмущения
3. **Структурные уравнения**: Vᵢ = fᵢ(PAᵢ, Uᵢ), где PAᵢ — родители (причины) Vᵢ

Например:
```
Рыночные настроения → Волатильность → Доходность
       ↓
    Объём

Структурные уравнения:
Волатильность = f₁(Настроения, U₁)
Доходность = f₂(Волатильность, U₂)
Объём = f₃(Настроения, U₃)
```

### Каузальные графы (DAG)

Каузальная структура представляется как направленный ациклический граф (DAG):
- Узлы = Переменные
- Рёбра = Прямые каузальные связи
- Нет циклов (причина предшествует следствию)

```
Пример DAG для факторной доходности:

  Рынок         Процентные ставки
     ↓                  ↓
  Бета-фактор    Облигационный фактор
     ↓                  ↓
     └──────→ Доходность акции ←──────┘
                    ↑
              Фактор размера
```

---

## От корреляции к причинности

### Фундаментальная проблема

**Корреляция**: P(Y | X) ≠ P(Y)
- X и Y статистически зависимы
- Может быть: X → Y, Y → X, X ← Z → Y (конфаундинг) или ошибка отбора

**Причинность**: P(Y | do(X)) — эффект *вмешательства* на X
- Что происходит с Y, если мы принудительно устанавливаем X в определённое значение?
- do(X) удаляет все входящие рёбра к X в каузальном графе

### do-исчисление

do-исчисление Перла позволяет вычислять каузальные эффекты из наблюдательных данных при выполнении определённых условий:

**Правило 1 (Вставка/удаление наблюдений):**
P(Y | do(X), Z, W) = P(Y | do(X), W) если Y ⊥ Z | X, W в модифицированном графе

**Правило 2 (Обмен действие/наблюдение):**
P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) если Y ⊥ Z | X, W в модифицированном графе

**Правило 3 (Вставка/удаление действий):**
P(Y | do(X), do(Z), W) = P(Y | do(X), W) если Y ⊥ Z | X, W в модифицированном графе

### Идентификация каузальных эффектов

Ключевые графические критерии:

**Критерий бэкдора**: Множество переменных Z удовлетворяет критерию бэкдора для (X, Y) если:
1. Ни один узел в Z не является потомком X
2. Z блокирует все бэкдор-пути от X к Y

Если Z удовлетворяет критерию бэкдора:
```
P(Y | do(X)) = Σ_z P(Y | X, Z=z) P(Z=z)
```

**Критерий фронтдора**: Когда корректировка по бэкдору невозможна, но есть опосредующая переменная:
```
X → M → Y  (с ненаблюдаемым конфаундером между X и Y)
P(Y | do(X)) = Σ_m P(M=m | X) Σ_x' P(Y | M=m, X=x') P(X=x')
```

---

## Математические основы

### Тестирование условной независимости

Алгоритмы каузального обнаружения в значительной степени опираются на тесты условной независимости (CI):

**Определение**: X ⊥ Y | Z означает, что X и Y независимы при условии Z:
```
P(X, Y | Z) = P(X | Z) P(Y | Z)
```

**Распространённые CI-тесты**:

1. **Тест частной корреляции** (линейный, гауссовский):
```
ρ(X,Y|Z) = (ρ(X,Y) - ρ(X,Z)ρ(Y,Z)) / √((1-ρ(X,Z)²)(1-ρ(Y,Z)²))

Тестовая статистика: z = (1/2)ln((1+ρ)/(1-ρ)) √(n-|Z|-3)
```

2. **Ядерные тесты** (нелинейные):
- HSIC (Критерий независимости Гильберта-Шмидта)
- KCI (Ядерный тест условной независимости)

3. **Тесты на взаимную информацию**:
```
I(X; Y | Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
```

### Классы марковской эквивалентности

Несколько DAG могут кодировать одинаковые условные независимости:

```
DAG 1: X → Y → Z    }
DAG 2: X ← Y ← Z    }  Одинаковые независимости: X ⊥ Z | Y
DAG 3: X ← Y → Z    }

Отличается: X → Y ← Z  (V-структура: X ⊥ Z, но X ⊬ Z | Y)
```

Каузальное обнаружение из наблюдательных данных может идентифицировать только **класс марковской эквивалентности**, представленный CPDAG (Завершённый частично направленный ациклический граф).

### Предположение о верности (Faithfulness)

Предположение о верности гласит, что все условные независимости в данных следуют из каузального графа (нет "случайных" сокращений).

**Пример нарушения**: X → Y, X → Z → Y, где прямой и косвенный эффекты сокращаются:
```
Y = αX + βZ, Z = γX
Y = αX + βγX = (α + βγ)X

Если α = -βγ, то Y ⊥ X несмотря на X → Y!
```

---

## Алгоритмы каузального обнаружения

### Алгоритм PC (Peter-Clark)

Алгоритм PC изучает CPDAG через тестирование условной независимости:

**Фаза 1: Изучение скелета**
```
Начать с полного ненаправленного графа
Для каждого ребра X — Y:
    Для размеров обуславливающих множеств k = 0, 1, 2, ...
        Для каждого подмножества S соседей X или Y с |S| = k:
            Если X ⊥ Y | S:
                Удалить ребро X — Y
                Записать S как разделяющее множество
```

**Фаза 2: Ориентация V-структур**
```
Для каждой тройки X — Z — Y, где X и Y не смежны:
    Если Z ∉ SepSet(X, Y):
        Ориентировать как X → Z ← Y (V-структура)
```

**Фаза 3: Ориентация остальных рёбер** (используя правила Мика):
```
R1: Если X → Y — Z и X, Z не смежны: ориентировать Y → Z
R2: Если X → Y → Z и X — Z: ориентировать X → Z
R3: Если X — Y → Z, X — W → Z, X — Z, Y, W не смежны: ориентировать X → Z
R4: Если X — Y → Z → W и X — W: ориентировать X → W
```

### Алгоритм FCI (Fast Causal Inference)

FCI расширяет PC для работы со скрытыми конфаундерами:

- Выводит PAG (Частичный родовой граф)
- Использует дополнительные типы рёбер: ◦→ (возможно направленное), ◦—◦ (неизвестно), ↔ (двунаправленное, указывает на скрытый конфаундер)

```
Пример PAG:
X ◦→ Y означает: X вызывает Y, или есть скрытый конфаундер, или оба
X ↔ Y означает: скрытый конфаундер между X и Y (нет прямого ребра)
```

### Алгоритм GES (Greedy Equivalence Search)

Алгоритм на основе оценки, который ищет по классам эквивалентности:

**Прямая фаза**:
```
Начать с пустого графа
Повторять:
    Найти добавление ребра, максимально увеличивающее BIC-оценку
    Добавить ребро, если оценка улучшается
Пока не существует улучшающего добавления
```

**Обратная фаза**:
```
Повторять:
    Найти удаление ребра, максимально увеличивающее BIC-оценку
    Удалить ребро, если оценка улучшается
Пока не существует улучшающего удаления
```

**BIC-оценка**:
```
BIC(G, D) = Σᵢ [log P(Xᵢ | PAᵢ) - (|θᵢ|/2) log n]
```

### NOTEARS

Подход непрерывной оптимизации:

**Ключевая идея**: Ацикличность может быть выражена как гладкое ограничение:
```
h(W) = tr(exp(W ◦ W)) - d = 0
```
где W — матрица смежности, а ◦ — поэлементное произведение.

**Задача оптимизации**:
```
минимизировать    (1/2n)||X - XW||²_F + λ||W||₁
при условии       h(W) = 0
```

Решается методом расширенного лагранжиана:
```
L(W, α, ρ) = (1/2n)||X - XW||²_F + λ||W||₁ + α·h(W) + (ρ/2)h(W)²
```

---

## Применение в торговле

### Каузальное обнаружение факторов для альфы

**Цель**: Идентифицировать факторы, каузально влияющие на доходность (не просто коррелирующие)

**Процесс**:
1. Собрать потенциальные факторы (технические, фундаментальные, альтернативные данные)
2. Применить каузальное обнаружение для идентификации рёбер фактор → доходность
3. Оценить каузальные эффекты с использованием бэкдор/фронтдор корректировки
4. Построить торговые сигналы из каузально валидированных факторов

```
Традиционный подход:
Фактор_1, Фактор_2, ..., Фактор_k → Все используются, если коррелируют с доходностью

Каузальный подход:
Фактор_1 → Доходность (каузальный)           → Использовать
Фактор_2 ← Конфаундер → Доходность           → Скорректировать на конфаундер
Фактор_3 ←─────────────── Доходность         → Отбросить (обратная причинность)
Фактор_4 ... нет ребра ... Доходность        → Отбросить (ложная связь)
```

### Режимно-устойчивые факторы

Каузальные связи должны быть стабильны между режимами:

```
Режим 1 (Бычий рынок):
  Моментум → Доходность (сильный положительный эффект)

Режим 2 (Медвежий рынок):
  Моментум → Доходность (отрицательный эффект, но каузальная связь сохраняется)

против Ложного фактора:
  Режим 1: Фактор X коррелирует с доходностью
  Режим 2: Корреляция исчезает (была искажена конфаундером)
```

### Каузальное построение портфеля

Использование каузальной структуры для оптимизации портфеля:
1. Идентифицировать каузальные драйверы доходности каждого актива
2. Диверсифицировать на основе каузальных факторов, а не только корреляций
3. Более устойчиво к разрушению корреляций при стрессе

```python
# Традиционный: на основе корреляции
weights = optimize(returns.cov())

# Каузальный: на основе факторов с каузальной корректировкой
causal_factors = discover_causal_factors(data)
factor_exposures = estimate_causal_effects(returns, causal_factors)
weights = optimize_causal(factor_exposures, factor_cov)
```

---

## Реализация на Python

### Модель каузального обнаружения

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations

@dataclass
class CausalGraph:
    """Представляет каузальный граф (DAG или CPDAG)."""
    nodes: List[str]
    edges: Dict[Tuple[str, str], str]  # (from, to): edge_type

    def add_edge(self, source: str, target: str, edge_type: str = "directed"):
        self.edges[(source, target)] = edge_type

    def remove_edge(self, source: str, target: str):
        self.edges.pop((source, target), None)
        self.edges.pop((target, source), None)

    def get_parents(self, node: str) -> List[str]:
        return [s for (s, t), e in self.edges.items() if t == node and e == "directed"]

    def get_children(self, node: str) -> List[str]:
        return [t for (s, t), e in self.edges.items() if s == node and e == "directed"]

    def get_neighbors(self, node: str) -> Set[str]:
        neighbors = set()
        for (s, t), e in self.edges.items():
            if s == node:
                neighbors.add(t)
            if t == node:
                neighbors.add(s)
        return neighbors


class PartialCorrelationTest:
    """Тест частной корреляции для условной независимости."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test(self, data: pd.DataFrame, x: str, y: str, z: List[str]) -> Tuple[bool, float]:
        """
        Тест X ⊥ Y | Z с использованием частной корреляции.
        Возвращает: (независимы, p-value)
        """
        if len(z) == 0:
            # Простая корреляция
            r, p = stats.pearsonr(data[x], data[y])
            return p > self.alpha, p

        # Частная корреляция через остатки регрессии
        from sklearn.linear_model import LinearRegression

        # Остатки X по Z
        reg_x = LinearRegression().fit(data[z], data[x])
        res_x = data[x] - reg_x.predict(data[z])

        # Остатки Y по Z
        reg_y = LinearRegression().fit(data[z], data[y])
        res_y = data[y] - reg_y.predict(data[z])

        # Корреляция остатков
        r, p = stats.pearsonr(res_x, res_y)

        # z-преобразование Фишера для корректного p-value
        n = len(data)
        z_stat = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
        se = 1 / np.sqrt(n - len(z) - 3)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat / se)))

        return p_value > self.alpha, p_value


class PCAlgorithm:
    """Алгоритм PC для каузального обнаружения."""

    def __init__(self, alpha: float = 0.05, max_cond_size: int = 4):
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.ci_test = PartialCorrelationTest(alpha)
        self.sep_sets: Dict[Tuple[str, str], Set[str]] = {}

    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """Изучить каузальную структуру из данных."""
        nodes = list(data.columns)

        # Фаза 1: Изучение скелета
        graph = self._learn_skeleton(data, nodes)

        # Фаза 2: Ориентация v-структур
        graph = self._orient_v_structures(graph, nodes)

        # Фаза 3: Применение правил Мика
        graph = self._apply_meek_rules(graph)

        return graph


class CausalFactorModel:
    """
    Каузальная факторная модель для торговли.
    Обнаруживает каузальные факторы и оценивает их эффекты на доходность.
    """

    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.01):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.graph: Optional[CausalGraph] = None
        self.causal_factors: List[str] = []
        self.factor_effects: Dict[str, float] = {}

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> 'CausalFactorModel':
        """
        Обнаружить каузальные факторы из признаков и оценить их эффекты.
        """
        # Объединить признаки и доходность
        data = features.copy()
        data['returns'] = returns.values

        # Обнаружить каузальную структуру
        pc = PCAlgorithm(alpha=self.alpha)
        self.graph = pc.fit(data)

        # Идентифицировать факторы, вызывающие доходность
        estimator = CausalEffectEstimator(self.graph)

        for factor in features.columns:
            try:
                effect = estimator.estimate_ate(data, factor, 'returns')

                if effect['ci_low'] * effect['ci_high'] > 0:  # Одинаковый знак
                    if abs(effect['ate']) >= self.min_effect_size:
                        self.causal_factors.append(factor)
                        self.factor_effects[factor] = effect['ate']
            except ValueError:
                continue

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Генерировать прогнозы доходности из каузальных факторов."""
        if not self.causal_factors:
            return np.zeros(len(features))

        predictions = np.zeros(len(features))
        for factor in self.causal_factors:
            if factor in features.columns:
                predictions += self.factor_effects[factor] * features[factor].values

        return predictions
```

### Конвейер данных и бэктестинг

```python
class FactorDataPipeline:
    """Конвейер для подготовки факторных данных для каузального обнаружения."""

    def compute_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Вычислить потенциальные каузальные факторы из ценовых данных."""
        factors = pd.DataFrame(index=prices.index)

        close = prices['close']
        volume = prices['volume'] if 'volume' in prices else pd.Series(1, index=prices.index)

        # Факторы моментума
        factors['momentum_5'] = close.pct_change(5)
        factors['momentum_20'] = close.pct_change(20)
        factors['momentum_60'] = close.pct_change(60)

        # Факторы возврата к среднему
        factors['ma_deviation_20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()

        # Факторы волатильности
        returns = close.pct_change()
        factors['volatility_20'] = returns.rolling(20).std()
        factors['volatility_ratio'] = returns.rolling(5).std() / returns.rolling(20).std()

        # RSI
        factors['rsi'] = self._compute_rsi(close, 14)

        return factors.dropna()


def fetch_bybit_data(symbol: str = "BTCUSDT", interval: str = "D", limit: int = 1000) -> pd.DataFrame:
    """Получить данные криптовалюты из API Bybit."""
    import requests
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": limit}

    resp = requests.get(url, params=params).json()
    records = resp['result']['list']

    df = pd.DataFrame(records, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
    df = df.sort_values('open_time').reset_index(drop=True)
    df.set_index('open_time', inplace=True)

    return df
```

---

## Реализация на Rust

### Структура проекта

```
109_causal_factor_discovery/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── graph.rs
│   ├── discovery/
│   │   ├── mod.rs
│   │   ├── pc.rs
│   │   └── ci_test.rs
│   ├── estimation/
│   │   ├── mod.rs
│   │   └── backdoor.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── factors.rs
│   │   └── bybit.rs
│   └── trading/
│       ├── mod.rs
│       ├── strategy.rs
│       └── backtest.rs
└── examples/
    ├── discover_factors.rs
    ├── crypto_causal.rs
    └── backtest_strategy.rs
```

### Реализация каузального графа (Rust)

```rust
// src/graph.rs
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub enum EdgeType {
    Directed,
    Undirected,
    Bidirected,  // Для скрытых конфаундеров
}

#[derive(Debug, Clone)]
pub struct CausalGraph {
    pub nodes: Vec<String>,
    pub edges: HashMap<(String, String), EdgeType>,
    node_set: HashSet<String>,
}

impl CausalGraph {
    pub fn new(nodes: Vec<String>) -> Self {
        let node_set: HashSet<String> = nodes.iter().cloned().collect();
        CausalGraph {
            nodes,
            edges: HashMap::new(),
            node_set,
        }
    }

    pub fn add_edge(&mut self, source: &str, target: &str, edge_type: EdgeType) {
        self.edges.insert((source.to_string(), target.to_string()), edge_type);
    }

    pub fn get_parents(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter_map(|((s, t), e)| {
                if t == node && *e == EdgeType::Directed {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_neighbors(&self, node: &str) -> HashSet<String> {
        let mut neighbors = HashSet::new();
        for ((s, t), _) in &self.edges {
            if s == node { neighbors.insert(t.clone()); }
            if t == node { neighbors.insert(s.clone()); }
        }
        neighbors
    }
}
```

### Тест условной независимости (Rust)

```rust
// src/discovery/ci_test.rs
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};

pub struct PartialCorrelationTest {
    pub alpha: f64,
}

impl PartialCorrelationTest {
    pub fn new(alpha: f64) -> Self {
        PartialCorrelationTest { alpha }
    }

    /// Тест X ⊥ Y | Z с использованием частной корреляции
    pub fn test(
        &self,
        data: &Array2<f64>,
        x_idx: usize,
        y_idx: usize,
        z_indices: &[usize],
    ) -> (bool, f64) {
        let n = data.nrows() as f64;

        if z_indices.is_empty() {
            let x = data.column(x_idx);
            let y = data.column(y_idx);
            let r = pearson_correlation(&x.to_owned(), &y.to_owned());
            let p_value = correlation_p_value(r, n as usize);
            return (p_value > self.alpha, p_value);
        }

        // Частная корреляция через остатки
        let x = data.column(x_idx).to_owned();
        let y = data.column(y_idx).to_owned();
        let z: Array2<f64> = Array2::from_shape_fn((data.nrows(), z_indices.len()), |(i, j)| {
            data[[i, z_indices[j]]]
        });

        let res_x = residualize(&x, &z);
        let res_y = residualize(&y, &z);
        let r = pearson_correlation(&res_x, &res_y);

        // z-преобразование Фишера
        let z_stat = 0.5 * ((1.0 + r) / (1.0 - r + 1e-10)).ln();
        let se = 1.0 / (n - z_indices.len() as f64 - 3.0).sqrt();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf((z_stat / se).abs()));

        (p_value > self.alpha, p_value)
    }
}
```

### Получение данных Bybit (Rust)

```rust
// src/data/bybit.rs
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct BybitKline {
    pub open_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<BybitKline>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: serde_json::Value = reqwest::get(&url).await?.json().await?;
    // Парсинг данных...
    Ok(vec![])
}
```

---

## Практические примеры с данными акций и криптовалют

### Пример 1: Обнаружение факторов для акций (Python)

```python
import yfinance as yf

# Загрузка данных
data = yf.download('AAPL', start='2018-01-01', end='2024-01-01')
data.columns = [c.lower() for c in data.columns]

# Вычисление факторов
pipeline = FactorDataPipeline()
factors = pipeline.compute_factors(data)
returns = data['close'].pct_change().shift(-1).dropna()

# Обнаружение каузальных факторов
model = CausalFactorModel(alpha=0.05, min_effect_size=0.001)
model.fit(factors, returns)

print("Обнаруженные каузальные факторы:")
print(model.get_factor_summary())
```

### Пример 2: Каузальный анализ криптовалют (Python)

```python
# Получение данных BTC и ETH
btc = fetch_bybit_data("BTCUSDT", "D", 1000)
eth = fetch_bybit_data("ETHUSDT", "D", 1000)

# Вычисление факторов для BTC
pipeline = FactorDataPipeline()
btc_factors = pipeline.compute_factors(btc)
btc_returns = btc['close'].pct_change().shift(-1).dropna()

# Обнаружение каузальных факторов
btc_model = CausalFactorModel(alpha=0.1, min_effect_size=0.005)
btc_model.fit(btc_factors, btc_returns)

print("Каузальные факторы BTC:")
print(btc_model.get_factor_summary())
```

---

## Оценка производительности

### Сводка метрик

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| Information Coefficient (IC) | Корреляция между прогнозами и доходностью | > 0.03 |
| IC Information Ratio | IC среднее / IC std | > 0.5 |
| Коэффициент Шарпа | Доходность с поправкой на риск | > 1.0 |
| Коэффициент Сортино | Доходность с поправкой на нисходящий риск | > 1.5 |
| Максимальная просадка | Наибольшее снижение от пика до впадины | > -20% |
| Стабильность факторов | % факторов, сохраняемых при переобучении | > 50% |

### Преимущества каузального обнаружения факторов

1. **Устойчивость**: Каузальные факторы должны оставаться предиктивными в разных рыночных режимах
2. **Интерпретируемость**: Каузальные связи имеют ясный экономический смысл
3. **Избегание переобучения**: Выбирается меньше ложных факторов
4. **Перенос обучения**: Каузальные факторы могут переноситься между активами/рынками

---

## Направления развития

1. **Изменяющиеся во времени каузальные структуры**: Обнаружение смен режимов в каузальных связях
2. **Глубокое каузальное обнаружение**: Обнаружение на основе нейронных сетей для нелинейных связей
3. **Контрфактическая торговля**: Анализ "что если" с использованием каузальных моделей
4. **Мультиактивные каузальные модели**: Кросс-активные каузальные связи для построения портфеля
5. **Каузальный мониторинг в реальном времени**: Онлайн каузальное обнаружение для адаптивных стратегий
6. **Каузальное обучение с подкреплением**: Комбинация каузального вывода с RL для торговли

---

## Список литературы

1. Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge University Press.
2. Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search. MIT Press.
3. Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference. MIT Press.
4. Zheng, X., et al. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. NeurIPS 2018.
5. Vowels, M. J., et al. (2022). D'ya Like DAGs? A Survey on Structure Learning and Causal Discovery. ACM Computing Surveys.

---

## Запуск примеров

### Python

```bash
cd 109_causal_factor_discovery/python
pip install -r requirements.txt
python model.py           # Тест каузального обнаружения
python backtest.py        # Запуск бэктестинга
```

### Rust

```bash
cd 109_causal_factor_discovery
cargo build
cargo run --example discover_factors
cargo run --example crypto_causal
cargo run --example backtest_strategy
```
