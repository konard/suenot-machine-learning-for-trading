# Глава 238: Разделённые VAE для трейдинга

## Введение

Разделённые вариационные автокодировщики (Disentangled VAE) представляют собой семейство генеративных моделей, направленных на изучение интерпретируемых латентных представлений, в которых каждое измерение латентного пространства соответствует независимому, осмысленному фактору вариации в данных. В контексте финансовых рынков это означает возможность автоматического обнаружения и разделения таких рыночных сил, как тренд, волатильность, ликвидность и моментум, без необходимости ручного проектирования факторов.

Стандартные VAE, несмотря на мощные генеративные способности, часто порождают запутанные латентные пространства, где одно измерение может одновременно кодировать несколько факторов, а один фактор может быть распределён по нескольким измерениям. Это делает интерпретацию и контролируемое использование латентного пространства практически невозможными. Семейство разделённых VAE — включая β-VAE, FactorVAE, DIP-VAE и β-TCVAE — решает эту проблему различными способами, каждый из которых имеет свои математические основания, преимущества и ограничения.

В этой главе мы систематически исследуем все основные методы разделения представлений, их математические основы, практические реализации и применение в трейдинге. Мы разработаем полную систему от подготовки данных до построения торговой стратегии, используя данные криптовалютных бирж (Bybit) и фондового рынка.

---

## Содержание

1. [Введение в Disentangled VAE](#1-введение-в-disentangled-vae)
2. [Методы разделения представлений](#2-методы-разделения-представлений)
3. [Математические основы](#3-математические-основы)
4. [Практические примеры](#4-практические-примеры)
5. [Реализация на Rust](#5-реализация-на-rust)
6. [Реализация на Python](#6-реализация-на-python)
7. [Лучшие практики](#7-лучшие-практики)
8. [Ресурсы](#8-ресурсы)
9. [Связанные главы](#9-связанные-главы)
10. [Уровень сложности](#10-уровень-сложности)

---

## 1. Введение в Disentangled VAE

### 1.1 Проблема запутанных представлений

Генеративные модели, обученные на финансовых данных, часто создают латентные пространства, в которых отдельные измерения не несут ясного интерпретируемого смысла. Это проявляется в нескольких формах:

**Многофакторное кодирование**: Одно латентное измерение может одновременно кодировать и тренд, и волатильность, делая невозможным анализ каждого фактора в изоляции. Например, при варьировании такого измерения мы наблюдаем одновременное изменение направления и амплитуды ценовых колебаний, что затрудняет понимание причинно-следственных связей.

**Распределённое кодирование**: Один рыночный фактор может быть распределён по нескольким латентным измерениям. Информация о волатильности, например, может частично кодироваться в трёх или четырёх измерениях, что делает невозможным целенаправленное манипулирование этим фактором.

**Избыточное кодирование**: Несколько латентных измерений могут нести дублирующую информацию, что расточительно с точки зрения информационной ёмкости модели и создаёт ложные корреляции при построении торговых стратегий.

**Непереносимость**: Запутанные представления плохо обобщаются на новые рыночные условия. Модель, обученная на бычьем рынке, может создать представления, непригодные для медвежьего, поскольку факторы запутаны с конкретным рыночным режимом.

### 1.2 Ключевые преимущества разделённых представлений

Разделённые VAE решают перечисленные проблемы, обеспечивая три фундаментальных преимущества:

#### Интерпретируемость

Каждое латентное измерение соответствует отдельному, осмысленному фактору вариации. В финансовом контексте это означает, что трейдер может проверить, какой рыночный фактор кодирует каждое измерение — тренд, волатильность, объём, моментум или спред — и использовать эту информацию для принятия решений. Латентные обходы (traversals) позволяют визуально или количественно оценить влияние каждого фактора на рыночные данные.

#### Контролируемая генерация

Благодаря независимости латентных измерений появляется возможность генерировать синтетические рыночные сценарии с точным контролем над отдельными факторами. Трейдер может задать вопрос: «Как поведёт себя мой портфель, если волатильность увеличится вдвое, а тренд останется прежним?» — и получить обоснованный ответ, варьируя только соответствующее латентное измерение.

#### Изоляция факторов

Разделённые представления позволяют строить торговые стратегии, нацеленные на конкретные рыночные факторы. Можно создать стратегию, торгующую исключительно моментум, и отдельную стратегию на волатильность, при этом каждая будет основана на чистом, изолированном сигнале. Это также упрощает хеджирование: зная, какому фактору подвержен портфель, можно построить целенаправленный хедж.

### 1.3 Сравнительная таблица методов

| Характеристика | Стандартный VAE | β-VAE | FactorVAE | DIP-VAE | β-TCVAE |
|---|---|---|---|---|---|
| **Целевая функция** | ELBO | Взвешенный KL | ELBO + TC-дискриминатор | ELBO + DIP-штраф | ELBO + взвешенная TC |
| **Гиперпараметры** | — | β | γ | λ_od, λ_d | β |
| **Механизм разделения** | Нет | Усиление KL-штрафа | Минимизация TC через adversarial | Штраф на ковариацию | Декомпозиция KL и взвешивание TC |
| **Качество реконструкции** | Высокое | Среднее-Низкое | Высокое | Высокое | Высокое |
| **Качество разделения** | Низкое | Среднее-Высокое | Высокое | Среднее | Высокое |
| **Стабильность обучения** | Высокая | Высокая | Средняя (GAN-компонент) | Высокая | Высокая |
| **Вычислительная стоимость** | Низкая | Низкая | Средняя | Низкая | Низкая |
| **Компромисс реконструкция-разделение** | Нет | Сильный | Слабый | Слабый | Умеренный |
| **Применимость к финансам** | Генерация | Факторный анализ | Чистые факторы | Робастные факторы | Баланс точности и разделения |

Каждый метод имеет свою область применения. β-VAE — наиболее простой и широко применяемый, но страдает от сильного компромисса между реконструкцией и разделением. FactorVAE обеспечивает наилучшее разделение при минимальной потере качества реконструкции, но требует обучения дополнительного дискриминатора. DIP-VAE предлагает прямой подход через штрафование ковариационной матрицы. β-TCVAE достигает хорошего баланса, нацеливаясь именно на полную корреляцию — ключевой фактор запутанности.

---

## 2. Методы разделения представлений

### 2.1 β-VAE: взвешивание KL-штрафа

β-VAE — простейшее и наиболее широко используемое расширение стандартного VAE для достижения разделённых представлений. Идея элегантна в своей простоте: ввести весовой коэффициент β перед членом KL-дивергенции в целевой функции.

#### Целевая функция

```
L_β-VAE = E_q(z|x)[log p(x|z)] - β · KL(q(z|x) || p(z))
```

При β = 1 мы получаем стандартный VAE. При β > 1 модель испытывает повышенное давление для приближения апостериорного распределения к изотропному гауссовскому приору, что побуждает каждое латентное измерение быть статистически независимым от остальных.

#### Интуиция через информационное узкое место

Параметр β контролирует ширину информационного канала между входом и латентным кодом. При высоком β этот канал сужается, и модель вынуждена передавать через него только наиболее существенную информацию. Каждое измерение должно «заслужить» свою информационную ёмкость, что естественно приводит к разложению, где каждое измерение несёт отдельный, неизбыточный сигнал.

#### Ограничения

Основной недостаток β-VAE — неизбежный компромисс между качеством реконструкции и степенью разделения. При высоких значениях β модель жертвует точностью реконструкции ради более чистых факторов. В финансах это может означать потерю тонких, но важных паттернов рыночной микроструктуры. Кроме того, параметр β воздействует на все компоненты KL-дивергенции одинаково, хотя для разделения критична только полная корреляция.

#### Стратегии отжига β

Вместо использования фиксированного значения β с начала обучения применяются стратегии постепенного увеличения:

- **Линейный отжиг**: β линейно возрастает от 0 до целевого значения за N шагов обучения
- **Циклический отжиг**: β колеблется между 0 и целевым значением, позволяя модели периодически «дышать»
- **Монотонный с прогревом**: β остаётся на 0 в течение начального периода, затем плавно нарастает

### 2.2 FactorVAE: полная корреляция с дискриминатором

FactorVAE решает проблему компромисса β-VAE, напрямую минимизируя полную корреляцию (Total Correlation, TC) латентного распределения с использованием adversarial-подхода.

#### Целевая функция

```
L_FactorVAE = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z)) - γ · TC(q(z))
```

где полная корреляция определяется как:

```
TC(q(z)) = KL(q(z) || ∏_j q(z_j))
```

Прямое вычисление TC затруднено, поскольку требует оценки маргинального распределения q(z). FactorVAE использует дискриминатор — нейронную сеть, обученную отличать выборки из совместного распределения q(z) от выборок из факторизованного распределения q̄(z) = ∏_j q(z_j).

#### Дискриминатор полной корреляции

Дискриминатор D(z) обучается классифицировать:
- Выборки из q(z) (совместное распределение) — класс «реальные»
- Выборки из q̄(z) (факторизованное, с перемешанными координатами) — класс «фейковые»

Оценка TC получается как:

```
TC(q(z)) ≈ E_q(z)[log D(z) - log(1 - D(z))]
```

#### Преимущества для финансов

FactorVAE позволяет достичь высокого качества разделения без существенной потери в реконструкции. Для финансовых данных это особенно важно: мы получаем чистые, интерпретируемые факторы и при этом сохраняем способность модели точно воспроизводить рыночную динамику. Однако обучение дискриминатора добавляет нестабильности (типичной для GAN-подходов), что требует тщательной настройки.

### 2.3 DIP-VAE: разделённый предполагаемый prior

DIP-VAE (Disentangled Inferred Prior) подходит к проблеме разделения с другой стороны: вместо штрафования KL-дивергенции или полной корреляции, метод накладывает ограничения непосредственно на ковариационную матрицу агрегированного апостериорного распределения.

#### Целевая функция

```
L_DIP-VAE = L_VAE + λ_od · ∑_{i≠j} [Cov_q(z)]²_{ij} + λ_d · ∑_i ([Cov_q(z)]_{ii} - 1)²
```

Два штрафных члена действуют на ковариационную матрицу q(z):
- **Внедиагональный штраф** (λ_od): минимизирует корреляции между латентными измерениями
- **Диагональный штраф** (λ_d): приближает дисперсии к 1, соответствуя стандартному нормальному приору

#### Два варианта: DIP-VAE-I и DIP-VAE-II

**DIP-VAE-I** штрафует ковариацию момента q(z), усреднённого по данным:

```
Cov_p(x)[E_q(z|x)[z]] = E_p(x)[μ(x) μ(x)^T] - E_p(x)[μ(x)] E_p(x)[μ(x)]^T
```

**DIP-VAE-II** штрафует полную ковариацию, включая вклад дисперсий кодировщика:

```
Cov_q(z) = Cov_p(x)[E_q(z|x)[z]] + E_p(x)[diag(σ²(x))]
```

DIP-VAE-II обычно даёт лучшее разделение, поскольку учитывает и дисперсию кодировщика, а не только средние.

#### Преимущества

- Отсутствие дополнительных нейронных сетей (в отличие от FactorVAE)
- Прямой контроль над ковариационной структурой
- Стабильное обучение
- Гибкое управление через два параметра λ_od и λ_d

### 2.4 β-TCVAE: декомпозиция полной корреляции

β-TCVAE основывается на ключевом наблюдении: не все компоненты KL-дивергенции одинаково важны для разделения. Декомпозируя KL-дивергенцию, можно прицельно усилить именно тот компонент, который отвечает за запутанность — полную корреляцию.

#### Декомпозиция KL-дивергенции

KL-дивергенция в целевой функции VAE раскладывается на три члена:

```
E_p(x)[KL(q(z|x) || p(z))] = I(x; z) + TC(q(z)) + ∑_j KL(q(z_j) || p(z_j))
```

где:
- **I(x; z)** — взаимная информация между данными и латентным кодом (Index-Code MI)
- **TC(q(z))** — полная корреляция, измеряющая зависимость между латентными измерениями
- **∑_j KL(q(z_j) || p(z_j))** — сумма расхождений маргиналов от приора (Dimension-wise KL)

#### Целевая функция

```
L_β-TCVAE = E_q(z|x)[log p(x|z)] - α · I(x; z) - β · TC(q(z)) - γ · ∑_j KL(q(z_j) || p(z_j))
```

Стандартная настройка: α = γ = 1, а β > 1. Таким образом, усиливается только штраф на полную корреляцию, не затрагивая другие компоненты. Это позволяет сохранить хорошую реконструкцию (за счёт взаимной информации) и при этом добиться разделения (за счёт минимизации TC).

#### Оценка TC с использованием мини-батчей

Для оценки TC используется метод взвешенной выборки на мини-батчах:

```
TC(q(z)) ≈ E_q(z|x_n)[log q(z) - log q̄(z)]
```

где q(z) оценивается по мини-батчу:

```
log q(z) ≈ log (1/NM) ∑_m q(z|x_m) - log N
```

Здесь N — размер датасета, M — размер мини-батча. Эта оценка не требует дополнительных нейронных сетей и может быть вычислена в рамках стандартного прохода обучения.

#### Преимущества для финансовых данных

β-TCVAE представляет оптимальный баланс для финансовых приложений:
- Минимальная потеря реконструкции (в отличие от β-VAE)
- Отсутствие adversarial-компонента (в отличие от FactorVAE)
- Один интуитивный гиперпараметр β
- Прочная теоретическая основа через декомпозицию KL

---

## 3. Математические основы

### 3.1 Декомпозиция ELBO

Нижняя граница свидетельства (Evidence Lower Bound, ELBO) является центральной целевой функцией для всех вариантов VAE:

```
ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
```

Первый член — логарифм правдоподобия реконструкции, побуждающий декодер точно воспроизводить входные данные. Второй член — дивергенция Кульбака-Лейблера, регуляризирующая апостериорное распределение.

Для понимания методов разделения критически важна более тонкая декомпозиция ожидаемого KL-члена. Усредняя по эмпирическому распределению данных p(x), получаем:

```
E_p(x)[KL(q(z|x) || p(z))] = I_q(x; z) + KL(q(z) || p(z))
```

Далее, член KL(q(z) || p(z)) можно разложить:

```
KL(q(z) || p(z)) = KL(q(z) || ∏_j q(z_j)) + ∑_j KL(q(z_j) || p(z_j))
```

Первый член — полная корреляция TC, второй — сумма маргинальных KL-дивергенций. Объединяя:

```
E_p(x)[KL(q(z|x) || p(z))] = I_q(x; z) + TC(q(z)) + ∑_j KL(q(z_j) || p(z_j))
```

Эта трёхчленная декомпозиция показывает, что стандартный VAE (и β-VAE) штрафует все три компонента равномерно, хотя для разделения важен именно TC.

### 3.2 Полная корреляция (Total Correlation)

Полная корреляция — мера статистической зависимости между случайными величинами, обобщающая взаимную информацию на произвольное число переменных:

```
TC(z_1, ..., z_d) = KL(q(z_1, ..., z_d) || ∏_j q(z_j))
```

Полная корреляция равна нулю тогда и только тогда, когда все компоненты z_j статистически независимы. Минимизация TC — прямой путь к факторизации латентного распределения, а значит, к разделённому представлению.

Для гауссовского распределения TC имеет простую форму:

```
TC = (1/2) · [∑_j log σ²_j - log det(Σ)]
```

где σ²_j — маргинальные дисперсии, а Σ — полная ковариационная матрица. TC равна нулю, когда ковариационная матрица диагональна, то есть все компоненты некоррелированы.

В контексте финансовых данных минимизация TC означает, что обнаруженные латентные факторы — тренд, волатильность, ликвидность и т.д. — будут статистически независимы. Это фундаментально для построения диверсифицированных факторных стратегий.

### 3.3 Цели оптимизации β-VAE и FactorVAE

#### β-VAE

Целевая функция β-VAE может быть записана через трёхчленную декомпозицию:

```
L_β-VAE = E_q(z|x)[log p(x|z)] - β · [I_q(x; z) + TC(q(z)) + ∑_j KL(q(z_j) || p(z_j))]
```

Видно, что β усиливает все три компонента одинаково. Увеличение β:
- Уменьшает взаимную информацию I(x; z), что ухудшает реконструкцию
- Уменьшает TC, что улучшает разделение
- Уменьшает маргинальные KL, что приближает каждый маргинал к приору

Первый и третий эффекты нежелательны для целей разделения — отсюда субоптимальность β-VAE.

#### FactorVAE

FactorVAE добавляет только штраф на TC, используя adversarial-оценку:

```
L_FactorVAE = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z)) - γ · E_q(z)[log D(z) / (1 - D(z))]
```

Дискриминатор D обучается на задаче бинарной классификации:

```
max_D E_q(z)[log D(z)] + E_q̄(z)[log(1 - D(z))]
```

где q̄(z) получается перестановкой координат z по различным примерам в мини-батче. Это создаёт выборки из факторизованного маргинального распределения ∏_j q(z_j) без явной оценки плотности.

### 3.4 Метрики разделения

Количественная оценка качества разделения — нетривиальная задача, особенно в финансах, где истинные генеративные факторы неизвестны. Тем не менее существуют стандартные метрики:

#### DCI (Disentanglement, Completeness, Informativeness)

Метрика DCI использует матрицу важности признаков R ∈ R^{d×k}, где d — число латентных измерений, k — число факторов. Элемент R_{ij} — важность i-го латентного измерения для предсказания j-го фактора (обычно через Gradient Boosting или Lasso).

- **Disentanglement**: Для каждого латентного измерения i вычисляется энтропия нормализованного вектора важности R_{i,:}. Низкая энтропия означает, что измерение связано с одним фактором.
- **Completeness**: Для каждого фактора j вычисляется энтропия нормализованного вектора R_{:,j}. Низкая энтропия означает, что фактор захвачен одним измерением.
- **Informativeness**: Точность предсказания факторов из латентных кодов.

```
DCI_D = 1 - H(ρ_i) / log(K)
```

где ρ_i — нормализованные веса важности для i-го латентного измерения, K — число факторов.

#### MIG (Mutual Information Gap)

Для каждого истинного фактора v_k вычисляется взаимная информация со всеми латентными измерениями z_j. MIG определяется как нормализованный разрыв между двумя наибольшими значениями:

```
MIG = (1/K) ∑_k [I(z_{j*(k)}; v_k) - max_{j ≠ j*(k)} I(z_j; v_k)] / H(v_k)
```

где j*(k) = argmax_j I(z_j; v_k). Высокий MIG означает, что каждый фактор преимущественно захвачен одним латентным измерением.

#### SAP (Separated Attribute Predictability)

SAP оценивает линейную предсказуемость каждого фактора из каждого латентного измерения с использованием SVM или линейной регрессии:

```
SAP = (1/K) ∑_k [S_{j*(k),k} - max_{j ≠ j*(k)} S_{j,k}]
```

где S_{j,k} — точность предсказания k-го фактора из j-го латентного измерения. Высокий SAP указывает на чёткое разделение.

#### Применение метрик в финансах

В финансах истинные генеративные факторы неизвестны, поэтому вместо них используются прокси:
- Известные технические индикаторы (RSI, ATR, OBV, MACD) как суррогаты факторов
- Корреляция латентных измерений с макроэкономическими показателями
- Прогнозная сила отдельных измерений для будущих доходностей
- Стабильность интерпретаций латентных обходов во времени

---

## 4. Практические примеры

### Пример 01: Подготовка данных (Bybit + фондовый рынок)

Подготовка данных — фундамент для всех последующих экспериментов. Мы работаем с двумя источниками: криптовалютные данные с Bybit (через REST API) и данные фондового рынка (через yfinance).

```python
import numpy as np
import pandas as pd
import requests
import time
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass


@dataclass
class MarketDataConfig:
    """Configuration for market data fetching and preprocessing."""
    symbols: List[str]
    interval: str = "60"  # 1h candles
    lookback_days: int = 365
    window_size: int = 24  # 24 hours lookback
    stride: int = 1
    normalize_method: str = "zscore"  # "zscore" or "minmax"
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = ["open", "high", "low", "close", "volume"]


class BybitDataFetcher:
    """Fetch historical OHLCV data from Bybit exchange."""

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def __init__(self, config: MarketDataConfig):
        self.config = config

    def fetch_klines(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch kline data for a single symbol."""
        end_time = int(time.time() * 1000)
        start_time = end_time - self.config.lookback_days * 24 * 3600 * 1000

        all_candles = []
        current_end = end_time

        while current_end > start_time:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": self.config.interval,
                "end": current_end,
                "limit": limit,
            }
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()

            if data["retCode"] != 0 or not data["result"]["list"]:
                break

            candles = data["result"]["list"]
            all_candles.extend(candles)

            current_end = int(candles[-1][0]) - 1
            time.sleep(0.1)  # Rate limiting

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df

    def fetch_multi_symbol(self) -> dict:
        """Fetch data for all configured symbols."""
        data = {}
        for symbol in self.config.symbols:
            print(f"Fetching {symbol}...")
            data[symbol] = self.fetch_klines(symbol)
            time.sleep(0.5)
        return data


class StockDataFetcher:
    """Fetch stock market data using yfinance."""

    def __init__(self, config: MarketDataConfig):
        self.config = config

    def fetch_stocks(self) -> dict:
        """Fetch daily OHLCV data for stock symbols."""
        import yfinance as yf

        data = {}
        for symbol in self.config.symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{self.config.lookback_days}d")
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].reset_index()
            df = df.rename(columns={"Date": "timestamp"})
            data[symbol] = df
        return data


class FinancialDataPreprocessor:
    """Preprocess financial data for Disentangled VAE training."""

    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.scalers = {}

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as additional features."""
        df = df.copy()

        # Returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility (rolling std of returns)
        df["volatility"] = df["log_return"].rolling(window=20).std()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()

        # OBV (On-Balance Volume)
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["obv"] = obv

        # VWAP approximation
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

        # Momentum
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

        df = df.dropna().reset_index(drop=True)
        return df

    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize features using configured method."""
        features = [c for c in df.columns if c != "timestamp"]

        if self.config.normalize_method == "zscore":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df[features] = scaler.fit_transform(df[features])
        self.scalers[symbol] = scaler
        return df

    def create_windows(self, df: pd.DataFrame) -> np.ndarray:
        """Create sliding windows for VAE input."""
        features = [c for c in df.columns if c != "timestamp"]
        values = df[features].values
        n_samples = (len(values) - self.config.window_size) // self.config.stride + 1

        windows = np.zeros((n_samples, self.config.window_size, len(features)))
        for i in range(n_samples):
            start = i * self.config.stride
            end = start + self.config.window_size
            windows[i] = values[start:end]

        return windows

    def prepare_dataset(
        self, raw_data: dict, add_technicals: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """Full preprocessing pipeline."""
        all_windows = []
        symbols_list = []

        for symbol, df in raw_data.items():
            if add_technicals:
                df = self.add_technical_features(df)

            df = self.normalize(df, symbol)
            windows = self.create_windows(df)
            all_windows.append(windows)
            symbols_list.extend([symbol] * len(windows))

        dataset = np.concatenate(all_windows, axis=0)
        return dataset, symbols_list


# --- Usage example ---
if __name__ == "__main__":
    # Crypto data from Bybit
    crypto_config = MarketDataConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        interval="60",
        lookback_days=180,
        window_size=24,
    )
    fetcher = BybitDataFetcher(crypto_config)
    crypto_data = fetcher.fetch_multi_symbol()

    preprocessor = FinancialDataPreprocessor(crypto_config)
    X_crypto, labels_crypto = preprocessor.prepare_dataset(crypto_data)
    print(f"Crypto dataset shape: {X_crypto.shape}")

    # Stock data
    stock_config = MarketDataConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
        interval="1d",
        lookback_days=365,
        window_size=20,
    )
    stock_fetcher = StockDataFetcher(stock_config)
    stock_data = stock_fetcher.fetch_stocks()

    stock_preprocessor = FinancialDataPreprocessor(stock_config)
    X_stocks, labels_stocks = stock_preprocessor.prepare_dataset(stock_data)
    print(f"Stock dataset shape: {X_stocks.shape}")
```

Этот пример создаёт полный конвейер подготовки данных, поддерживающий как криптовалютные данные с Bybit, так и данные фондового рынка. Ключевые особенности:
- Скользящие окна с настраиваемым размером и шагом
- Автоматическое добавление технических индикаторов (RSI, ATR, OBV, VWAP, моментум)
- Гибкая нормализация (z-score или min-max)
- Обработка ограничений по частоте запросов API

---

### Пример 02: Архитектура β-VAE для финансовых временных рядов

Реализация полной архитектуры β-VAE с поддержкой различных методов разделения.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from enum import Enum


class DisentanglementMethod(Enum):
    STANDARD_VAE = "standard"
    BETA_VAE = "beta_vae"
    FACTOR_VAE = "factor_vae"
    DIP_VAE_I = "dip_vae_i"
    DIP_VAE_II = "dip_vae_ii"
    BETA_TCVAE = "beta_tcvae"


class TemporalEncoder(nn.Module):
    """Encoder with 1D convolutions for time series data."""

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_dims: list,
        latent_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim

        # Temporal convolution layers
        conv_layers = []
        in_channels = input_dim
        current_length = window_size

        for i, h_dim in enumerate(hidden_dims):
            conv_layers.append(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1)
            )
            conv_layers.append(nn.BatchNorm1d(h_dim))
            conv_layers.append(nn.LeakyReLU(0.2))
            in_channels = h_dim
            current_length = (current_length + 1) // 2

        self.conv_net = nn.Sequential(*conv_layers)
        self.flat_dim = hidden_dims[-1] * current_length

        # Latent projections
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, window_size, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, window_size)
        h = self.conv_net(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class TemporalDecoder(nn.Module):
    """Decoder with transposed 1D convolutions."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list,
        output_dim: int,
        window_size: int,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.window_size = window_size

        # Calculate dimensions
        hidden_dims_rev = list(reversed(hidden_dims))
        n_layers = len(hidden_dims_rev)
        self.init_length = window_size
        for _ in range(n_layers):
            self.init_length = (self.init_length + 1) // 2

        self.fc = nn.Linear(latent_dim, hidden_dims_rev[0] * self.init_length)
        self.init_channels = hidden_dims_rev[0]

        # Transposed convolution layers
        deconv_layers = []
        for i in range(len(hidden_dims_rev) - 1):
            deconv_layers.append(
                nn.ConvTranspose1d(
                    hidden_dims_rev[i], hidden_dims_rev[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                )
            )
            deconv_layers.append(nn.BatchNorm1d(hidden_dims_rev[i + 1]))
            deconv_layers.append(nn.LeakyReLU(0.2))

        # Final layer
        deconv_layers.append(
            nn.ConvTranspose1d(
                hidden_dims_rev[-1], output_dim,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            )
        )

        self.deconv_net = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.init_channels, self.init_length)
        h = self.deconv_net(h)
        # Trim or pad to exact window_size
        h = h[:, :, :self.window_size]
        return h.permute(0, 2, 1)  # (batch, window_size, output_dim)


class TCDiscriminator(nn.Module):
    """Discriminator for FactorVAE Total Correlation estimation."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DisentangledVAE(nn.Module):
    """Unified Disentangled VAE supporting multiple disentanglement methods."""

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        latent_dim: int = 10,
        hidden_dims: list = None,
        method: DisentanglementMethod = DisentanglementMethod.BETA_VAE,
        beta: float = 4.0,
        gamma: float = 10.0,
        lambda_od: float = 10.0,
        lambda_d: float = 1.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.latent_dim = latent_dim
        self.method = method
        self.beta = beta
        self.gamma = gamma
        self.lambda_od = lambda_od
        self.lambda_d = lambda_d

        self.encoder = TemporalEncoder(input_dim, window_size, hidden_dims, latent_dim)
        self.decoder = TemporalDecoder(latent_dim, hidden_dims, input_dim, window_size)

        if method == DisentanglementMethod.FACTOR_VAE:
            self.discriminator = TCDiscriminator(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}

    def compute_loss(
        self,
        x: torch.Tensor,
        output: Dict[str, torch.Tensor],
        dataset_size: int = 10000,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss based on the selected disentanglement method."""
        x_recon = output["x_recon"]
        mu = output["mu"]
        logvar = output["logvar"]
        z = output["z"]

        # Reconstruction loss (MSE for continuous financial data)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # Standard KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if self.method == DisentanglementMethod.STANDARD_VAE:
            total_loss = recon_loss + kl_loss
            return {"loss": total_loss, "recon": recon_loss, "kl": kl_loss}

        elif self.method == DisentanglementMethod.BETA_VAE:
            total_loss = recon_loss + self.beta * kl_loss
            return {"loss": total_loss, "recon": recon_loss, "kl": kl_loss}

        elif self.method == DisentanglementMethod.BETA_TCVAE:
            return self._beta_tcvae_loss(x, x_recon, mu, logvar, z, dataset_size)

        elif self.method in (DisentanglementMethod.DIP_VAE_I, DisentanglementMethod.DIP_VAE_II):
            return self._dip_vae_loss(x_recon, x, mu, logvar)

        elif self.method == DisentanglementMethod.FACTOR_VAE:
            return self._factor_vae_loss(x_recon, x, mu, logvar, z)

    def _beta_tcvae_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        dataset_size: int,
    ) -> Dict[str, torch.Tensor]:
        """β-TCVAE loss with decomposed KL."""
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        batch_size = z.shape[0]
        d = z.shape[1]

        # log q(z|x) for the current batch
        log_qz_given_x = -0.5 * (logvar + (z - mu).pow(2) / logvar.exp()).sum(dim=1)

        # log q(z) using minibatch weighted sampling
        # log q(z_j) for marginals
        log_qz = []
        log_qz_product = torch.zeros(batch_size, device=z.device)

        for j in range(d):
            z_j = z[:, j].unsqueeze(1)  # (batch, 1)
            mu_j = mu[:, j].unsqueeze(0)  # (1, batch)
            logvar_j = logvar[:, j].unsqueeze(0)  # (1, batch)

            log_qz_j = -0.5 * (
                logvar_j + (z_j - mu_j).pow(2) / logvar_j.exp() + np.log(2 * np.pi)
            )
            log_qz_j = torch.logsumexp(log_qz_j, dim=1) - np.log(batch_size * dataset_size)
            log_qz.append(log_qz_j)
            log_qz_product += log_qz_j

        # log q(z) joint
        log_qz_joint_parts = []
        for j in range(d):
            z_j = z[:, j].unsqueeze(1)
            mu_j = mu[:, j].unsqueeze(0)
            logvar_j = logvar[:, j].unsqueeze(0)
            log_part = -0.5 * (logvar_j + (z_j - mu_j).pow(2) / logvar_j.exp() + np.log(2 * np.pi))
            log_qz_joint_parts.append(log_part)

        log_qz_joint = sum(
            torch.logsumexp(part, dim=1) for part in log_qz_joint_parts
        ) - d * np.log(batch_size * dataset_size)

        # Decomposed terms
        mi = (log_qz_given_x - log_qz_joint).mean()  # Index-Code MI
        tc = (log_qz_joint - log_qz_product).mean()  # Total Correlation
        dwkl = (log_qz_product + 0.5 * d * np.log(2 * np.pi * np.e)).mean()  # Dimension-wise KL

        total_loss = recon_loss + mi + self.beta * tc + dwkl
        return {
            "loss": total_loss,
            "recon": recon_loss,
            "mi": mi,
            "tc": tc,
            "dwkl": dwkl,
        }

    def _dip_vae_loss(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """DIP-VAE loss with covariance penalties."""
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if self.method == DisentanglementMethod.DIP_VAE_I:
            # Covariance of means
            cov = torch.mm(mu.t(), mu) / mu.shape[0] - torch.mm(
                mu.mean(0, keepdim=True).t(), mu.mean(0, keepdim=True)
            )
        else:  # DIP_VAE_II
            # Full covariance including variance term
            centered_mu = mu - mu.mean(0, keepdim=True)
            cov_mu = torch.mm(centered_mu.t(), centered_mu) / mu.shape[0]
            mean_var = torch.mean(logvar.exp(), dim=0)
            cov = cov_mu + torch.diag(mean_var)

        # Off-diagonal penalty
        off_diag = cov - torch.diag(torch.diag(cov))
        off_diag_penalty = (off_diag ** 2).sum()

        # Diagonal penalty (push towards 1)
        diag_penalty = ((torch.diag(cov) - 1) ** 2).sum()

        dip_loss = self.lambda_od * off_diag_penalty + self.lambda_d * diag_penalty
        total_loss = recon_loss + kl_loss + dip_loss

        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
            "dip": dip_loss,
            "off_diag": off_diag_penalty,
            "diag": diag_penalty,
        }

    def _factor_vae_loss(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """FactorVAE loss (VAE part only, discriminator trained separately)."""
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # TC estimated via discriminator
        d_z = self.discriminator(z)
        tc_estimate = (d_z[:, 0] - d_z[:, 1]).mean()

        total_loss = recon_loss + kl_loss + self.gamma * tc_estimate
        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
            "tc": tc_estimate,
        }
```

Эта архитектура объединяет все четыре метода разделения в единой модели с переключаемой целевой функцией. Ключевые элементы:

- **Временной кодировщик** на основе одномерных свёрток для захвата паттернов во временных рядах
- **Транспонированный декодер** для восстановления последовательностей
- **Дискриминатор полной корреляции** для метода FactorVAE
- **Единый интерфейс потерь** с автоматическим выбором формулы в зависимости от метода

---

### Пример 03: Обучение модели с разделением

Полный цикл обучения с поддержкой отжига β, мониторингом метрик разделения и ранней остановкой.

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional
import json
import os


class BetaScheduler:
    """Scheduler for beta annealing during training."""

    def __init__(
        self,
        target_beta: float,
        warmup_steps: int = 1000,
        anneal_steps: int = 5000,
        strategy: str = "linear",  # "linear", "cyclic", "monotonic"
        n_cycles: int = 4,
    ):
        self.target_beta = target_beta
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.strategy = strategy
        self.n_cycles = n_cycles

    def get_beta(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0

        progress = (step - self.warmup_steps) / max(self.anneal_steps, 1)

        if self.strategy == "linear":
            return min(self.target_beta, self.target_beta * progress)

        elif self.strategy == "cyclic":
            cycle_length = self.anneal_steps / self.n_cycles
            cycle_progress = ((step - self.warmup_steps) % cycle_length) / cycle_length
            return self.target_beta * min(1.0, cycle_progress * 2)

        elif self.strategy == "monotonic":
            return self.target_beta * (1 - np.exp(-5 * progress))

        return self.target_beta


class DisentanglementMetrics:
    """Compute disentanglement metrics using proxy factors."""

    @staticmethod
    def compute_correlation_matrix(
        z: np.ndarray, factors: np.ndarray
    ) -> np.ndarray:
        """Compute correlation between latent dims and proxy factors."""
        n_latent = z.shape[1]
        n_factors = factors.shape[1]
        corr_matrix = np.zeros((n_latent, n_factors))

        for i in range(n_latent):
            for j in range(n_factors):
                corr_matrix[i, j] = np.abs(
                    np.corrcoef(z[:, i], factors[:, j])[0, 1]
                )

        return corr_matrix

    @staticmethod
    def mutual_information_gap(corr_matrix: np.ndarray) -> float:
        """Approximate MIG using correlation as proxy for MI."""
        n_factors = corr_matrix.shape[1]
        mig = 0.0

        for j in range(n_factors):
            sorted_corr = np.sort(corr_matrix[:, j])[::-1]
            if len(sorted_corr) >= 2:
                mig += sorted_corr[0] - sorted_corr[1]

        return mig / n_factors

    @staticmethod
    def dci_disentanglement(corr_matrix: np.ndarray) -> float:
        """Compute DCI disentanglement score."""
        n_latent = corr_matrix.shape[0]
        dci = 0.0

        for i in range(n_latent):
            row = corr_matrix[i]
            if row.sum() > 0:
                p = row / (row.sum() + 1e-10)
                entropy = -np.sum(p * np.log(p + 1e-10))
                max_entropy = np.log(corr_matrix.shape[1])
                dci += 1 - entropy / (max_entropy + 1e-10)

        return dci / n_latent

    @staticmethod
    def latent_correlation(z: np.ndarray) -> float:
        """Average absolute off-diagonal correlation between latent dims."""
        corr = np.corrcoef(z.T)
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return np.abs(corr[mask]).mean()


class DisentangledVAETrainer:
    """Training loop for Disentangled VAE models."""

    def __init__(
        self,
        model: "DisentangledVAE",
        learning_rate: float = 1e-3,
        beta_scheduler: Optional[BetaScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.beta_scheduler = beta_scheduler

        self.optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if "discriminator" not in n],
            lr=learning_rate,
        )

        if hasattr(model, "discriminator"):
            self.disc_optimizer = optim.Adam(
                model.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9)
            )

        self.history: Dict[str, List[float]] = {
            "loss": [], "recon": [], "kl": [],
            "tc": [], "mig": [], "dci": [],
            "latent_corr": [], "beta": [],
        }
        self.global_step = 0

    def train_step(
        self, batch: torch.Tensor, dataset_size: int
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        batch = batch.to(self.device)

        # Update beta if using scheduler
        if self.beta_scheduler is not None:
            self.model.beta = self.beta_scheduler.get_beta(self.global_step)

        # Forward pass
        output = self.model(batch)
        losses = self.model.compute_loss(batch, output, dataset_size)

        # Train FactorVAE discriminator
        if self.model.method == DisentanglementMethod.FACTOR_VAE:
            self._train_discriminator(batch)

        # Backward pass
        self.optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.global_step += 1
        return {k: v.item() for k, v in losses.items()}

    def _train_discriminator(self, batch: torch.Tensor):
        """Train FactorVAE discriminator."""
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encoder(batch)
            z = self.model.reparameterize(mu, logvar)

        # Permuted z (factorized marginal)
        z_perm = z.clone()
        for j in range(z.shape[1]):
            perm_idx = torch.randperm(z.shape[0])
            z_perm[:, j] = z[perm_idx, j]

        self.disc_optimizer.zero_grad()

        d_real = self.model.discriminator(z)
        d_fake = self.model.discriminator(z_perm)

        disc_loss = (
            F.cross_entropy(d_real, torch.zeros(z.shape[0], dtype=torch.long, device=self.device))
            + F.cross_entropy(d_fake, torch.ones(z.shape[0], dtype=torch.long, device=self.device))
        ) * 0.5

        disc_loss.backward()
        self.disc_optimizer.step()
        self.model.train()

    def train_epoch(
        self, dataloader: DataLoader, dataset_size: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {}

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            step_losses = self.train_step(batch, dataset_size)

            for k, v in step_losses.items():
                epoch_losses.setdefault(k, []).append(v)

        return {k: np.mean(v) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(
        self, dataloader: DataLoader, proxy_factors: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model and compute disentanglement metrics."""
        self.model.eval()

        all_z = []
        total_recon = 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)

            output = self.model(batch)
            total_recon += F.mse_loss(output["x_recon"], batch).item()
            all_z.append(output["mu"].cpu().numpy())
            n_batches += 1

        z = np.concatenate(all_z, axis=0)
        metrics = {
            "recon_mse": total_recon / n_batches,
            "latent_corr": DisentanglementMetrics.latent_correlation(z),
        }

        if proxy_factors is not None:
            corr_matrix = DisentanglementMetrics.compute_correlation_matrix(
                z[: len(proxy_factors)], proxy_factors
            )
            metrics["mig"] = DisentanglementMetrics.mutual_information_gap(corr_matrix)
            metrics["dci"] = DisentanglementMetrics.dci_disentanglement(corr_matrix)

        return metrics

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        n_epochs: int = 100,
        batch_size: int = 64,
        proxy_factors: Optional[np.ndarray] = None,
        patience: int = 20,
        save_path: str = "checkpoints",
    ) -> Dict[str, List[float]]:
        """Full training loop with validation and early stopping."""
        train_dataset = TensorDataset(torch.FloatTensor(train_data))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data is not None:
            val_dataset = TensorDataset(torch.FloatTensor(val_data))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_loss = float("inf")
        patience_counter = 0
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, len(train_data))

            # Log
            for k, v in train_metrics.items():
                self.history.setdefault(k, []).append(v)
            self.history["beta"].append(self.model.beta)

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, proxy_factors)
                for k, v in val_metrics.items():
                    self.history.setdefault(f"val_{k}", []).append(v)

                # Early stopping
                val_loss = val_metrics["recon_mse"]
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(save_path, "best_model.pt"),
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Recon: {train_metrics['recon']:.4f} | "
                    f"Beta: {self.model.beta:.2f}"
                )

        return self.history


# --- Usage example ---
if __name__ == "__main__":
    # Assume X_train, X_val are prepared numpy arrays
    # Shape: (n_samples, window_size, n_features)
    input_dim = 12  # OHLCV + technical indicators
    window_size = 24
    latent_dim = 10

    # Create model with beta-TCVAE
    model = DisentangledVAE(
        input_dim=input_dim,
        window_size=window_size,
        latent_dim=latent_dim,
        hidden_dims=[32, 64, 128],
        method=DisentanglementMethod.BETA_TCVAE,
        beta=6.0,
    )

    scheduler = BetaScheduler(
        target_beta=6.0,
        warmup_steps=500,
        anneal_steps=3000,
        strategy="linear",
    )

    trainer = DisentangledVAETrainer(
        model=model,
        learning_rate=1e-3,
        beta_scheduler=scheduler,
    )

    # Generate synthetic data for demonstration
    X_train = np.random.randn(5000, window_size, input_dim).astype(np.float32)
    X_val = np.random.randn(1000, window_size, input_dim).astype(np.float32)

    history = trainer.train(
        train_data=X_train,
        val_data=X_val,
        n_epochs=50,
        batch_size=64,
    )
```

Процесс обучения включает:
- **Планировщик β** с тремя стратегиями отжига (линейный, циклический, монотонный)
- **Метрики разделения** на основе корреляционного анализа с прокси-факторами
- **Раннюю остановку** по валидационной ошибке реконструкции
- **Отдельное обучение дискриминатора** для метода FactorVAE
- **Градиентное обрезание** для стабильности обучения

---

### Пример 04: Анализ латентных факторов (траверсы и интерпретация)

После обучения модели критически важно проанализировать, какие рыночные факторы кодирует каждое латентное измерение. Латентные обходы (traversals) — основной инструмент для этого.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import stats


class LatentAnalyzer:
    """Tools for analyzing and interpreting learned latent factors."""

    def __init__(
        self,
        model: "DisentangledVAE",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode_dataset(self, data: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Encode entire dataset to latent space."""
        all_mu = []
        all_logvar = []

        for i in range(0, len(data), batch_size):
            batch = torch.FloatTensor(data[i : i + batch_size]).to(self.device)
            mu, logvar = self.model.encoder(batch)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())

        return np.concatenate(all_mu), np.concatenate(all_logvar)

    @torch.no_grad()
    def latent_traversal(
        self,
        reference: np.ndarray,
        dim: int,
        n_steps: int = 11,
        range_std: float = 3.0,
    ) -> np.ndarray:
        """Traverse a single latent dimension while fixing others."""
        ref_tensor = torch.FloatTensor(reference).unsqueeze(0).to(self.device)
        mu, logvar = self.model.encoder(ref_tensor)
        mu = mu.cpu().numpy()[0]

        std = np.exp(0.5 * logvar.cpu().numpy()[0])
        values = np.linspace(
            mu[dim] - range_std * std[dim],
            mu[dim] + range_std * std[dim],
            n_steps,
        )

        traversals = []
        for val in values:
            z = mu.copy()
            z[dim] = val
            z_tensor = torch.FloatTensor(z).unsqueeze(0).to(self.device)
            recon = self.model.decoder(z_tensor)
            traversals.append(recon.cpu().numpy()[0])

        return np.array(traversals)

    def full_traversal_analysis(
        self,
        reference: np.ndarray,
        feature_names: List[str] = None,
        n_steps: int = 11,
    ) -> Dict[int, np.ndarray]:
        """Perform traversal for all latent dimensions."""
        results = {}
        for dim in range(self.model.latent_dim):
            results[dim] = self.latent_traversal(reference, dim, n_steps)
        return results

    def compute_traversal_sensitivity(
        self, traversals: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Compute how much each feature changes during each traversal."""
        n_dims = len(traversals)
        n_features = traversals[0].shape[-1]
        sensitivity = np.zeros((n_dims, n_features))

        for dim, trav in traversals.items():
            # Mean across time steps, std across traversal steps
            feature_means = trav.mean(axis=1)  # (n_steps, n_features)
            sensitivity[dim] = feature_means.std(axis=0)

        return sensitivity

    def interpret_factors(
        self,
        sensitivity: np.ndarray,
        feature_names: List[str],
        top_k: int = 3,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Interpret each latent dimension based on feature sensitivity."""
        interpretations = {}
        for dim in range(sensitivity.shape[0]):
            scores = sensitivity[dim]
            top_indices = np.argsort(scores)[::-1][:top_k]
            interpretations[dim] = [
                (feature_names[i], scores[i]) for i in top_indices
            ]
        return interpretations

    def compute_factor_stability(
        self,
        data: np.ndarray,
        n_splits: int = 5,
    ) -> np.ndarray:
        """Assess temporal stability of factor interpretations."""
        split_size = len(data) // n_splits
        all_sensitivities = []

        for i in range(n_splits):
            ref = data[i * split_size]
            traversals = self.full_traversal_analysis(ref)
            sensitivity = self.compute_traversal_sensitivity(traversals)
            all_sensitivities.append(sensitivity)

        stacked = np.array(all_sensitivities)
        stability = 1.0 - stacked.std(axis=0) / (stacked.mean(axis=0) + 1e-10)
        return stability

    def factor_return_correlation(
        self,
        z: np.ndarray,
        returns: np.ndarray,
        horizons: List[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Correlate each latent dimension with future returns at various horizons."""
        if horizons is None:
            horizons = [1, 5, 10, 20]

        correlations = {}
        for h in horizons:
            corr = np.zeros(z.shape[1])
            for dim in range(z.shape[1]):
                valid = min(len(z), len(returns) - h)
                c, p = stats.pearsonr(z[:valid, dim], returns[h : h + valid])
                corr[dim] = c if p < 0.05 else 0.0
            correlations[h] = corr

        return correlations

    def plot_traversals(
        self,
        traversals: Dict[int, np.ndarray],
        feature_names: List[str],
        save_path: Optional[str] = None,
    ):
        """Visualize latent traversals for all dimensions."""
        n_dims = len(traversals)
        n_features = min(5, traversals[0].shape[-1])

        fig, axes = plt.subplots(n_dims, n_features, figsize=(4 * n_features, 3 * n_dims))
        if n_dims == 1:
            axes = axes[np.newaxis, :]

        for dim in range(n_dims):
            trav = traversals[dim]
            for feat_idx in range(n_features):
                ax = axes[dim, feat_idx]
                for step in range(trav.shape[0]):
                    alpha = 0.3 + 0.7 * step / trav.shape[0]
                    ax.plot(trav[step, :, feat_idx], alpha=alpha, color="blue")
                ax.set_title(f"z_{dim} -> {feature_names[feat_idx]}")
                ax.set_xlabel("Time step")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_latent_space(
        self,
        z: np.ndarray,
        labels: Optional[np.ndarray] = None,
        dims: Tuple[int, int] = (0, 1),
        save_path: Optional[str] = None,
    ):
        """Plot 2D projection of the latent space."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            scatter = ax.scatter(
                z[:, dims[0]], z[:, dims[1]],
                c=labels, cmap="viridis", alpha=0.5, s=10,
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(z[:, dims[0]], z[:, dims[1]], alpha=0.3, s=10)

        ax.set_xlabel(f"z_{dims[0]}")
        ax.set_ylabel(f"z_{dims[1]}")
        ax.set_title("Latent Space Projection")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# --- Usage example ---
if __name__ == "__main__":
    feature_names = [
        "open", "high", "low", "close", "volume",
        "log_return", "volatility", "rsi", "atr", "obv", "vwap", "momentum",
    ]

    # Assume trained model and data are available
    analyzer = LatentAnalyzer(model)

    # Encode dataset
    z_mu, z_logvar = analyzer.encode_dataset(X_train)
    print(f"Latent space shape: {z_mu.shape}")

    # Traversal analysis
    ref_sample = X_train[0]
    traversals = analyzer.full_traversal_analysis(ref_sample)

    # Sensitivity analysis
    sensitivity = analyzer.compute_traversal_sensitivity(traversals)
    interpretations = analyzer.interpret_factors(sensitivity, feature_names)

    for dim, factors in interpretations.items():
        top_factors = ", ".join([f"{name}: {score:.3f}" for name, score in factors])
        print(f"Latent dim {dim}: {top_factors}")

    # Factor stability
    stability = analyzer.compute_factor_stability(X_train)
    print(f"Mean stability: {stability.mean():.3f}")

    # Visualizations
    analyzer.plot_traversals(traversals, feature_names, "traversals.png")
    analyzer.plot_latent_space(z_mu, save_path="latent_space.png")
```

Анализ латентных факторов включает:
- **Латентные обходы**: варьирование одного измерения при фиксации остальных для понимания влияния каждого фактора
- **Анализ чувствительности**: какие признаки (close, volume, RSI и т.д.) наиболее чувствительны к каждому латентному измерению
- **Интерпретация факторов**: автоматическое сопоставление латентных измерений с финансовыми концепциями
- **Стабильность факторов**: проверка устойчивости интерпретаций во времени
- **Корреляция с доходностями**: прогнозная сила отдельных факторов для будущих доходностей

---

### Пример 05: Торговая стратегия на основе разделённых факторов

Построение полной торговой стратегии, использующей разделённые латентные факторы для генерации сигналов.

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import torch


class SignalType(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    signal: SignalType
    confidence: float
    dominant_factor: int
    factor_values: np.ndarray


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    position_size: float = 0.1  # fraction of capital per trade
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_positions: int = 5
    signal_threshold: float = 1.5  # std deviations
    rebalance_frequency: int = 1  # every N periods
    stop_loss: float = 0.02
    take_profit: float = 0.05


class DisentangledFactorStrategy:
    """Trading strategy based on disentangled latent factors."""

    def __init__(
        self,
        model: "DisentangledVAE",
        analyzer: "LatentAnalyzer",
        config: BacktestConfig,
        factor_assignments: Dict[int, str] = None,
        tradeable_factors: List[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.analyzer = analyzer
        self.config = config
        self.device = device

        # Factor assignments: latent_dim -> factor_name
        self.factor_assignments = factor_assignments or {}
        # Which factors to use for trading signals
        self.tradeable_factors = tradeable_factors or list(range(model.latent_dim))

        # Historical statistics for z-scoring
        self.z_mean = None
        self.z_std = None

    def calibrate(self, historical_data: np.ndarray):
        """Calibrate strategy on historical data."""
        z_mu, _ = self.analyzer.encode_dataset(historical_data)
        self.z_mean = z_mu.mean(axis=0)
        self.z_std = z_mu.std(axis=0)

    def generate_signal(
        self,
        window: np.ndarray,
        timestamp: pd.Timestamp,
    ) -> TradeSignal:
        """Generate trading signal from a single data window."""
        with torch.no_grad():
            x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            mu, _ = self.model.encoder(x)
            z = mu.cpu().numpy()[0]

        # Z-score the latent values
        z_scored = (z - self.z_mean) / (self.z_std + 1e-10)

        # Aggregate signal from tradeable factors
        factor_signals = {}
        for dim in self.tradeable_factors:
            factor_signals[dim] = z_scored[dim]

        # Find dominant factor (highest absolute z-score)
        dominant = max(factor_signals, key=lambda d: abs(factor_signals[d]))
        dominant_value = factor_signals[dominant]

        # Generate signal based on dominant factor
        if abs(dominant_value) > self.config.signal_threshold:
            if dominant_value > 0:
                signal = SignalType.LONG
            else:
                signal = SignalType.SHORT
            confidence = min(1.0, abs(dominant_value) / (2 * self.config.signal_threshold))
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.0

        return TradeSignal(
            timestamp=timestamp,
            signal=signal,
            confidence=confidence,
            dominant_factor=dominant,
            factor_values=z_scored,
        )

    def generate_multi_factor_signal(
        self,
        window: np.ndarray,
        timestamp: pd.Timestamp,
        weights: Optional[Dict[int, float]] = None,
    ) -> TradeSignal:
        """Generate signal using weighted combination of factors."""
        with torch.no_grad():
            x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            mu, _ = self.model.encoder(x)
            z = mu.cpu().numpy()[0]

        z_scored = (z - self.z_mean) / (self.z_std + 1e-10)

        if weights is None:
            weights = {d: 1.0 / len(self.tradeable_factors) for d in self.tradeable_factors}

        composite_signal = sum(
            weights.get(d, 0) * z_scored[d] for d in self.tradeable_factors
        )

        dominant = max(self.tradeable_factors, key=lambda d: abs(z_scored[d]))

        if abs(composite_signal) > self.config.signal_threshold:
            signal = SignalType.LONG if composite_signal > 0 else SignalType.SHORT
            confidence = min(1.0, abs(composite_signal) / (2 * self.config.signal_threshold))
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.0

        return TradeSignal(
            timestamp=timestamp,
            signal=signal,
            confidence=confidence,
            dominant_factor=dominant,
            factor_values=z_scored,
        )


class FactorBacktester:
    """Backtesting engine for factor-based strategies."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run_backtest(
        self,
        signals: List[TradeSignal],
        prices: np.ndarray,
        timestamps: List[pd.Timestamp],
    ) -> pd.DataFrame:
        """Run backtest and return equity curve with metrics."""
        capital = self.config.initial_capital
        position = 0.0
        entry_price = 0.0
        trades = []

        equity_curve = []
        positions_hist = []

        for i, (signal, price) in enumerate(zip(signals, prices)):
            # Check stop-loss / take-profit
            if position != 0:
                pnl_pct = (price - entry_price) / entry_price * np.sign(position)
                if pnl_pct <= -self.config.stop_loss or pnl_pct >= self.config.take_profit:
                    # Close position
                    trade_pnl = position * (price - entry_price)
                    cost = abs(position * price) * self.config.transaction_cost
                    capital += trade_pnl - cost
                    trades.append({
                        "exit_time": signal.timestamp,
                        "exit_price": price,
                        "pnl": trade_pnl - cost,
                        "type": "stop_loss" if pnl_pct <= -self.config.stop_loss else "take_profit",
                    })
                    position = 0.0

            # Process signal
            if i % self.config.rebalance_frequency == 0:
                target_position = 0.0

                if signal.signal == SignalType.LONG:
                    target_position = (
                        capital * self.config.position_size * signal.confidence / price
                    )
                elif signal.signal == SignalType.SHORT:
                    target_position = (
                        -capital * self.config.position_size * signal.confidence / price
                    )

                # Execute trade
                delta = target_position - position
                if abs(delta) > 1e-10:
                    cost = abs(delta * price) * (
                        self.config.transaction_cost + self.config.slippage
                    )
                    capital -= cost
                    position = target_position
                    entry_price = price

            # Mark to market
            portfolio_value = capital + position * price
            equity_curve.append({
                "timestamp": signal.timestamp,
                "equity": portfolio_value,
                "capital": capital,
                "position": position,
                "price": price,
                "signal": signal.signal.value,
                "dominant_factor": signal.dominant_factor,
                "confidence": signal.confidence,
            })

        return pd.DataFrame(equity_curve)

    @staticmethod
    def compute_metrics(equity_df: pd.DataFrame) -> Dict[str, float]:
        """Compute comprehensive trading metrics."""
        equity = equity_df["equity"].values
        returns = np.diff(equity) / equity[:-1]

        total_return = (equity[-1] - equity[0]) / equity[0]
        n_periods = len(returns)

        # Annualization factor (assume hourly data, 24/7 for crypto)
        ann_factor = np.sqrt(365 * 24)

        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * ann_factor

        # Sortino ratio
        downside = returns[returns < 0]
        sortino = np.mean(returns) / (np.std(downside) + 1e-10) * ann_factor if len(downside) > 0 else 0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = drawdown.max()

        # Calmar ratio
        calmar = (total_return / max_drawdown) if max_drawdown > 0 else 0

        # Win rate
        winning = returns[returns > 0]
        win_rate = len(winning) / len(returns) if len(returns) > 0 else 0

        # Profit factor
        gross_profit = winning.sum() if len(winning) > 0 else 0
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-10)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": int((np.diff(equity_df["signal"].values) != 0).sum()),
            "avg_return": float(np.mean(returns)),
            "volatility": float(np.std(returns) * ann_factor),
        }

    @staticmethod
    def factor_attribution(
        equity_df: pd.DataFrame, n_factors: int
    ) -> Dict[int, float]:
        """Attribute returns to individual latent factors."""
        attribution = {}
        for factor in range(n_factors):
            mask = equity_df["dominant_factor"] == factor
            if mask.sum() > 0:
                factor_equity = equity_df.loc[mask, "equity"].values
                if len(factor_equity) > 1:
                    factor_return = (factor_equity[-1] - factor_equity[0]) / factor_equity[0]
                else:
                    factor_return = 0.0
                attribution[factor] = factor_return
        return attribution


# --- Usage example ---
if __name__ == "__main__":
    # Configuration
    bt_config = BacktestConfig(
        initial_capital=100_000,
        position_size=0.1,
        transaction_cost=0.001,
        signal_threshold=1.5,
        stop_loss=0.02,
        take_profit=0.05,
    )

    # Initialize strategy
    strategy = DisentangledFactorStrategy(
        model=model,
        analyzer=analyzer,
        config=bt_config,
        factor_assignments={0: "trend", 1: "volatility", 2: "momentum"},
        tradeable_factors=[0, 2],  # Trade only trend and momentum
    )

    # Calibrate on training data
    strategy.calibrate(X_train)

    # Generate signals on test data
    signals = []
    timestamps = pd.date_range("2024-01-01", periods=len(X_test), freq="h")

    for i in range(len(X_test)):
        signal = strategy.generate_signal(X_test[i], timestamps[i])
        signals.append(signal)

    # Extract close prices from test data
    close_prices = X_test[:, -1, 3]  # Last timestep, close feature

    # Run backtest
    backtester = FactorBacktester(bt_config)
    equity_df = backtester.run_backtest(signals, close_prices, timestamps)

    # Compute metrics
    metrics = backtester.compute_metrics(equity_df)
    print("\n=== Backtest Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Factor attribution
    attribution = backtester.factor_attribution(equity_df, model.latent_dim)
    print("\n=== Factor Attribution ===")
    for factor, ret in attribution.items():
        name = strategy.factor_assignments.get(factor, f"factor_{factor}")
        print(f"  {name}: {ret:.4f}")
```

Торговая стратегия включает:
- **Генерацию сигналов** на основе z-скоров латентных факторов
- **Многофакторные сигналы** с настраиваемыми весами
- **Полный бэктестинг** со стоп-лоссами, тейк-профитами, транзакционными издержками и проскальзыванием
- **Атрибуцию доходности** по факторам для понимания источников прибыли
- **Метрики производительности**: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor

---

## 5. Реализация на Rust

### Структура проекта

```
238_disentangled_vae_trading/
└── rust/
    ├── Cargo.toml
    └── src/
        ├── lib.rs              # Core library
        ├── model.rs            # VAE architecture (encoder, decoder)
        ├── loss.rs             # Loss functions for all methods
        ├── training.rs         # Training loop and schedulers
        ├── data.rs             # Data fetching (Bybit API) and preprocessing
        ├── latent.rs           # Latent analysis and traversals
        ├── strategy.rs         # Trading strategy and backtesting
        ├── metrics.rs          # Disentanglement metrics
        └── main.rs             # CLI entry point
```

### Зависимости (Cargo.toml)

```toml
[package]
name = "disentangled-vae-trading"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = { version = "0.16", features = ["blas"] }
ndarray-rand = "0.15"
ndarray-linalg = { version = "0.17", features = ["openblas-static"] }
rand = "0.8"
rand_distr = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.12", features = ["json", "blocking"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1"
thiserror = "1"
log = "0.4"
env_logger = "0.11"
clap = { version = "4", features = ["derive"] }
```

### Быстрый старт

```bash
# Сборка проекта
cd rust && cargo build --release

# Запуск с параметрами по умолчанию (β-VAE, BTCUSDT, 1h)
cargo run --release -- \
    --method beta-vae \
    --beta 4.0 \
    --latent-dim 10 \
    --symbols BTCUSDT,ETHUSDT \
    --interval 60 \
    --epochs 100

# Запуск с β-TCVAE
cargo run --release -- \
    --method beta-tcvae \
    --beta 6.0 \
    --latent-dim 8 \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --interval 60 \
    --epochs 200 \
    --annealing linear

# Анализ латентных факторов
cargo run --release -- analyze \
    --model-path checkpoints/best_model.bin \
    --data-path data/btcusdt_1h.csv \
    --output traversals/

# Бэктестинг стратегии
cargo run --release -- backtest \
    --model-path checkpoints/best_model.bin \
    --symbols BTCUSDT \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --capital 100000 \
    --position-size 0.1
```

### Ключевые особенности реализации на Rust

- **Производительность**: Вычисления на ndarray с BLAS-ускорением для матричных операций
- **Безопасность типов**: Строгая типизация гарантирует корректность вычислений потерь и метрик
- **Параллелизм**: Обработка данных и генерация сигналов используют tokio для асинхронного выполнения
- **Интеграция с Bybit**: Нативный HTTP-клиент для получения рыночных данных без внешних зависимостей
- **Сериализация моделей**: Сохранение и загрузка весов через serde для развёртывания в production

---

## 6. Реализация на Python

### Структура проекта

```
238_disentangled_vae_trading/
└── python/
    ├── pyproject.toml
    ├── requirements.txt
    └── src/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   ├── bybit_fetcher.py      # Bybit API client
        │   ├── stock_fetcher.py       # yfinance integration
        │   └── preprocessor.py        # Feature engineering, normalization
        ├── models/
        │   ├── __init__.py
        │   ├── encoder.py             # Temporal encoder
        │   ├── decoder.py             # Temporal decoder
        │   ├── disentangled_vae.py     # Unified model
        │   └── discriminator.py       # TC discriminator for FactorVAE
        ├── training/
        │   ├── __init__.py
        │   ├── trainer.py             # Training loop
        │   ├── scheduler.py           # Beta scheduling
        │   └── losses.py              # All loss functions
        ├── analysis/
        │   ├── __init__.py
        │   ├── traversals.py          # Latent traversals
        │   ├── interpretation.py      # Factor interpretation
        │   └── metrics.py             # DCI, MIG, SAP metrics
        ├── strategy/
        │   ├── __init__.py
        │   ├── signal_generator.py    # Signal generation
        │   ├── backtester.py          # Backtesting engine
        │   └── portfolio.py           # Portfolio management
        └── utils/
            ├── __init__.py
            ├── visualization.py       # Plotting utilities
            └── config.py              # Configuration management
```

### Зависимости (requirements.txt)

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
yfinance>=0.2.18
scipy>=1.11.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.14.0
```

### Быстрый старт

```bash
# Установка зависимостей
cd python && pip install -r requirements.txt

# Обучение β-VAE
python -m src.training.trainer \
    --method beta_vae \
    --beta 4.0 \
    --latent-dim 10 \
    --symbols BTCUSDT ETHUSDT \
    --epochs 100 \
    --batch-size 64

# Обучение β-TCVAE
python -m src.training.trainer \
    --method beta_tcvae \
    --beta 6.0 \
    --latent-dim 8 \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --epochs 200 \
    --annealing linear

# Анализ факторов
python -m src.analysis.traversals \
    --model-path checkpoints/best_model.pt \
    --data-path data/btcusdt_1h.csv \
    --output figures/

# Бэктестинг
python -m src.strategy.backtester \
    --model-path checkpoints/best_model.pt \
    --symbols BTCUSDT \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --capital 100000

# Визуализация результатов
python -m src.utils.visualization \
    --results-path results/backtest.csv \
    --output figures/
```

### Ключевые особенности реализации на Python

- **PyTorch**: Гибкая реализация с автоматическим дифференцированием
- **Модульная архитектура**: Каждый компонент может использоваться независимо
- **TensorBoard**: Логирование метрик обучения и визуализация латентного пространства
- **Конфигурация через YAML**: Воспроизводимость экспериментов
- **Jupyter-совместимость**: Все компоненты работают в интерактивном режиме

---

## 7. Лучшие практики

### 7.1 Когда использовать каждый метод

#### β-VAE

Используйте, когда:
- Нужен быстрый прототип с минимальной сложностью реализации
- Допустима потеря качества реконструкции ради более чистых факторов
- Латентное пространство невелико (5-10 измерений)
- Важна стабильность обучения

Не используйте, когда:
- Критична точность реконструкции рыночных данных
- Необходимо тонкое управление компромиссом разделение/реконструкция

#### FactorVAE

Используйте, когда:
- Требуется высокое качество разделения без потери реконструкции
- Имеются достаточные вычислительные ресурсы для обучения дискриминатора
- Работаете с многомерными данными (много активов, много признаков)

Не используйте, когда:
- Датасет мал (дискриминатор требует достаточно данных)
- Нестабильность GAN-обучения неприемлема для production

#### DIP-VAE

Используйте, когда:
- Необходим прямой контроль над ковариационной структурой
- Важна стабильность обучения
- Работаете с данными, где важны конкретные корреляционные свойства

Не используйте, когда:
- Зависимости между факторами нелинейны (DIP штрафует только линейные корреляции)

#### β-TCVAE

Используйте, когда:
- Нужен оптимальный баланс между всеми критериями
- Важна теоретическая обоснованность подхода
- Требуется декомпозиция потерь для диагностики

Не используйте, когда:
- Размер мини-батча мал (оценка TC через мини-батчи неточна при малых размерах)

### 7.2 Настройка гиперпараметров

#### Латентная размерность

- **Начинайте с 8-12 измерений** для финансовых данных
- Если несколько измерений коллапсируют (дисперсия ≈ 1, среднее ≈ 0), уменьшите размерность
- Если все измерения активны и интерпретируемость низкая, увеличьте размерность
- Используйте кривую KL по измерениям: сортируйте измерения по KL-дивергенции и ищите «локоть»

#### Параметр β (для β-VAE и β-TCVAE)

- **Начинайте с β = 4** для β-VAE
- **Начинайте с β = 6** для β-TCVAE
- Обучайте сетку моделей (β ∈ {1, 2, 4, 8, 16}) и сравнивайте:
  - Ошибку реконструкции
  - Среднюю абсолютную корреляцию между латентными измерениями
  - Прогнозную силу для будущих доходностей
- Выбирайте β в «локте» кривой компромисса

#### Параметр γ (FactorVAE)

- **Начинайте с γ = 10**
- Увеличивайте, если латентные измерения остаются коррелированными
- Уменьшайте, если обучение дискриминатора нестабильно

#### Параметры λ (DIP-VAE)

- **λ_od = 10, λ_d = 1** — стандартная отправная точка
- Увеличьте λ_od, если внедиагональные элементы ковариационной матрицы велики
- Увеличьте λ_d, если диагональные элементы сильно отклоняются от 1

#### Размер окна

- **1 час, окно 24**: захват суточных паттернов для криптовалют
- **1 день, окно 20**: захват месячных паттернов для акций
- **5 минут, окно 60**: захват внутридневных паттернов
- Размер окна должен быть достаточным для проявления всех интересующих паттернов

#### Стратегия отжига

- **Линейный отжиг с прогревом 20%** — универсальная рекомендация
- Прогрев позволяет кодировщику и декодеру инициализировать качественные веса до наложения давления разделения
- Для нестабильного обучения попробуйте циклический отжиг

### 7.3 Типичные ошибки

#### Ошибка 1: Коллапс апостериорного распределения

**Симптомы**: Несколько или все латентные измерения имеют KL ≈ 0, модель игнорирует латентный код.

**Причины**: Слишком высокое значение β, слишком мощный декодер, отсутствие прогрева.

**Решение**:
- Уменьшите β
- Используйте отжиг с прогревом
- Ослабьте декодер (уменьшите число слоёв или ширину)
- Используйте KL-free bits: гарантируйте минимальную информационную ёмкость каждому измерению

#### Ошибка 2: Нестабильность обучения FactorVAE

**Симптомы**: Осцилляции потерь, коллапс дискриминатора.

**Причины**: Дисбаланс скоростей обучения VAE и дискриминатора.

**Решение**:
- Уменьшите скорость обучения дискриминатора (обычно в 10 раз меньше, чем у VAE)
- Используйте gradient penalty для дискриминатора
- Обновляйте дискриминатор чаще, чем VAE (например, 5:1)

#### Ошибка 3: Ложная интерпретация факторов

**Симптомы**: Латентные обходы показывают чёткие паттерны, но факторы не имеют прогнозной силы.

**Причины**: Переобучение, разделение шума, а не сигнала.

**Решение**:
- Проверяйте стабильность факторов на out-of-sample данных
- Используйте cross-validation для оценки прогнозной силы
- Добавьте регуляризацию (dropout, weight decay)

#### Ошибка 4: Неправильная нормализация

**Симптомы**: Некоторые признаки доминируют в латентном пространстве, другие игнорируются.

**Причины**: Разный масштаб признаков (цена в тысячах, RSI от 0 до 100, объём в миллионах).

**Решение**:
- Используйте z-score нормализацию для каждого признака
- Вычисляйте статистики нормализации только на обучающих данных
- Используйте rolling нормализацию для сохранения темпоральной валидности

#### Ошибка 5: Утечка данных из будущего

**Симптомы**: Отличные результаты на бэктесте, плохие в реальной торговле.

**Причины**: Использование всего датасета для нормализации, обучение на тестовых данных.

**Решение**:
- Строго разделяйте данные на train/val/test по времени
- Используйте expanding window для нормализации
- Никогда не используйте тестовые данные для настройки гиперпараметров

---

## 8. Ресурсы

### Основные научные статьи

1. **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**
   - Авторы: Higgins et al.
   - Год: 2017
   - URL: https://openreview.net/forum?id=Sy2fzU9gl

2. **Disentangling by Factorising (FactorVAE)**
   - Авторы: Kim & Mnih
   - Год: 2018
   - URL: https://arxiv.org/abs/1802.05983

3. **Variational Inference of Disentangled Latent Concepts from Unlabeled Observations (DIP-VAE)**
   - Авторы: Kumar, Sattigeri, Balakrishnan
   - Год: 2018
   - URL: https://arxiv.org/abs/1711.00848

4. **Isolating Sources of Disentanglement in Variational Autoencoders (β-TCVAE)**
   - Авторы: Chen et al.
   - Год: 2018
   - URL: https://arxiv.org/abs/1802.04942

5. **Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations**
   - Авторы: Locatello et al.
   - Год: 2019
   - URL: https://arxiv.org/abs/1811.12359

6. **A Framework for the Quantitative Evaluation of Disentangled Representations**
   - Авторы: Eastwood & Williams
   - Год: 2018
   - URL: https://openreview.net/forum?id=By-7dz-AZ

### Применение в финансах

7. **Deep Generative Models for Financial Data**
   - Тематика: Обзор генеративных моделей для финансовых временных рядов

8. **Variational Autoencoders for Financial Factor Models**
   - Тематика: Использование VAE для обнаружения латентных факторов в финансах

9. **Disentangled Representations for Risk Factor Discovery**
   - Тематика: Применение разделённых представлений для обнаружения факторов риска

### Реализации и библиотеки

10. **disentanglement_lib** (Google Research)
    - URL: https://github.com/google-research/disentanglement_lib
    - Библиотека для обучения и оценки разделённых представлений

11. **Pythae** — Python библиотека для автокодировщиков
    - URL: https://github.com/clementchadebec/benchmark_VAE
    - Включает реализации β-VAE, FactorVAE, DIP-VAE, β-TCVAE

12. **PyTorch VAE**
    - URL: https://github.com/AntixK/PyTorch-VAE
    - Коллекция реализаций различных VAE на PyTorch

---

## 9. Связанные главы

- **[Глава 231: VAE факторная модель](../231_vae_factor_model/)** — Основы вариационных автокодировщиков для факторного моделирования. Рекомендуется изучить перед этой главой как введение в VAE и их применение к финансовым факторам.

- **[Глава 234: Торговля с Beta-VAE](../234_beta_vae_trading/)** — Углублённое изучение β-VAE, одного из методов, рассмотренных в этой главе. Содержит детальную реализацию на Rust и подробный анализ выбора параметра β.

- **[Глава 236: Условный VAE для трейдинга](../236_conditional_vae_trading/)** — Условные вариационные автокодировщики, позволяющие генерировать рыночные данные с условием на определённые факторы. Комплементарный подход к разделённым VAE.

- **[Глава 237: Иерархический VAE для трейдинга](../237_hierarchical_vae_trading/)** — Иерархические VAE с многоуровневой структурой латентного пространства. Может быть объединён с разделёнными методами для более выразительных представлений.

---

## 10. Уровень сложности

**Продвинутый**

Эта глава предполагает уверенное знание:
- Вариационных автокодировщиков (стандартный VAE, ELBO, трюк репараметризации)
- Глубокого обучения (PyTorch, обучение нейронных сетей, регуляризация)
- Теории информации (KL-дивергенция, взаимная информация, энтропия)
- Финансовых рынков (OHLCV данные, технические индикаторы, бэктестинг)
- Линейной алгебры (ковариационные матрицы, собственные значения)

Рекомендуемый порядок изучения:
1. Глава 231 (VAE факторная модель) — основы
2. Глава 234 (Beta-VAE) — углублённое изучение β-VAE
3. **Глава 238 (эта глава)** — полный обзор методов разделения
4. Глава 236 (Условный VAE) — расширение через условную генерацию
5. Глава 237 (Иерархический VAE) — расширение через иерархию
