# Глава 39: Конформное предсказание — Торговля с калиброванной неопределённостью

## Обзор

Конформное предсказание (Conformal Prediction, CP) — это мощный фреймворк для количественной оценки неопределённости, предоставляющий **калиброванные интервалы предсказания с гарантированным покрытием**. В отличие от стандартных моделей машинного обучения, которые часто дают чрезмерно самоуверенные прогнозы, конформное предсказание обеспечивает честные оценки неопределённости при минимальных предположениях.

Ключевая идея для торговли: **торгуем только когда модель уверена, и размер позиции обратно пропорционален неопределённости**. Этот подход естественным образом избегает сделок в периоды высокой неопределённости (рыночный стресс, смена режимов) и концентрирует капитал, когда прогнозы надёжны.

### Почему конформное предсказание для торговли?

1. **Гарантированное покрытие**: Если вы целитесь на 90% покрытие, примерно 90% ваших интервалов предсказания будут содержать истинное значение
2. **Свобода от распределения**: Работает с любой базовой моделью без параметрических предположений
3. **Валидность на конечной выборке**: Гарантии выполняются для любого размера выборки, а не только асимптотически
4. **Адаптивность**: Интервалы естественно расширяются в волатильные периоды и сужаются в стабильные
5. **Модель-агностичность**: Обёртывает любую ML-модель (нейросети, градиентный бустинг и др.) конформным предсказанием

## Содержание

1. [Теоретические основы](#теоретические-основы)
    * [Взаимозаменяемость и гарантии покрытия](#взаимозаменяемость-и-гарантии-покрытия)
    * [Баллы неконформности](#баллы-неконформности)
2. [Методы конформного предсказания](#методы-конформного-предсказания)
    * [Раздельное конформное предсказание](#раздельное-конформное-предсказание)
    * [Конформизированная квантильная регрессия (CQR)](#конформизированная-квантильная-регрессия-cqr)
    * [Адаптивный конформный вывод для временных рядов](#адаптивный-конформный-вывод-для-временных-рядов)
3. [Проектирование торговой стратегии](#проектирование-торговой-стратегии)
    * [Генерация сигналов с неопределённостью](#генерация-сигналов-с-неопределённостью)
    * [Определение размера позиции с калиброванной уверенностью](#определение-размера-позиции-с-калиброванной-уверенностью)
    * [Критерий Келли с интервалами предсказания](#критерий-келли-с-интервалами-предсказания)
4. [Реализация](#реализация)
    * [Примеры кода](#примеры-кода)
    * [Ноутбуки](#ноутбуки)
5. [Бэктестинг и оценка](#бэктестинг-и-оценка)
6. [Ресурсы и литература](#ресурсы-и-литература)

---

## Теоретические основы

### Взаимозаменяемость и гарантии покрытия

Конформное предсказание опирается на предположение о **взаимозаменяемости** (exchangeability): совместное распределение точек данных инвариантно к перестановкам. Это слабее, чем предположение i.i.d., и допускает зависимые данные при определённых условиях.

**Ключевая теорема (Vovk et al., 2005)**: Для взаимозаменяемых данных $(X_1, Y_1), \ldots, (X_n, Y_n), (X_{n+1}, Y_{n+1})$ множество конформного предсказания $C(X_{n+1})$, построенное на уровне $1-\alpha$, удовлетворяет:

$$P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$$

Эта гарантия является **маргинальной** (усреднённой по всем тестовым точкам) и выполняется точно на конечных выборках.

### Баллы неконформности

Ядро конформного предсказания — **балл неконформности** — функция, измеряющая насколько «необычна» точка данных относительно других. Распространённые варианты:

- **Абсолютный остаток**: $s(x, y) = |y - \hat{f}(x)|$
- **Нормализованный остаток**: $s(x, y) = \frac{|y - \hat{f}(x)|}{\hat{\sigma}(x)}$
- **На основе квантилей**: $s(x, y) = \max(\hat{q}_{\alpha/2}(x) - y, y - \hat{q}_{1-\alpha/2}(x))$

Выбор функции баллов влияет на форму и адаптивность интервалов предсказания.

---

## Методы конформного предсказания

### Раздельное конформное предсказание

Простейший и наиболее практичный метод:

1. **Разделение** данных на обучающую и калибровочную выборки
2. **Обучение** базовой модели на обучающей выборке
3. **Калибровка** путём вычисления баллов неконформности на калибровочной выборке
4. **Предсказание** путём нахождения $1-\alpha$ квантиля калибровочных баллов

```python
import numpy as np
from sklearn.model_selection import train_test_split

class SplitConformalPredictor:
    """
    Раздельное конформное предсказание для регрессии с гарантированным покрытием.

    Гарантия покрытия: P(Y ∈ [lower, upper]) ≥ 1 - alpha
    """
    def __init__(self, model, alpha=0.1):
        self.model = model
        self.alpha = alpha  # Уровень непокрытия (1 - alpha = покрытие)
        self.calibration_scores = None
        self.q_hat = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Обучение модели и калибровка на отложенной выборке.

        Параметры:
        ----------
        X_train : array-like, Обучающие признаки
        y_train : array-like, Обучающие цели
        X_calib : array-like, Калибровочные признаки (отложенные от обучения)
        y_calib : array-like, Калибровочные цели
        """
        # Шаг 1: Обучение базовой модели
        self.model.fit(X_train, y_train)

        # Шаг 2: Получение предсказаний на калибровочной выборке
        y_pred_calib = self.model.predict(X_calib)

        # Шаг 3: Вычисление баллов неконформности (абсолютные остатки)
        self.calibration_scores = np.abs(y_calib - y_pred_calib)

        # Шаг 4: Вычисление квантиля для интервалов предсказания
        # Квантиль (1-alpha)(1 + 1/n) обеспечивает покрытие на конечной выборке
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Ограничение сверху
        self.q_hat = np.quantile(self.calibration_scores, q_level)

        return self

    def predict(self, X):
        """
        Возвращает точечное предсказание и интервал предсказания.

        Возвращает:
        -----------
        dict с ключами: 'prediction', 'lower', 'upper', 'interval_width'
        """
        y_pred = self.model.predict(X)

        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat

        return {
            'prediction': y_pred,
            'lower': lower,
            'upper': upper,
            'interval_width': np.full_like(y_pred, 2 * self.q_hat)
        }

    def coverage(self, X_test, y_test):
        """Вычисление эмпирического покрытия на тестовой выборке."""
        pred = self.predict(X_test)
        covered = (y_test >= pred['lower']) & (y_test <= pred['upper'])
        return covered.mean()
```

### Конформизированная квантильная регрессия (CQR)

CQR создаёт **адаптивные интервалы**, ширина которых варьируется в зависимости от входных признаков. Это критически важно для финансовых данных, где неопределённость существенно различается в разных рыночных режимах.

```python
from sklearn.ensemble import GradientBoostingRegressor

class ConformizedQuantileRegression:
    """
    CQR: Конформизированная квантильная регрессия

    Создаёт гетероскедастические интервалы, адаптирующиеся к локальной неопределённости.
    Более информативна, чем раздельное конформное предсказание для финансовых данных.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        # Обучение квантильных моделей для нижней и верхней границ
        self.lower_model = GradientBoostingRegressor(
            loss='quantile', alpha=alpha/2, n_estimators=100
        )
        self.upper_model = GradientBoostingRegressor(
            loss='quantile', alpha=1-alpha/2, n_estimators=100
        )
        self.q_hat = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        """Обучение квантильных моделей и калибровка."""
        # Обучение квантильных моделей
        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)

        # Получение начальных интервалов на калибровочной выборке
        lower_calib = self.lower_model.predict(X_calib)
        upper_calib = self.upper_model.predict(X_calib)

        # Вычисление баллов конформности
        # Балл = насколько интервал должен расшириться для покрытия истинного значения
        scores = np.maximum(
            lower_calib - y_calib,  # Нижняя граница слишком высока
            y_calib - upper_calib   # Верхняя граница слишком низка
        )

        # Квантиль для гарантированного покрытия
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = np.quantile(scores, q_level)

        return self

    def predict(self, X):
        """Возвращает адаптивные интервалы предсказания."""
        lower = self.lower_model.predict(X) - self.q_hat
        upper = self.upper_model.predict(X) + self.q_hat

        return {
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower,
            'midpoint': (upper + lower) / 2
        }
```

### Адаптивный конформный вывод для временных рядов

Стандартное конформное предсказание предполагает взаимозаменяемость, которая нарушается во временных рядах. **Адаптивный конформный вывод (ACI)** решает эту проблему путём динамической корректировки уровня покрытия на основе недавней производительности.

```python
class AdaptiveConformalPredictor:
    """
    Адаптивный конформный вывод (ACI) для временных рядов.

    Динамически корректирует уровень покрытия на основе недавних ошибок,
    поддерживая приблизительное покрытие при сдвиге распределения.

    Ссылка: Gibbs & Candès (2021) "Adaptive Conformal Inference
            Under Distribution Shift"
    """
    def __init__(self, model, target_coverage=0.9, gamma=0.05):
        self.model = model
        self.target_coverage = target_coverage
        self.gamma = gamma  # Скорость обучения для адаптации
        self.alpha_t = 1 - target_coverage  # Текущий уровень непокрытия
        self.history = []  # Отслеживание покрытия во времени

    def update(self, y_true, lower, upper):
        """
        Обновление alpha на основе того, было ли y_true покрыто.

        Реализует онлайн-обучение уровня покрытия:
        - Если покрытие больше целевого: уменьшаем интервалы (снижаем alpha)
        - Если покрытие меньше целевого: увеличиваем интервалы (повышаем alpha)
        """
        covered = (lower <= y_true) and (y_true <= upper)
        self.history.append(covered)

        # Градиентное обновление: двигаем alpha к достижению целевого покрытия
        if covered:
            # Покрыто -> можем позволить более узкие интервалы
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha_t - 0)
        else:
            # Не покрыто -> нужны более широкие интервалы
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha_t - 1)

        # Ограничение допустимого диапазона
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.5)

        return covered

    def predict(self, X, calibration_scores):
        """Генерация предсказания с адаптивным интервалом."""
        y_pred = self.model.predict(X)

        # Вычисление ширины интервала на основе текущего alpha
        q_level = 1 - self.alpha_t
        q_hat = np.quantile(calibration_scores, min(q_level, 1.0))

        return {
            'prediction': y_pred,
            'lower': y_pred - q_hat,
            'upper': y_pred + q_hat,
            'interval_width': 2 * q_hat,
            'current_alpha': self.alpha_t,
            'recent_coverage': np.mean(self.history[-100:]) if self.history else None
        }
```

---

## Проектирование торговой стратегии

### Генерация сигналов с неопределённостью

Основная идея: **торгуем только когда интервал предсказания узкий (высокая уверенность) И направление ясно**.

```python
class ConformalTradingStrategy:
    """
    Торговая стратегия с использованием калиброванных интервалов предсказания.

    Ключевые принципы:
    1. Торгуем только когда интервал узкий (высокая уверенность)
    2. Направление должно быть ясным (интервал не пересекает ноль или далеко от него)
    3. Размер позиции обратно пропорционален неопределённости
    """
    def __init__(self, predictor, width_threshold=0.02, min_edge=0.005):
        """
        Параметры:
        ----------
        predictor : Конформный предсказатель с методом predict()
        width_threshold : Максимальная ширина интервала для сделки (напр., 2% для доходностей)
        min_edge : Минимальное ожидаемое преимущество для торговли (напр., 0.5% ожидаемая доходность)
        """
        self.predictor = predictor
        self.width_threshold = width_threshold
        self.min_edge = min_edge

    def generate_signal(self, X):
        """
        Генерация торгового сигнала на основе интервала предсказания.

        Возвращает:
        -----------
        dict с: prediction, interval_width, confidence, trade, direction, size
        """
        pred = self.predictor.predict(X)

        # Обработка скалярных и массивных входов
        if hasattr(pred['interval_width'], '__len__'):
            interval_width = pred['interval_width'][0]
            lower = pred['lower'][0]
            upper = pred['upper'][0]
            midpoint = pred.get('midpoint', (lower + upper) / 2)
            if hasattr(midpoint, '__len__'):
                midpoint = midpoint[0]
        else:
            interval_width = pred['interval_width']
            lower = pred['lower']
            upper = pred['upper']
            midpoint = pred.get('midpoint', (lower + upper) / 2)

        signal = {
            'prediction': midpoint,
            'interval_width': interval_width,
            'lower': lower,
            'upper': upper,
            'confidence': 1 / (1 + interval_width * 10),  # Преобразование в шкалу 0-1
            'trade': False,
            'direction': 0,
            'size': 0.0
        }

        # Условие 1: Интервал должен быть достаточно узким
        if interval_width >= self.width_threshold:
            signal['skip_reason'] = 'interval_too_wide'
            return signal

        # Условие 2: Направление должно быть ясным с достаточным преимуществом
        if lower > self.min_edge:
            # Весь интервал положительный с достаточной величиной
            signal['direction'] = 1  # Лонг
            signal['trade'] = True
            signal['edge'] = lower  # Худший случай ожидаемой доходности
        elif upper < -self.min_edge:
            # Весь интервал отрицательный с достаточной величиной
            signal['direction'] = -1  # Шорт
            signal['trade'] = True
            signal['edge'] = -upper  # Худший случай ожидаемой доходности
        else:
            signal['skip_reason'] = 'unclear_direction'
            return signal

        # Размер позиции обратно пропорционален ширине интервала
        # Уже интервал -> выше уверенность -> больше позиция
        signal['size'] = self._compute_size(interval_width, signal['edge'])

        return signal

    def _compute_size(self, interval_width, edge):
        """
        Вычисление размера позиции на основе неопределённости и преимущества.

        Использует упрощённый подход: size = edge / interval_width
        Ограничен 1.0 (100% капитала)
        """
        if interval_width <= 0:
            return 0.0

        # Размер пропорционален отношению преимущества к неопределённости
        raw_size = edge / interval_width

        # Применение ограничений
        size = min(raw_size, 1.0)
        size = max(size, 0.0)

        return size
```

### Определение размера позиции с калиброванной уверенностью

```python
def kelly_with_conformal(prediction, lower, upper, risk_free_rate=0):
    """
    Критерий Келли, адаптированный для интервалов конформного предсказания.

    Ключевая идея: Ширина интервала предоставляет калиброванную оценку
    неопределённости, которую можно использовать для корректировки доли Келли.

    Параметры:
    ----------
    prediction : Точечное предсказание (ожидаемая доходность)
    lower : Нижняя граница интервала предсказания
    upper : Верхняя граница интервала предсказания
    risk_free_rate : Безрисковая ставка для расчёта избыточной доходности

    Возвращает:
    -----------
    kelly_fraction : Рекомендуемый размер позиции как доля капитала
    """
    interval_width = upper - lower
    expected_excess = prediction - risk_free_rate

    # Граничный случай: нет ожидаемого преимущества
    if expected_excess <= 0:
        return 0.0

    # Граничный случай: вырожденный интервал
    if interval_width <= 0:
        return 0.0

    # Ширина интервала служит прокси для волатильности
    # Доля Келли = ожидаемая_доходность / дисперсия
    # Используем interval_width как прокси для стандартного отклонения
    implied_variance = (interval_width / 2) ** 2  # Полуширина как прокси std

    kelly_fraction = expected_excess / implied_variance

    # Применяем половину Келли для безопасности (распространённая практика)
    kelly_fraction = kelly_fraction / 2

    # Ограничение разумными уровнями
    kelly_fraction = min(kelly_fraction, 2.0)  # Макс 200% (с плечом)
    kelly_fraction = max(kelly_fraction, -2.0)  # Макс -200% (шорт)

    return kelly_fraction


class ConfidenceBasedSizing:
    """
    Определение размера позиции на основе уверенности интервала предсказания.

    Отображает ширину интервала на размер позиции различными способами.
    """
    def __init__(self, method='inverse', max_size=1.0, min_size=0.0):
        self.method = method
        self.max_size = max_size
        self.min_size = min_size

    def compute_size(self, interval_width, baseline_width=None):
        """
        Вычисление размера позиции на основе ширины интервала.

        Параметры:
        ----------
        interval_width : Текущая ширина интервала предсказания
        baseline_width : Эталонная ширина для нормализации
        """
        if baseline_width is None:
            baseline_width = interval_width

        if self.method == 'inverse':
            # Размер = базовая / текущая (уже = больше)
            size = baseline_width / max(interval_width, 1e-6)

        elif self.method == 'linear':
            # Линейное уменьшение: size = 1 - width/baseline
            size = 1 - interval_width / max(baseline_width, 1e-6)

        elif self.method == 'exponential':
            # Экспоненциальное затухание на основе ширины
            size = np.exp(-interval_width / baseline_width)

        elif self.method == 'threshold':
            # Бинарный: полный размер если ниже порога, ноль иначе
            size = self.max_size if interval_width < baseline_width else 0

        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

        # Ограничение границами
        return np.clip(size, self.min_size, self.max_size)
```

---

## Реализация

### Ноутбуки

| # | Ноутбук | Описание |
|---|---------|----------|
| 1 | `01_conformal_theory.ipynb` | Теория: взаимозаменяемость, гарантии покрытия, баллы неконформности |
| 2 | `02_split_conformal.ipynb` | Реализация и анализ раздельного конформного предсказания |
| 3 | `03_conformalized_quantile.ipynb` | Конформизированная квантильная регрессия для адаптивных интервалов |
| 4 | `04_adaptive_conformal.ipynb` | Адаптивный конформный вывод для временных рядов |
| 5 | `05_financial_application.ipynb` | Применение к прогнозированию доходностей на реальных рыночных данных |
| 6 | `06_interval_analysis.ipynb` | Анализ паттернов ширины интервалов и рыночных режимов |
| 7 | `07_trading_rules.ipynb` | Торговые правила на основе интервалов предсказания |
| 8 | `08_position_sizing.ipynb` | Определение размера позиции по Келли с калиброванной неопределённостью |
| 9 | `09_backtesting.ipynb` | Полный бэктест стратегии конформного предсказания |
| 10 | `10_comparison.ipynb` | Сравнение со стандартным ML без оценки неопределённости |

### Примеры кода

Смотрите директорию `rust_examples/` для готовых к продакшену реализаций на Rust:

- **Клиент API Bybit** для криптовалютных данных в реальном времени
- **Модульные алгоритмы конформного предсказания** (Split CP, CQR, ACI)
- **Фреймворк торговой стратегии** с сигналами на основе интервалов
- **Движок бэктестинга** с правильной обработкой временных рядов

---

## Бэктестинг и оценка

### Ключевые метрики

**Метрики покрытия:**
- **Эмпирическое покрытие**: Доля истинных значений внутри интервалов предсказания
- **Условное покрытие**: Покрытие, стратифицированное по ширине интервала, режиму волатильности и т.д.
- **Стабильность покрытия**: Насколько согласованно покрытие во времени?

**Качество интервалов:**
- **Средняя ширина**: Средняя ширина интервала предсказания
- **Вариабельность ширины**: Стандартное отклонение ширины интервалов
- **Резкость (Sharpness)**: Обратная средней ширине (уже = резче)
- **Балл Винклера**: Комбинированная мера покрытия и резкости

**Торговая результативность:**
- **Коэффициент Шарпа**: Доходность с поправкой на риск
- **Процент выигрышей**: Доля прибыльных сделок
- **Средняя сделка**: Средняя доходность на сделку
- **Селективность торговли**: Доля периодов с торговым сигналом

```python
def evaluate_conformal_strategy(results_df):
    """
    Комплексная оценка стратегии конформного предсказания.

    Параметры:
    ----------
    results_df : DataFrame с колонками:
        - prediction, lower, upper, actual, direction, size, pnl
    """
    metrics = {}

    # Метрики покрытия
    covered = (results_df['actual'] >= results_df['lower']) & \
              (results_df['actual'] <= results_df['upper'])
    metrics['coverage'] = covered.mean()

    # Метрики интервалов
    widths = results_df['upper'] - results_df['lower']
    metrics['avg_width'] = widths.mean()
    metrics['width_std'] = widths.std()
    metrics['sharpness'] = 1 / widths.mean()

    # Торговые метрики (только для фактических сделок)
    trades = results_df[results_df['direction'] != 0]
    if len(trades) > 0:
        metrics['n_trades'] = len(trades)
        metrics['trade_frequency'] = len(trades) / len(results_df)
        metrics['avg_pnl'] = trades['pnl'].mean()
        metrics['sharpe'] = trades['pnl'].mean() / trades['pnl'].std() * np.sqrt(252)
        metrics['win_rate'] = (trades['pnl'] > 0).mean()
        metrics['total_return'] = trades['pnl'].sum()

        # Условное покрытие для сделок
        trades_covered = covered[results_df['direction'] != 0]
        metrics['coverage_on_trades'] = trades_covered.mean()

    return metrics
```

---

## Ресурсы и литература

### Академические статьи

- **Vovk, Gammerman, Shafer (2005)**: "Algorithmic Learning in a Random World" — Основополагающий учебник по конформному предсказанию
- **Romano, Patterson, Candès (2019)**: "Conformalized Quantile Regression" — CQR для адаптивных интервалов
- **Gibbs & Candès (2021)**: "Adaptive Conformal Inference Under Distribution Shift" — ACI для невзаимозаменяемых данных
- **Barber et al. (2022)**: "Conformal Prediction Beyond Exchangeability" — Расширения для зависимых данных

### Программные библиотеки

- [MAPIE](https://mapie.readthedocs.io/): Комплексная библиотека конформного предсказания для Python
- [Crepes](https://github.com/henrikbostrom/crepes): Конформные регрессоры и предиктивные системы
- [ConformalPrediction.jl](https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl): Реализация на Julia

### Туториалы и курсы

- [A Tutorial on Conformal Prediction](https://www.jmlr.org/papers/v9/shafer08a.html) — Shafer & Vovk, JMLR 2008
- [Conformal Prediction in 2020](https://arxiv.org/abs/2107.07511) — Недавний обзорный обзор

---

## Уровень сложности

**Средний** (3/5)

### Предварительные требования

- Статистический вывод и проверка гипотез
- Интервалы предсказания vs. доверительные интервалы
- Основы квантильной регрессии
- Основы анализа временных рядов
- Управление рисками и определение размера позиции

### Результаты обучения

После завершения этой главы вы сможете:

1. Реализовать конформное предсказание для прогнозирования доходностей
2. Строить интервалы предсказания с гарантированным покрытием
3. Проектировать торговые стратегии, использующие оценку неопределённости
4. Определять размер позиции на основе калиброванной уверенности
5. Оценивать стратегии с помощью метрик покрытия и результативности
6. Адаптировать методы конформного предсказания для нестационарных финансовых временных рядов
