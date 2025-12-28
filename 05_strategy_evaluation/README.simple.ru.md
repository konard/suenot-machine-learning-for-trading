# Оценка торговых стратегий - Просто и понятно

## Главный вопрос: Как понять, хороша ли моя стратегия?

Представь, что ты создал торгового робота. Как узнать, хорош ли он, **ДО** того, как рискнуть реальными деньгами?

### Простая аналогия

**Тестирование стратегии = Тест-драйв автомобиля перед покупкой**

- **Бэктестинг** = Проехать по знакомой дороге (исторические данные)
- **Форвард-тестинг** = Проехать по новому маршруту (свежие данные)
- **Живая торговля** = Купить и ездить каждый день (реальные деньги!)

## Две главные цели: Доходность и Риск

### 1. Доходность - Сколько заработал?

**Простое объяснение:** Прибыль в процентах.

```python
# Начальный капитал
initial_capital = 10000  # $10,000

# Конечный капитал через год
final_capital = 12000  # $12,000

# Доходность
returns = (final_capital - initial_capital) / initial_capital
print(f"Доходность: {returns:.1%}")  # Доходность: 20.0%
```

### 2. Риск - Насколько сильно "трясло" в процессе?

**Аналогия:** Две дороги в один пункт назначения:
- **Дорога A:** Ровная, спокойная → Низкий риск
- **Дорога B:** Американские горки, стресс → Высокий риск

```python
import numpy as np

# Стратегия A: Стабильная (+1%, +1%, +1%, +1%)
strategy_a = [0.01, 0.01, 0.01, 0.01]

# Стратегия B: Волатильная (+10%, -5%, +8%, -3%)
strategy_b = [0.10, -0.05, 0.08, -0.03]

# Обе заработали 10% итого, но риск разный!
print(f"A доходность: {sum(strategy_a):.1%}")  # 4%
print(f"B доходность: {sum(strategy_b):.1%}")  # 10%

# Волатильность (риск) = стандартное отклонение
risk_a = np.std(strategy_a)
risk_b = np.std(strategy_b)

print(f"\nРиск A: {risk_a:.4f}")  # 0.0000 (нет колебаний)
print(f"Риск B: {risk_b:.4f}")  # 0.0615 (сильные колебания!)
```

**Вывод:** Стратегия B заработала больше, но с ОГРОМНЫМ стрессом!

## Коэффициент Шарпа - Король всех метрик

### Что это?

**Формула:**
```
Коэффициент Шарпа = (Доходность - Безрисковая ставка) / Риск
```

**Простыми словами:** Сколько прибыли ты получаешь за каждую единицу риска?

### Аналогия

**Коэффициент Шарпа = Очки за мастерство в игре**

Представь два способа заработать $100:
- **Способ A:** Пройти 1 км пешком (легко, низкий риск)
- **Способ B:** Прыгнуть с парашютом (опасно, высокий риск)

Какой способ лучше? **Способ A!** Меньше риска для той же награды.

### Расчет Sharpe Ratio

```python
import numpy as np

# Данные стратегии (месячная доходность)
strategy_returns = np.array([0.02, 0.03, -0.01, 0.04, 0.02, 0.01,
                              0.03, 0.02, -0.02, 0.05, 0.01, 0.02])

# Безрисковая ставка (например, государственные облигации = 2% годовых = 0.17% в месяц)
risk_free_rate = 0.02 / 12  # 0.0017

# 1. Средняя доходность
mean_return = strategy_returns.mean()

# 2. Риск (волатильность)
volatility = strategy_returns.std()

# 3. Sharpe Ratio
sharpe = (mean_return - risk_free_rate) / volatility

print(f"Средняя доходность: {mean_return:.2%}")
print(f"Волатильность: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

# Результат:
# Средняя доходность: 1.83%
# Волатильность: 1.97%
# Sharpe Ratio: 0.84
```

### Как интерпретировать Sharpe Ratio?

| Sharpe Ratio | Оценка | Пример |
|--------------|--------|--------|
| < 0 | Ужасно | Лучше держать деньги в банке |
| 0 - 0.5 | Плохо | Слишком много риска для такой прибыли |
| 0.5 - 1.0 | Приемлемо | Нормальная стратегия |
| 1.0 - 2.0 | Хорошо | Профессиональный уровень |
| 2.0 - 3.0 | Отлично | Топ-фонды |
| > 3.0 | Невероятно | Warren Buffett, Renaissance Tech |

**Реальные примеры:**
- **S&P 500:** Sharpe ≈ 0.5 (долгосрочно)
- **Warren Buffett:** Sharpe ≈ 0.76
- **Renaissance Medallion Fund:** Sharpe ≈ 2.5 - 3.0 (легенда!)

### Проблема: Автокорреляция

Финансовые данные не случайны! Есть паттерны:
- **Momentum:** Если вчера рост → сегодня тоже рост
- **Mean Reversion:** Если вчера большой рост → сегодня возврат

**Эндрю Ло** создал скорректированный Sharpe Ratio для таких случаев.

```python
# Простая корректировка для автокорреляции
def adjusted_sharpe_ratio(returns, risk_free_rate=0):
    mean_return = returns.mean()
    std = returns.std()

    # Проверить автокорреляцию
    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

    # Корректировка
    q = len(returns)
    adjustment = np.sqrt(1 + 2 * autocorr)

    sharpe = (mean_return - risk_free_rate) / std
    adjusted_sharpe = sharpe / adjustment

    return adjusted_sharpe

adjusted_sr = adjusted_sharpe_ratio(strategy_returns)
print(f"Скорректированный Sharpe: {adjusted_sr:.2f}")
```

## Фундаментальный закон активного управления

### Загадка: Buffett vs Renaissance Technologies

**Warren Buffett (Berkshire Hathaway):**
- Держит: 100-150 акций
- Период владения: Годы
- Сделок в день: ~0

**Renaissance Technologies (Medallion Fund):**
- Держит: Тысячи позиций
- Период владения: Минуты-часы
- Сделок в день: 100,000+

**Оба показывают ~30-40% годовых!** Как такое возможно?

### Формула фундаментального закона

```
IR = IC × √Breadth

где:
IR = Information Ratio (похож на Sharpe)
IC = Information Coefficient (насколько хороши прогнозы)
Breadth = Количество независимых ставок
```

**Простое объяснение:**

```
Прибыль = Мастерство × √Количество_попыток
```

### Аналогия: Баскетбол

**Игрок A (Buffett):**
- Бросает: 10 раз за игру
- Точность: 90% попадания
- Очки = 0.90 × √10 ≈ 2.8

**Игрок B (RenTec):**
- Бросает: 100 раз за игру
- Точность: 55% попадания
- Очки = 0.55 × √100 = 5.5

**Игрок B выигрывает!** Меньше мастерство, но МНОГО больше попыток!

### Практический пример

```python
import numpy as np

def calculate_IR(IC, breadth):
    """
    IC = корреляция между прогнозами и реальностью
    breadth = количество независимых ставок в год
    """
    return IC * np.sqrt(breadth)

# Стратегия 1: Long-term value investing (как Buffett)
ic_buffett = 0.15  # 15% корреляция (очень хорошо для долгосрочного!)
breadth_buffett = 10  # 10 независимых решений в год

IR_buffett = calculate_IR(ic_buffett, breadth_buffett)
print(f"Buffett IR: {IR_buffett:.3f}")

# Стратегия 2: High-frequency trading (как RenTec)
ic_rentech = 0.02  # 2% корреляция (низко, но есть!)
breadth_rentech = 10000  # 10,000 сделок в год

IR_rentech = calculate_IR(ic_rentech, breadth_rentech)
print(f"RenTec IR: {IR_rentech:.3f}")

# Результат:
# Buffett IR: 0.474
# RenTec IR: 2.000

# RenTec побеждает благодаря огромному breadth!
```

### Главный вывод

**Два пути к успеху:**
1. **Путь Buffett:** Будь ОЧЕНЬ точным (высокий IC), делай мало ставок
2. **Путь RenTec:** Будь чуть-чуть точным (низкий IC), делай ТЫСЯЧИ ставок

## Управление портфелем: Теория Марковица

### Идея диверсификации

**Аналогия:** Не клади все яйца в одну корзину

```python
import numpy as np
import matplotlib.pyplot as plt

# Акция A: Стабильная (коммунальные услуги)
returns_A = np.random.normal(0.08, 0.10, 1000)  # 8% ± 10%

# Акция B: Волатильная (технологии)
returns_B = np.random.normal(0.12, 0.30, 1000)  # 12% ± 30%

# Портфель 50/50
portfolio_returns = 0.5 * returns_A + 0.5 * returns_B

print("СРАВНЕНИЕ РИСКОВ:")
print(f"Только A - Риск: {returns_A.std():.2%}")
print(f"Только B - Риск: {returns_B.std():.2%}")
print(f"50/50 - Риск: {portfolio_returns.std():.2%}")

# Результат:
# Только A - Риск: 10.05%
# Только B - Риск: 29.87%
# 50/50 - Риск: 17.23%  <- МЕНЬШЕ, чем среднее (20%)!
```

**Магия:** Риск портфеля (17%) меньше, чем средний риск (20%)! Это и есть диверсификация!

### Эффективная граница (Efficient Frontier)

**Что это:** График всех "умных" портфелей, где ты получаешь максимальную доходность для данного риска.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Исторические данные (годовая доходность)
returns = np.array([0.10, 0.12, 0.08, 0.15])  # 4 акции
cov_matrix = np.array([
    [0.10, 0.02, 0.01, 0.03],
    [0.02, 0.15, 0.02, 0.04],
    [0.01, 0.02, 0.08, 0.01],
    [0.03, 0.04, 0.01, 0.20]
])

def portfolio_stats(weights, returns, cov_matrix):
    """Вычислить доходность и риск портфеля"""
    portfolio_return = np.dot(weights, returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std

def minimize_volatility(weights):
    """Минимизировать риск"""
    return portfolio_stats(weights, returns, cov_matrix)[1]

# Ограничения: веса от 0 до 1, сумма = 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(len(returns)))

# Найти минимальный риск
initial_guess = [0.25] * 4
result = minimize(minimize_volatility, initial_guess,
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
opt_return, opt_risk = portfolio_stats(optimal_weights, returns, cov_matrix)

print("ОПТИМАЛЬНЫЙ ПОРТФЕЛЬ:")
for i, weight in enumerate(optimal_weights):
    print(f"Акция {i+1}: {weight:.1%}")

print(f"\nДоходность: {opt_return:.2%}")
print(f"Риск: {opt_risk:.2%}")

# Результат:
# ОПТИМАЛЬНЫЙ ПОРТФЕЛЬ:
# Акция 1: 28.5%
# Акция 2: 15.3%
# Акция 3: 52.7%
# Акция 4: 3.5%
#
# Доходность: 9.82%
# Риск: 8.23%
```

### Визуализация эффективной границы

```python
# Создать 1000 случайных портфелей
num_portfolios = 1000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    # Случайные веса
    weights = np.random.random(4)
    weights /= np.sum(weights)

    # Вычислить статистику
    portfolio_return, portfolio_std = portfolio_stats(weights, returns, cov_matrix)
    sharpe = portfolio_return / portfolio_std

    results[0, i] = portfolio_std  # Риск
    results[1, i] = portfolio_return  # Доходность
    results[2, i] = sharpe  # Sharpe Ratio

# График
plt.figure(figsize=(10, 6))
plt.scatter(results[0, :], results[1, :], c=results[2, :],
            cmap='viridis', marker='o', alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Риск (волатильность)')
plt.ylabel('Доходность')
plt.title('Эффективная граница')

# Отметить оптимальный портфель
plt.scatter(opt_risk, opt_return, c='red', marker='*', s=500,
            label='Оптимальный портфель')
plt.legend()
plt.grid()
plt.show()
```

## Альтернативные стратегии

### 1. Портфель 1/N (Равновзвешенный)

**Идея:** Просто раздели деньги поровну!

```python
# У меня 4 акции → 25% в каждую
weights = [0.25, 0.25, 0.25, 0.25]
```

**Почему это работает:**
- Не нужно прогнозировать доходность (сложно!)
- Избегает переобучения
- Исследования показывают: часто лучше сложных моделей!

### 2. Минимальная дисперсия (Min-Variance)

**Идея:** Игнорируй доходность, минимизируй только риск.

```python
# Найти портфель с наименьшим риском (уже показали выше)
# Подходит для консервативных инвесторов
```

### 3. Black-Litterman модель

**Идея:** Начни с рыночных весов, потом корректируй по своим прогнозам.

**Аналогия:** Мудрость толпы + твое мнение

```python
# Рыночные веса (market cap)
market_weights = [0.40, 0.30, 0.20, 0.10]  # Apple, Microsoft, Google, Amazon

# Твое мнение: "Я думаю, Apple вырастет на 5% больше рынка"
views = {0: 0.05}  # Акция 0 (+5%)

# Black-Litterman комбинирует оба
# (реализация сложна, но идея проста)
```

### 4. Правило Келли - Размер ставки

**История:** Джон Келли работал в Bell Labs с Клодом Шенноном (отец теории информации). Он решил задачу: "Сколько ставить на каждую игру в казино, чтобы максимизировать богатство?"

**Формула Келли:**
```
f* = (p × b - q) / b

где:
f* = доля капитала для ставки
p = вероятность выигрыша
q = вероятность проигрыша (1 - p)
b = коэффициент выигрыша
```

### Пример: Подбрасывание монеты с преимуществом

```python
# Нечестная монета
p = 0.55  # 55% шанс выигрыша (орел)
q = 0.45  # 45% шанс проигрыша (решка)
b = 1  # 1:1 выплата (ставишь $1, выигрываешь $1)

# Формула Келли
kelly_fraction = (p * b - q) / b
print(f"Kelly говорит: ставь {kelly_fraction:.1%} капитала")

# Результат: ставь 10.0% капитала

# Симуляция
capital = 1000
bets = 100

for i in range(bets):
    bet_amount = capital * kelly_fraction
    coin_flip = np.random.random() < p

    if coin_flip:
        capital += bet_amount  # Выигрыш
    else:
        capital -= bet_amount  # Проигрыш

    if i % 20 == 0:
        print(f"После {i} ставок: ${capital:.2f}")

print(f"\nИтоговый капитал: ${capital:.2f}")

# Результат:
# После 0 ставок: $1000.00
# После 20 ставок: $1087.45
# После 40 ставок: $1215.83
# После 60 ставок: $1298.77
# После 80 ставок: $1456.92
#
# Итоговый капитал: $1623.41
```

### Правило Келли для акций

```python
# Акция с преимуществом
expected_return = 0.15  # Ожидаем 15%
volatility = 0.25  # Риск 25%
risk_free_rate = 0.02  # Безрисковая ставка 2%

# Формула Келли для акций (упрощенная)
kelly_fraction = (expected_return - risk_free_rate) / (volatility ** 2)

print(f"Kelly рекомендует: {kelly_fraction:.1%} капитала в эту акцию")

# Результат: 20.8% капитала

# НО! Многие используют "половину Келли" для безопасности
half_kelly = kelly_fraction / 2
print(f"Половина Kelly (безопаснее): {half_kelly:.1%}")

# Результат: 10.4% капитала
```

**Почему половина Келли?**
- Полный Келли слишком агрессивен
- Ошибка в оценке вероятности → большие потери
- Половина Келли: меньше доходность, но намного меньше риск банкротства

## Иерархический паритет рисков (HRP)

**Кто создал:** Marcos Lopez de Prado (легенда квантовых финансов)

### Проблемы классической оптимизации Марковица

1. **Нестабильность:** Маленькое изменение данных → огромное изменение весов
2. **Концентрация:** Все деньги в 1-2 акциях
3. **Недостаточная эффективность:** Плохо работает на реальных данных

### Идея HRP

**Использовать машинное обучение для группировки похожих активов**

**Аналогия:** Семейное древо активов

```
ПОРТФЕЛЬ
├── Технологии
│   ├── Apple
│   └── Microsoft
└── Энергетика
    ├── Exxon
    └── Chevron
```

**Алгоритм:**
1. Кластеризация активов (найти похожие)
2. Создать иерархию (дерево)
3. Распределить риск равномерно по ветвям

```python
# Упрощенная концепция HRP
import scipy.cluster.hierarchy as sch

# Матрица корреляций
correlation_matrix = np.array([
    [1.00, 0.85, 0.10, 0.15],  # Apple
    [0.85, 1.00, 0.12, 0.18],  # Microsoft (похож на Apple)
    [0.10, 0.12, 1.00, 0.75],  # Exxon
    [0.15, 0.18, 0.75, 1.00]   # Chevron (похож на Exxon)
])

# Преобразовать в расстояние
distance = np.sqrt(0.5 * (1 - correlation_matrix))

# Иерархическая кластеризация
linkage = sch.linkage(distance, method='single')

# Визуализация дендрограммы
plt.figure(figsize=(10, 5))
sch.dendrogram(linkage, labels=['Apple', 'Microsoft', 'Exxon', 'Chevron'])
plt.title('Иерархия активов')
plt.show()

# HRP затем распределяет веса на основе этой иерархии
```

**Преимущества HRP:**
✅ Работает даже если корреляции нестабильны
✅ Не требует обратимости ковариационной матрицы
✅ Меньше риск вне выборки
✅ Более диверсифицированные портфели

## Бэктестинг с Zipline

**Zipline** - это библиотека для тестирования торговых стратегий, созданная Quantopian.

### Простой пример стратегии

```python
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol, record
import pandas as pd

def initialize(context):
    """Запускается один раз в начале"""
    context.asset = symbol('AAPL')
    context.sma_window = 20  # 20-дневная скользящая средняя

def handle_data(context, data):
    """Запускается каждый день"""

    # Получить историю цен
    history = data.history(context.asset, 'price',
                           context.sma_window + 1, '1d')

    # Вычислить скользящую среднюю
    sma = history[:-1].mean()
    current_price = history[-1]

    # Торговая логика
    if current_price > sma:
        # Цена выше средней → купить
        order_target_percent(context.asset, 1.0)
        record(action='BUY')
    else:
        # Цена ниже средней → продать
        order_target_percent(context.asset, 0.0)
        record(action='SELL')

    # Записать данные
    record(price=current_price, sma=sma)

def analyze(context, perf):
    """Анализ после окончания"""
    print(f"Итоговая стоимость: ${perf['portfolio_value'][-1]:,.2f}")
    print(f"Доходность: {(perf['portfolio_value'][-1]/10000 - 1):.2%}")

# Запустить бэктест
result = run_algorithm(
    start=pd.Timestamp('2020-01-01'),
    end=pd.Timestamp('2023-01-01'),
    initialize=initialize,
    handle_data=handle_data,
    analyze=analyze,
    capital_base=10000,
    data_frequency='daily',
    bundle='quandl'
)

# Результат:
# Итоговая стоимость: $12,456.78
# Доходность: 24.57%
```

### Оптимизация портфеля в Zipline

```python
from zipline.api import order_optimal_portfolio
from zipline.finance import commission, slippage
import cvxpy as cp

def initialize(context):
    """Настройка"""
    context.stocks = [symbol('AAPL'), symbol('MSFT'),
                      symbol('GOOGL'), symbol('AMZN')]

    # Реалистичные издержки
    context.set_commission(commission.PerShare(cost=0.001, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())

def handle_data(context, data):
    """Ежедневная торговля"""

    # Получить доходности
    prices = data.history(context.stocks, 'price', 252, '1d')
    returns = prices.pct_change().dropna()

    # Вычислить ковариацию
    cov_matrix = returns.cov()

    # Mean-Variance оптимизация
    n = len(context.stocks)
    weights = cp.Variable(n)
    risk = cp.quad_form(weights, cov_matrix.values)
    objective = cp.Minimize(risk)

    # Ограничения
    constraints = [
        cp.sum(weights) == 1,  # Сумма весов = 1
        weights >= 0  # Без шорта
    ]

    # Решить
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Применить веса
    if weights.value is not None:
        target_weights = {stock: weight
                          for stock, weight in zip(context.stocks, weights.value)}
        order_optimal_portfolio(target_weights)

# Запустить...
```

## Оценка с Pyfolio

**Pyfolio** анализирует результаты бэктеста и создает красивые отчеты.

```python
import pyfolio as pf

# Извлечь данные из Zipline
returns = result['returns']
positions = result['positions']
transactions = result['transactions']

# Создать полный tear sheet (отчет)
pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions,
    live_start_date='2022-01-01'  # Дата начала форвард-теста
)
```

### Что показывает Pyfolio?

**1. Основная статистика:**
```
Annual return: 15.2%
Cumulative returns: 52.3%
Annual volatility: 18.5%
Sharpe ratio: 0.82
Calmar ratio: 1.15
Max drawdown: -13.2%
```

**2. Графики:**
- Кривая капитала
- Drawdown график (просадки)
- Роллинг Sharpe Ratio
- Месячная доходность (тепловая карта)

**3. Анализ рисков:**
- VaR (Value at Risk): "С 95% уверенностью не потеряешь больше 5% за день"
- Стресс-тесты: Как бы стратегия пережила 2008 кризис?

**4. Анализ сделок:**
- Средняя прибыль на сделку
- Win rate (% прибыльных сделок)
- Profit factor

### Пример интерпретации

```python
# После создания tear sheet, читаем:

# Хорошо:
✅ Sharpe ratio: 1.2 (отлично!)
✅ Max drawdown: -8% (терпимо)
✅ Calmar ratio: 2.5 (очень хорошо)

# Плохо:
❌ Win rate: 32% (только треть сделок прибыльны)
❌ Tail risk: высокий (редкие, но большие потери)

# Вывод: Стратегия работает, но нужно:
# - Улучшить точность сигналов (поднять win rate)
# - Добавить стоп-лосс для защиты от больших потерь
```

## Практический чеклист: Оценка стратегии

### До запуска стратегии:

**1. Бэктест на истории (минимум 3-5 лет)**
```python
✓ Sharpe ratio > 1.0
✓ Max drawdown < 20%
✓ Положительная доходность каждый год
✓ Win rate > 40%
```

**2. Форвард-тест на свежих данных**
```python
✓ Результаты похожи на бэктест
✓ Стратегия не сломалась на новых данных
```

**3. Анализ издержек**
```python
✓ Учтены комиссии
✓ Учтено проскальзывание (slippage)
✓ Достаточно ликвидности
```

**4. Стресс-тесты**
```python
✓ Как пережила COVID-19 crash (март 2020)?
✓ Как пережила 2008 кризис?
✓ Что если волатильность удвоится?
```

### Красные флаги (не запускай!):

❌ Sharpe < 0.5 (слишком много риска)
❌ Drawdown > 30% (психологически невыносимо)
❌ Результаты сильно отличаются в бэктесте vs форвард-тесте
❌ Слишком много сделок (комиссии съедят прибыль)
❌ Переобучение (идеально на истории, ужасно на новых данных)

## Заключение

**Главные уроки:**

1. **Sharpe Ratio - король метрик:** Доходность/Риск
2. **Диверсификация - бесплатный обед:** Снижай риск без потери доходности
3. **Бэктест - обязателен:** Никогда не запускай стратегию без теста!
4. **Kelly Criterion - управление размером:** Не ставь слишком много
5. **Pyfolio - твой друг:** Подробный анализ стратегии

**Золотое правило:**
> "Если не можешь объяснить, почему стратегия работает, она скорее всего не будет работать на реальных деньгах"

**Следующий шаг:** Открой ноутбуки в этой папке и протестируй свою первую стратегию!
