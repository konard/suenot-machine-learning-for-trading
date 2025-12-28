# Инжиниринг финансовых признаков: Как исследовать альфа-факторы

Алгоритмические торговые стратегии управляются сигналами, которые указывают, когда покупать или продавать активы для получения превосходящей доходности относительно бенчмарка, такого как индекс. Часть доходности актива, которая не объясняется подверженностью этому бенчмарку, называется альфой, и, следовательно, сигналы, направленные на получение такой некоррелированной доходности, также называются альфа-факторами.

Если вы уже знакомы с машинным обучением, вы можете знать, что инжиниринг признаков является ключевым ингредиентом для успешных прогнозов. В трейдинге это не отличается. Однако инвестирование особенно богато десятилетиями исследований того, как работают рынки и какие признаки могут работать лучше других для объяснения или прогнозирования движения цен в результате. Эта глава предоставляет обзор в качестве отправной точки для вашего собственного поиска альфа-факторов.

Эта глава также представляет ключевые инструменты, которые облегчают вычисление и тестирование альфа-факторов. Мы выделим, как библиотеки NumPy, pandas и TA-Lib облегчают манипуляцию данными, и представим популярные методы сглаживания, такие как вейвлеты и фильтр Калмана, которые помогают уменьшить шум в данных.

Мы также предварительно рассмотрим, как вы можете использовать торговый симулятор Zipline для оценки прогнозной производительности (традиционных) альфа-факторов. Мы обсудим ключевые метрики альфа-факторов, такие как информационный коэффициент и оборот фактора. Углубленное введение в бэктестинг торговых стратегий, использующих машинное обучение, следует в [Главе 6](../08_ml4t_workflow), которая охватывает **рабочий процесс ML4T**, который мы будем использовать на протяжении всей книги для оценки торговых стратегий.

Пожалуйста, см. [Приложение - Библиотека альфа-факторов](../24_alpha_factor_library) для дополнительного материала по этой теме, включая многочисленные примеры кода, которые вычисляют широкий спектр альфа-факторов.

## Содержание

1. [Альфа-факторы на практике: от данных к сигналам](#альфа-факторы-на-практике-от-данных-к-сигналам)
2. [Опираясь на десятилетия исследований факторов](#опираясь-на-десятилетия-исследований-факторов)
    * [Ссылки](#ссылки)
3. [Инжиниринг альфа-факторов, которые предсказывают доходность](#инжиниринг-альфа-факторов-которые-предсказывают-доходность)
    * [Пример кода: Как создавать факторы с использованием pandas и NumPy](#пример-кода-как-создавать-факторы-с-использованием-pandas-и-numpy)
    * [Пример кода: Как использовать TA-Lib для создания технических альфа-факторов](#пример-кода-как-использовать-ta-lib-для-создания-технических-альфа-факторов)
    * [Пример кода: Как устранять шум в альфа-факторах с помощью фильтра Калмана](#пример-кода-как-устранять-шум-в-альфа-факторах-с-помощью-фильтра-калмана)
    * [Пример кода: Как предобрабатывать шумные сигналы с использованием вейвлетов](#пример-кода-как-предобрабатывать-шумные-сигналы-с-использованием-вейвлетов)
    * [Ресурсы](#ресурсы)
4. [От сигналов к сделкам: бэктестинг с Zipline](#от-сигналов-к-сделкам-бэктестинг-с-zipline)
    * [Пример кода: Как использовать Zipline для бэктестинга однофакторной стратегии](#пример-кода-как-использовать-zipline-для-бэктестинга-однофакторной-стратегии)
    * [Пример кода: Комбинирование факторов из различных источников данных на платформе Quantopian](#пример-кода-комбинирование-факторов-из-различных-источников-данных-на-платформе-quantopian)
    * [Пример кода: Разделение сигнала и шума – как использовать alphalens](#пример-кода-разделение-сигнала-и-шума--как-использовать-alphalens)
5. [Альтернативные библиотеки и платформы для алгоритмической торговли](#альтернативные-библиотеки-и-платформы-для-алгоритмической-торговли)

## Альфа-факторы на практике: от данных к сигналам

Альфа-факторы — это трансформации рыночных, фундаментальных и альтернативных данных, которые содержат прогнозные сигналы. Они предназначены для захвата рисков, которые управляют доходностью активов. Один набор факторов описывает фундаментальные, общеэкономические переменные, такие как рост, инфляция, волатильность, производительность и демографический риск. Другой набор состоит из торгуемых инвестиционных стилей, таких как рыночный портфель, стоимостное-ростовое инвестирование и импульсное инвестирование.

Существуют также факторы, которые объясняют движение цен на основе экономики или институциональной среды финансовых рынков, или поведения инвесторов, включая известные искажения этого поведения. Экономическая теория, стоящая за факторами, может быть рациональной, где факторы имеют высокую доходность в долгосрочной перспективе для компенсации их низкой доходности во время плохих времен, или поведенческой, где премии за риск фактора являются результатом возможно искаженного или не полностью рационального поведения агентов, которое не арбитражируется.

## Опираясь на десятилетия исследований факторов

В идеализированном мире категории факторов риска должны быть независимыми друг от друга (ортогональными), давать положительные премии за риск и формировать полный набор, охватывающий все измерения риска и объясняющий систематические риски для активов в данном классе. На практике эти требования будут выполняться только приблизительно.

### Ссылки

- [Dissecting Anomalies](http://schwert.ssb.rochester.edu/f532/ff_JF08.pdf) by Eugene Fama and Ken French (2008)
- [Explaining Stock Returns: A Literature Review](https://www.ifa.com/pdfs/explainingstockreturns.pdf) by James L. Davis (2001)
- [Market Efficiency, Long-Term Returns, and Behavioral Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=15108) by Eugene Fama (1997)
- [The Efficient Market Hypothesis and It's Critics](https://pubs.aeaweb.org/doi/pdf/10.1257/089533003321164958) by Burton Malkiel (2003)
- [The New Palgrave Dictionary of Economics](https://www.palgrave.com/us/book/9780333786765) (2008) by Steven Durlauf and Lawrence Blume, 2nd ed.
- [Anomalies and Market Efficiency](https://www.nber.org/papers/w9277.pdf) by G. William Schwert25 (Ch. 15 in Handbook of the Economics of Finance, by Constantinides, Harris, and Stulz, 2003)
- [Investor Psychology and Asset Pricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=265132), by David Hirshleifer (2001)
- [Practical advice for analysis of large, complex data sets](https://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html), Patrick Riley, Unofficial Google Data Science Blog

## Инжиниринг альфа-факторов, которые предсказывают доходность

На основе концептуального понимания ключевых категорий факторов, их обоснования и популярных метрик ключевой задачей является идентификация новых факторов, которые могут лучше захватить риски, воплощенные драйверами доходности, изложенными ранее, или найти новые. В любом случае будет важно сравнить производительность инновационных факторов с производительностью известных факторов для выявления дополнительных приростов сигнала.

### Пример кода: Как создавать факторы с использованием pandas и NumPy

Ноутбук [feature_engineering.ipynb](00_data/feature_engineering.ipynb) в директории [data](00_data) иллюстрирует, как создавать базовые факторы.

### Пример кода: Как использовать TA-Lib для создания технических альфа-факторов

Ноутбук [how_to_use_talib](02_how_to_use_talib.ipynb) иллюстрирует использование TA-Lib, которая включает широкий спектр общих технических индикаторов. Эти индикаторы объединяет то, что они используют только рыночные данные, т.е. информацию о цене и объеме.

Ноутбук [common_alpha_factors](../24_alpha_factor_library/02_common_alpha_factors.ipynb) в **приложении** содержит десятки дополнительных примеров.

### Пример кода: Как устранять шум в альфа-факторах с помощью фильтра Калмана

Ноутбук [kalman_filter_and_wavelets](03_kalman_filter_and_wavelets.ipynb) демонстрирует использование фильтра Калмана с использованием пакета `PyKalman` для сглаживания; мы также будем использовать его в [Главе 9](../09_time_series_models), когда будем разрабатывать стратегию парной торговли.

### Пример кода: Как предобрабатывать шумные сигналы с использованием вейвлетов

Ноутбук [kalman_filter_and_wavelets](03_kalman_filter_and_wavelets.ipynb) также демонстрирует, как работать с вейвлетами, используя пакет `PyWavelets`.

### Ресурсы

- [Fama French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) Data Library
- [numpy](https://numpy.org/) website
    - [Quickstart Tutorial](https://numpy.org/devdocs/user/quickstart.html)
- [pandas](https://pandas.pydata.org/) website
    - [User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
    - [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
    - [Python Pandas Tutorial: A Complete Introduction for Beginners](https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/)
- [alphatools](https://github.com/marketneutral/alphatools) - Quantitative finance research tools in Python
- [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) - Package based on the work of Dr Marcos Lopez de Prado regarding his research with respect to Advances in Financial Machine Learning
- [PyKalman](https://pykalman.github.io/) documentation
- [Tutorial: The Kalman Filter](http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf)
- [Understanding and Applying Kalman Filtering](http://biorobotics.ri.cmu.edu/papers/sbp_papers/integrated3/kleeman_kalman_basics.pdf)
- [How a Kalman filter works, in pictures](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) - Wavelet Transforms in Python
- [An Introduction to Wavelets](https://www.eecis.udel.edu/~amer/CISC651/IEEEwavelet.pdf)
- [The Wavelet Tutorial](http://web.iitd.ac.in/~sumeet/WaveletTutorial.pdf)
- [Wavelets for Kids](http://www.gtwavelet.bme.gatech.edu/wp/kidsA.pdf)
- [The Barra Equity Risk Model Handbook](https://www.alacra.com/alacra/help/barra_handbook_GEM.pdf)
- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [Modern Investment Management: An Equilibrium Approach](https://www.amazon.com/Modern-Investment-Management-Equilibrium-Approach/dp/0471124109) by Bob Litterman, 2003
- [Quantitative Equity Portfolio Management: Modern Techniques and Applications](https://www.crcpress.com/Quantitative-Equity-Portfolio-Management-Modern-Techniques-and-Applications/Qian-Hua-Sorensen/p/book/9781584885580) by Edward Qian, Ronald Hua, and Eric Sorensen
- [Spearman Rank Correlation](https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php)

## От сигналов к сделкам: бэктестинг с Zipline

Библиотека с открытым исходным кодом [zipline](https://zipline.ml4trading.io/index.html) представляет собой событийно-ориентированную систему бэктестинга, поддерживаемую и используемую в производстве фондом количественных инвестиций с краудсорсингом [Quantopian](https://www.quantopian.com/) для облегчения разработки алгоритмов и живой торговли. Она автоматизирует реакцию алгоритма на торговые события и предоставляет ему текущие и исторические данные на определенный момент времени, что позволяет избежать смещения заглядывания вперед.

- [Глава 8](../08_ml4t_workflow) содержит более полное введение в Zipline.
- Пожалуйста, следуйте [инструкциям](../installation) в папке `installation`, включая решение **известных проблем**.

### Пример кода: Как использовать Zipline для бэктестинга однофакторной стратегии

Ноутбук [single_factor_zipline](04_single_factor_zipline.ipynb) разрабатывает и тестирует простой фактор возврата к среднему, который измеряет, насколько недавняя производительность отклонилась от исторического среднего. Краткосрочный разворот — это общая стратегия, которая использует слабо прогнозируемый паттерн, что увеличение цены акций, вероятно, будет возвращаться к среднему обратно вниз на горизонтах от менее чем минуты до одного месяца.

### Пример кода: Комбинирование факторов из различных источников данных на платформе Quantopian

Исследовательская среда Quantopian адаптирована для быстрого тестирования прогнозных альфа-факторов. Процесс очень похож, потому что он основан на `zipline`, но предлагает гораздо более богатый доступ к источникам данных.

Ноутбук [multiple_factors_quantopian_research](05_multiple_factors_quantopian_research.ipynb) иллюстрирует, как вычислять альфа-факторы не только из рыночных данных, как ранее, но также из фундаментальных и альтернативных данных.

### Пример кода: Разделение сигнала и шума – как использовать alphalens

Ноутбук [performance_eval_alphalens](06_performance_eval_alphalens.ipynb) представляет библиотеку [alphalens](http://quantopian.github.io/alphalens/) для анализа производительности прогнозных (альфа) факторов, выпущенную с открытым исходным кодом Quantopian. Он демонстрирует, как она интегрируется с библиотекой бэктестинга `zipline` и библиотекой анализа производительности и рисков портфеля `pyfolio`, которую мы рассмотрим в следующей главе.

`alphalens` облегчает анализ прогнозной силы альфа-факторов в отношении:
- Корреляции сигналов с последующей доходностью
- Прибыльности равновзвешенного или взвешенного по фактору портфеля на основе (подмножества) сигналов
- Оборота факторов для указания потенциальных торговых издержек
- Производительности фактора во время конкретных событий
- Разбивок предыдущего по секторам

Анализ может проводиться с использованием `tearsheets` или отдельных вычислений и графиков. Tearsheets проиллюстрированы в онлайн-репозитории для экономии места.

- См. [здесь](https://github.com/quantopian/alphalens/blob/master/alphalens/examples/alphalens_tutorial_on_quantopian.ipynb) для подробного учебника по `alphalens` от Quantopian

## Альтернативные библиотеки и платформы для алгоритмической торговли

- [QuantConnect](https://www.quantconnect.com/)
- [Alpha Trading Labs](https://www.alphalabshft.com/)
    - Alpha Trading Labs больше не активна
- [WorldQuant](https://www.worldquantvrc.com/en/cms/wqc/home/)
- Python Algorithmic Trading Library [PyAlgoTrade](http://gbeced.github.io/pyalgotrade/)
- [pybacktest](https://github.com/ematvey/pybacktest)
- [Trading with Python](http://www.tradingwithpython.com/)
- [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=5041)
