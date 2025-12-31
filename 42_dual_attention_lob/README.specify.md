# Chapter 42: TLOB - Dual Attention for Limit Order Book

## Описание

TLOB (Transformer for Limit Order Book) — модель на основе трансформера с двойным механизмом внимания для захвата пространственных и временных зависимостей в данных книги заявок (LOB).

## Техническое задание

### Цели
1. Реализовать TLOB архитектуру с dual attention
2. Обработка spatial dependencies (уровни цен)
3. Обработка temporal dependencies (временная динамика)
4. Предсказание тренда цены на разных горизонтах

### Ключевые компоненты
- Spatial Attention для уровней LOB
- Temporal Attention для исторической динамики
- Multi-horizon prediction head
- LOB data preprocessing pipeline

### Метрики
- F1-score для предсказания направления
- Precision/Recall по горизонтам
- Сравнение с DeepLOB baseline

## Научные работы

1. **TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data**
   - arXiv: https://arxiv.org/abs/2502.15757
   - Год: 2025
   - Ключевые идеи: dual attention для spatial-temporal LOB анализа

2. **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books**
   - Baseline модель для сравнения

## Данные
- LOBSTER dataset
- FI-2010 dataset
- Binance/Crypto LOB data

## Реализация

### Python
- PyTorch реализация TLOB
- LOB data loader

### Rust
- High-performance LOB processing
- Real-time inference

## Структура
```
42_dual_attention_lob/
├── README.specify.md
├── docs/ru/
├── python/
│   ├── tlob_model.py
│   ├── lob_dataloader.py
│   └── train.py
└── rust/src/
```
