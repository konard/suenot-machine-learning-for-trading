# Линейное внимание и SSM (Краткое руководство)

Краткое введение в Главу 134: как использовать линейное внимание и модели пространства состояний в трейдинге.

## Содержание
1. Теоретические основы
2. Основные архитектуры
3. Примеры кода
4. Реализация на Rust
5. Метрики оценки
6. Ресурсы

## Примеры кода
- PyTorch: [`python/model.py`](python/model.py)
```bash
python python/model.py
```

- Обучение (Bybit Crypto и Акции): [`python/train.py`](python/train.py)
```bash
python python/train.py
```

- Бэктест: [`python/backtest.py`](python/backtest.py)
```bash
python python/backtest.py
```

## Реализация на Rust
Код обработки рыночных данных Bybit и акций на Rust: 
- Библиотека: [`rust/src/lib.rs`](rust/src/lib.rs).
- Исполняемый файл: [`rust/src/main.rs`](rust/src/main.rs).

**Запуск:**
```bash
cd rust
cargo run
```
