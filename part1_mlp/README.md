# Часть 1. ООП-реализация многослойного перцептрона

Простая нейросеть (MLP) на чистом numpy, без ML-фреймворков.

## Структура

- `app/models.py` - класс `NeuralNetwork` (forward, backward, train, predict, save/load весов)
- `app/manager.py` - класс `DatasetManager` (загрузка CSV, нормализация, train/test split)
- `app/exceptions.py` - свои исключения (`InvalidLayerSizeError`, `MismatchedDataError`, `UnknownActivationError`, `WeightsFileError`)
- `app/main.py` - консольное меню

## Запуск

```
python -m app.main
```

Запускать из папки `part1_mlp` (используются относительные импорты внутри пакета `app`).

## Пример

Тестовые данные: `../data/input/sample_dataset.csv` (300 строк, 2 признака, бинарная цель).
Создать сеть `[2, 6, 1]`, активация `sigmoid`, обучить 300 эпох со скоростью 0.1 - ошибка падает
с ~0.26 до ~0.04 (график см. в отчёте).
