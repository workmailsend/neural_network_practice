# Часть 2. CLI-утилита для препроцессинга данных

## Структура

- `app/cli.py` - точка входа, argparse с командами `prepare` и `doctor`
- `app/scanner.py` - обход каталога, чтение изображений (PIL) и CSV, сборка в массив numpy
- `app/config.py` - настройки из переменных окружения / `.env` (без хардкода путей)
- `app/logging_config.py` - настройка логирования

## Запуск

Из папки `part2_prepare`:

```
python -m app.cli prepare --path ../data/input/part2_sample --ext png csv --output ../data/output/prepared.npy
python -m app.cli doctor
```

`--path`, `--output` и уровень логирования можно не указывать - тогда берутся значения из
`.env` (см. `.env.example` в корне) или значения по умолчанию.

## Пример

Тестовые данные: `../data/input/part2_sample/` (3 картинки разного размера + 2 csv-таблицы).
`prepare` находит все 5 файлов, приводит их к единой длине (дополняя нулями более короткие)
и сохраняет как один массив `(5, 1200)` в `.npy`.
