# neural_network_practice

Практическое задание по дисциплине «Основы нейронных сетей» - три части:

1. [`part1_mlp`](part1_mlp/README.md) - объектно-ориентированная нейросеть (MLP) с нуля на numpy
2. [`part2_prepare`](part2_prepare/README.md) - CLI-утилита препроцессинга данных (`prepare`, `doctor`)
3. [`part3_parallel`](part3_parallel/README.md) - многопоточное чтение + многопроцессорная аугментация

## Установка

```
pip install -r requirements.txt
cp .env.example .env
```

Каждая часть запускается отдельно из своей папки (`python -m app.main` / `python -m app.cli ...`) -
подробности в README каждой части.
