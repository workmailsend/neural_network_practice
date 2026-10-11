"""Собственные исключения для пакета part1_mlp.

Используются вместо общих ValueError/RuntimeError, чтобы вызывающий код
(консольное меню в main.py) мог отличить ошибку конфигурации сети от
ошибки несовместимых данных и показать пользователю понятное сообщение.
"""


class NeuralNetworkError(Exception):
    """Базовый класс для всех ошибок пакета - удобно ловить одним except,
    если вызывающему коду не важна конкретная причина."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Некорректное описание архитектуры сети: меньше двух слоёв, либо
    размер какого-то слоя не является положительным целым числом."""

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        super().__init__(
            f"Некорректные размеры слоёв {layer_sizes!r}: нужно минимум 2 слоя "
            "(вход и выход), все размеры должны быть положительными целыми числами."
        )


class MismatchedDataError(NeuralNetworkError):
    """Размерность поданных данных не совпадает с ожидаемой архитектурой
    сети (например, число признаков в X не равно размеру входного слоя)."""

    def __init__(self, expected, got, context=""):
        self.expected = expected
        self.got = got
        suffix = f" ({context})" if context else ""
        super().__init__(f"Ожидалась размерность {expected}, получено {got}{suffix}.")


class UnknownActivationError(NeuralNetworkError):
    """Запрошена функция активации, которую сеть не умеет использовать."""

    def __init__(self, name):
        self.name = name
        super().__init__(f"Неизвестная функция активации: {name!r}. Доступны: 'sigmoid', 'relu'.")


class WeightsFileError(NeuralNetworkError):
    """Не удалось прочитать/записать файл весов (повреждён, не найден,
    либо не совпадает с текущей архитектурой сети)."""
