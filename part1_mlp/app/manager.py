"""Загрузка и подготовка табличных данных из CSV для обучения сети."""
import csv

import numpy as np

from .exceptions import MismatchedDataError


class DatasetManager:
    """Читает CSV (первая строка - заголовки, последний столбец - целевая
    переменная), умеет нормализовать признаки и делить выборку на train/test.
    """

    def __init__(self):
        self.zagolovki = None
        self.X = None
        self.y = None
        self.sposob_normalizacii = None
        self._param_normalizacii = None

    def load_csv(self, put_k_faylu):
        """Загружает данные из CSV-файла по указанному пути."""
        with open(put_k_faylu, encoding="utf-8") as f:
            chitatel = csv.reader(f)
            stroki = list(chitatel)

        if len(stroki) < 2:
            raise MismatchedDataError("минимум 1 строка данных", f"{len(stroki) - 1} строк", context=put_k_faylu)

        self.zagolovki = stroki[0]
        dannye = np.array([[float(znachenie) for znachenie in stroka] for stroka in stroki[1:]])
        self.X = dannye[:, :-1]
        self.y = dannye[:, -1:]
        return self.X, self.y

    def normalize(self, sposob="standard"):
        """Нормализует признаки X. sposob = 'standard' (вычесть среднее,
        поделить на стандартное отклонение) или 'minmax' (привести к [0, 1])."""
        if self.X is None:
            raise MismatchedDataError("загруженные данные", None, context="сначала вызовите load_csv")

        if sposob == "standard":
            srednee = self.X.mean(axis=0)
            std = self.X.std(axis=0)
            std[std == 0] = 1.0  # защита от деления на ноль для constant-признака
            self.X = (self.X - srednee) / std
            self._param_normalizacii = {"srednee": srednee.tolist(), "std": std.tolist()}
        elif sposob == "minmax":
            minimum = self.X.min(axis=0)
            maximum = self.X.max(axis=0)
            razmah = maximum - minimum
            razmah[razmah == 0] = 1.0
            self.X = (self.X - minimum) / razmah
            self._param_normalizacii = {"minimum": minimum.tolist(), "maximum": maximum.tolist()}
        else:
            raise ValueError(f"Неизвестный способ нормализации: {sposob!r} (доступны 'standard', 'minmax')")

        self.sposob_normalizacii = sposob
        return self.X

    def train_test_split(self, dolya_test=0.2, seed=None):
        """Делит данные на обучающую и тестовую выборки в заданной пропорции."""
        if self.X is None:
            raise MismatchedDataError("загруженные данные", None, context="сначала вызовите load_csv")

        n = self.X.shape[0]
        generator = np.random.default_rng(seed)
        poryadok = generator.permutation(n)
        n_test = int(n * dolya_test)
        test_idx = poryadok[:n_test]
        train_idx = poryadok[n_test:]

        X_train, X_test = self.X[train_idx], self.X[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]
        return X_train, X_test, y_train, y_test
