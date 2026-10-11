"""Простая нейросеть (многослойный перцептрон) на numpy, без фреймворков."""
import json
from pathlib import Path

import numpy as np

from .exceptions import InvalidLayerSizeError, MismatchedDataError, UnknownActivationError, WeightsFileError


def sigmoida(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoida_proizv(a):
    # производная сигмоиды через саму активацию a = sigmoida(z)
    return a * (1.0 - a)


def relu(z):
    return np.maximum(0.0, z)


def relu_proizv(a):
    return (a > 0).astype(a.dtype)


FUNKCII_AKTIVACII = {
    "sigmoid": (sigmoida, sigmoida_proizv),
    "relu": (relu, relu_proizv),
}


class NeuralNetwork:
    """Полносвязная сеть с одним или несколькими скрытыми слоями.

    razmery_sloev - список размеров слоёв, например [4, 8, 1]:
    4 входа, скрытый слой на 8 нейронов, 1 выход.
    aktivacia - 'sigmoid' или 'relu'.
    """

    def __init__(self, razmery_sloev, aktivacia="sigmoid", seed=None):
        if len(razmery_sloev) < 2 or any(not isinstance(n, int) or n <= 0 for n in razmery_sloev):
            raise InvalidLayerSizeError(razmery_sloev)
        if aktivacia not in FUNKCII_AKTIVACII:
            raise UnknownActivationError(aktivacia)

        self.razmery_sloev = list(razmery_sloev)
        self.aktivacia = aktivacia
        self.f_aktiv, self.f_aktiv_proizv = FUNKCII_AKTIVACII[aktivacia]

        generator = np.random.default_rng(seed)
        self.vesa = []
        self.smeshenia = []
        for n_vh, n_vyh in zip(razmery_sloev[:-1], razmery_sloev[1:]):
            masshtab = 1.0 / np.sqrt(n_vh)
            self.vesa.append(generator.normal(0, masshtab, size=(n_vh, n_vyh)))
            self.smeshenia.append(np.zeros((1, n_vyh)))

        self.istoria_oshibok = []

    def smenit_aktivaciu(self, aktivacia):
        """Сменить функцию активации без пересоздания сети (веса остаются)."""
        if aktivacia not in FUNKCII_AKTIVACII:
            raise UnknownActivationError(aktivacia)
        self.aktivacia = aktivacia
        self.f_aktiv, self.f_aktiv_proizv = FUNKCII_AKTIVACII[aktivacia]

    def _proverit_razmer_vhoda(self, X):
        nado = self.razmery_sloev[0]
        est = X.shape[1] if X.ndim == 2 else None
        if est != nado:
            raise MismatchedDataError(nado, est, context="число признаков во входных данных X")

    def forward(self, X):
        """Прямой проход: считаем выход сети слой за слоем. Возвращаем
        сам выход и список активаций всех слоёв (нужен для backward)."""
        self._proverit_razmer_vhoda(X)
        aktivacii = [X]
        a = X
        for W, b in zip(self.vesa, self.smeshenia):
            z = a @ W + b
            a = self.f_aktiv(z)
            aktivacii.append(a)
        return a, aktivacii

    def backward(self, y, aktivacii, skorost_obuchenia):
        """Обратное распространение ошибки + шаг градиентного спуска.
        Ошибка - среднеквадратичная (MSE). Возвращает значение ошибки."""
        y_pred = aktivacii[-1]
        if y.shape != y_pred.shape:
            raise MismatchedDataError(y_pred.shape, y.shape, context="форма y")

        m = y.shape[0]
        oshibka = float(np.mean((y_pred - y) ** 2))

        delta = (2.0 / m) * (y_pred - y) * self.f_aktiv_proizv(y_pred)
        for sloi in reversed(range(len(self.vesa))):
            a_prev = aktivacii[sloi]
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            if sloi > 0:
                delta = (delta @ self.vesa[sloi].T) * self.f_aktiv_proizv(aktivacii[sloi])
            self.vesa[sloi] -= skorost_obuchenia * dW
            self.smeshenia[sloi] -= skorost_obuchenia * db
        return oshibka

    def train(self, X, y, epohi, skorost_obuchenia, razmer_batcha=None, pechat_kazhdye=0):
        """Обучение сети. Если razmer_batcha не задан - считаем сразу по
        всем примерам, иначе данные бьются на случайные батчи каждую эпоху."""
        self._proverit_razmer_vhoda(X)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise MismatchedDataError(X.shape[0], y.shape[0], context="число строк X и y должно совпадать")

        n = X.shape[0]
        generator = np.random.default_rng()
        for epoha in range(1, epohi + 1):
            if razmer_batcha is None:
                _, aktivacii = self.forward(X)
                oshibka_epohi = self.backward(y, aktivacii, skorost_obuchenia)
            else:
                poryadok = generator.permutation(n)
                oshibki_batchei = []
                for start in range(0, n, razmer_batcha):
                    idx = poryadok[start:start + razmer_batcha]
                    _, aktivacii = self.forward(X[idx])
                    oshibki_batchei.append(self.backward(y[idx], aktivacii, skorost_obuchenia))
                oshibka_epohi = float(np.mean(oshibki_batchei))
            self.istoria_oshibok.append(oshibka_epohi)
            if pechat_kazhdye and epoha % pechat_kazhdye == 0:
                print(f"  эпоха {epoha:5d}/{epohi} - ошибка (MSE): {oshibka_epohi:.6f}")
        return self.istoria_oshibok

    def predict(self, X):
        """Только прямой проход, без обучения."""
        y_pred, _ = self.forward(X)
        return y_pred

    def save_weights(self, path):
        """Сохраняет архитектуру и веса в JSON."""
        dannye = {
            "razmery_sloev": self.razmery_sloev,
            "aktivacia": self.aktivacia,
            "vesa": [w.tolist() for w in self.vesa],
            "smeshenia": [b.tolist() for b in self.smeshenia],
        }
        try:
            Path(path).write_text(json.dumps(dannye, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            raise WeightsFileError(f"Не удалось сохранить веса в {path}: {e}") from e

    def load_weights(self, path):
        """Загружает веса из файла, созданного save_weights()."""
        try:
            dannye = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise WeightsFileError(f"Не удалось прочитать файл весов {path}: {e}") from e

        if dannye.get("razmery_sloev") != self.razmery_sloev:
            raise WeightsFileError(
                f"Файл весов описывает архитектуру {dannye.get('razmery_sloev')}, "
                f"а у текущей сети {self.razmery_sloev} - загрузка невозможна."
            )
        self.smenit_aktivaciu(dannye["aktivacia"])
        self.vesa = [np.array(w) for w in dannye["vesa"]]
        self.smeshenia = [np.array(b) for b in dannye["smeshenia"]]
