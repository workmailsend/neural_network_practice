"""Аугментация изображений (повороты, отражения, шум) - CPU-bound работа,
поэтому распараллеливается процессами (multiprocessing), а не потоками:
из-за GIL чистые вычисления на потоках Python не ускоряются."""
import multiprocessing as mp

import numpy as np


def povorot_90(massiv):
    return np.rot90(massiv).copy()


def otrazhenie(massiv):
    return np.fliplr(massiv).copy()


def dobavit_shum(massiv, uroven=0.05):
    shum = np.random.default_rng().normal(0, uroven, massiv.shape)
    return np.clip(massiv + shum, 0, 1)


def augmentirovat_odin(imya_i_massiv):
    """Применяет набор аугментаций к одному изображению. Функция должна
    быть определена на верхнем уровне модуля (не вложенная), иначе
    multiprocessing не сможет её сериализовать (pickle) для передачи в
    дочерний процесс."""
    imya, massiv = imya_i_massiv
    return {
        f"{imya}_original": massiv,
        f"{imya}_rotated": povorot_90(massiv),
        f"{imya}_flipped": otrazhenie(massiv),
        f"{imya}_noisy": dobavit_shum(massiv),
    }


def augmentirovat_posledovatelno(spisok_massivov):
    """Обычная последовательная версия - для сравнения скорости."""
    itog = {}
    for item in spisok_massivov:
        itog.update(augmentirovat_odin(item))
    return itog


def augmentirovat_mnogoprocessno(spisok_massivov, chislo_processov=4):
    """Параллельная версия через пул процессов. apply_async + callback
    используется вместо pool.map(), чтобы явно показать, как собирать
    результаты из разных процессов по мере готовности. callback всегда
    выполняется в основном процессе, поэтому доп. блокировка для itog
    не нужна - вызовы callback идут строго по очереди."""
    itog = {}

    def na_zavershenii(rezultat):
        itog.update(rezultat)

    with mp.Pool(processes=chislo_processov) as pool:
        zadachi = [pool.apply_async(augmentirovat_odin, (item,), callback=na_zavershenii) for item in spisok_massivov]
        for zadacha in zadachi:
            zadacha.wait()
    return itog
