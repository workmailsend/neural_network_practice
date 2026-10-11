"""Обход каталога с данными и преобразование файлов в массивы numpy."""
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

RASSHIRENIA_IZOBRAZHENIY = {".png", ".jpg", ".jpeg", ".bmp"}


class ScanError(Exception):
    """Ошибка при сканировании каталога или чтении файла (отсутствует
    каталог, файл повреждён и т.п.)."""


def naiti_fayly(put_k_papke, rasshirenia):
    """Рекурсивно обходит put_k_papke и возвращает список файлов с
    указанными расширениями (например ['.png', '.csv'])."""
    papka = Path(put_k_papke)
    if not papka.is_dir():
        raise ScanError(f"Каталог не найден: {put_k_papke}")
    rasshirenia = {r if r.startswith(".") else f".{r}" for r in rasshirenia}
    naideno = [f for f in papka.rglob("*") if f.is_file() and f.suffix.lower() in rasshirenia]
    return sorted(naideno)


def zagruzit_izobrazhenie(put_k_faylu):
    """Читает изображение через PIL, приводит к RGB, нормализует (/255)."""
    with Image.open(put_k_faylu) as img:
        massiv = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return massiv


def zagruzit_csv_kak_massiv(put_k_faylu):
    """Читает CSV-файл (с заголовком) и возвращает числовой массив numpy."""
    with open(put_k_faylu, encoding="utf-8") as f:
        chitatel = csv.reader(f)
        next(chitatel, None)  # пропускаем заголовок
        stroki = [[float(znachenie) for znachenie in stroka] for stroka in chitatel if stroka]
    return np.array(stroki, dtype=np.float32)


def obrabotat_fayly(fayly, logger=None):
    """Загружает список файлов (изображения и/или CSV), приводит все к
    единому плоскому виду и объединяет в один массив. Возвращает массив и
    словарь статистики (число файлов, время, формы)."""
    nachalo = time.perf_counter()
    massivy = []
    propushcheno = 0
    for put in fayly:
        try:
            if put.suffix.lower() in RASSHIRENIA_IZOBRAZHENIY:
                massiv = zagruzit_izobrazhenie(put)
            elif put.suffix.lower() == ".csv":
                massiv = zagruzit_csv_kak_massiv(put)
            else:
                propushcheno += 1
                continue
            massivy.append(massiv.flatten())
            if logger:
                logger.info("обработан файл %s, форма %s", put.name, massiv.shape)
        except (OSError, ValueError) as e:
            propushcheno += 1
            if logger:
                logger.warning("не удалось обработать %s: %s", put, e)

    vremya = time.perf_counter() - nachalo
    if not massivy:
        raise ScanError("Ни одного файла не удалось обработать.")

    # разные файлы могут давать массивы разной длины (разные размеры
    # картинок/таблиц) - дополняем нулями до максимальной длины, чтобы
    # можно было сохранить всё одним прямоугольным массивом numpy.
    maks_dlina = max(len(m) for m in massivy)
    vyravnennye = [np.pad(m, (0, maks_dlina - len(m))) for m in massivy]
    itogovyi_massiv = np.stack(vyravnennye)

    statistika = {
        "vsego_naideno": len(fayly),
        "obrabotano": len(massivy),
        "propushcheno": propushcheno,
        "forma_itogovogo_massiva": list(itogovyi_massiv.shape),
        "vremya_sekund": round(vremya, 4),
    }
    return itogovyi_massiv, statistika
