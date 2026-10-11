"""Сравнение времени последовательной и многопроцессорной аугментации,
сохранение результата в performance_report.json."""
import json
import time
from pathlib import Path

from .augment import augmentirovat_mnogoprocessno, augmentirovat_posledovatelno


def sravnit_proizvoditelnost(spisok_massivov, chislo_processov=4, put_otcheta="performance_report.json"):
    nachalo = time.perf_counter()
    augmentirovat_posledovatelno(spisok_massivov)
    vremya_posledovatelno = time.perf_counter() - nachalo

    nachalo = time.perf_counter()
    augmentirovat_mnogoprocessno(spisok_massivov, chislo_processov)
    vremya_parallelno = time.perf_counter() - nachalo

    otchet = {
        "chislo_izobrazheniy": len(spisok_massivov),
        "chislo_processov": chislo_processov,
        "vremya_posledovatelno_sek": round(vremya_posledovatelno, 4),
        "vremya_parallelno_sek": round(vremya_parallelno, 4),
        "uskorenie": round(vremya_posledovatelno / vremya_parallelno, 2) if vremya_parallelno > 0 else None,
    }
    Path(put_otcheta).write_text(json.dumps(otchet, ensure_ascii=False, indent=2), encoding="utf-8")
    return otchet
