"""CLI-утилита препроцессинга данных: команды prepare и doctor."""
import argparse
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

from .config import PUT_PO_UMOLCHANIU, VYHOD_PO_UMOLCHANIU
from .logging_config import nastroit_logging
from .scanner import ScanError, naiti_fayly, obrabotat_fayly


def komanda_prepare(args):
    """prepare --path <директория> --ext <расширения> --output <файл.npy>:
    обходит директорию, превращает найденные файлы в единый numpy-массив,
    сохраняет его и печатает статистику обработки."""
    logger = nastroit_logging()
    put = args.path or PUT_PO_UMOLCHANIU
    vyhod = args.output or str(Path(VYHOD_PO_UMOLCHANIU) / "prepared.npy")

    logger.info("скан каталога %s (расширения: %s)", put, args.ext)
    try:
        fayly = naiti_fayly(put, args.ext)
        if not fayly:
            logger.warning("не найдено ни одного файла с расширениями %s в %s", args.ext, put)
            return 1
        massiv, statistika = obrabotat_fayly(fayly, logger=logger)
    except ScanError as e:
        logger.error("ошибка сканирования: %s", e)
        return 1

    Path(vyhod).parent.mkdir(parents=True, exist_ok=True)
    np.save(vyhod, massiv)

    print()
    print("=== Статистика обработки ===")
    print(f"Найдено файлов:     {statistika['vsego_naideno']}")
    print(f"Обработано:         {statistika['obrabotano']}")
    print(f"Пропущено с ошибкой:{statistika['propushcheno']}")
    print(f"Форма итогового массива: {statistika['forma_itogovogo_massiva']}")
    print(f"Время обработки:    {statistika['vremya_sekund']} с")
    print(f"Сохранено в:        {vyhod}")
    return 0


def _proverit_biblioteku(imya):
    try:
        modul = import_module(imya)
        versia = getattr(modul, "__version__", "версия неизвестна")
        print(f"  [OK]  {imya} - {versia}")
        return True
    except ImportError:
        print(f"  [НЕТ] {imya} не установлен")
        return False


def komanda_doctor(args):
    """doctor: проверяет окружение - версию Python, нужные библиотеки,
    доступность CUDA через nvidia-smi, текущую рабочую директорию."""
    print("=== Проверка окружения ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Рабочая директория: {Path.cwd()}")

    print("Библиотеки:")
    for imya in ("numpy", "PIL", "dotenv"):
        _proverit_biblioteku(imya)
    # tensorflow/pytorch - тяжёлые необязательные зависимости, не критично,
    # если их нет
    for imya in ("tensorflow", "torch"):
        _proverit_biblioteku(imya)

    print("GPU (CUDA):")
    try:
        rezultat = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5,
        )
        if rezultat.returncode == 0:
            pervaya_stroka = rezultat.stdout.strip().splitlines()[0] if rezultat.stdout.strip() else ""
            print(f"  [OK]  nvidia-smi доступен: {pervaya_stroka}")
        else:
            print("  [НЕТ] nvidia-smi вернул ошибку - GPU NVIDIA не обнаружена")
    except FileNotFoundError:
        print("  [НЕТ] nvidia-smi не найден в системе - похоже, GPU NVIDIA/драйверы не установлены")
    except subprocess.TimeoutExpired:
        print("  [НЕТ] nvidia-smi не ответил вовремя")
    return 0


def postroit_parser():
    parser = argparse.ArgumentParser(prog="prepare", description="Подготовка данных для нейросети")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser("prepare", help="обработать каталог с данными")
    p_prepare.add_argument("--path", default=None, help="каталог с исходными файлами")
    p_prepare.add_argument("--ext", nargs="+", default=["csv", "png"], help="расширения файлов, например csv png")
    p_prepare.add_argument("--output", default=None, help="путь к итоговому .npy файлу")
    p_prepare.set_defaults(func=komanda_prepare)

    p_doctor = subparsers.add_parser("doctor", help="проверить окружение")
    p_doctor.set_defaults(func=komanda_doctor)

    return parser


def main():
    parser = postroit_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
