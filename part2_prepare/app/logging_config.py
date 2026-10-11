"""Настройка логирования: вывод в консоль и, по желанию, в файл."""
import logging
from pathlib import Path

from .config import UROVEN_LOGOV


def nastroit_logging(uroven=None, fayl_loga=None):
    """Настраивает логирование в консоль и (опционально) в файл.
    Возвращает готовый logger с именем 'prepare'."""
    uroven = uroven or UROVEN_LOGOV
    obrabotchiki = [logging.StreamHandler()]
    if fayl_loga:
        Path(fayl_loga).parent.mkdir(parents=True, exist_ok=True)
        obrabotchiki.append(logging.FileHandler(fayl_loga, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, uroven.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=obrabotchiki,
        force=True,
    )
    return logging.getLogger("prepare")
