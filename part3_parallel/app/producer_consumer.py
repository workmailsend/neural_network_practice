"""Паттерн "производитель-потребитель" на потоках (threading):
один поток сканирует каталог и кладёт пути файлов в очередь, несколько
потоков читают файлы и превращают их в массивы numpy."""
import queue
import threading
from pathlib import Path

import numpy as np
from PIL import Image

RASSHIRENIA = {".png", ".jpg", ".jpeg", ".bmp"}
STOP_SIGNAL = None  # кладём в очередь, чтобы сказать потребителю "файлов больше нет"


def naiti_fayly_izobrazheniy(put_k_papke):
    papka = Path(put_k_papke)
    return sorted(f for f in papka.rglob("*") if f.suffix.lower() in RASSHIRENIA)


def proizvoditel(put_k_papke, ochered_putey, chislo_potrebiteley):
    """Поток-производитель: находит файлы и кладёт пути в очередь, в конце
    кладёт по одному стоп-сигналу на каждого потребителя."""
    fayly = naiti_fayly_izobrazheniy(put_k_papke)
    for put in fayly:
        ochered_putey.put(put)
    for _ in range(chislo_potrebiteley):
        ochered_putey.put(STOP_SIGNAL)
    return len(fayly)


def potrebitel(ochered_putey, ochered_massivov, schetchik, schetchik_lock, imya_potoka):
    """Поток-потребитель: читает файлы из очереди путей, превращает в
    массивы numpy и кладёт в очередь для аугментации. Счётчик
    обработанных файлов общий для всех потоков, поэтому защищён Lock."""
    while True:
        put = ochered_putey.get()
        if put is STOP_SIGNAL:
            ochered_putey.task_done()
            break
        try:
            with Image.open(put) as img:
                massiv = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            ochered_massivov.put((put.name, massiv))
            with schetchik_lock:
                schetchik["obrabotano"] += 1
        except (OSError, ValueError) as e:
            print(f"[{imya_potoka}] не удалось прочитать {put}: {e}")
        finally:
            ochered_putey.task_done()


def sobrat_massivy_cherez_potoki(put_k_papke, chislo_potrebiteley=3):
    """Запускает 1 поток-производитель + N потоков-потребителей, ждёт
    завершения всех и возвращает список (имя_файла, массив)."""
    ochered_putey = queue.Queue()
    ochered_massivov = queue.Queue()
    schetchik = {"obrabotano": 0}
    schetchik_lock = threading.Lock()

    potok_proizvoditel = threading.Thread(
        target=proizvoditel, args=(put_k_papke, ochered_putey, chislo_potrebiteley), name="producer"
    )
    potoki_potrebiteli = [
        threading.Thread(
            target=potrebitel,
            args=(ochered_putey, ochered_massivov, schetchik, schetchik_lock, f"consumer-{i}"),
            name=f"consumer-{i}",
        )
        for i in range(chislo_potrebiteley)
    ]

    potok_proizvoditel.start()
    for potok in potoki_potrebiteli:
        potok.start()

    # join() гарантирует, что мы дождёмся честного завершения всех
    # потоков и не останется "подвисших" фоновых задач
    potok_proizvoditel.join()
    for potok in potoki_potrebiteli:
        potok.join()

    rezultaty = []
    while not ochered_massivov.empty():
        rezultaty.append(ochered_massivov.get())

    return rezultaty, schetchik["obrabotano"]
