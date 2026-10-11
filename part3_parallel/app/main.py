"""Точка входа части 3: сканируем каталог потоками (producer-consumer),
затем сравниваем последовательную и многопроцессорную аугментацию."""
import argparse

from .performance import sravnit_proizvoditelnost
from .producer_consumer import sobrat_massivy_cherez_potoki


def main():
    parser = argparse.ArgumentParser(description="Многопоточное чтение + многопроцессорная аугментация")
    parser.add_argument("--path", default="../data/input/part2_sample/images", help="каталог с изображениями")
    parser.add_argument("--consumers", type=int, default=3, help="число потоков-потребителей")
    parser.add_argument("--processes", type=int, default=4, help="число процессов для аугментации")
    parser.add_argument("--report", default="performance_report.json", help="куда сохранить отчёт о скорости")
    args = parser.parse_args()

    print(f"Сканирование {args.path} ({args.consumers} потоков-потребителей)...")
    rezultaty, obrabotano = sobrat_massivy_cherez_potoki(args.path, args.consumers)
    print(f"Потоками обработано файлов: {obrabotano} (прочитано массивов: {len(rezultaty)})")

    if not rezultaty:
        print("Нет изображений для аугментации - проверьте --path.")
        return 1

    print(f"Сравнение производительности на {len(rezultaty)} изображениях, {args.processes} процессов...")
    otchet = sravnit_proizvoditelnost(rezultaty, args.processes, args.report)
    print()
    print("=== Отчёт о производительности ===")
    for klyuch, znachenie in otchet.items():
        print(f"  {klyuch}: {znachenie}")
    print(f"Сохранено в {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
