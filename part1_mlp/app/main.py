"""Консольное меню для работы с нейросетью: создать, обучить, предсказать,
сохранить/загрузить веса, посмотреть график ошибки."""
from .exceptions import NeuralNetworkError
from .manager import DatasetManager
from .models import NeuralNetwork

nejroset = None  # сеть создаётся позже, через пункт меню 1
menedzher = DatasetManager()
X_train = X_test = y_train = y_test = None


def menu():
    print()
    print("1. Создать сеть")
    print("2. Загрузить данные из CSV")
    print("3. Обучить сеть")
    print("4. Сделать предсказание")
    print("5. Сохранить/загрузить веса")
    print("6. Показать график ошибки")
    print("7. Выход")


def punkt_sozdat_set():
    global nejroset
    sloi_str = input("Размеры слоёв через запятую (например 2,4,1): ")
    razmery = [int(x.strip()) for x in sloi_str.split(",")]
    aktivacia = input("Функция активации (sigmoid/relu) [sigmoid]: ").strip() or "sigmoid"
    nejroset = NeuralNetwork(razmery, aktivacia=aktivacia)
    print(f"Сеть создана: слои {razmery}, активация {aktivacia}")


def punkt_zagruzit_dannye():
    global X_train, X_test, y_train, y_test
    put = input("Путь к CSV-файлу: ").strip()
    menedzher.load_csv(put)
    sposob = input("Нормализация (standard/minmax) [standard]: ").strip() or "standard"
    menedzher.normalize(sposob)
    dolya_test = input("Доля тестовой выборки [0.2]: ").strip()
    dolya_test = float(dolya_test) if dolya_test else 0.2
    X_train, X_test, y_train, y_test = menedzher.train_test_split(dolya_test)
    print(f"Загружено {menedzher.X.shape[0]} строк, {menedzher.X.shape[1]} признаков. "
          f"Train: {X_train.shape[0]}, test: {X_test.shape[0]}")


def punkt_obuchit():
    if nejroset is None:
        print("Сначала создайте сеть (пункт 1).")
        return
    if X_train is None:
        print("Сначала загрузите данные (пункт 2).")
        return
    skorost = float(input("Скорость обучения (learning rate) [0.1]: ").strip() or 0.1)
    epohi = int(input("Число эпох [500]: ").strip() or 500)
    batch_str = input("Размер батча (пусто - весь датасет сразу): ").strip()
    razmer_batcha = int(batch_str) if batch_str else None
    print("Обучение...")
    nejroset.train(X_train, y_train, epohi, skorost, razmer_batcha, pechat_kazhdye=max(1, epohi // 10))
    print(f"Готово. Ошибка на последней эпохе: {nejroset.istoria_oshibok[-1]:.6f}")


def punkt_predskazat():
    if nejroset is None:
        print("Сначала создайте сеть (пункт 1).")
        return
    if X_test is not None:
        print(f"Предсказание на {X_test.shape[0]} тестовых примерах (первые 5):")
        predskazano = nejroset.predict(X_test)
        for i in range(min(5, len(predskazano))):
            print(f"  пример {i}: предсказано={predskazano[i]}, факт={y_test[i]}")
    else:
        print("Тестовых данных нет, предсказание пропущено.")


def punkt_vesa():
    if nejroset is None:
        print("Сначала создайте сеть (пункт 1).")
        return
    deystvie = input("Сохранить (s) или загрузить (l) веса? ").strip().lower()
    put = input("Путь к файлу весов [weights.json]: ").strip() or "weights.json"
    if deystvie == "s":
        nejroset.save_weights(put)
        print(f"Веса сохранены в {put}")
    elif deystvie == "l":
        nejroset.load_weights(put)
        print(f"Веса загружены из {put}")
    else:
        print("Неизвестное действие.")


def punkt_grafik():
    if nejroset is None or not nejroset.istoria_oshibok:
        print("Сеть ещё не обучалась - графика нет.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен, график пропущен (это необязательный пункт).")
        return
    put = input("Куда сохранить график [loss.png]: ").strip() or "loss.png"
    plt.figure()
    plt.plot(nejroset.istoria_oshibok)
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка (MSE)")
    plt.title("График ошибки обучения")
    plt.savefig(put)
    plt.close()
    print(f"График сохранён в {put}")


DEYSTVIA = {
    "1": punkt_sozdat_set,
    "2": punkt_zagruzit_dannye,
    "3": punkt_obuchit,
    "4": punkt_predskazat,
    "5": punkt_vesa,
    "6": punkt_grafik,
}


def main():
    print("=== Учебная нейросеть (MLP) ===")
    while True:
        menu()
        vybor = input("Выберите пункт: ").strip()
        if vybor == "7":
            print("Выход.")
            break
        deystvie = DEYSTVIA.get(vybor)
        if deystvie is None:
            print("Нет такого пункта меню, попробуйте ещё раз.")
            continue
        try:
            deystvie()
        except NeuralNetworkError as e:
            print(f"Ошибка: {e}")
        except (ValueError, OSError) as e:
            print(f"Ошибка ввода: {e}")


if __name__ == "__main__":
    main()
