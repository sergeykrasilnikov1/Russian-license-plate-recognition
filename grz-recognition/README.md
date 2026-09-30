# GRZ Recognition

Офлайн-распознавание российских автомобильных номеров для полуфинала
«Искусственный интеллект и анализ данных» Volga IT.

Проект находит номерные знаки на изображениях, определяет их тип и распознаёт
текст. Поддерживаются обычные номера (`type1`), квадратные двухстрочные
(`type1a`), жёлтые (`type1b`) и прочие знаки (`other`). Результат сохраняется
в CSV. Во время распознавания доступ к интернету не требуется.

## Быстрый старт

Python **3.10–3.12**, Linux или Windows с WSL2. Для GPU-инференса используется
NVIDIA с установленным драйвером; при отсутствии CUDA выбирается CPU.

Из корня репозитория:

```bash
cd grz-recognition
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-runtime.txt
python scripts/run_inference.py --input /path/to/images --output result.csv
```

Зависимости устанавливаются один раз. Веса моделей входят в репозиторий;
инференс не загружает модели и не обращается к внешним API.

Входной каталог может содержать `.jpg`, `.jpeg` и `.png`, включая вложенные
каталоги. Имена файлов должны быть уникальными. Повреждённые изображения
пропускаются с сообщением в журнале.

Пути также можно передать через переменные окружения:

```bash
export GRZ_INPUT_DIR=/path/to/images
export GRZ_OUTPUT_CSV=result.csv
python scripts/run_inference.py
```

## Выходной формат

CSV: UTF-8, разделитель `;`, заголовок в первой строке.

```csv
image;plate_num;plate_type;confidence
car_01.jpg;A123BC777;type1;0.9300
car_02.jpg;K555OP25;type1a;0.8800
car_03.jpg;H741C###;type1b;0.6200
```

`image` содержит только имя файла. Каждому найденному знаку соответствует
отдельная строка; если знаки не найдены, строк для изображения нет.
Нечитаемые символы обозначаются `#`. Уверенность находится в диапазоне `[0, 1]`.

## Как это работает

1. YOLO-pose находит номер и четыре угла.
2. Контекстный фильтр проверяет расположение знака на транспортном средстве.
3. Перспективное преобразование выравнивает номер; квадратные номера читаются
   по строкам.
4. CRNN распознаёт текст, а геометрия, цвет и маска номера уточняют тип.

Основные настройки находятся в `configs/pipeline.yaml`. Альтернативный
ONNX-пайплайн запускается так:

```bash
python scripts/run_inference.py --input /path/to/images --output result.csv --prefer-onnx
```

## Проверка скорости

Замер выбранного пайплайна на своих изображениях, без эталонной разметки:

```bash
python scripts/benchmark_latency.py --latency-only --input /path/to/images --n 200 --warmup 3
```

Команда выводит среднее время, медиану и p95 в мс/изображение.
Чтение с диска и первые изображения для прогрева исключаются из замера.
Для ONNX-варианта добавьте `--prefer-onnx`.

## Датасет и генератор

Разметка хранится в `dataset/meta.csv`: одна строка на знак, разделитель `;`.
Поля: `image`, `plate_num`, `plate_type`, `bbox`, `quad`, `is_vehicle`,
`is_synthetic`, `source`, `license`, `conditions`.

Фотографии и метки доступны в
[архиве датасета](https://github.com/sergeykrasilnikov1/volgaIT/releases/tag/v1.0.0).
Источники и методика описаны в [даташите](dataset/README.md).

Для работы с данными и запуска тестов:

```bash
pip install -r requirements-dev.txt
python scripts/build_synthetic.py --out dataset --count 5000 --seed 42
python scripts/validate_dataset.py --dataset dataset
python -m pytest tests -q
```

Синтетика воспроизводится по seed. Генератор и его ресурсы находятся
в `dataset/generator/`. Локальный валидатор сохраняет результат
в `dataset/validation_report.txt`.
Архив для передачи создаётся командой:

```bash
python scripts/package_dataset.py --output dist/dataset.zip
```

## Структура проекта

```text
configs/       настройки детектора, OCR и пайплайна
src/           детекция, OCR, обработка и сбор данных
scripts/       команды запуска, проверки и генерации
weights/       локальные веса моделей
dataset/      метаданные, генератор и даташит
docs/         пояснительная записка и технические отчёты
tests/        автоматические проверки
```

## Источники и лицензии

Данные описаны в `dataset/README.md`, лицензия датасета — CC BY 4.0.
Основной runtime использует код и веса
[Lin-Lini/volga-it-2026-lpr](https://github.com/Lin-Lini/volga-it-2026-lpr).
Версия, происхождение весов и MIT License сохранены в
`src/vendor/volga_it_2026_lpr/`. Компонент Ultralytics распространяется
по AGPL-3.0. Атрибуция ресурсов генератора находится рядом с ресурсами.
