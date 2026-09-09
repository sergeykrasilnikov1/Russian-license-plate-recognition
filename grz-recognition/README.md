# GRZ Recognition — нестандартные российские ГРЗ

Offline-пайплайн для полуфинала олимпиады **АИС Город**: детекция и распознавание
государственных регистрационных знаков типов **type1**, **type1a**, **type1b**
и отсев **other**.

## Две среды: локальная (CPU) и обучающая (GPU)

Проект намеренно разделён на два окружения, зависимости не смешиваются:

| Файл | Где ставится | Что покрывает |
|------|--------------|---------------|
| `requirements-dev.txt` | локальная машина **без GPU** | сбор данных, генерация синтетики, фильтры (mediapipe/imagehash), тесты |
| `requirements-train.txt` | отдельный **GPU-сервер** | torch, ultralytics, paddleocr, onnx/onnxruntime-gpu — обучение, экспорт, бенчмарк, инференс |

Отсутствие `torch`/`paddlepaddle` на локальной машине — это ожидаемое состояние, а не
проблема: Этапы 2–3 (данные) полностью CPU-only, Этапы 4–6 выполняются на сервере.

### Локальная установка (Этапы 2–3, тесты)

```bash
cd grz-recognition
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Проверка каркаса:

```bash
python -c "from src.utils.plate_mask import validate_plate; print(validate_plate('A123BC777'))"
python scripts/validate_dataset.py --dataset dataset
python -m pytest tests -q
```

## Обучение выполняется на GPU-сервере

Локально код Этапов 4–6 только пишется и проверяется логически (импорты, CLI,
чистые функции, mock-прогоны). Реальное обучение, экспорт в ONNX, бенчмарк
latency и инференс запускаются на сервере с NVIDIA GPU.

Установка на сервере:

```bash
git clone <repo> && cd grz-recognition
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r requirements-train.txt
# paddlepaddle: выбрать сборку под CUDA сервера, см. комментарий в requirements-train.txt
```

Команды на сервере:

```bash
# Этап 3 (если синтетику генерируем там же)
python scripts/build_synthetic.py --out dataset --count 6000 --seed 42

# Этап 4: обучение детектора + экспорт в ONNX
python scripts/train_detector.py --config configs/detector.yaml --data configs/data.yaml

# Этап 5: дообучение OCR
python scripts/train_ocr.py --config configs/ocr.yaml --crops-dir dataset/crops

# Этап 5/6: замер latency (бюджет 100 мс/изображение)
python scripts/benchmark_latency.py --input dataset/images/real --n 200 --config configs/pipeline.yaml

# Этап 6: финальный инференс
python scripts/run_inference.py --input /path/to/images --output result.csv
```

## Правило артефактов

Все генерируемые данные — скачанные датасеты, синтетика, кропы, веса, логи,
отчёты валидатора — пишутся **только внутрь `grz-recognition/`**. Ничего не
создаётся в `/tmp` или вне репозитория; логи идут в `logs/`, кэш загрузок в
`.cache/` (оба в `.gitignore`). После каждого завершённого этапа делается
git-коммит, чтобы прогресс не терялся.

## Требования

- Python 3.10+
- Linux / Windows 11 (WSL2)
- NVIDIA GPU — только для обучения; инференс — ONNX Runtime (CUDA или CPU)
- Референсный бюджет latency: **≤ 100 мс / изображение** на GTX 1050 Ti

## Запуск инференса (после Этапа 6)

```bash
python scripts/run_inference.py --input /path/to/images --output result.csv
# или:
export GRZ_INPUT_DIR=/path/to/images
export GRZ_OUTPUT_CSV=result.csv
python scripts/run_inference.py
```

Выходной CSV: `image;plate_num;plate_type;confidence` (UTF-8, `;`).

## Структура

См. дерево в `agent_prompt` / задание полуфинала. Ключевые точки входа:

| Скрипт | Назначение |
|--------|------------|
| `scripts/scrape_real_data.py` | Автосбор реальных фото (`--probe-only`, `--dry-run`, `--plate-types`) |
| `scripts/build_synthetic.py` | Синтетика ≥ 5000, `--seed` |
| `scripts/train_detector.py` | YOLOv11n |
| `scripts/train_ocr.py` | Дообучение OCR |
| `scripts/benchmark_latency.py` | Замер ≤ 100 мс |
| `scripts/run_inference.py` | Финальный offline inference |
| `scripts/validate_dataset.py` | Валидация `dataset/` |

## Этапы реализации

По каждому завершённому этапу в `docs/reports/stageN.md` лежит подробный отчёт:
что сделано, какими командами протестировано, какой был вывод, что осталось.

| Этап | Тема | Отчёт |
|------|------|-------|
| 1 | Структура и окружение | [stage1.md](docs/reports/stage1.md) |
| 2 | Автосбор реальных данных | [stage2.md](docs/reports/stage2.md) |
| 3 | Генератор синтетики | — |
| 4 | Детекция (YOLO) + тип как класс | — |
| 5 | OCR + benchmark | — |
| 6 | Inference CLI | — |
| 7 | Документация и валидатор | — |

## Лицензия данных

Датасет сдаётся под **CC BY 4.0** (`dataset/LICENSE`). Все источники — с лицензией,
допускающей некоммерческое использование, либо собственная синтетика.
