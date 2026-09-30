# Этап 4 — Детекция и классификация типа знака (YOLOv11n)

- **Дата:** 2026-09-14
- **Статус:** train остановлен вручную на 34/80; domain-gap eval на `best.pt` выполнен (Kaggle eval v3 COMPLETE)
- **Где выполнялось:** Kaggle GPU (Tesla P100), локально — CPU dry-run / unit-тесты
- **Окружение (local):** Python 3.10.12, pytest, PyYAML, OpenCV — **без** torch/ultralytics
- **Окружение (Kaggle train):** Python 3.12.13, torch 2.3.1+cu118, numpy 2.0.2, ultralytics 8.3.40, Tesla P100

## 1. Цель этапа

Один детектор YOLOv11n с 4 классами = типами знака (`type1` / `type1a` / `type1b` /
`other`), стратифицированный train/val, экспорт ONNX, эвристика `is_vehicle`,
плюс честная проверка domain gap по type1a (100% синтетика на Этапе 3).

## 2. Что реализовано локально (без GPU)

### 2.1 Классы из одного реестра

`src/utils/geometry.py`: `CLASS_NAME_TO_ID` / `class_names()` строятся из порядка
ключей `PLATE_TYPES`. В `configs/detector.yaml` **нет** захардкоженного списка
имён — они попадают в `configs/data.yaml` только через
`scripts/prepare_detector_split.py`.

### 2.2 Конфиг обучения

Один режим: YOLOv11n, `imgsz=640`, `batch=16`, `epochs=80`. Отдельные профили 512/416
не нужны: nano на 640 и так укладывается в 100 мс на детекторе, а меньший вход режет
мелкие номера.

### 2.3 Сплит

`src/detection/split.py` + `scripts/prepare_detector_split.py`:

- стратификация по `(plate_type, is_synthetic)` на уровне **изображения**;
- seed=42, val_ratio=0.2;
- holdout `synthetic_generator_style_b` **никогда не попадает в train**.

Прогон локально:

```text
built style_b type1a images: 150 (seed=777)
train=5825  val=1606
```

Списки: `dataset/splits/{train,val,val_real_type1,val_syn_type1a,val_syn_type1a_style_b,val_real_type1b}.txt`.

### 2.4 Domain gap для type1a

Этап 3 дал **один** визуальный стиль синтетики. Для честной оценки добавлен
второй стиль (`--build-style-b`): ночь, меньший масштаб номера, чаще dirt/blur,
каталог `images/synthetic_style_b/`, `source=synthetic_generator_style_b`.

Метрики по четырём спискам — §3.1 (eval 2026-09-14).

### 2.5 CLI обучения и экспорта

| Скрипт | Локально | На сервере |
|--------|----------|------------|
| `train_detector.py` | `--dry-run` проверяет data.yaml, классы | реальное `YOLO.train` |
| `export_onnx.py` | `--dry-run` | `model.export(format=onnx)` → `weights/detector.onnx` |

### 2.6 Эвристика `is_vehicle`

`src/detection/vehicle.py` — классический CV (относительный размер bbox,
положение по Y, тёмная/текстурированная полоса под номером). Порог
`min_context_score` в `detector.yaml` / `pipeline.yaml`. Лёгкая CNN — только
если эвристика провалит валидацию на `is_vehicle=0` (сейчас 148 рядов `other`).

### 2.7 Тесты

`pytest tests/test_stage4.py` (+ регресс stage1): 18 passed.
Проверены: imgsz 640, holdout style_b, dry-run CLI, эвристика bumper vs poster.

## 3. GPU: обучение и eval

Веса: `weights/detector_best.pt`. Сырой JSON eval: `weights/eval_metrics.json`.
Kaggle: Tesla P100, torch 2.3.1+cu118, numpy 2.0.2, ultralytics 8.3.40.

### 3.0 Почему 34/80, а не полный прогон

Это **не** лимит сессии Kaggle (9–12 ч) и **не** краш. Сессия остановлена вручную
примерно через 50 мин (~90 с/эпоха с val). `patience=15` не сработал.

Кривая `runs/grz_accurate/results.csv` **ещё росла** к остановке, плато нет:

| epoch | mAP50 | mAP50-95 | train/cls_loss |
|------:|------:|---------:|---------------:|
| 1 | 0.884 | 0.745 | 1.86 |
| 15 | 0.956 | 0.876 | 0.451 |
| 22 | 0.964 | 0.883 | 0.401 |
| 30 | 0.968 | 0.892 | 0.385 |
| **34** | **0.969** | **0.893** | **0.362** |

`--resume` с `last.pt` мог бы ещё поднять качество; по запросу обучение **не**
перезапускалось. Eval ниже — на checkpoint эпохи 34.

### 3.1 Domain-gap (отдельный eval, без train)

`model.val()` на готовом `best.pt`, imgsz 640. На одноклассовых срезах Ultralytics
иногда копирует одно число во все `maps[]` — в таблице ниже для срезов указан
**агрегат среза** (это и есть mAP по картинкам этого списка).

| Срез | N img | mAP50 | mAP50-95 | Комментарий |
|------|------:|------:|---------:|-------------|
| val (все классы) | 1606 | 0.969 | 0.894 | то же, что в results.csv эпохи 34 |
| real type1 | 402 | **0.934** | **0.680** | эталон «реального» мира: IoU/локализация хуже |
| syn type1a same style | 384 | 0.995 | 0.995 | потолок на своей синтетике, не показатель |
| syn type1a style_b | 150 | 0.995 | 0.994 | holdout-стиль **не** дал просадки mAP50 |
| real type1b | **6** | 0.995 | 0.729 | слишком мало кадров; mAP50-95 шумный |

style_b не отделил type1a от train-синтетики (процедурный night/dirt всё ещё
тот же генератор). Реальный gap — **real type1 mAP50-95 0.68 vs 0.89 на смеси**.

### 3.2 mAP по классам на общем val

| Класс | AP50 | mAP50-95 |
|-------|-----:|---------:|
| type1 | 0.959 | 0.804 |
| type1a | 0.995 | 0.995 |
| type1b | 0.995 | 0.994 |
| other | 0.926 | 0.783 |

Агрегат 0.969 маскирует слабое место: **реальный type1 по строгой IoU (0.80 / на срезе 0.68)**, а type1a/type1b на val почти целиком синтетика.

### 3.3 Latency (detector)

Экспорт ONNX с пути `/kaggle/input/.../detector_best.pt` упал: read-only FS
(`detector_best.onnx` рядом с весами). TensorRT на P100 не снимался.

Ориентир по `YOLO.predict` на P100, imgsz 640 (smoke, 12 кадров): первый кадр
131 мс (прогрев), далее **13–16 мс/кадр**. Это не onnxruntime и не 1050 Ti, но
порядок величины: детектор один не вылезает за 100 мс на более сильном GPU.

| Режим | mAP50 | mAP50-95 | ms/img (P100, YOLO.predict, после прогрева) | ONNX CUDA | TRT | fp16 |
|-------|------:|---------:|-------------------------------------------:|-----------|-----|------|
| YOLOv11n 640 epoch 34 | 0.969 | 0.894 | ~15 | не экспортирован | — | — |

## 4. Ограничения

- Локально ultralytics/torch **не установлены** — ожидаемо.
- style_b — procedural alternate, не «другой реальный мир»; при просадке метрики
  честно писать domain gap в пояснительную записку.
- Эвристика vehicle не заменяет разметку `is_vehicle` и может ошибаться на
  крупных кропах пластин без бампера.

## 5. Следующий этап

Этап 5: OCR (warp + PP-OCRv4/v5 / PARSeq), `benchmark_latency.py` end-to-end.
