# Этап 4 — Детекция и классификация типа знака (YOLOv11n)

- **Дата:** 2026-09-14
- **Статус:** код и сплит готовы локально; **обучение на GPU-сервере ещё не запускалось**
- **Где выполнялось:** локальная машина **без GPU** (dry-run / unit-тесты)
- **Окружение (local):** Python 3.10.12, pytest, PyYAML, OpenCV — **без** torch/ultralytics

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

### 2.2 Конфиг и профили latency

`configs/detector.yaml`:

| Профиль | imgsz | batch | epochs | Назначение |
|---------|-------|-------|--------|------------|
| `accurate` | 640 | 16 | 80 | качество |
| `balanced` | 512 | 20 | 70 | компромисс под 100 мс |
| `fast` | 416 | 24 | 60 | запас по latency на GTX 1050 Ti |

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

Метрики на сервере снимать отдельно по четырём спискам (см. §3).

### 2.5 CLI обучения и экспорта

| Скрипт | Локально | На сервере |
|--------|----------|------------|
| `train_detector.py` | `--dry-run` проверяет профиль, data.yaml, классы | реальное `YOLO.train` |
| `export_onnx.py` | `--dry-run` | `model.export(format=onnx)` → `weights/detector.onnx` |

### 2.6 Эвристика `is_vehicle`

`src/detection/vehicle.py` — классический CV (относительный размер bbox,
положение по Y, тёмная/текстурированная полоса под номером). Порог
`min_context_score` в `detector.yaml` / `pipeline.yaml`. Лёгкая CNN — только
если эвристика провалит валидацию на `is_vehicle=0` (сейчас 148 рядов `other`).

### 2.7 Тесты

`pytest tests/test_stage4.py` (+ регресс stage1): 18 passed.
Проверены: профили, holdout style_b, dry-run CLI, эвристика bumper vs poster.

## 3. Что выполняется на GPU-сервере (ещё не запускалось)

Точные команды: **`docs/server_training_guide.md`**.

Кратко:

```bash
pip install -r requirements-train.txt
python scripts/prepare_detector_split.py --seed 42 --val-ratio 0.2 --build-style-b 150
python scripts/train_detector.py --profile accurate --device 0 --seed 42
python scripts/export_onnx.py --weights runs/detect/grz_yolo11n/weights/best.pt --imgsz 640 --out weights/detector.onnx
python scripts/export_onnx.py --weights runs/detect/grz_yolo11n/weights/best.pt --imgsz 640 --half --out weights/detector_fp16.onnx
```

### 3.1 Метрики, которые нужно зафиксировать после обучения

| Подмножество | Ожидание |
|--------------|----------|
| val real type1 | эталон «нормального» качества |
| val syn type1a (тот же стиль) | завышенно высокая mAP — не показатель |
| val syn type1a style_b | ближе к честной оценке; просадка = domain gap |
| val real type1b (N≈35) | единственный реальный сигнал класса |

Также: mAP50 / mAP50-95 **по каждому из 4 классов** на общем val.

### 3.2 Latency (detector only)

Замер onnxruntime `CUDAExecutionProvider` vs `TensorrtExecutionProvider` (если
есть) на imgsz 640/512/416. fp16 на Pascal 1050 Ti — только по факту замера,
не по чужим бенчмаркам.

В этом отчёте ячейки метрик/latency: **TBD (сервер)**.

| Профиль | mAP50 | mAP50-95 | ms/img CUDA | ms/img TRT | fp16 delta |
|---------|-------|----------|-------------|------------|------------|
| accurate 640 | TBD | TBD | TBD | TBD | TBD |
| balanced 512 | TBD | TBD | TBD | TBD | TBD |
| fast 416 | TBD | TBD | TBD | TBD | TBD |

| Domain-gap slice | mAP50 | mAP50-95 |
|------------------|-------|----------|
| real type1 | TBD | TBD |
| syn type1a same style | TBD | TBD |
| syn type1a style_b | TBD | TBD |
| real type1b | TBD | TBD |

Версии на сервере (заполнить после прогона): CUDA __ / PyTorch __ / ultralytics __ / GPU __.

## 4. Ограничения

- Локально ultralytics/torch **не установлены** — ожидаемо.
- style_b — procedural alternate, не «другой реальный мир»; при просадке метрики
  честно писать domain gap в пояснительную записку.
- Эвристика vehicle не заменяет разметку `is_vehicle` и может ошибаться на
  крупных кропах пластин без бампера.

## 5. Следующий этап

Этап 5: OCR (warp + PP-OCRv4/v5 / PARSeq), `benchmark_latency.py` end-to-end.
