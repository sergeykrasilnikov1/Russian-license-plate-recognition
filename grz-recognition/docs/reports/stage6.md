# Этап 6 — Финальный inference-пайплайн

> **Обновление 27.09.2026:** этот отчёт фиксирует исторический ONNX baseline.
> После leakage-aware сравнения production-конфиг переключён на YOLO-pose +
> plate-specific CRNN (`engine: reference_pose_crnn`), который достиг 56.52%
> e2e exact против 24.77% у bbox+PP-OCRv5. Старый путь доступен только через
> `--prefer-onnx`; актуальные результаты — в `docs/ocr_selection_progress.md`.

- **Дата:** 2026-09-14
- **Статус:** CLI и CSV-контракт готовы; веса OCR и детектора локальные ONNX; e2e замер на CPU
- **Где выполнялось:** локально CPU (AMD Ryzen 7 5700U); экспорт детектора — Kaggle Tesla P100
- **Окружение:** onnxruntime CPU, OpenCV. Нет CUDA / нет ultralytics на локальной машине

## 1. Цель этапа

Точка входа жюри: каталог `.jpg`/`.png` → CSV `image;plate_num;plate_type;confidence`
без сети, без хардкода ответов, с замером времени относительно 100 мс.

## 2. Что сделано

### 2.1 CLI

`scripts/run_inference.py`:

- `--input` / `GRZ_INPUT_DIR`, `--output` / `GRZ_OUTPUT_CSV`
- 0 / 1 / N знаков → 0 / 1 / N строк; пустой кадр **не** попадает в CSV
- UTF-8, `;`, заголовок фиксирован
- печать `mean_ms` и `fits` против `latency_budget_ms: 100`
- `--from-meta` — только отладка без детектора
- `--prefer-onnx` — `weights/detector.onnx`

### 2.2 Offline-веса (никаких скачиваний в runtime)

OCR больше не идёт через RapidOCR (тот тянул modelscope при init). Rec-only
onnxruntime + CTC:

- `weights/ocr/en_PP-OCRv4_rec_mobile.onnx` (active)
- `weights/ocr/en_PP-OCRv5_rec_mobile.onnx` (для сравнения)

Детектор: экспорт с Kaggle (`sergeniy/grz-export-detector-onnx`, COMPLETE) —
копия `.pt` в writable FS, `YOLO.export(onnx, imgsz=640, opset=12)` →
`weights/detector.onnx` (10.1 МБ, выход `(1, 8, 8400)` = 4 box + 4 класса).

Тест `tests/test_stage6.py`: нет `import requests/urllib/httpx` в runtime-модулях;
нет lookup по именам файлов; пустые детекции не пишут строки.

### 2.3 Замер latency (доступное железо)

| Режим | n img | mean ms | ≤ 100 мс |
|-------|------:|--------:|----------|
| `--from-meta` warp+OCR | 80 (warmup 2) | **26.4** | да |
| detect ONNX 640 + warp + OCR | 40 (warmup 2) | **145.3** | **нет** (CPU) |
| только детектор ONNX 640 | 14 | ~104 | — |
| warp+OCR после детекции | 14 | ~44 | — |
| YOLO.predict P100 (Этап 4) | 12 | ~15 после прогрева | детектор на GPU |

ONNX собран со **статическим** 640×640 — уменьшить imgsz без переэкспорта нельзя.
INT8 / TensorRT на этой машине нет. На референсе 1050 Ti бюджет, скорее всего,
держится за счёт GPU-детектора (ориентир P100 ~15 мс), но это **не** заменено
на i5-7600 + 1050 Ti.

### 2.4 Тесты

`pytest tests/test_stage6.py` (+ stage1/5): 28 passed в общем прогоне stage1+5+6.

## 3. Команды проверки

```bash
python -m pytest tests/test_stage6.py -q
python scripts/run_inference.py --input dataset/images/synthetic --limit 40 \
  --output logs/e2e.csv --prefer-onnx
# debug без детектора:
python scripts/run_inference.py --input dataset/images/synthetic --from-meta dataset/meta.csv \
  --limit 80 --no-vehicle-filter --output logs/meta.csv
```

## 4. Ограничения

- Чекпоинт детектора по-прежнему 34/80, заглушка.
- Full-plate OCR ~46 % без fine-tune.
- CPU e2e 145 мс > 100 мс; уложиться на CPU без квантования/меньшего ONNX нельзя.
- PARSeq по-прежнему без весов (выбор OCR только v4 vs v5).
- Domain gap type1a/type1b не пересчитывался.

## 5. Дальше (Этап 7)

Датащит, валидатор, пояснительная записка ≤ 5 страниц. Fine-tune OCR и дообучение
детектора — не часть Этапа 6.
