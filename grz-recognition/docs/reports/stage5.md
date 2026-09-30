# Этап 5 — OCR символов и сборка инференс-пайплайна

- **Дата:** 2026-09-14
- **Статус:** инференс-пайплайн собран и прогнан на CPU; детектор не дообучался
- **Где выполнялось:** локально CPU (AMD Ryzen 7 5700U, 16 потоков), Python 3.10.12
- **Окружение:** OpenCV, RapidOCR 3.x + onnxruntime 1.23 (CPU). **Нет** torch / ultralytics / CUDA
- **Детектор:** `weights/detector_best.pt` (34/80) — рабочая заглушка Этапа 4, не финальный артефакт

## 1. Цель этапа

Warp кропа по quad, сравнение трёх OCR-backend'ов с одним интерфейсом, ограничение
словаря ГРЗ, сборка CSV-пайплайна и честный замер latency относительно бюджета 100 мс.
Датасет и дообучение детектора **не** трогались. Формулировки domain gap в
`docs/explanatory_note.md` не переоценивались — только помечены как открытые риски Этапа 4.

## 2. Что реализовано локально (без GPU)

### 2.1 Perspective warp

`src/ocr/warp.py`: `order_quad` сортирует углы по `atan2` от центроида, приводит
кольцо к TL→TR→BR→BL, затем `cv2.getPerspectiveTransform` в прямоугольник
`geometry.warp_size(type)` (мм × 0.8).

Вырожденный quad (`area < 4`) → `DegenerateQuadError`.

type1a:

| Режим | Поведение |
|-------|-----------|
| `flatten` | отдельные warp верхней и нижней строки по субквадам, склейка в одну линию (CRNN) |
| `stacked` | один warp всего знака в 2-строчный прямоугольник |

На 8 синтетических type1a `choose_type1a_mode` выбрал **flatten** (один горизонтальный
чернильный пояс vs две полосы у stacked). В `configs/ocr.yaml`: `type1a_mode: flatten`.

Тесты (`tests/test_stage5.py`): clockwise / сдвиг точек, вырожденная линия, поворот
180°, зеркало, flatten vs stacked на синтетическом двухстрочнике.

### 2.2 OCR backends

Единый контракт `recognize(crop) -> (str, float)` в `src/ocr/backends/`:

| Имя конфига | Реализация | Веса |
|-------------|------------|------|
| `ppocrv4_mobile` | RapidOCR ONNX, `LangRec.EN`, PP-OCRv4 mobile | `en_PP-OCRv4_rec_mobile.onnx` |
| `ppocrv5_mobile` | то же, PP-OCRv5 mobile | `en_PP-OCRv5_rec_mobile.onnx` |
| `parseq` | onnxruntime, вход 128×32 | `weights/ocr/parseq.onnx` (**нет файла**) |

Переключение — поле `active` в `configs/ocr.yaml`, без правки кода.

**TODO:** PARSeq не участвовал в сравнении (нет ONNX). Выбор `active` — только
v4 vs v5. Экспорт с патчем `bool → float` у causal mask отложен.

PaddlePaddle-стек не поднимался.

### 2.3 Словарь ГРЗ

`configs/ocr_dict.txt` и `charset: "0123456789ABEKMHOPCTYX#"`.
Маска — `src/utils/plate_mask.py`. Порог символа `min_char_conf: 0.35` → `#`.

Урезать словарь **внутри** RapidOCR rec-head без дообучения нельзя: индексы CTC
зашиты в ONNX. В конфиге `dict_truncation: postfilter_only` и TODO на fine-tune
головы на следующем этапе.

Агрегат уверенности: `min(det_score, min(char_ocr))` (`confidence_mode: det_ocr_min`).

### 2.4 Пайплайн и CLI

`src/pipeline/infer.py` + `scripts/run_inference.py`:

детектор → `is_vehicle` → warp по типу → OCR → CSV `image;plate_num;plate_type;confidence` (UTF-8, `;`).

Локально `.pt` не исполняется (нет ultralytics). `weights/detector.onnx` не
экспортирован (Этап 4: read-only FS на Kaggle). Для CPU-отладки:
`--from-meta dataset/meta.csv` (GT quad). Жюри-путь — ONNX/`.pt` на машине с runtime.

`--from-meta --limit 40 --no-vehicle-filter` на синтетике: 40 строк, заголовок CSV
корректный. После смены `active` на v4 rec-веса — `en_PP-OCRv4_rec_mobile.onnx`.

### 2.5 Тесты

`pytest tests/test_stage5.py`: 13 passed (плюс регресс stage1/stage4).

## 3. Бенчмарк OCR-кандидатов (факт)

Первый прогон `--n 80` дал 60 кропов: v5 38.3% vs v4 33.3% full-plate при
одинаковых **15.9** мс (один знак после запятой). Этого мало: Wilson 95% CI при
n=60 для p≈0.38 — порядка ±12 п.п., интервалы перекрываются.

Повтор: `python3 scripts/benchmark_latency.py --n 500 --warmup 3 --seed 42`
(страты type1 / type1a / type1b / other, только readable GT, skipped_warp=0).

Sanity rec-тензора на одном кропе (не decode-текст):

| | graph | shape | sha256_16 |
|--|-------|-------|-----------|
| v4 | `PaddlePaddle Graph 0` | `(1, 40, 97)` | `3d31de266afb0578` |
| v5 | `PaddlePaddle Graph in PIR mode` | `(1, 40, 438)` | `f30a3a0833b8538c` |

Тензоры **разные** (в т.ч. размер словаря 97 vs 438). Совпадение 15.9 мс на n=60 —
округление и шум малой выборки, не одна сессия.

| backend | n | acc@char | acc@full-plate | full 95% Wilson CI | mean ms | median ms | p95 ms | GPU |
|---------|--:|---------:|---------------:|-------------------:|--------:|----------:|-------:|-----|
| **ppocrv4_mobile** | 500 | 0.739 | **0.460** | **[0.417, 0.504]** | 25.73 | 23.88 | 43.60 | н/д |
| ppocrv5_mobile | 500 | **0.795** | 0.376 | [0.335, 0.419] | 26.85 | 22.97 | 57.05 | н/д |
| parseq | — | — | — | — | — | — | — | нет `parseq.onnx` |

На n=500 знак full-plate **сменился**: v4 лучше по exact match, CI почти не
перекрываются. v5 по-прежнему лучше посимвольно. Latency одного порядка, не
идентична. `active: ppocrv4_mobile` — целевая метрика номера это полный номер.

**TODO:** PARSeq не участвовал в сравнении; финальный выбор сделан только между
v4 и v5. Экспорт `weights/ocr/parseq.onnx` (bool→float mask) — отдельная итерация.

## 4. End-to-end latency

| Режим | железо | ms/img | ≤ 100 мс? |
|-------|--------|-------:|-----------|
| warp + OCR, n=500 mean (v4) | Ryzen 7 5700U CPU | **25.7** | да для OCR-части |
| warp + OCR, n=80 subset | тот же CPU | 35.3 mean / 28.9 median | да для OCR-части |
| detect + warp + OCR | локальный CPU | **не измерен** | нет ultralytics / нет `detector.onnx` |
| detect only (Этап 4) | Tesla P100 `YOLO.predict` | ~15 после прогрева | детектор один в бюджете на P100 |

Экспорт ONNX и e2e на одном железе — не сейчас.

Локальный CSV-путь: `run_inference.py --from-meta dataset/meta.csv --limit 40
--no-vehicle-filter` → 40 строк, заголовок `image;plate_num;plate_type;confidence`.

## 5. Ограничения

- Чекпоинт детектора 34/80 — отладочная заглушка.
- Открытые риски type1a/type1b **не** пересчитывались (пояснительная записка §6).
- RapidOCR при init грузит det/cls ONNX; в `recognize` они выключены.
- PARSeq в таблице — честный пробел; **не** сравнивался наравне с v4/v5.
- n=60 недостаточно для выбора v4 vs v5 по full-plate; ориентир — n=500.

## 6. Следующее (не сейчас)

- Экспорт `detector.onnx` и полный e2e на CPU и GPU.
- Fine-tune rec / экспорт PARSeq ONNX, урезание словаря в голове.
- Вернуться к дообучению детектора и формулировкам domain gap.
