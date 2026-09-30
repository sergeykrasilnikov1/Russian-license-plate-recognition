# GRZ Recognition · Volga IT

Офлайн-распознавание обычных, квадратных и жёлтых российских автомобильных
номеров. На вход подаётся каталог изображений, на выходе формируется CSV
с текстом номера, типом знака и уверенностью.

Код проекта и веса моделей находятся в [`grz-recognition/`](grz-recognition/).

## Запуск

Python 3.10–3.12, Linux или Windows с WSL2:

```bash
git clone --depth 1 https://github.com/sergeykrasilnikov1/volgaIT.git
cd volgaIT/grz-recognition
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-runtime.txt
python scripts/run_inference.py --input /path/to/images --output result.csv
```

[Полная инструкция](grz-recognition/README.md) ·
[Описание датасета](grz-recognition/dataset/README.md) ·
[Пояснительная записка](grz-recognition/docs/explanatory_note.md)

[Архив датасета](https://github.com/sergeykrasilnikov1/volgaIT/releases/tag/v1.0.0)
доступен в релизе.
