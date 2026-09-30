#!/usr/bin/env python3
"""Validate and package the annotated dataset without caches or raw downloads."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_dataset import validate, format_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/dataset.zip')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    errors, warnings, stats = validate(dataset)
    report = format_report(errors, warnings, stats)
    (dataset / 'validation_report.txt').write_text(report, encoding='utf-8')
    if errors:
        print(report)
        return 1
    with (dataset / 'meta.csv').open(encoding='utf-8', newline='') as stream:
        images = {row['image'] for row in csv.DictReader(stream, delimiter=';')}
    files = {dataset / name for name in ('meta.csv', 'README.md', 'LICENSE', 'validation_report.txt')}
    for image in images:
        files.add(dataset / image)
        files.add(dataset / 'labels' / f'{Path(image).stem}.txt')
    files.update(path for path in (dataset / 'generator').rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts
                 and path.suffix not in {'.pyc', '.pyo'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for directory in ('images/real/', 'images/synthetic/', 'labels/', 'generator/'):
            archive.writestr('dataset/' + directory, '')
        for path in sorted(files):
            archive.write(path, 'dataset/' + path.relative_to(dataset).as_posix())
    print(f'Packaged {stats["images"]} images: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
