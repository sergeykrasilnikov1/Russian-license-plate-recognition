# Synthetic GRZ generator

Entry: `../../scripts/build_synthetic.py`

```bash
python scripts/build_synthetic.py --count 5000 --seed 42
```

Same seed → same images and `meta.csv` rows.

## What it is based on

Not drawn from scratch. Type 1 is
[maerty1/Russia-number-plate-generator](https://github.com/maerty1/Russia-number-plate-generator)
(`License_plate_in_Russia.svg`, `RoadNumbers2.0.ttf`, `FONT_SIZE=135`).
Type 1Б is that plate recoloured yellow. Type 1А fills `assets/type1a_blank.png`
(region frame + RUS/flag already on the blank). `other` uses the type-1 layout
in diplomatic red.

Isolated previews (white studio background): `references/preview_type1.png`,
`preview_type1a.png`, `preview_type1b.png`.

## Default mix

`type1=0.22, type1a=0.38, type1b=0.32, other=0.08` — type1a/type1b boosted
because Stage 2 could not fill those groups from open sources.
