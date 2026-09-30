import csv

from scripts.audit_ocr_corpus import audit, split_key


def test_foreign_split_paths_relocate_without_basename_guess(tmp_path):
    assert split_key("/kaggle/input/data/images/real/a.jpg", tmp_path) == "images/real/a.jpg"
    assert split_key("./images/real/a.jpg", tmp_path) == "images/real/a.jpg"
    assert split_key("/unrelated/a.jpg", tmp_path) == "/unrelated/a.jpg"


def test_content_leakage_and_missing_target_labels(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "splits").mkdir()
    for name in ("a", "b"):
        (tmp_path / f"images/{name}.jpg").write_bytes(b"identical image content")
    (tmp_path / "splits/train.txt").write_text("images/a.jpg\n")
    (tmp_path / "splits/val.txt").write_text("images/b.jpg\n")
    with (tmp_path / "meta.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "plate_num", "plate_type", "is_synthetic"], delimiter=";")
        writer.writeheader()
        for name in ("a", "b"):
            writer.writerow(dict(image=f"images/{name}.jpg", plate_num="A123BC77", plate_type="type1", is_synthetic="0"))
    result = audit(tmp_path)
    assert result["train_val_path_overlap"] == []
    assert len(result["train_val_exact_content_overlap"]) == 1
    assert result["real_readable_val_rows"] == {"type1": 1}
    assert result["target_types_without_real_readable_val"] == ["type1a", "type1b"]
    assert not result["ready_for_full_target_comparison"]
