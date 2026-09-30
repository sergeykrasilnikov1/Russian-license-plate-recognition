from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from scripts.scrape_platesmania_square import (
    GalleryCard,
    ScrapeStats,
    parse_args,
    parse_gallery,
    photo_filename,
    process_card,
)


def _image_bytes(size: tuple[int, int], fmt: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=fmt)
    return buffer.getvalue()


class _FakeClient:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requested: list[str] = []

    def get(self, url: str):
        self.requested.append(url)
        return SimpleNamespace(content=self.payloads[url])


def test_parse_gallery_pairs_images_only_inside_same_card():
    html = """
    <img src="https://example.test/m/sidebar.jpg">
    <div class="panel panel-grey">
      <div class="panel-body">
        <div><a><img src="https://img.test/260920/m/123.jpg" width="460" height="350"></a></div>
        <div><a><img src="https://img.test/260920/inf/123.png" alt="к 074 ту 122"></a></div>
      </div>
    </div>
    <div class="panel panel-blue">
      <img src="https://img.test/260920/m/wrong.jpg">
      <img src="https://img.test/260920/inf/wrong.png">
    </div>
    """

    assert parse_gallery(html) == [
        GalleryCard(
            photo_url="https://img.test/260920/m/123.jpg",
            plate_preview_url="https://img.test/260920/inf/123.png",
            plate_text="к 074 ту 122",
        )
    ]


def test_parse_gallery_removes_repeated_photo_cards():
    card = """
    <div class="panel panel-grey">
      <img src="https://img.test/m/123.jpg">
      <img src="https://img.test/inf/123.png" alt="а 001 аа 01">
    </div>
    """

    assert len(parse_gallery(card + card + card)) == 1


def test_process_card_saves_photo_when_preview_height_is_80(tmp_path):
    card = GalleryCard("https://img.test/m/123.jpg", "https://img.test/inf/123.png", "а 001 аа 01")
    client = _FakeClient(
        {
            card.plate_preview_url: _image_bytes((160, 80), "PNG"),
            card.photo_url: _image_bytes((460, 350), "JPEG"),
        }
    )
    stats = ScrapeStats()

    process_card(card, client=client, output=tmp_path, target_height=80, dry_run=False, stats=stats)

    assert (tmp_path / "123.jpg").is_file()
    assert stats.square == 1
    assert stats.saved == 1
    assert client.requested == [card.plate_preview_url, card.photo_url]


def test_process_card_does_not_download_photo_for_other_height(tmp_path):
    card = GalleryCard("https://img.test/m/456.jpg", "https://img.test/inf/456.png")
    client = _FakeClient({card.plate_preview_url: _image_bytes((160, 79), "PNG")})
    stats = ScrapeStats()

    process_card(card, client=client, output=tmp_path, target_height=80, dry_run=False, stats=stats)

    assert list(tmp_path.iterdir()) == []
    assert stats.square == 0
    assert stats.saved == 0
    assert client.requested == [card.plate_preview_url]


def test_photo_filename_discards_path_traversal_and_query():
    assert photo_filename("https://img.test/m/../33779164.JPG?token=x") == "33779164.jpg"


def test_cli_accepts_several_saved_html_pages():
    args = parse_args(["--html-file", "pages/1.html", "pages/2.html"])

    assert [path.as_posix() for path in args.html_file] == ["pages/1.html", "pages/2.html"]
