"""Generate tiny deterministic PNGs for the sample fixture.

Run this once to populate data/samples/images/. The resulting files are
checked in so the smoke test works with no network access.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


HERE = Path(__file__).parent
IMG_DIR = HERE / "images"


def _make_image(name: str, label: str, color: tuple[int, int, int]) -> Path:
    IMG_DIR.mkdir(exist_ok=True)
    img = Image.new("RGB", (160, 96), color)
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, 152, 88], outline=(0, 0, 0), width=2)
    d.text((16, 40), label, fill=(0, 0, 0))
    path = IMG_DIR / name
    img.save(path)
    return path


def main() -> None:
    _make_image("login.png", "Login", (220, 230, 250))
    _make_image("submit.png", "Submit", (220, 250, 220))
    _make_image("diagram.png", "H2O", (250, 250, 220))


if __name__ == "__main__":
    main()
