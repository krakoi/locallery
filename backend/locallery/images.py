"""Read-only image processing with the installed cjpegli encoder."""

import io
import tempfile
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

from .cancellation import run_command

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".tif", ".tiff"}
Image.MAX_IMAGE_PIXELS = 134201344


def make_preview(
    source: Path, asset_id: str, storage: Path, check_running=lambda: None
):
    directory = storage / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / (asset_id + ".jpg")
    check_running()
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=directory) as temporary_dir:
        return _make_preview(source, cache, Path(temporary_dir), check_running)


def _make_preview(source, cache, directory, check_running):
    temporary, encoded = directory / "input.png", directory / "output.jpg"
    try:
        with Image.open(source) as original:
            original.seek(0)
            image = ImageOps.exif_transpose(original)
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            if profile := original.info.get("icc_profile"):
                alpha = (
                    image.convert("RGBA").getchannel("A")
                    if "A" in image.getbands()
                    else None
                )
                image = ImageCms.profileToProfile(
                    image,
                    ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                    ImageCms.createProfile("sRGB"),
                    outputMode="RGB",
                )
                if alpha is not None:
                    image.putalpha(alpha)
            rgba = image.convert("RGBA")
            background = Image.new("RGB", image.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            background.save(temporary, format="PNG")
            width, height = image.size
        check_running()
        result = run_command(
            ["cjpegli", str(temporary), str(encoded), "--quality=90"],
            timeout=120,
            check_running=check_running,
        )
        if result.returncode:
            raise ValueError(
                f"cjpegli failed: {result.stderr.decode(errors='replace').strip()}"
            )
        check_running()
        encoded.replace(cache)
        return str(cache), width, height
    finally:
        temporary.unlink(missing_ok=True)
