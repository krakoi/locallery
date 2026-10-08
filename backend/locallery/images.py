"""Read-only image processing with the installed cjpegli encoder."""

import io
import subprocess
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".tif", ".tiff"}
Image.MAX_IMAGE_PIXELS = 134201344


def make_preview(source: Path, asset_id: str, storage: Path):
    directory = storage / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    cache, temporary = directory / (asset_id + ".jpg"), directory / (asset_id + ".png")
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
        result = subprocess.run(
            ["cjpegli", str(temporary), str(cache), "--quality=90"],
            capture_output=True,
            timeout=120,
        )
        if result.returncode:
            raise ValueError(
                f"cjpegli failed: {result.stderr.decode(errors='replace').strip()}"
            )
        return str(cache), width, height
    finally:
        temporary.unlink(missing_ok=True)
