"""Image compression and optimization utilities for detention memo uploads."""

import io
import logging

from PIL import Image, UnidentifiedImageError
from django.core.files.uploadedfile import InMemoryUploadedFile
from rest_framework.exceptions import ValidationError

logger = logging.getLogger(__name__)


class InvalidImageUpload(ValidationError):
    """Raised when a file sent as a photo is not a readable image."""


def compress_image(
    uploaded_file,
    max_width: int = 1920,
    max_height: int = 1080,
    quality: int = 85,
    format: str = "JPEG",
) -> InMemoryUploadedFile:
    """
    Compress and resize an uploaded image.

    Args:
        uploaded_file: Django UploadedFile object
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        quality: JPEG quality (1-100)
        format: Output format (JPEG, PNG, WEBP)

    Returns:
        Compressed InMemoryUploadedFile

    Raises:
        InvalidImageUpload: the file is not an image (or is a decompression bomb).
        A non-image is never passed through as a "photo".
    """
    name = getattr(uploaded_file, "name", "") or "image"
    try:
        # Open the image
        img = Image.open(uploaded_file)
        img.load()

        # Convert RGBA to RGB if needed (for JPEG)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1])
            img = rgb_img
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize while maintaining aspect ratio
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Save to bytes buffer
        output = io.BytesIO()
        img.save(output, format=format, quality=quality, optimize=True)
        output.seek(0)
    except Image.DecompressionBombError as exc:
        logger.warning("Rejected oversized image upload %s: %s", name, exc)
        raise InvalidImageUpload({"image": f"{name} is too large to process."}) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        logger.warning("Rejected non-image upload %s: %s", name, exc)
        raise InvalidImageUpload({"image": f"{name} is not a valid image file."}) from exc

    # Create new uploaded file object
    filename = f"{name.rsplit('.', 1)[0]}.{format.lower()}"
    return InMemoryUploadedFile(
        file=output,
        field_name=getattr(uploaded_file, "field_name", None),
        name=filename,
        content_type=f"image/{format.lower()}",
        size=output.getbuffer().nbytes,
        charset=None,
    )
