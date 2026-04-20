"""
Adaptable document processing helpers.
Supports PDF and image formats by returning page images for downstream AI extraction.
"""

import os
from pathlib import Path

from PIL import Image
from pdf2image import convert_from_path


def get_pages(file_path: str) -> list:
    path = Path(file_path)
    ext = path.suffix.lower()
    pages = []

    if ext == ".pdf":
        pages = convert_from_path(file_path, dpi=300)
    elif ext in [".tif", ".tiff"]:
        img = Image.open(file_path)
        try:
            while True:
                pages.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        pages = [Image.open(file_path)]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return pages


def group_into_sets(pages_data: list, pages_per_record: int = 3) -> list:
    groups = []
    for i in range(0, len(pages_data), pages_per_record):
        chunk = pages_data[i : i + pages_per_record]
        groups.append({"record": (i // pages_per_record) + 1, "pages": chunk})
    return groups


def process_document(file_path: str, pages_per_record: int = 3):
    pages = get_pages(file_path)
    return group_into_sets(pages, pages_per_record)


if __name__ == "__main__":
    folder = "./forms"
    supported = [".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"]

    if os.path.exists(folder):
        for file in os.listdir(folder):
            if Path(file).suffix.lower() in supported:
                process_document(os.path.join(folder, file), pages_per_record=3)
