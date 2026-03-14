"""
Adaptable Document Processing Workflow
Supports: PDF, multi-page TIFF, single JPEG/PNG/BMP/TIFF
Automatically detects format, slices pages, and extracts text via OCR
"""

import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path


def get_pages(file_path: str) -> list:
    """
    Accepts any supported file format and returns a list of PIL Image objects,
    one per page. Works for:
      - PDF          (multi-page)
      - TIFF / TIF   (multi-page supported)
      - JPEG / JPG   (single page only)
      - PNG          (single page only)
      - BMP          (single page only)
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    pages = []

    if ext == '.pdf':
        # PDF — convert each page to image at 300 DPI
        print(f"[PDF] Converting pages at 300 DPI...")
        images = convert_from_path(file_path, dpi=300)
        pages = images
        print(f"  → {len(pages)} page(s) found")

    elif ext in ['.tif', '.tiff']:
        # TIFF — may be multi-page, iterate through frames
        print(f"[TIFF] Reading frames...")
        img = Image.open(file_path)
        try:
            while True:
                pages.append(img.copy())
                img.seek(img.tell() + 1)  # move to next frame
        except EOFError:
            pass  # end of frames
        print(f"  → {len(pages)} page(s) found")

    elif ext in ['.jpg', '.jpeg']:
        # JPEG — always single page
        print(f"[JPEG] Single page format...")
        pages = [Image.open(file_path)]
        print(f"  → 1 page found")

    elif ext in ['.png', '.bmp']:
        # PNG/BMP — always single page
        print(f"[{ext.upper().strip('.')}] Single page format...")
        pages = [Image.open(file_path)]
        print(f"  → 1 page found")

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return pages


def ocr_pages(pages: list) -> list:
    """
    Runs OCR on each page image and returns a list of extracted text strings.
    """
    results = []
    for i, page in enumerate(pages):
        print(f"  OCR processing page {i + 1}/{len(pages)}...")
        # Convert to grayscale for better OCR accuracy
        gray = page.convert('L')
        text = pytesseract.image_to_string(gray)
        results.append({
            'page': i + 1,
            'text': text.strip()
        })
    return results


def group_into_sets(pages_text: list, pages_per_record: int = 3) -> list:
    """
    Groups extracted pages into logical record sets.
    e.g. if each student has 3 forms (Academic + Medical + Leave),
    group every 3 pages together as one student record.

    Adjust pages_per_record to match your document structure.
    """
    groups = []
    for i in range(0, len(pages_text), pages_per_record):
        chunk = pages_text[i:i + pages_per_record]
        groups.append({
            'record': (i // pages_per_record) + 1,
            'pages': [c['page'] for c in chunk],
            'combined_text': '\n\n--- PAGE BREAK ---\n\n'.join(
                [c['text'] for c in chunk]
            )
        })
    return groups


def process_document(file_path: str, pages_per_record: int = 3):
    """
    Main entry point. Pass any supported file and it will:
    1. Detect format
    2. Slice into pages
    3. OCR each page
    4. Group pages into logical records
    5. Print a summary
    """
    print(f"\n{'='*50}")
    print(f"Processing: {file_path}")
    print(f"{'='*50}")

    # Step 1 — get pages regardless of format
    pages = get_pages(file_path)

    # Step 2 — OCR all pages
    print(f"\nRunning OCR on {len(pages)} page(s)...")
    ocr_results = ocr_pages(pages)

    # Step 3 — group into logical records
    groups = group_into_sets(ocr_results, pages_per_record)

    # Step 4 — summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total pages   : {len(pages)}")
    print(f"Pages/record  : {pages_per_record}")
    print(f"Total records : {len(groups)}")

    for group in groups:
        print(f"\n--- Record {group['record']} (pages {group['pages']}) ---")
        # Print just first 300 chars as preview
        preview = group['combined_text'][:300].replace('\n', ' ')
        print(f"Preview: {preview}...")

    return groups


# ---------------------------------------------------------------
# USAGE EXAMPLES
# ---------------------------------------------------------------
if __name__ == '__main__':

    # Example 1 — PDF
    # records = process_document('scan.pdf', pages_per_record=3)

    # Example 2 — Multi-page TIFF
    # records = process_document('scan.tif', pages_per_record=3)

    # Example 3 — Single JPEG (1 page only)
    # records = process_document('form.jpg', pages_per_record=1)

    # Example 4 — Batch process a folder of mixed files
    folder = './forms'
    supported = ['.pdf', '.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']

    if os.path.exists(folder):
        for file in os.listdir(folder):
            if Path(file).suffix.lower() in supported:
                process_document(
                    os.path.join(folder, file),
                    pages_per_record=3
                )
    else:
        print("Drop your files in a 'forms/' folder and run this script.")
        print("Or call process_document('yourfile.pdf') directly.")
