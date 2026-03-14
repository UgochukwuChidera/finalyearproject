"""
TIFF Page Joining & Manipulation
Covers:
  - Merge multiple TIFFs into one multi-page TIFF
  - Append pages to existing TIFF
  - Extract specific pages
  - Delete specific pages
  - Mix TIFFs and other image formats (JPG, PNG) into one TIFF
  - Convert PDF pages into a multi-page TIFF
  - Split a multi-page TIFF into individual files
"""

from PIL import Image
from pathlib import Path


# ================================================================
# 1. MERGE MULTIPLE TIFF FILES INTO ONE MULTI-PAGE TIFF
# ================================================================

def merge_tiffs(input_files: list, output_path: str):
    """
    Joins multiple TIFF files (or any image files) into a single
    multi-page TIFF. Equivalent to merging PDFs.

    input_files : list of file paths (can mix .tif, .jpg, .png etc.)
    output_path : where to save the merged TIFF
    """
    pages = []

    for file in input_files:
        img = Image.open(file)
        # Handle multi-page TIFFs inside the input list too
        try:
            while True:
                # Convert to RGB to ensure consistency
                pages.append(img.copy().convert('RGB'))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

    if not pages:
        raise ValueError("No pages found in input files.")

    # Save: first page saved normally, rest appended
    first = pages[0]
    rest  = pages[1:]

    first.save(
        output_path,
        format='TIFF',
        save_all=True,          # enable multi-page
        append_images=rest,     # all other pages
        compression='tiff_lzw' # lossless LZW compression
    )
    print(f"Merged {len(pages)} pages → {output_path}")


# ================================================================
# 2. APPEND PAGES TO AN EXISTING MULTI-PAGE TIFF
# ================================================================

def append_to_tiff(existing_tiff: str, new_files: list, output_path: str):
    """
    Loads an existing multi-page TIFF, then appends additional
    pages from new_files. Saves result to output_path.
    """
    pages = []

    # Load existing TIFF pages first
    existing = Image.open(existing_tiff)
    try:
        while True:
            pages.append(existing.copy().convert('RGB'))
            existing.seek(existing.tell() + 1)
    except EOFError:
        pass

    # Append new files
    for file in new_files:
        img = Image.open(file)
        try:
            while True:
                pages.append(img.copy().convert('RGB'))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

    first = pages[0]
    rest  = pages[1:]
    first.save(
        output_path,
        format='TIFF',
        save_all=True,
        append_images=rest,
        compression='tiff_lzw'
    )
    print(f"Appended pages. Total: {len(pages)} → {output_path}")


# ================================================================
# 3. EXTRACT SPECIFIC PAGES FROM A MULTI-PAGE TIFF
# ================================================================

def extract_pages(input_tiff: str, page_indices: list, output_path: str):
    """
    Extracts only the specified pages (0-indexed) from a TIFF.

    e.g. extract_pages('scan.tif', [0, 2, 4], 'extracted.tif')
    extracts pages 1, 3, and 5
    """
    img = Image.open(input_tiff)
    selected = []

    frame = 0
    try:
        while True:
            if frame in page_indices:
                selected.append(img.copy().convert('RGB'))
            frame += 1
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    if not selected:
        raise ValueError("No pages matched the given indices.")

    first = selected[0]
    rest  = selected[1:]
    first.save(
        output_path,
        format='TIFF',
        save_all=True,
        append_images=rest,
        compression='tiff_lzw'
    )
    print(f"Extracted pages {page_indices} → {output_path}")


# ================================================================
# 4. DELETE SPECIFIC PAGES FROM A MULTI-PAGE TIFF
# ================================================================

def delete_pages(input_tiff: str, page_indices: list, output_path: str):
    """
    Removes specific pages (0-indexed) from a TIFF and saves the rest.

    e.g. delete_pages('scan.tif', [2, 5], 'cleaned.tif')
    removes pages 3 and 6
    """
    img = Image.open(input_tiff)
    kept = []

    frame = 0
    try:
        while True:
            if frame not in page_indices:
                kept.append(img.copy().convert('RGB'))
            frame += 1
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    if not kept:
        raise ValueError("All pages were deleted — nothing to save.")

    first = kept[0]
    rest  = kept[1:]
    first.save(
        output_path,
        format='TIFF',
        save_all=True,
        append_images=rest,
        compression='tiff_lzw'
    )
    print(f"Deleted pages {page_indices}. Remaining: {len(kept)} → {output_path}")


# ================================================================
# 5. SPLIT MULTI-PAGE TIFF INTO INDIVIDUAL FILES
# ================================================================

def split_tiff(input_tiff: str, output_folder: str, prefix: str = 'page'):
    """
    Splits a multi-page TIFF into individual single-page TIFF files.
    Saves them as page_001.tif, page_002.tif, etc.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    img = Image.open(input_tiff)

    frame = 0
    try:
        while True:
            out_path = f"{output_folder}/{prefix}_{frame + 1:03d}.tif"
            page = img.copy().convert('RGB')
            page.save(out_path, format='TIFF', compression='tiff_lzw')
            print(f"  Saved: {out_path}")
            frame += 1
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    print(f"Split complete. {frame} pages → {output_folder}/")


# ================================================================
# 6. CONVERT PDF TO MULTI-PAGE TIFF
# ================================================================

def pdf_to_tiff(pdf_path: str, output_path: str, dpi: int = 300):
    """
    Converts a PDF into a single multi-page TIFF.
    Useful for archiving or feeding into TIFF-based pipelines.
    """
    from pdf2image import convert_from_path

    print(f"Converting PDF to TIFF at {dpi} DPI...")
    pages = convert_from_path(pdf_path, dpi=dpi)
    pages_rgb = [p.convert('RGB') for p in pages]

    first = pages_rgb[0]
    rest  = pages_rgb[1:]
    first.save(
        output_path,
        format='TIFF',
        save_all=True,
        append_images=rest,
        compression='tiff_lzw'
    )
    print(f"PDF converted: {len(pages)} pages → {output_path}")


# ================================================================
# 7. COUNT PAGES IN A TIFF
# ================================================================

def count_pages(tiff_path: str) -> int:
    """Returns the number of pages/frames in a TIFF file."""
    img = Image.open(tiff_path)
    count = 0
    try:
        while True:
            count += 1
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    print(f"{tiff_path} has {count} page(s)")
    return count


# ================================================================
# QUICK COMPARISON: TIFF vs PDF operations
# ================================================================
"""
OPERATION          PDF (pypdf)                    TIFF (Pillow)
-----------        ----------                     -------------
Merge              PdfWriter + add_page()         save(save_all=True, append_images=[...])
Split              PdfWriter per page             img.seek(frame), save individually
Delete pages       skip index in loop             skip index in loop
Extract pages      select index in loop           select index in loop
Count pages        len(PdfReader.pages)           seek until EOFError
Compression        built-in                       tiff_lzw (lossless)
Quality loss       none (for scanned PDFs)        none (LZW is lossless)
"""


# ================================================================
# USAGE EXAMPLES
# ================================================================
if __name__ == '__main__':

    # Merge two TIFFs (or mix with JPG/PNG)
    merge_tiffs(
        ['scan0001.tif', 'scan0002.tif'],
        'merged.tif'
    )

    # Append more pages to existing TIFF
    append_to_tiff('merged.tif', ['extra_page.jpg'], 'merged_v2.tif')

    # Extract only pages 1, 3, 5 (0-indexed: 0, 2, 4)
    extract_pages('merged.tif', [0, 2, 4], 'odd_pages.tif')

    # Delete pages 3 and 6 (0-indexed: 2, 5)
    delete_pages('merged.tif', [2, 5], 'cleaned.tif')

    # Split into individual files
    split_tiff('merged.tif', './split_pages/', prefix='form')

    # Convert PDF to TIFF
    pdf_to_tiff('scan.pdf', 'scan_archive.tif', dpi=300)

    # Count pages
    count_pages('merged.tif')
