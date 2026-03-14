"""DPI-aware kernel scaling. All defaults calibrated for 300 DPI."""
_BASE = 300

def scale(n, dpi):
    s = max(1, round(n * dpi / _BASE))
    return s if s % 2 == 1 else s + 1

def kernel_sizes(dpi: int) -> dict:
    r = dpi / _BASE
    return {
        "skew_hough_threshold":  round(200 * r),
        "binarization_block":    scale(31, dpi),
        "illumination_blur":     scale(51, dpi),
        "clahe_tile":            max(1, round(8 * r)),
        "diff_threshold":        30,
        "morph_open":            scale(3, dpi),
        "morph_close":           scale(5, dpi),
        "morph_dilate":          scale(7, dpi),
        "min_region_area":       round(50 * r**2),
        "max_features":          round(1000 * r),
        "checkbox_max_area":     round(3000 * r**2),
        "textbox_min_area":      round(3000 * r**2),
        "line_min_width":        round(60 * r),
        "dpi":                   dpi,
        "scale_factor":          round(r, 4),
    }
