"""DAPE pipeline entrypoint — thin re-export module for backward compatibility."""
import logging

from pipeline import (  # noqa: F401
    _load_config,
    _load_template_extraction,
    _to_png_bytes,
    _safe_crop,
    _deskew_image,
    _validation_ok,
    _is_checkbox,
    _differs_from_template,
    MIN_DESKEW_ANGLE_DEG,
    process_form,
)

from batching import (  # noqa: F401
    _wrap_job_call,
    _worker,
    _process_batch_parallel_inner,
    _process_batch_sequential,
)

from cli import (  # noqa: F401
    parse_args,
    main,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
