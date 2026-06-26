"""Batch processing utilities for parallel form processing."""

import concurrent.futures
import logging
import time
from functools import partial

logger = logging.getLogger(__name__)


def _wrap_job_call(func, *args, **kwargs):
    """Execute a callable, logging and returning errors instead of raising."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.exception("Job call failed: %s", e)
        return {"error": str(e), "fields": [], "confidence": 0.0}


def _worker(config_name, config_path, image_path, output_dir, log_dir,
            dictionaries_dir, dpi, original_filename, job_id, api_key):
    """Worker function for multiprocessing. Lazily imports process_form."""
    from pipeline import process_form
    return _wrap_job_call(
        process_form,
        image_path=image_path,
        config_name=config_name,
        config_path=config_path,
        output_dir=output_dir,
        log_dir=log_dir,
        dictionaries_dir=dictionaries_dir,
        dpi=dpi,
        original_filename=original_filename,
        job_id=job_id,
        api_key=api_key,
    )


def _process_batch_parallel_inner(jobs: dict[str, dict], max_workers: int = 3) -> dict[str, dict]:
    """Process a batch of forms in parallel using ProcessPoolExecutor.
    
    Uses concurrent.futures.ProcessPoolExecutor with submit() + as_completed()
    so results are properly collected and exceptions propagate back to the caller.
    
    Args:
        jobs: dict mapping job_id -> kwargs dict for _worker
        max_workers: number of parallel workers (default 3, controlled by batch settings)
    
    Returns:
        dict mapping job_id -> result dict
    """
    results: dict[str, dict] = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(_worker, **kwargs): job_id
            for job_id, kwargs in jobs.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_job):
            job_id = future_to_job[future]
            try:
                results[job_id] = future.result()
            except Exception as e:
                logger.exception("Unhandled exception in batch job %s", job_id)
                results[job_id] = {"error": str(e), "fields": [], "confidence": 0.0}
    
    return results


def _process_batch_sequential(jobs: dict[str, dict]) -> dict[str, dict]:
    """Process a batch of forms sequentially (fallback for non-parallel contexts)."""
    results: dict[str, dict] = {}
    for job_id, kwargs in jobs.items():
        results[job_id] = _worker(**kwargs)
    return results
