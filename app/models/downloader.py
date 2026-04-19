"""
Model artifact downloader.

At startup, registry.py calls into this module to make sure required model
directories are present on disk. If a directory is already in `model_dir`,
we leave it alone (supports local dev and models bundled into the image
or HF Space). Otherwise we pull it from HuggingFace Hub.

Currently only the DistilBERT review classifier directory is sourced from
HF Hub; the single-file artifacts (.pkl, .pt) are expected to live in
`model_dir` directly (bundled via git-lfs on HF Spaces).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_hf_snapshot(
    repo_id: Optional[str],
    local_dir: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    """
    Download every file in `repo_id` into `local_dir` (idempotent — HF Hub
    skips files whose ETag matches what's already on disk). Returns the
    local directory path, or None if no repo_id was configured.
    """
    if not repo_id:
        return None

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed; cannot pull %s", repo_id)
        return None

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Snapshotting HF repo %s (rev=%s) -> %s", repo_id, revision or "main", local_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            revision=revision,
            token=token,
        )
    except Exception as exc:
        logger.error("HF snapshot failed for %s: %s", repo_id, exc)
        return None

    return local_dir
