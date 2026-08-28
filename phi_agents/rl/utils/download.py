#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import errors, repo_exists

import phi_agents.utils.file_utils as fu
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


def locate_hf_cli() -> str | None:
    bin_dir = Path(sys.executable).resolve().parent  # .../env/bin
    cli_binary = "huggingface-cli"
    cli = bin_dir / cli_binary

    if cli.exists() and cli.is_file():
        logger.debug(f"Found {cli_binary} at {cli}")
        return str(cli)

    logger.warning(f"{cli_binary} not found at {cli}! Is `huggingface-hub` installed?")
    logger.info(f"Fallback to {cli_binary} in PATH ({shutil.which('huggingface-cli')=})")
    return cli_binary


def is_offline() -> bool:
    return (
        os.getenv("HF_HUB_OFFLINE") == "1"
        or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    )


# Resolve specific HF exceptions dynamically to avoid import errors across versions
_HF_EXPECTED_EXCEPTIONS: tuple[type[BaseException], ...] = (OSError,)
for _exc_name in ("LocalEntryNotFoundError", "RepositoryNotFoundError", "HFValidationError", "LocalTokenNotFoundError"):
    _exc = getattr(errors, _exc_name, None)
    if _exc is not None:
        _HF_EXPECTED_EXCEPTIONS += (_exc,)


def download_model(
    name_or_path: str, hf_args: list[str] | None = None, base_dir: Path | None = None
) -> str:
    if fu.exists(name_or_path):
        return name_or_path

    offline = is_offline()

    name_or_path_parts = Path(name_or_path).parts
    base_dir = base_dir or Path.cwd()
    if len(name_or_path_parts) >= 2:
        dst_name = (
            base_dir / ".model_cache" / name_or_path_parts[-2] / name_or_path_parts[-1]
        ).as_posix()
    else:
        dst_name = (base_dir / ".model_cache" / name_or_path_parts[-1]).as_posix()

    if fu.exists(dst_name):
        logger.info(f"Model already exists at local cache: {dst_name}")
        return dst_name

    # If offline mode is enabled, try to locate the model in the Hugging Face local cache.
    if offline:
        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(name_or_path, local_files_only=True)
            logger.info(f"Offline mode: located cached model for {name_or_path} at {local_dir}")
            return local_dir
        except Exception as e:
            logger.warning(f"Offline mode: could not locate cached model for {name_or_path} via local_files_only: {e}")

    if safe_hf_repo_exists(name_or_path):
        cmd = [
            locate_hf_cli(),
            "download",
            name_or_path,
            "--local-dir",
            dst_name,
            "--exclude=*consolidated*",  # Unnecessary consoldated files contain duplicate weights to the safetensor files used by hf
        ]
        if hf_args:
            cmd += hf_args

        print(" ".join(cmd))
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        subprocess.check_call(cmd)
        name_or_path = dst_name

    elif fu.get_scheme(name_or_path) != "file":
        print(f"Downloading {name_or_path} to {dst_name}")
        if not fu.exists(dst_name):
            fu.copy(name_or_path, dst_name)
        name_or_path = dst_name
    else:
        assert fu.exists(name_or_path), f"Could not figure out how to download model {name_or_path}"

    return name_or_path


def distributed_download_models(model_name_or_path: str, local_rank: int) -> str:
    paths: list[str | None] = []
    if local_rank == 0:
        paths.append(download_model(model_name_or_path))
    else:
        paths = [None]

    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(paths, src=0)

    assert isinstance(paths[0], str)
    return paths[0]


def download_adapter(adapter_path: Path) -> Path:
    if fu.get_scheme(adapter_path) == "s3":
        temp_dir = tempfile.mkdtemp()
        fu.copy(adapter_path / "*", temp_dir)
        adapter_path = Path(temp_dir)
    else:
        assert fu.exists(adapter_path)

    return adapter_path


def safe_hf_repo_exists(repo_id: str) -> bool:
    if is_offline():
        return False
    try:
        return repo_exists(repo_id)  # type: ignore # (hf's lib doesn't have type stubs)
    except _HF_EXPECTED_EXCEPTIONS as e:
        logger.debug(f"HF repo check skipped: {e}")
    except Exception as e:
        logger.warning(f"HF repo check failed unexpectedly: {e}")
    return False
