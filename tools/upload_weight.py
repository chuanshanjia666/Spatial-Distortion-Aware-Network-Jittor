#!/usr/bin/env python3
"""
Upload trained model checkpoints from ``OUTPUT_DIR`` to ModelScope Hub.

The model is uploaded to ``chuanshanjia666/SDA-Net`` under the path::

    $BACKEND-$TRAIN_DATASETS/model_file

where ``BACKEND`` and ``TRAIN_DATASETS`` come from ``config.py``.

Usage:
    python tools/upload_weight.py

Requirements:
    pip install modelscope

Authentication:
    Set environment variable ``MODELSCOPE_API_TOKEN``, or provide ``--token``.
    Get your token at: https://www.modelscope.cn/my/myaccesstoken
"""

import os
import sys
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BACKEND, TRAIN_DATASETS, OUTPUT_DIR


def build_repo_path():
    """Build the remote directory path: e.g. ``jittor-habbof[train]``."""
    datasets_str = "+".join(TRAIN_DATASETS)           # e.g. "habbof[train]" or "cepdof[train]+wepdtof[train]"
    datasets_str = datasets_str.replace("[", "-").replace("]", "")  # clean brackets for URL safety
    return f"{BACKEND}-{datasets_str}"                # e.g. "jittor-habbof-train"


def find_model_files(output_dir, backend):
    """Find files matching backend checkpoint extension."""
    ext = "pkl" if backend == "jittor" else "pth"
    pattern = os.path.join(output_dir, f"*.{ext}")
    files = sorted(glob.glob(pattern))
    # prefer latest.* first, then sorted by name
    latest_files = [f for f in files if os.path.basename(f).startswith("latest.")]
    iter_files = [f for f in files if not os.path.basename(f).startswith("latest.")]
    return latest_files + iter_files


def upload_via_cli(model_dir, repo_id, revision, token=None):
    """
    Upload files via ModelScope CLI (fallback, more robust for large files).
    Files must be placed in a temp dir matching the revision structure.
    """
    import tempfile
    import shutil
    import subprocess

    # ModelScope CLI uploads the whole directory as a revision.
    # repo_path becomes the revision (e.g. "jittor-habbof-train").
    cmd = [
        "modelscope", "model", "-act", "upload",
        "-gid", repo_id.split("/")[0],
        "-mid", repo_id.split("/")[1],
        "-md", model_dir,
        "-vt", revision,
        "-vi", f"SDANet trained with {BACKEND} backend on {', '.join(TRAIN_DATASETS)}",
    ]
    if token:
        cmd.extend(["-tk", token])

    print(f"[CLI] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"ModelScope CLI upload failed with code {result.returncode}")
    print("[CLI] Upload completed successfully.")


def _upload_single(api, repo_id, local_path, remote_path, retries=5, initial_backoff=2.0):
    """Upload a single file with retry on 429 commit-lock conflict."""
    import time
    import random

    size_mb = os.path.getsize(local_path) / 1e6
    for attempt in range(retries):
        try:
            print(f"[SDK] Uploading {os.path.basename(local_path)} ({size_mb:.1f} MB) [{attempt+1}/{retries}] ...")
            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_type="model",
            )
            print(f"[SDK]   ✓ {os.path.basename(local_path)} done.")
            return
        except Exception as e:
            # Check for commit lock 429
            err_str = str(e)
            if "429" in err_str or "commit lock" in err_str or "10030000001" in err_str:
                wait = initial_backoff * (2 ** attempt) + random.uniform(0, 1)
                print(f"[SDK]   ⚠  commit lock busy, retrying in {wait:.1f}s ...")
                time.sleep(wait)
            else:
                print(f"[SDK]   ✗ {os.path.basename(local_path)} FAILED: {e}")
                raise
    # All retries exhausted
    raise RuntimeError(f"{os.path.basename(local_path)} failed after {retries} retries")


def upload_via_sdk(files, repo_id, revision, token=None, max_workers=8):
    """Upload files via ModelScope Python SDK with parallel threads."""
    from modelscope.hub.api import HubApi

    api = HubApi(token=token) if token else HubApi()

    tasks = []
    for local_path in files:
        remote_path = f"{revision}/{os.path.basename(local_path)}"
        tasks.append((local_path, remote_path))

    print(f"[SDK] Starting {max_workers} parallel upload threads for {len(tasks)} file(s) ...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_upload_single, api, repo_id, lp, rp): lp
            for lp, rp in tasks
        }
        failed = []
        for future in as_completed(futures):
            local_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[SDK]   ✗ {os.path.basename(local_path)} FAILED: {e}")
                failed.append(local_path)

    if failed:
        print(f"\n[SDK] {len(failed)} file(s) failed, see above.")
        raise RuntimeError(f"Upload failed for: {', '.join(os.path.basename(f) for f in failed)}")
    print(f"[SDK] All {len(files)} file(s) uploaded successfully.")


def main():
    parser = argparse.ArgumentParser(description="Upload SDANet checkpoints to ModelScope")
    parser.add_argument("--token", default=os.environ.get("MODELSCOPE_API_TOKEN"),
                        help="ModelScope access token (env: MODELSCOPE_API_TOKEN)")
    parser.add_argument("--repo", default="chuanshanjia666/SDA-Net",
                        help="ModelScope repo ID (default: chuanshanjia666/SDA-Net)")
    parser.add_argument("--mode", choices=["sdk", "cli"], default="sdk",
                        help="Upload method: sdk (parallel API, default) or cli")
    parser.add_argument("--latest-only", action="store_true",
                        help="Upload only the latest checkpoint")
    parser.add_argument("-w", "--workers", type=int, default=2,
                        help="Number of parallel upload threads (default: 2; keep low to avoid commit-lock conflict)")
    args = parser.parse_args()

    if not args.token:
        print("⚠  MODELSCOPE_API_TOKEN not set. Upload may fail.")
        print("   Get your token at: https://www.modelscope.cn/my/myaccesstoken")
        print("   Set it via: export MODELSCOPE_API_TOKEN='your-token'")
        print("   Or pass --token 'your-token'")

    revision = build_repo_path()
    print(f"[Config] BACKEND        = {BACKEND}")
    print(f"[Config] TRAIN_DATASETS = {TRAIN_DATASETS}")
    print(f"[Config] OUTPUT_DIR     = {OUTPUT_DIR}")
    print(f"[Upload] Remote repo    = {args.repo}")
    print(f"[Upload] Remote rev     = {revision}")

    files = find_model_files(OUTPUT_DIR, BACKEND)
    if not files:
        print(f"ERROR: No checkpoint files (*.{'pkl' if BACKEND == 'jittor' else 'pth'}) found in '{OUTPUT_DIR}'.")
        sys.exit(1)

    if args.latest_only:
        files = [f for f in files if os.path.basename(f).startswith("latest.")]
        if not files:
            print("ERROR: No latest checkpoint found in '{OUTPUT_DIR}' with --latest-only.")
            sys.exit(1)

    print(f"[Upload] {len(files)} file(s) to upload:")
    for f in files:
        print(f"         - {os.path.basename(f)} ({os.path.getsize(f)/1e6:.1f} MB)")

    if args.mode == "sdk":
        upload_via_sdk(files, args.repo, revision, args.token, max_workers=args.workers)
    else:
        # CLI mode: put selected files into a temp dir, then upload
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in files:
                shutil.copy2(f, os.path.join(tmpdir, os.path.basename(f)))
            upload_via_cli(tmpdir, args.repo, revision, args.token)

    print(f"\n✓ Done. Model available at: https://www.modelscope.cn/models/{args.repo}/summary")
    print(f"  Revision (subdirectory): {revision}")


if __name__ == "__main__":
    main()