import os
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 min timeout, avoid timeout after large file download
from huggingface_hub import snapshot_download


DATASETS = ["CEPDOF", "HABBOF", "FishEye8k"]

HF_REPOS = {
    "FishEye8k": "Voxel51/fisheye8k",
}

MANUAL_URL = "https://vip.bu.edu"

MANUAL_DATASETS = ["CEPDOF", "HABBOF"]

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def download_manual(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  This dataset cannot be downloaded from HuggingFace.")
    print(f"  Please download it manually from:")
    print(f"    {MANUAL_URL}")
    print(f"  After downloading, place the files in:")
    print(f"    datasets/{name}/")
    print(f"{'='*60}\n")


def download_from_hf(name: str, repo_id: str):
    target_dir = os.path.join("datasets", name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Downloading {name} from HuggingFace...")
    print(f"  Repo: {repo_id}")
    print(f"{'='*60}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=target_dir,
                resume_download=True,
                max_workers=4,
            )
            print(f"  ✓ {name} downloaded successfully to datasets/{name}/\n")
            return
        except Exception as e:
            print(f"  ✗ Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ✗ Failed to download {name} after {MAX_RETRIES} attempts.\n")


def main():
    print("=" * 60)
    print("  Spatial Distortion Aware Network - Dataset Downloader")
    print("=" * 60)

    for name in DATASETS:
        if name in MANUAL_DATASETS:
            download_manual(name)
        elif name in HF_REPOS:
            download_from_hf(name, HF_REPOS[name])

    print("\nDone! Summary:")
    print(f"  - CEPDOF, HABBOF: download manually from {MANUAL_URL}")
    print(f"  - FishEye8k: should now be available in datasets/")


if __name__ == "__main__":
    main()
