"""Dataset download and SHA-256 verification utility for flumen."""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "MANIFEST.sha256"

# Default fallback URL for remote asset distribution (e.g. Zenodo or GitHub Releases)
DEFAULT_BASE_URL = os.environ.get(
    "FLUMEN_DATA_URL",
    "https://github.com/baltabaygal/flumen/releases/download/v3.0.0/"
)


def compute_sha256(filepath):
    """Compute hex SHA-256 digest of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path=MANIFEST_PATH, base_dir=DATA_DIR):
    """Verify all files declared in MANIFEST.sha256 against their expected checksums."""
    manifest_path = Path(manifest_path)
    base_dir = Path(base_dir)

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        return False

    all_ok = True
    checked = 0
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            expected_hash, rel_path = parts[0], parts[1].lstrip("*")
            target = base_dir / rel_path

            if not target.exists():
                print(f"MISSING: {rel_path}")
                all_ok = False
                continue

            actual_hash = compute_sha256(target)
            if actual_hash == expected_hash:
                print(f"OK: {rel_path}")
            else:
                print(f"FAILED: {rel_path} (expected {expected_hash}, got {actual_hash})")
                all_ok = False
            checked += 1

    print(f"\nManifest verification: {checked} files checked. Status: {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


def download_assets(base_url=DEFAULT_BASE_URL, manifest_path=MANIFEST_PATH, target_dir=DATA_DIR):
    """Download any missing assets listed in the manifest and verify their checksums."""
    target_dir = Path(target_dir)
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        return False

    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            expected_hash, rel_path = line.split(None, 1)
            rel_path = rel_path.lstrip("*")
            dest = target_dir / rel_path

            if dest.exists() and compute_sha256(dest) == expected_hash:
                print(f"Already verified: {rel_path}")
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            url = base_url.rstrip("/") + "/" + Path(rel_path).name
            print(f"Downloading {url} -> {dest} ...")
            try:
                urllib.request.urlretrieve(url, dest)
                actual_hash = compute_sha256(dest)
                if actual_hash != expected_hash:
                    print(f"Checksum mismatch for {dest}!", file=sys.stderr)
                    return False
                print(f"Verified: {rel_path}")
            except Exception as e:
                print(f"Failed to download {url}: {e}", file=sys.stderr)
                return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify or download flumen data assets.")
    parser.add_argument("--download", action="store_true", help="Download missing assets from remote release")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL for asset download")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to MANIFEST.sha256")
    args = parser.parse_args()

    if args.download:
        success = download_assets(base_url=args.base_url, manifest_path=args.manifest)
    else:
        success = verify_manifest(manifest_path=args.manifest)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
