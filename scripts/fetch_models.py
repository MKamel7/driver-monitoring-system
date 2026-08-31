"""Download the two model checkpoints this project needs.

The checkpoints are 68 MiB together and are published as assets on the
GitHub Release rather than tracked in git, so cloning the repository does
not cost 68 MiB of blobs that most readers never run.

    python scripts/fetch_models.py

Every file is verified against a SHA-256 recorded here before it is accepted,
and an existing file that already matches is left alone. Nothing is
overwritten silently: a file that exists with the wrong hash is reported and
kept, so a half-finished download or a hand-edited checkpoint is visible
rather than replaced behind your back.
"""

import argparse
import hashlib
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

RELEASE = "v1.0.0"
BASE_URL = f"https://github.com/MKamel7/driver-monitoring-system/releases/download/{RELEASE}"

# name -> (sha256, size in bytes)
MODELS = {
    "eye_predictor.dat": (
        "9b2b9fccb3253c8c3a164f9d9a668fd1a5afa7d7042450ed4824ef3d3eb3518a",
        19060214,
    ),
    "yolov8m_custom.pt": (
        "8e9fb1ff56f1b939b6b3b61f9e83430d02f2c83e03a96012a40d586a00e45791",
        51993632,
    ),
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MiB"


def fetch(name: str, expected: str, size: int, dest_dir: Path) -> bool:
    """Return True if dest_dir/name ends up present and correct."""
    dest = dest_dir / name

    if dest.exists():
        actual = sha256(dest)
        if actual == expected:
            print(f"  {name}: already present and verified")
            return True
        print(f"  {name}: EXISTS BUT DOES NOT MATCH", file=sys.stderr)
        print(f"    expected {expected}", file=sys.stderr)
        print(f"    found    {actual}", file=sys.stderr)
        print("    Leaving it alone. Delete it and re-run if you want a fresh copy.",
              file=sys.stderr)
        return False

    url = f"{BASE_URL}/{name}"
    print(f"  {name}: downloading {human(size)} from {url}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Download to a temporary file in the same directory, verify, then move.
    # A partial download is never left behind under the real name.
    handle, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=f".{name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with open(handle, "wb") as out, urllib.request.urlopen(url) as response:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        print(f"    download failed: {exc}", file=sys.stderr)
        return False
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        print(f"    CHECKSUM MISMATCH, discarded. expected {expected}, got {actual}",
              file=sys.stderr)
        return False

    tmp.replace(dest)
    print(f"    verified {expected[:16]}...")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help=f"where to put the checkpoints (default: {MODELS_DIR})")
    args = parser.parse_args()

    print(f"Fetching model checkpoints into {args.models_dir}")
    ok = [fetch(name, digest, size, args.models_dir)
          for name, (digest, size) in MODELS.items()]

    if all(ok):
        print("All checkpoints present and verified.")
        return 0

    print(f"{ok.count(False)} of {len(ok)} checkpoints are missing or wrong.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
