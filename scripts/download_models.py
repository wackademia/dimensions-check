"""Download fixed model revisions and verify SHA-256; safe to re-run or use offline."""

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("lygitdata/garmentiq", "5f02016e9ad3a4aa171fa9199423a437170f5afe",
     "tiny_vit_inditex_finetuned.pt", "",
     "354518859d576e13623cd61224d3b6bdc1bf3c18318cab2a0d0b55f6e0ca67d9"),
    ("lygitdata/garmentiq", "5f02016e9ad3a4aa171fa9199423a437170f5afe",
     "hrnet.pth", "",
     "5b29ada40632cb5ce1aaa38e4896054329c42d0f6c6649a5b9d0b53e41ee04f6"),
    ("lygitdata/BiRefNet_garmentiq_backup", "5f1a10e72d46e2f596e010ca8a0b6896b0ed1d03",
     "model.safetensors", "birefnet",
     "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"),
)


def matches(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


def ensure_models(check_only=False):
    for repo, revision, filename, subdir, checksum in MODELS:
        folder = ROOT / "models" / subdir
        path = folder / filename
        if not matches(path, checksum):
            if check_only:
                raise RuntimeError(f"Missing or damaged model: {path}. Run scripts/download_models.py")
            from huggingface_hub import hf_hub_download
            print(f"Downloading {repo}/{filename} (partial downloads can be resumed)...", flush=True)
            hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                            local_dir=folder, force_download=path.exists())
            if not matches(path, checksum):
                raise RuntimeError(f"SHA-256 mismatch: {path}. Re-run this script to repair it.")
        print(f"SHA-256 OK: {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify existing files without network access")
    args = parser.parse_args()
    ensure_models(args.check_only)
