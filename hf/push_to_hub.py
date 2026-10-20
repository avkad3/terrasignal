"""Export the trained models and publish them to the Hugging Face Hub.

Log in first (once): `hf auth login` with a *write* token from
https://huggingface.co/settings/tokens

  python hf/push_to_hub.py --repo <your-hf-username>/terrasignal-water-unet
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

from terrasignal.model import export_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="user/name on the Hugging Face Hub")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("Not logged in. Run `hf auth login` with a write token first.", file=sys.stderr)
        return 1
    print(f"logged in as {user}")

    out = ROOT / "hf_export"
    if out.exists():
        shutil.rmtree(out)
    export_checkpoint(ROOT / "checkpoints" / "water_unet.pt", out)
    export_checkpoint(ROOT / "checkpoints" / "water_unet_gee.pt", out / "gee_baseline")
    card = (ROOT / "hf" / "MODEL_CARD.md").read_text(encoding="utf-8").replace("REPO_ID", args.repo)
    (out / "README.md").write_text(card, encoding="utf-8")

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    info = api.upload_folder(repo_id=args.repo, folder_path=out, commit_message="Upload TerraSignal water U-Net")
    print(f"uploaded: https://huggingface.co/{args.repo}  ({info.oid[:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
