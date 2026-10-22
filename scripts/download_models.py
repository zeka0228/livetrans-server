"""모델 가중치 내려받기 (저장소에는 넣지 않는다 — 권리 원칙 3, 출처·라이선스는 THIRD_PARTY_NOTICES.md).

사용:
  uv run python scripts/download_models.py            # STT 기본 세트 (DiCoW_v3_3 + faster-whisper large-v3)
  uv run python scripts/download_models.py --all      # 전부
  uv run python scripts/download_models.py --stt --kotoba --diar --enhance

Hugging Face 모델은 HF 캐시(~/.cache/huggingface)에, Zenodo 파일은 저장소의 models/ (gitignore) 에 받는다.
pyannote 는 Hugging Face 에서 이용 조건에 동의한 뒤 토큰이 필요하다 (--token 또는 HF_TOKEN, 또는 `hf auth login`).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

HF_MODELS = {
    "stt": [
        ("BUT-FIT/DiCoW_v3_3", "DiCoW v3.3 — 화자 조건부 Whisper (CC BY 4.0)"),
        ("Systran/faster-whisper-large-v3", "faster-whisper large-v3 — 폴백 STT (MIT)"),
    ],
    "kotoba": [
        (
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            "kotoba-whisper v2.0 — 일본어 대안 STT (MIT / Apache-2.0)",
        ),
    ],
    "diar": [
        (
            "pyannote/speaker-diarization-community-1",
            "pyannote 화자 분할 (CC BY 4.0, 이용 조건 동의 + 토큰 필요)",
        ),
    ],
}

ZENODO_FILES = {
    "enhance": [
        (
            "https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt?download=1",
            MODELS_DIR / "bandit-v2" / "checkpoint-multi.ckpt",
            "abcfccf65446752a057f4a302c941479a54b7560ebf8d7bca039d2ea98e64cfc",
            "Bandit v2 다국어 CASS 체크포인트 (CC BY-SA 4.0, 446.7 MB)",
        ),
    ],
}


def download_hf(repo_id: str, desc: str, token: str | None) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    print(f"[HF] {repo_id} — {desc}", flush=True)
    try:
        path = snapshot_download(repo_id, token=token)
    except GatedRepoError:
        print(
            f"   ✖ 접근 제한 모델입니다. https://huggingface.co/{repo_id} 에서 이용 조건에 동의한 뒤\n"
            "     --token <hf_토큰> 또는 HF_TOKEN 환경 변수로 다시 실행하세요.",
            file=sys.stderr,
        )
        return False
    except HfHubHTTPError as e:
        print(f"   ✖ 실패: {e}", file=sys.stderr)
        return False
    print(f"   ✔ {path}")
    return True


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path, sha256: str, desc: str) -> bool:
    print(f"[URL] {dest.relative_to(ROOT)} — {desc}", flush=True)
    if dest.exists() and sha256_of(dest) == sha256:
        print("   ✔ 이미 있음 (체크섬 일치)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def report(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r   … {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    print()
    digest = sha256_of(tmp)
    if digest != sha256:
        tmp.unlink(missing_ok=True)
        print(f"   ✖ 체크섬 불일치: {digest}", file=sys.stderr)
        return False
    tmp.replace(dest)
    print(f"   ✔ {dest}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--all", action="store_true", help="모든 그룹")
    ap.add_argument("--stt", action="store_true", help="DiCoW + faster-whisper large-v3 (기본)")
    ap.add_argument("--kotoba", action="store_true", help="kotoba-whisper v2.0 (일본어 대안)")
    ap.add_argument("--diar", action="store_true", help="pyannote 화자 분할 (토큰 필요)")
    ap.add_argument("--enhance", action="store_true", help="Bandit v2 CASS 체크포인트 (Zenodo)")
    ap.add_argument(
        "--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face 토큰 (gated 모델용)"
    )
    args = ap.parse_args()

    groups = [g for g in ("stt", "kotoba", "diar", "enhance") if args.all or getattr(args, g)]
    if not groups:
        groups = ["stt"]

    ok = True
    for g in groups:
        for repo_id, desc in HF_MODELS.get(g, []):
            ok &= download_hf(repo_id, desc, args.token)
        for url, dest, sha, desc in ZENODO_FILES.get(g, []):
            ok &= download_url(url, dest, sha, desc)
    print("\n완료" if ok else "\n일부 실패 — 위 메시지를 확인하세요", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
