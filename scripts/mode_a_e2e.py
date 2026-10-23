"""모드 A 실서버 E2E: 파일(영상 포함)을 obs_stream 처럼 넣고 WS 이벤트 + HLS 산출물(세그먼트·VTT·재생 가능)을 확인한다.

사용: uv run python -X utf8 scripts\\mode_a_e2e.py <media path> [base_url=http://127.0.0.1:8765] [delay_s=8] [--no-video]
테스트 영상은 scripts\\make_test_media.py 로 만든다. 서버의 output.hls_dir 이 기본(runtime/hls)이어야 산출물 점검이 된다.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livetrans.output.hls import find_ffmpeg

MEDIA = sys.argv[1]
ARGS = [a for a in sys.argv[2:] if not a.startswith("--")]
BASE = ARGS[0] if ARGS else "http://127.0.0.1:8765"
DELAY = float(ARGS[1]) if len(ARGS) > 1 else 8.0
VIDEO = "--no-video" not in sys.argv
FFMPEG = find_ffmpeg(None) or "ffmpeg"
FFPROBE = str(Path(FFMPEG).with_name("ffprobe" + Path(FFMPEG).suffix))
HLS_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "hls"


async def main() -> None:
    health = httpx.get(BASE + "/api/health", timeout=5).json()
    print("models:", health["models_loaded"])
    r = httpx.post(
        BASE + "/api/sessions",
        json={
            "mode": "A",
            "source": {"type": "file", "params": {"url": MEDIA, "video": VIDEO}},
            "src_lang": "auto",
            "tgt_lang": "ko",
            "delay_s": DELAY,
            "tts": {"enabled": True, "voice_ref": None, "duck_db": -12},
            "display": {"show_original": True, "show_speaker": True},
        },
        timeout=20,
    )
    print("create:", r.status_code, r.text[:200])
    r.raise_for_status()
    body = r.json()
    sid, hls_url = body["session_id"], BASE + body["hls_url"]
    ws_url = BASE.replace("http", "ws") + body["ws_url"]
    t0 = time.time()
    events = []
    async with websockets.connect(ws_url, max_size=None) as ws:
        print("[  0.0s]", json.loads(await ws.recv())["payload"])
        await ws.send(json.dumps({"type": "control", "payload": {"action": "start"}}))
        polled = False
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
            except TimeoutError:
                print("timeout waiting events")
                break
            ev = json.loads(msg)
            events.append(ev)
            p = ev["payload"]
            el = time.time() - t0
            if ev["type"] == "session.state":
                print(
                    f"[{el:5.1f}s] STATE {p['state']} delay={p['delay_s']} play_start_wall_ms={p['play_start_wall_ms']}"
                )
                if p["state"] == "stopped":
                    break
            elif ev["type"] == "segment.final":
                print(
                    f"[{el:5.1f}s] FINAL seg={p['seg_id']} spk={p.get('speaker')} {p['t_start_ms'] / 1000:.1f}-{p['t_end_ms'] / 1000:.1f}s {p['lang']} | {p['text']}"
                )
            elif ev["type"] == "translation.final":
                print(
                    f"[{el:5.1f}s]   TRANS seg={p['seg_id']} {p['engine']} {p['mt_ms']}ms | {p['text']}"
                )
            elif ev["type"] == "tts.ready":
                print(
                    f"[{el:5.1f}s]   TTS seg={p['seg_id']} {p['duration_ms']}ms rate={p['rate']} {p['voice_ref']} ({p['tts_ms']}ms)"
                )
            elif ev["type"] == "deadline.event":
                print(
                    f"[{el:5.1f}s]   DEADLINE seg={p['seg_id']} {p['kind']} {p['level']} late={p['late_ms']}ms"
                )
            elif ev["type"] == "error":
                print(f"[{el:5.1f}s] ERROR {p}")
            elif ev["type"] == "metrics" and int(el) % 10 < 2:
                print(
                    f"[{el:5.1f}s] metrics p50={p['p50_ms']} miss={p['miss_rate']:.2f} q={p['queue_depth']} gpu={p['gpu_util_stream']}"
                )
            if not polled and ev["type"] == "session.state" and p["state"] == "playing":
                polled = True
                m = httpx.get(hls_url, timeout=5)
                media = httpx.get(hls_url.replace("master.m3u8", "media.m3u8"), timeout=5)
                print(
                    f"[{el:5.1f}s] HLS master {m.status_code}; media.m3u8 {media.status_code} segs={media.text.count('.ts')}"
                )
    d = HLS_ROOT / sid
    vtts = sorted(d.glob("sub_tgt_*.vtt"))
    cue_lines = [
        line
        for path in vtts
        for line in path.read_text(encoding="utf-8").splitlines()
        if "-->" in line or (line and not line.startswith(("WEBVTT", "X-TIMESTAMP")))
    ]
    print(f"\nVTT 세그먼트 {len(vtts)}개, 큐 줄 {len(cue_lines)}")
    for line in cue_lines[:8]:
        print("   ", line)
    media_files = sorted(d.glob("media_*.ts"))
    print(f"미디어 세그먼트 {len(media_files)}개")
    if media_files:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "stream=codec_type,codec_name,width", "-of", "json", str(media_files[len(media_files) // 2])],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        print(
            "세그먼트 스트림:",
            [
                (s["codec_type"], s["codec_name"], s.get("width"))
                for s in json.loads(out.stdout)["streams"]
            ],
        )
    chk = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(d / "master.m3u8"), "-t", "10", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    print(
        "ffmpeg 재생 검사:",
        "OK" if chk.returncode == 0 else f"rc={chk.returncode} {chk.stderr[:300]}",
    )
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print("이벤트 수:", kinds)


if __name__ == "__main__":
    asyncio.run(main())
