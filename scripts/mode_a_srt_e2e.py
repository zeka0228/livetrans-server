"""모드 A SRT 입력 E2E (OBS 없이): 서버가 `srt://127.0.0.1:<port>?mode=listener` 로 기다리고, ffmpeg 가 테스트 영상을 caller 로 밀어 넣는다.

사용: uv run python -X utf8 scripts\\mode_a_srt_e2e.py <media path> [base_url=http://127.0.0.1:8765] [delay_s=8] [port=9000]
OBS 설정과 같은 경로(SRT → 서버)를 검증한다. OBS 에서는 출력 URL 을 `srt://<서버 IP>:<port>?mode=caller` 로 둔다.
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
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8765"
DELAY = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 9000
FFMPEG = find_ffmpeg(None) or "ffmpeg"
HLS_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "hls"


async def main() -> None:
    r = httpx.post(
        BASE + "/api/sessions",
        json={
            "mode": "A",
            "source": {
                "type": "obs_stream",
                "params": {"url": f"srt://127.0.0.1:{PORT}?mode=listener"},
            },
            "src_lang": "auto",
            "tgt_lang": "ko",
            "delay_s": DELAY,
            "tts": {"enabled": True},
        },
        timeout=20,
    )
    print("create:", r.status_code, r.text[:160])
    r.raise_for_status()
    body = r.json()
    sid = body["session_id"]
    ws_url = BASE.replace("http", "ws") + body["ws_url"]
    t0 = time.time()
    counts: dict[str, int] = {}
    pusher = None
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.recv()  # starting
        await ws.send(json.dumps({"type": "control", "payload": {"action": "start"}}))
        # 서버(ffmpeg listener)가 포트를 연 뒤 OBS 역할의 ffmpeg caller 를 띄운다
        await asyncio.sleep(2.0)
        pusher = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-re", "-i", MEDIA, "-c", "copy", "-f", "mpegts",
             f"srt://127.0.0.1:{PORT}?mode=caller"],
        )  # fmt: skip
        print(f"[{time.time() - t0:5.1f}s] OBS 역할 ffmpeg 송출 시작 (pid {pusher.pid})")
        while True:
            try:
                ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
            except TimeoutError:
                print("timeout")
                break
            counts[ev["type"]] = counts.get(ev["type"], 0) + 1
            p = ev["payload"]
            el = time.time() - t0
            if ev["type"] == "session.state":
                print(f"[{el:5.1f}s] STATE {p['state']}")
                if p["state"] == "stopped":
                    break
            elif ev["type"] in ("segment.final", "translation.final", "tts.ready", "error"):
                short = {
                    k: v
                    for k, v in p.items()
                    if k
                    in ("seg_id", "speaker", "text", "engine", "duration_ms", "code", "message")
                }
                print(f"[{el:5.1f}s] {ev['type']} {short}")
    if pusher is not None:
        pusher.wait(timeout=30)
    d = HLS_ROOT / sid
    print(
        "미디어 세그먼트:",
        len(list(d.glob("media_*.ts"))),
        "VTT:",
        len(list(d.glob("sub_tgt_*.vtt"))),
    )
    chk = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(d / "master.m3u8"), "-t", "8", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    print("ffmpeg 재생 검사:", "OK" if chk.returncode == 0 else chk.stderr[:200])
    print("이벤트 수:", counts)


if __name__ == "__main__":
    asyncio.run(main())
