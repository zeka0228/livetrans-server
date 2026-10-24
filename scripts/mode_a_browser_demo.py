"""모드 A 를 브라우저(hls.js)에서 바로 보는 데모 (서버 검증용, 앱 없이).

사용: uv run python -X utf8 scripts\\mode_a_browser_demo.py <media path or srt://…> [delay_s=20] [base=http://127.0.0.1:8765] [--port 8780] [--no-video] [--no-open]

하는 일: ① `scripts/hls_view.html` 을 127.0.0.1:<port> 로 정적 서빙 ② 모드 A 세션 생성 → start → `playing` 대기
③ 기본 브라우저로 `http://127.0.0.1:<port>/hls_view.html?src=<hls_url>` 열기 ④ `stopped` 까지 이벤트를 콘솔에 찍고 정리.
포트 기본값 8780 은 서버 `server.cors_origins` 기본 허용 출처라 CORS 없이 된다. 앱이 8780 을 쓰고 있으면 `--port` 를 바꾸고
`config/default.local.yaml` 의 `server.cors_origins` 에 그 출처를 추가한다.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

import httpx
import websockets

print = functools.partial(print, flush=True)  # 리다이렉트돼도 바로 보이게

HERE = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS = sys.argv[1:]
if not ARGS:
    sys.exit(__doc__)
MEDIA = ARGS[0]
DELAY = float(ARGS[1]) if len(ARGS) > 1 else 20.0
BASE = ARGS[2] if len(ARGS) > 2 else "http://127.0.0.1:8765"
PORT = int(FLAGS[FLAGS.index("--port") + 1]) if "--port" in FLAGS else 8780
VIDEO = "--no-video" not in FLAGS
OPEN = "--no-open" not in FLAGS


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:
        pass


def serve_static(port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(_QuietHandler, directory=str(HERE))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


async def main() -> None:
    is_stream = "://" in MEDIA
    r = httpx.post(
        BASE + "/api/sessions",
        json={
            "mode": "A",
            "source": {
                "type": "obs_stream" if is_stream else "file",
                "params": {"url": MEDIA, "video": VIDEO},
            },
            "src_lang": "auto",
            "tgt_lang": "ko",
            "delay_s": DELAY,
            "tts": {"enabled": True, "voice_ref": None, "duck_db": -12},
            "display": {"show_original": True, "show_speaker": True},
        },
        timeout=20,
    )
    r.raise_for_status()
    body = r.json()
    hls_url = BASE + body["hls_url"]
    page = f"http://127.0.0.1:{PORT}/hls_view.html?src={hls_url}"
    print(f"session {body['session_id']}  D={body['delay_s']} s  hls {hls_url}")
    print(f"브라우저 페이지: {page}")
    t0 = time.time()
    async with websockets.connect(BASE.replace("http", "ws") + body["ws_url"], max_size=None) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "control", "payload": {"action": "start"}}))
        opened = False
        while True:
            ev = json.loads(await ws.recv())
            p = ev["payload"]
            el = time.time() - t0
            if ev["type"] == "session.state":
                late = ""
                if p.get("play_start_wall_ms"):
                    late = (
                        f" (play_start 대비 {time.time() - p['play_start_wall_ms'] / 1000:+.1f}s)"
                    )
                print(f"[{el:6.1f}s] STATE {p['state']}{late}")
                if p["state"] == "playing" and not opened:
                    opened = True
                    if OPEN:
                        webbrowser.open(page)
                if p["state"] == "stopped":
                    break
            elif ev["type"] in ("segment.final", "translation.final"):
                print(f"[{el:6.1f}s] {ev['type']:17s} {p.get('speaker') or '':3s} {p['text'][:70]}")
            elif ev["type"] == "deadline.event":
                print(
                    f"[{el:6.1f}s] deadline {p['kind']} {p['level']} late={p['late_ms']} ms (seg {p['seg_id']})"
                )
            elif ev["type"] == "error":
                print(f"[{el:6.1f}s] error {p['code']}: {p['message']}")


if __name__ == "__main__":
    srv = serve_static(PORT)
    try:
        asyncio.run(main())
    finally:
        srv.shutdown()
