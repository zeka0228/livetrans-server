"""모드 A 전체 경로 (실제 ffmpeg 입력·HLS, 가짜 STT·번역·TTS): REST 생성 → WS → HLS 파일 서빙. ffmpeg 없으면 skip."""

import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from livetrans.core.config import OutputConfig, Settings, SttConfig, TimelineConfig
from livetrans.main import create_app
from livetrans.output.hls import find_ffmpeg
from livetrans.plugins.stt.fake import FakeSTT
from livetrans.plugins.tts.fake import FakeTTS
from tests.integration.test_mic_session import energy_vad, silence, tone
from tests.integration.test_mt_session import FakeTranslator

FFMPEG = find_ffmpeg(None)
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg 없음")


def write_wav(path: Path, audio16: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio16, -1, 1) * 32767).astype("<i2").tobytes())


def test_mode_a_file_source_serves_hls(tmp_path: Path):
    wav = tmp_path / "in.wav"
    write_wav(wav, np.concatenate([silence(0.5), tone(1.5), silence(0.8), tone(1.0), silence(1.2)]))
    settings = Settings(
        stt=SttConfig(engine="fake"),
        timeline=TimelineConfig(holdback_s=0.5),
        output=OutputConfig(hls_dir=str(tmp_path / "hls")),
    )
    settings.tts.defaults = {"ko-KR": "fake:auto"}
    app = create_app(
        settings,
        stt=FakeSTT(lang="ja"),
        vad_factory=lambda: energy_vad,
        tts_engines={"fake": FakeTTS()},
        translator=FakeTranslator(),
    )
    with TestClient(app) as client:
        r = client.post(
            "/api/sessions",
            json={
                "mode": "A",
                "source": {"type": "file", "params": {"url": str(wav)}},
                "src_lang": "auto",
                "tgt_lang": "ko",
                "delay_s": 2,
                "tts": {"enabled": True},
                "display": {"show_speaker": True},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        sid = body["session_id"]
        assert body["delay_s"] == 2.0 and body["hls_url"] == f"/hls/{sid}/master.m3u8"
        with client.websocket_connect(body["ws_url"]) as ws:
            events = [json.loads(ws.receive_text())]
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "start"}}))
            ws.send_text(
                json.dumps({"type": "config.update", "payload": {"volume": {"original": 0.7}}})
            )
            # 첫 세그먼트는 compose_wall(0) = +2.5 s 뒤, ffmpeg 가 1 s 오디오를 받은 뒤에야 플레이리스트에 뜬다
            deadline = time.time() + 10
            media = None
            while time.time() < deadline:
                media = client.get(f"/hls/{sid}/media.m3u8")
                if media.status_code == 200 and ".ts" in media.text:
                    break
                time.sleep(0.3)
            assert media is not None and media.status_code == 200 and ".ts" in media.text, (
                media.text
            )
            master = client.get(body["hls_url"])
            assert master.status_code == 200 and "sub_tgt.m3u8" in master.text
            seg_name = next(line for line in media.text.splitlines() if line.endswith(".ts"))
            seg = client.get(f"/hls/{sid}/{seg_name}")
            assert seg.status_code == 200 and seg.headers["content-type"].startswith("video/mp2t")
            assert client.get(f"/hls/{sid}/../etc").status_code in (404, 422)
            # 파일 끝 → 서버가 세션을 끝낸다 (E_CAPTURE_LOST 뒤 stopped)
            while True:
                ev = json.loads(ws.receive_text())
                events.append(ev)
                if ev["type"] == "session.state" and ev["payload"]["state"] == "stopped":
                    break
    states = [e["payload"]["state"] for e in events if e["type"] == "session.state"]
    assert states[:3] == ["starting", "buffering", "playing"] and states[-1] == "stopped"
    assert [e for e in events if e["type"] == "segment.final"]
    assert [e for e in events if e["type"] == "translation.final"]
    assert [e for e in events if e["type"] == "tts.ready"]
    vtt_list = (tmp_path / "hls" / sid / "sub_tgt.m3u8").read_text(encoding="utf-8")
    assert "#EXT-X-ENDLIST" in vtt_list
    vtts = sorted((tmp_path / "hls" / sid).glob("sub_tgt_*.vtt"))
    assert any("-->" in p.read_text(encoding="utf-8") for p in vtts)
