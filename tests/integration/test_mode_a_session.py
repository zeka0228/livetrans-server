"""모드 A 세션 (명세 §3·§4.5): 가짜 입력(실시간 속도) + 가짜 HLS + 가짜 TTS·번역·STT — 상태 전이, 컴포저 타이밍, 큐·TTS·데드라인·metrics."""

import asyncio
import time

import numpy as np
import pytest

from livetrans.core.config import Settings, SttConfig, TimelineConfig
from livetrans.core.voices import VoiceRegistry
from livetrans.pipeline.session import Session
from livetrans.plugins.source.base import AudioFrame
from livetrans.plugins.stt.fake import FakeSTT
from livetrans.plugins.tts.fake import FakeTTS
from tests.integration.test_mic_session import energy_vad, silence, tone
from tests.integration.test_mt_session import FakeTranslator


class FakeLiveSource:
    """16 kHz 프레임을 실시간 속도로 흘리는 입력 (출력 콜백 없음 → 세션이 48 kHz 로 올려 링에 넣는다)."""

    name = "file"
    has_video = False

    def __init__(self, audio: np.ndarray, chunk_ms: int = 100) -> None:
        self.audio = audio
        self.chunk = chunk_ms * 16
        self.started = False
        self.stopped = False
        self.lost_reason = None

    async def start(self) -> None:
        self.started = True

    async def frames(self):
        t = 0
        for i in range(0, len(self.audio), self.chunk):
            if self.stopped:
                return
            await asyncio.sleep(self.chunk / 16000)
            pcm = self.audio[i : i + self.chunk]
            yield AudioFrame(pcm=pcm, t_start_ms=t)
            t += len(pcm) * 1000 // 16000

    def raw_media_dir(self):
        return None

    async def stop(self) -> None:
        self.stopped = True


class FakeHls:
    def __init__(self) -> None:
        self.segments: list[tuple[int, np.ndarray, dict]] = []
        self.started = False
        self.closed = False
        self.removed = False

    async def start(self) -> None:
        self.started = True

    async def write_segment(self, k, pcm, cues) -> None:
        self.segments.append((k, pcm, {kind: list(v) for kind, v in cues.items()}))

    async def close(self) -> None:
        self.closed = True

    def remove(self) -> None:
        self.removed = True

    @property
    def alive(self) -> bool:
        return self.started and not self.closed


async def drain(session: Session) -> list[dict]:
    events = []
    while not session.outbox.empty():
        events.append(session.outbox.get_nowait().to_dict())
    return events


async def run_mode_a(audio_seconds=4.5, delay=2.0, holdback=0.5, tts=True):
    settings = Settings(
        stt=SttConfig(engine="fake"), timeline=TimelineConfig(holdback_s=holdback, max_lag_s=4.0)
    )
    settings.tts.defaults = {"ko-KR": "fake:auto"}
    voices = VoiceRegistry({"fake": FakeTTS()}, settings.tts.defaults)
    await voices.refresh()
    audio = np.concatenate([silence(0.5), tone(1.5), silence(0.8), tone(0.8), silence(0.9)])
    audio = audio[: int(audio_seconds * 16000)]
    source = FakeLiveSource(audio)
    hls = FakeHls()
    stt = FakeSTT(lang="ja")
    mt = FakeTranslator()
    session = Session(
        settings,
        stt,
        energy_vad,
        source,
        mode="A",
        src_lang=None,
        tgt_lang="ko",
        session_id="modea1",
        translator=mt,
        delay_s=delay,
        hls_writer=hls,
        voices=voices,
        tts_enabled=tts,
    )
    await session.start()
    t_begin = time.time()
    await session.begin()
    events = await drain(session)
    # 입력이 끝날 때까지 (실시간) + 여유
    await asyncio.sleep(audio_seconds + 0.5)
    session.update_config({"volume": {"original": 0.5, "tts": 0.8}})
    await session.stop()
    events += await drain(session)
    return session, events, hls, stt, mt, t_begin


async def test_mode_a_states_timeline_and_outputs():
    session, events, hls, _stt, _mt, t_begin = await run_mode_a()
    states = [e["payload"] for e in events if e["type"] == "session.state"]
    assert [s["state"] for s in states] == ["starting", "buffering", "playing", "stopped"]
    buf = states[1]
    assert buf["delay_s"] == 2.0
    assert abs(buf["play_start_wall_ms"] - (t_begin + 2.0) * 1000) < 400
    assert states[2]["play_start_wall_ms"] == buf["play_start_wall_ms"]
    # 세그먼트: 1 s 스테레오 48 kHz, 입력(4.5 s) 끝까지 + 여유 1개 이내
    assert hls.started and hls.closed
    ks = [k for k, _, _ in hls.segments]
    assert ks == list(range(len(ks))) and 5 <= len(ks) <= 7
    assert all(pcm.shape == (48000, 2) for _, pcm, _ in hls.segments)
    # 원문·번역 큐가 세그먼트에 실렸고, 늦은 결과는 L2 로 밀렸다
    src_cues = [c for _, _, cues in hls.segments for c in cues.get("src", [])]
    tgt_cues = [c for _, _, cues in hls.segments for c in cues.get("tgt", [])]
    assert src_cues and tgt_cues
    assert all(c.text.startswith("[ja->ko]") for c in tgt_cues)
    deadlines = [e["payload"] for e in events if e["type"] == "deadline.event"]
    assert deadlines and all(d["level"] in ("L2", "L3") for d in deadlines)
    assert any(d["kind"] == "subtitle" for d in deadlines)
    # TTS: 번역문으로 합성돼 tts.ready, 세그먼트 믹스에 들어감 (톤 0.2 + 원본)
    ready = [e["payload"] for e in events if e["type"] == "tts.ready"]
    assert ready and all(r["voice_ref"] == "fake:ko-KR" and r["duration_ms"] > 0 for r in ready)
    assert any(abs(pcm[:, 0]).max() > 0.15 for _, pcm, _ in hls.segments)
    # metrics 가 2 s 간격으로 왔고 형식이 맞다
    metrics = [e["payload"] for e in events if e["type"] == "metrics"]
    assert metrics and {
        "p50_ms",
        "p99_ms",
        "gpu_util_stream",
        "miss_rate",
        "queue_depth",
        "gpu_mem_used_mb",
    } <= set(metrics[0])
    assert session.composer.volume == {"original": 0.5, "tts": 0.8}
    assert not hls.removed  # 삭제는 2 분 뒤 예약


async def test_mode_a_without_tts_sends_no_tts_ready():
    _, events, hls, _, _, _ = await run_mode_a(audio_seconds=3.0, tts=False)
    assert not [e for e in events if e["type"] == "tts.ready"]
    assert [e["payload"]["state"] for e in events if e["type"] == "session.state"][-1] == "stopped"
    assert hls.closed


async def test_mode_a_hls_start_failure_stops_session():
    class BrokenHls(FakeHls):
        async def start(self) -> None:
            raise RuntimeError("ffmpeg 없음")

    settings = Settings(stt=SttConfig(engine="fake"))
    source = FakeLiveSource(tone(1.0))
    session = Session(
        settings,
        FakeSTT(),
        energy_vad,
        source,
        mode="A",
        src_lang=None,
        session_id="modea2",
        delay_s=2.0,
        hls_writer=BrokenHls(),
    )
    await session.start()
    await session.begin()
    await asyncio.sleep(3.5)  # HLS 는 첫 세그먼트 직전(compose_wall(0) ≈ +2.5 s)에 시작된다
    events = await drain(session)
    errors = [e["payload"] for e in events if e["type"] == "error"]
    assert errors and errors[0]["code"] == "E_OUTPUT_START" and not errors[0]["recoverable"]
    states = [e["payload"]["state"] for e in events if e["type"] == "session.state"]
    assert states[-1] == "stopped"


@pytest.mark.parametrize("delay", [1.0])
async def test_mode_a_timeline_compose_times(delay):
    """세그먼트 k 는 벽시계 compose_wall(k) 이후에 합성된다 (앞당겨지지 않는다)."""
    settings = Settings(stt=SttConfig(engine="fake"), timeline=TimelineConfig(holdback_s=0.5))
    hls = FakeHls()
    stamped: list[tuple[int, float]] = []

    async def write_segment(k, pcm, cues):
        stamped.append((k, time.time()))
        hls.segments.append((k, pcm, cues))

    hls.write_segment = write_segment
    session = Session(
        settings,
        FakeSTT(),
        energy_vad,
        FakeLiveSource(tone(3.0)),
        mode="A",
        src_lang=None,
        session_id="modea3",
        delay_s=delay,
        hls_writer=hls,
    )
    await session.start()
    await session.begin()
    await asyncio.sleep(3.2)
    tl = session.timeline
    await session.stop()
    live = [(k, t) for k, t in stamped if tl.compose_wall(k) <= t <= tl.compose_wall(k) + 0.6]
    assert len(live) >= 2, stamped  # 입력 중 합성된 것은 제시간에
