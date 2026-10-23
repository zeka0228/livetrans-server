"""세션: 입력 → 세그먼터 → (화자 분할) → STT → WS 이벤트.

모드 C = 앱 마이크(`mic_ws`), 모드 B = PC 에서 재생 중인 소리 캡처(`loopback`, D-016). 입력만 다르고 처리는 같다.

두 개의 태스크로 돈다.
- ingest: 소스 프레임을 세그먼터에 넣고, 확정 발화와 중간 결과 요청을 STT 작업으로 넘긴다.
- stt: 확정 발화를 먼저, 그다음 가장 최근 중간 결과 요청 하나만 처리한다 (우선순위 `final > partial`, §2.3).
  밀린 중간 결과는 버린다 — 최신 것만 의미가 있다.

화자 분할(Diarizer, §4.8)이 있으면 확정 발화마다 화자 구간을 붙이고, 화자 마스크를 지원하는 STT(DiCoW)면
**화자마다 1회** 인식해 같은 `seg_id` 로 `segment.final` 을 화자별로 보낸다 (§6.2, API 계약 S-17).
중간 결과는 화자 분할 없이 전체를 인식한다.

이벤트는 `outbox` 큐에 봉투로 쌓이고, WS 핸들러가 꺼내 보낸다.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import Counter, deque
from collections.abc import Callable

import numpy as np

from livetrans.core.config import Settings
from livetrans.core.events import (
    Envelope,
    EventFactory,
    error_payload,
    session_state_payload,
)
from livetrans.core.speaker import select_speakers, speaker_extent
from livetrans.pipeline.segmenter import PartialRequest, Segmenter, SpeechSegment
from livetrans.plugins.diar.base import Diarizer
from livetrans.plugins.source.base import Source
from livetrans.plugins.source.loopback import CaptureError
from livetrans.plugins.stt.base import STTEngine, Transcript

log = logging.getLogger("livetrans.session")

PARTIAL_INTERVAL_S = 1.0
LANG_CONFIDENT = 0.7  # 자동 감지 확률이 이보다 낮으면 최근 다수결 언어로 다시 인식 (§4.3)
LANG_VOTES = 10
LANG_MIN_VOTES = 3


def final_payload(
    seg: SpeechSegment,
    tr: Transcript,
    stt_ms: int,
    *,
    speaker: str | None = None,
    diar_ms: int | None = None,
) -> dict:
    """`segment.final` payload. `speaker` 가 있으면 그 화자가 말한 범위로 `t_start_ms`/`t_end_ms` 를 좁힌다."""
    t_start, t_end = seg.t_start_ms, seg.t_end_ms
    if speaker is not None:
        extent = speaker_extent(seg.speaker_turns, speaker)
        if extent is not None:
            t_start, t_end = extent
    payload = {
        "seg_id": seg.seg_id,
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "text": tr.text,
        "lang": tr.lang,
        "lang_prob": round(tr.lang_prob, 4),
        "words": [{"w": w.text, "t0_ms": w.t0_ms, "t1_ms": w.t1_ms} for w in tr.words],
        "forced_cut": seg.forced_cut,
        "stt_ms": stt_ms,
    }
    if speaker is not None:
        payload["speaker"] = speaker  # 없으면 생략 (명세 §6.2)
    if diar_ms is not None:
        payload["diar_ms"] = diar_ms
    return payload


class Session:
    def __init__(
        self,
        settings: Settings,
        stt: STTEngine,
        vad: Callable[[np.ndarray], float],
        source: Source,
        *,
        mode: str,
        src_lang: str | None,
        session_id: str | None = None,
        diarizer: Diarizer | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.mode = mode
        self.src_lang = src_lang
        self.stt = stt
        self.source = source
        self.diarizer = diarizer
        self._max_speakers_per_seg = settings.diar.max_speakers_per_seg
        # 캡처 소스는 control start 때 켠다 (그 전 소리는 받지 않음). 마이크는 앱이 보내는 대로 받는다.
        self._start_source_on_begin = mode == "B"
        self.segmenter = Segmenter(
            settings.vad,
            vad,
            sample_rate=settings.audio.sample_rate,
            partial_interval_s=PARTIAL_INTERVAL_S,
        )
        self.events = EventFactory(self.session_id)
        self.outbox: asyncio.Queue[Envelope] = asyncio.Queue()
        self.state = "starting"

        self._finals: deque[SpeechSegment] = deque()
        self._partial: PartialRequest | None = None
        self._wake = asyncio.Event()
        self._ingest_done = False
        self._last_final_id = -1
        self._prev_text: dict[
            str | None, str
        ] = {}  # 화자별 직전 확정 문장 (프롬프트·반복 제거, §4.3)
        self._lang_votes: deque[str] = deque(maxlen=LANG_VOTES)
        self._tasks: list[asyncio.Task] = []
        self._stopping: asyncio.Task | None = None

    # --- 수명 주기 ---

    async def start(self) -> None:
        if not self._start_source_on_begin:
            await self.source.start()
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name=f"ingest-{self.session_id}"),
            asyncio.create_task(self._stt_loop(), name=f"stt-{self.session_id}"),
        ]
        self._emit_state("starting")

    async def begin(self) -> None:
        """앱의 `control: start`. 모드 B·C 에는 지연 버퍼가 없으므로 바로 playing."""
        if self.state != "starting":
            return
        if self._start_source_on_begin:
            try:
                await self.source.start()
            except (CaptureError, TimeoutError) as e:
                self._emit("error", error_payload("E_CAPTURE_START", f"캡처 시작 실패: {e}", False))
                await self.stop()
                return
        self._emit_state("playing")

    def feed(self, data: bytes) -> None:
        """모드 C 바이너리 PCM. start 없이 오디오가 먼저 와도 시작으로 본다 (API 계약 Q-API-5)."""
        if self.mode != "C":
            return
        if self.state == "starting":
            self._emit_state("playing")
        self.source.feed(data)

    async def stop(self) -> None:
        """입력을 닫고, 남은 발화를 끝까지 인식한 뒤 `stopped` 를 보낸다. 여러 번 불러도 한 번만 실행."""
        if self._stopping is None:
            self._stopping = asyncio.create_task(self._stop())
        await asyncio.shield(self._stopping)

    async def _stop(self) -> None:
        await self.source.stop()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        lost = getattr(self.source, "lost_reason", None)
        if lost:
            # 남은 final 을 전부 보낸 뒤, stopped 바로 앞에 보낸다 (앱은 이 오류를 받으면 바로 세션을 끝낸다)
            self._emit("error", error_payload("E_CAPTURE_LOST", lost, False))
        self._discard_speakers()
        self._emit_state("stopped")

    async def abort(self) -> None:
        """연결이 끊겨 결과를 받을 곳이 없을 때: 남은 작업을 버린다."""
        await self.source.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._discard_speakers()
        self.state = "stopped"

    @property
    def stopped(self) -> bool:
        return self.state == "stopped"

    # --- 내부 ---

    def _discard_speakers(self) -> None:
        """세션 종료: 화자 임베딩 폐기 (§13)."""
        if self.diarizer is not None:
            self.diarizer.reset()

    def _emit(self, type_: str, payload: dict) -> None:
        self.outbox.put_nowait(self.events.make(type_, payload))

    def _emit_state(self, state: str) -> None:
        self.state = state
        self._emit("session.state", session_state_payload(state, delay_s=None))

    async def _ingest_loop(self) -> None:
        async for frame in self.source.frames():
            for ev in self.segmenter.push(frame.pcm):
                if isinstance(ev, SpeechSegment):
                    self._finals.append(ev)
                else:
                    self._partial = ev
            self._wake.set()
        self._finals.extend(self.segmenter.flush())
        self._ingest_done = True
        self._wake.set()
        if getattr(self.source, "lost_reason", None) and self._stopping is None:
            # 캡처 대상이 사라짐(창 종료 등): 남은 발화까지 인식하고 세션을 끝낸다. 오류는 _stop 에서 보낸다
            asyncio.get_running_loop().create_task(self.stop())

    async def _stt_loop(self) -> None:
        while True:
            if self._finals:
                await self._run_final(self._finals.popleft())
            elif self._partial is not None:
                req, self._partial = self._partial, None
                await self._run_partial(req)
            elif self._ingest_done:
                return
            else:
                await self._wake.wait()
                self._wake.clear()

    async def _run_partial(self, req: PartialRequest) -> None:
        if req.seg_id <= self._last_final_id:
            return
        seg = SpeechSegment(
            seg_id=req.seg_id,
            t_start_ms=req.t_start_ms,
            t_end_ms=req.t_start_ms + len(req.pcm) * 1000 // 16000,
            pcm=req.pcm,
        )
        try:
            tr = await self.stt.transcribe(
                seg, lang=self._lang_hint(), prompt=self._prev_text.get(None) or None, final=False
            )
        except Exception as e:  # noqa: BLE001 — 중간 결과 실패는 경고만 남기고 넘긴다
            log.warning("partial STT 실패 seg=%s: %s", req.seg_id, e)
            return
        # 인식하는 동안 같은 발화의 확정 결과가 먼저 나갔으면 보내지 않는다
        if req.seg_id <= self._last_final_id or not tr.text:
            return
        self._emit(
            "segment.partial",
            {"seg_id": req.seg_id, "t_start_ms": req.t_start_ms, "text": tr.text, "lang": tr.lang},
        )

    async def _run_final(self, seg: SpeechSegment) -> None:
        diar_ms = await self._attach_speakers(seg)
        # 화자 마스크를 지원하는 엔진이면 화자마다 1회, 아니면 구간 전체 1회 (주 화자만 표시)
        if seg.speakers and getattr(self.stt, "supports_speaker_mask", False):
            targets: list[str | None] = list(seg.speakers)
        else:
            targets = [None]
        results: list[tuple[str | None, Transcript, int]] = []
        for speaker in targets:
            got = await self._transcribe_final(seg, speaker)
            if got is not None:
                results.append((speaker, *got))
        self._last_final_id = max(self._last_final_id, seg.seg_id)
        if not results:
            return
        # 화자별 결과 중 빈 문장은 빼되, 전부 비었으면 하나는 보낸다 (앱이 같은 seg_id 의 partial 을 지울 수 있게)
        kept = [r for r in results if r[1].text] or results[:1]
        for speaker, tr, stt_ms in kept:
            shown = speaker if speaker is not None else (seg.speakers[0] if seg.speakers else None)
            self._emit(
                "segment.final",
                final_payload(seg, tr, stt_ms, speaker=shown, diar_ms=diar_ms),
            )

    async def _attach_speakers(self, seg: SpeechSegment) -> int | None:
        """Diarizer 로 화자 구간을 받아 발화에 붙인다. 실패하면 화자 없이 진행 (경고만)."""
        if self.diarizer is None:
            return None
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            turns = await self.diarizer.diarize(seg)
        except Exception as e:  # noqa: BLE001 — 화자 분할 실패는 자막을 막지 않는다
            log.warning("화자 분할 실패 seg=%s: %s", seg.seg_id, e)
            return None
        diar_ms = int((loop.time() - t0) * 1000)
        if not turns:
            return diar_ms
        seg.speakers = select_speakers(turns, self._max_speakers_per_seg)
        seg.speaker_turns = [t for t in turns if t.speaker in seg.speakers]
        return diar_ms

    async def _transcribe_final(
        self, seg: SpeechSegment, speaker: str | None
    ) -> tuple[Transcript, int] | None:
        """확정 인식 1회 (+ 저확률 언어 재인식). 실패하면 `E_STT_FAILED` 를 보내고 None."""
        prompt = self._prev_text.get(speaker) or None
        try:
            tr = await self.stt.transcribe(
                seg, lang=self.src_lang, prompt=prompt, final=True, speaker=speaker
            )
            stt_ms = tr.latency_ms
            if self.src_lang is None:
                majority = self._majority_lang()
                if tr.lang_prob < LANG_CONFIDENT and majority and majority != tr.lang:
                    tr = await self.stt.transcribe(
                        seg, lang=majority, prompt=prompt, final=True, speaker=speaker
                    )
                    stt_ms += tr.latency_ms
                elif tr.lang_prob >= LANG_CONFIDENT:
                    self._lang_votes.append(tr.lang)
        except Exception as e:
            log.exception("final STT 실패 seg=%s speaker=%s", seg.seg_id, speaker)
            who = f" ({speaker})" if speaker else ""
            self._emit(
                "error",
                error_payload("E_STT_FAILED", f"발화 {seg.seg_id}{who} 인식 실패: {e}", True),
            )
            return None
        # 환각 방지: 같은 화자의 직전과 똑같은 문장 반복은 비운다 (§4.3)
        if tr.text and tr.text == self._prev_text.get(speaker):
            tr.text, tr.words = "", []
        if tr.text:
            self._prev_text[speaker] = tr.text
            if speaker is not None:
                self._prev_text[None] = tr.text  # 중간 결과 프롬프트용 (화자 무관 최신 문장)
        return tr, stt_ms

    def _majority_lang(self) -> str | None:
        if len(self._lang_votes) < LANG_MIN_VOTES:
            return None
        return Counter(self._lang_votes).most_common(1)[0][0]

    def _lang_hint(self) -> str | None:
        return self.src_lang or self._majority_lang()
