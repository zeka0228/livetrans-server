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
import json
import logging
import uuid
import wave
from collections import Counter, deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from livetrans.core.config import Settings
from livetrans.core.events import (
    Envelope,
    EventFactory,
    error_payload,
    session_state_payload,
)
from livetrans.core.speaker import select_speakers, speaker_extent
from livetrans.pipeline.lanes import LanePartial, LaneSegmenter, LaneUtterance
from livetrans.pipeline.segmenter import PartialRequest, Segmenter, SpeechSegment
from livetrans.plugins.diar.base import Diarizer
from livetrans.plugins.mt import FallbackTranslator, load_glossary
from livetrans.plugins.mt.base import Translation, TranslationError, Translator
from livetrans.plugins.source.base import Source
from livetrans.plugins.source.loopback import CaptureError
from livetrans.plugins.stt.base import STTEngine, Transcript
from livetrans.plugins.stt.guard import hallucination_reason, salvage_prefix

log = logging.getLogger("livetrans.session")

PARTIAL_INTERVAL_S = 1.0
LANG_CONFIDENT = 0.7  # 자동 감지 확률이 이보다 낮으면 최근 다수결 언어로 다시 인식 (§4.3)
LANG_VOTES = 10
LANG_MIN_VOTES = 3
REPEAT_SUPPRESS_MAX_MS = 1500  # 직전 문장과 같은 결과를 비우는 발화 길이 상한


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
        tgt_lang: str = "ko",
        session_id: str | None = None,
        diarizer: Diarizer | None = None,
        rescue_stt: STTEngine | None = None,
        stream_diarizer=None,
        translator: Translator | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.mode = mode
        self.src_lang = src_lang
        self.tgt_lang = (tgt_lang or "ko").split("-")[0].lower()
        self.stt = stt
        self.source = source
        self.diarizer = diarizer
        self.rescue_stt = rescue_stt  # 환각 판정 시 그 발화만 다시 인식 (없으면 살리기만)
        # D-028 스트림 모드: 연속 오디오 화자 분할 + 화자 레인. 있으면 VAD 세그먼터 대신 레인이 발화를 자른다
        self.stream = stream_diarizer
        self._vad_cfg = settings.vad
        self._lanes: dict[str, LaneSegmenter] = {}
        self._lane_partials: dict[str, LanePartial] = {}
        self._next_seg_id = 0
        # 번역 (명세 §4.4): 확정 자막마다 비동기로 번역해 translation.final. 화자별 문맥 쌍 유지
        self.translator = translator
        self._mt_cfg = settings.mt
        self._mt_queue: asyncio.Queue[tuple[int, str | None, str, str] | None] = asyncio.Queue()
        self._mt_context: dict[str | None, deque[tuple[str, str]]] = {}
        self._glossary_cache: dict[str, dict[str, str]] = {}
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
        self._final_done: set[int] = set()  # 확정을 보낸 seg_id (중간 결과 억제용)
        self._prev_text: dict[
            str | None, str
        ] = {}  # 화자별 직전 확정 문장 (프롬프트·반복 제거, §4.3)
        self._lang_votes: dict[str | None, deque[str]] = {}  # 화자별 최근 감지 언어 (다수결 힌트)
        self._tasks: list[asyncio.Task] = []
        self._stopping: asyncio.Task | None = None
        self._log_file = None
        self._dump_dir: Path | None = None
        self._open_logs(settings.logging)

    # --- 수명 주기 ---

    async def start(self) -> None:
        if not self._start_source_on_begin:
            await self.source.start()
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name=f"ingest-{self.session_id}"),
            asyncio.create_task(self._stt_loop(), name=f"stt-{self.session_id}"),
            asyncio.create_task(self._mt_loop(), name=f"mt-{self.session_id}"),
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
        self._close_logs()

    async def abort(self) -> None:
        """연결이 끊겨 결과를 받을 곳이 없을 때: 남은 작업을 버린다."""
        await self.source.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._discard_speakers()
        self.state = "stopped"
        self._close_logs()

    @property
    def stopped(self) -> bool:
        return self.state == "stopped"

    # --- 세션 기록 (명세 §11) ---

    def _open_logs(self, cfg) -> None:
        """이벤트 JSONL·발화 오디오 덤프 파일을 연다. 못 열면 경고만 남기고 기록 없이 간다."""
        if not (cfg.per_session_jsonl or cfg.dump_audio):
            return
        stamp = f"{datetime.now():%Y%m%d}_{self.session_id}"
        base = Path(cfg.dir)
        try:
            if cfg.per_session_jsonl:
                base.mkdir(parents=True, exist_ok=True)
                self._log_file = (base / f"{stamp}.jsonl").open("a", encoding="utf-8")
            if cfg.dump_audio:
                self._dump_dir = base / "audio" / stamp
                self._dump_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:  # 디스크·권한 문제로 세션이 죽으면 안 된다
            log.warning("세션 기록을 열 수 없음 (%s): %s", base, e)
            self._log_file, self._dump_dir = None, None

    def _log_line(self, record: dict) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self._log_file.flush()
        except OSError as e:
            log.warning("세션 기록 쓰기 실패: %s", e)
            self._log_file = None

    def _dump_segment(self, seg: SpeechSegment) -> None:
        """확정 발화 PCM 을 WAV 로 남기고, 화자 구간과 함께 JSONL 에 적는다 (오프라인 엔진 비교·화자 분할 분석용)."""
        if self._dump_dir is None:
            return
        path = self._dump_dir / f"seg{seg.seg_id:04d}.wav"
        try:
            pcm16 = (np.clip(seg.pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm16)
        except OSError as e:
            log.warning("발화 덤프 실패 seg=%s: %s", seg.seg_id, e)
            return
        self._log_line(
            {
                "type": "debug.segment",
                "seg_id": seg.seg_id,
                "t_start_ms": seg.t_start_ms,
                "t_end_ms": seg.t_end_ms,
                "forced_cut": seg.forced_cut,
                "file": str(path),
                "speakers": list(seg.speakers),
                "speaker_turns": [
                    {
                        "speaker": t.speaker,
                        "t_start_ms": t.t_start_ms,
                        "t_end_ms": t.t_end_ms,
                        "overlap": t.overlap,
                    }
                    for t in seg.speaker_turns
                ],
            }
        )

    def _close_logs(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def _guard(self, tr: Transcript, seg_id: int, kind: str, outcome: str = "비움") -> Transcript:
        """환각(반복·저엔트로피) 텍스트는 비운다. 비운 문장은 프롬프트로도 쓰이지 않는다 (guard.py)."""
        why = hallucination_reason(tr.text)
        if why:
            log.warning(
                "환각 의심 %s seg=%s (%s) → %s: %r", kind, seg_id, why, outcome, tr.text[:80]
            )
            self._log_line(
                {
                    "type": "debug.hallucination",
                    "kind": kind,
                    "seg_id": seg_id,
                    "reason": why,
                    "outcome": outcome,
                    "text": tr.text[:200],
                }
            )
            tr.text, tr.words = "", []
        return tr

    async def _recover_final(
        self, seg: SpeechSegment, tr: Transcript, speaker: str | None, lang: str | None
    ) -> tuple[Transcript, int]:
        """확정 결과가 환각이면 복구 체인: ① 같은 엔진 반복 억제 재시도 ② 구조 엔진으로 화자 구간만 재인식 ③ 반복 직전까지 살리기.

        돌려주는 Transcript 는 환각이 아니거나 빈 문장이다. 두 번째 값은 추가로 쓴 ms.
        """
        why = hallucination_reason(tr.text)
        if not why:
            return tr, 0
        extra = 0
        bad = tr.text
        # ① 반복 억제 재시도 (같은 엔진, 같은 마스크)
        try:
            retry = await self.stt.transcribe(
                seg, lang=lang, prompt=None, final=True, speaker=speaker, anti_repeat=True
            )
            extra += retry.latency_ms
            if retry.text and not hallucination_reason(retry.text):
                self._note_hallucination(seg.seg_id, why, bad, "재시도 성공")
                return retry, extra
        except Exception as e:  # noqa: BLE001
            log.warning("환각 재시도 실패 seg=%s: %s", seg.seg_id, e)
        # ② 구조 엔진: 화자가 있으면 그 화자가 말한 범위만 잘라서
        if self.rescue_stt is not None:
            sub = self._speaker_crop(seg, speaker)
            try:
                res = await self.rescue_stt.transcribe(sub, lang=lang, prompt=None, final=True)
                extra += res.latency_ms
                if res.text and not hallucination_reason(res.text):
                    res.speaker = speaker
                    self._note_hallucination(
                        seg.seg_id, why, bad, f"구조 엔진 {self.rescue_stt.name}"
                    )
                    return res, extra
            except Exception as e:  # noqa: BLE001
                log.warning("구조 STT 실패 seg=%s: %s", seg.seg_id, e)
        # ③ 반복 직전까지 살리기
        kept = salvage_prefix(bad)
        self._note_hallucination(seg.seg_id, why, bad, f"살림 {kept!r}" if kept else "비움")
        tr.text, tr.words = kept, []
        return tr, extra

    def _note_hallucination(self, seg_id: int, why: str, text: str, outcome: str) -> None:
        log.warning("환각 의심 final seg=%s (%s) → %s: %r", seg_id, why, outcome, text[:80])
        self._log_line(
            {
                "type": "debug.hallucination",
                "kind": "final",
                "seg_id": seg_id,
                "reason": why,
                "outcome": outcome,
                "text": text[:200],
            }
        )

    @staticmethod
    def _speaker_crop(seg: SpeechSegment, speaker: str | None) -> SpeechSegment:
        """화자가 말한 범위(첫 시작~마지막 끝)만 잘라 낸 발화. 화자가 없으면 그대로."""
        extent = speaker_extent(seg.speaker_turns, speaker) if speaker else None
        if extent is None:
            return seg
        a, b = extent
        i0 = max(0, (a - seg.t_start_ms) * 16)
        i1 = min(len(seg.pcm), (b - seg.t_start_ms) * 16)
        if i1 - i0 < 1600:  # 0.1 s 미만이면 자르지 않는다
            return seg
        return SpeechSegment(
            seg_id=seg.seg_id,
            t_start_ms=a,
            t_end_ms=b,
            pcm=seg.pcm[i0:i1],
            forced_cut=seg.forced_cut,
            prev_seg_id=seg.prev_seg_id,
        )

    # --- 내부 ---

    def _discard_speakers(self) -> None:
        """세션 종료: 화자 임베딩·문맥 오디오 폐기 (§13)."""
        if self.diarizer is not None:
            self.diarizer.reset()
        if self.stream is not None:
            self.stream.reset()

    def _emit(self, type_: str, payload: dict) -> None:
        env = self.events.make(type_, payload)
        self.outbox.put_nowait(env)
        self._log_line(env.to_dict())

    def _emit_state(self, state: str) -> None:
        self.state = state
        self._emit("session.state", session_state_payload(state, delay_s=None))

    async def _ingest_loop(self) -> None:
        if self.stream is not None:
            await self._ingest_stream()
        else:
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
            elif self._lane_partials:
                speaker = next(iter(self._lane_partials))
                await self._run_partial(self._lane_partials.pop(speaker))
            elif self._partial is not None:
                req, self._partial = self._partial, None
                await self._run_partial(req)
            elif self._ingest_done:
                self._mt_queue.put_nowait(None)  # STT 끝 → 번역 루프도 남은 것만 처리하고 끝낸다
                return
            else:
                await self._wake.wait()
                self._wake.clear()

    # --- 스트림 모드 (D-028) ---

    def _alloc_seg_id(self) -> int:
        seg_id = self._next_seg_id
        self._next_seg_id += 1
        return seg_id

    async def _ingest_stream(self) -> None:
        sd = self.stream
        async for frame in self.source.frames():
            sd.push(frame.pcm)
            if sd.ready:
                await self._stream_step(final=False)
        await self._stream_step(final=True)
        for lane in self._lanes.values():
            for utt in lane.flush():
                self._finals.append(self._lane_segment(utt))
        self._lane_partials.clear()
        self._wake.set()

    async def _stream_step(self, *, final: bool) -> None:
        sd = self.stream
        try:
            turns = await sd.step(final=final)
        except Exception as e:  # noqa: BLE001 — 화자 분할이 죽어도 세션은 유지 (이 창은 건너뜀)
            log.warning("스트리밍 화자 분할 실패: %s", e)
            return
        by_speaker: dict[str, list[tuple[int, int]]] = {}
        for t in turns:
            by_speaker.setdefault(t.speaker, []).append((t.t_start_ms, t.t_end_ms))
        committed = sd.committed_ms
        for speaker in set(by_speaker) | set(self._lanes):
            lane = self._lanes.get(speaker)
            if lane is None:
                lane = self._lanes[speaker] = LaneSegmenter(
                    speaker, self._vad_cfg, self._alloc_seg_id
                )
            for utt in lane.push(by_speaker.get(speaker, []), committed):
                self._finals.append(self._lane_segment(utt))
            span = lane.open_span
            if span is not None and not final:
                seg_id, a, b = span
                self._lane_partials[speaker] = LanePartial(
                    seg_id=seg_id,
                    speaker=speaker,
                    t_start_ms=a,
                    t_end_ms=b,
                    pcm=sd.audio(a, b),
                    turns=sd.committed_turns(a, b),
                )
        self._wake.set()

    def _lane_segment(self, utt: LaneUtterance) -> SpeechSegment:
        """레인 발화 → STT 입력. 다른 화자의 겹친 구간도 함께 넣어 화자 마스크가 서게 한다."""
        turns = self.stream.committed_turns(utt.t_start_ms, utt.t_end_ms)
        if not any(t.speaker == utt.speaker for t in turns):
            turns = utt.turns + turns
        others = tuple(sorted({t.speaker for t in turns if t.speaker != utt.speaker}))
        return SpeechSegment(
            seg_id=utt.seg_id,
            t_start_ms=utt.t_start_ms,
            t_end_ms=utt.t_end_ms,
            pcm=self.stream.audio(utt.t_start_ms, utt.t_end_ms),
            forced_cut=utt.forced_cut,
            prev_seg_id=utt.prev_seg_id,
            speakers=(utt.speaker, *others),
            speaker_turns=turns,
        )

    # --- 번역 (명세 §4.4) ---

    async def _mt_loop(self) -> None:
        while True:
            item = await self._mt_queue.get()
            if item is None:
                return
            seg_id, speaker, text, lang = item
            await self._translate_one(seg_id, speaker, text, lang)

    def _queue_translation(self, seg_id: int, speaker: str | None, text: str, lang: str) -> None:
        if text:
            self._mt_queue.put_nowait((seg_id, speaker, text, lang))

    def _glossary(self, src: str) -> dict[str, str]:
        key = f"{src}->{self.tgt_lang}"
        if key not in self._glossary_cache:
            self._glossary_cache[key] = load_glossary(self._mt_cfg.glossary, src, self.tgt_lang)
        return self._glossary_cache[key]

    async def _translate_one(self, seg_id: int, speaker: str | None, text: str, lang: str) -> None:
        src = (lang or "").split("-")[0].lower()
        if self.translator is None or not src or src == self.tgt_lang:
            # 번역 엔진이 없거나 같은 언어: 원문을 그대로 (앱은 translation.final 이 있어야 줄이 완성된다)
            tr = Translation(
                seg_id, text, src or self.tgt_lang, self.tgt_lang, "passthrough", False, 0, speaker
            )
        else:
            context = list(self._mt_context.get(speaker, ()))
            glossary = self._glossary(src)
            try:
                if isinstance(self.translator, FallbackTranslator):
                    tr = await self.translator.translate_full(
                        seg_id,
                        text,
                        src=src,
                        tgt=self.tgt_lang,
                        context=context,
                        glossary=glossary,
                        speaker=speaker,
                    )
                else:
                    t0 = asyncio.get_running_loop().time()
                    out = await self.translator.translate(
                        text, src=src, tgt=self.tgt_lang, context=context, glossary=glossary
                    )
                    tr = Translation(
                        seg_id,
                        out,
                        src,
                        self.tgt_lang,
                        self.translator.name,
                        False,
                        int((asyncio.get_running_loop().time() - t0) * 1000),
                        speaker,
                    )
            except TranslationError as e:
                log.warning("번역 실패 seg=%s → 원문: %s", seg_id, e)
                tr = Translation(seg_id, text, src, self.tgt_lang, "passthrough", True, 0, speaker)
            except Exception:
                log.exception("번역 오류 seg=%s", seg_id)
                tr = Translation(seg_id, text, src, self.tgt_lang, "passthrough", True, 0, speaker)
            if tr.text and tr.engine != "passthrough":
                ctx = self._mt_context.setdefault(speaker, deque(maxlen=self._mt_cfg.context_pairs))
                ctx.append((text, tr.text))
        payload = {
            "seg_id": tr.seg_id,
            "text": tr.text,
            "src_lang": tr.src_lang,
            "tgt_lang": tr.tgt_lang,
            "engine": tr.engine,
            "fallback": tr.fallback,
            "mt_ms": tr.latency_ms,
        }
        if speaker is not None:
            payload["speaker"] = speaker
        self._emit("translation.final", payload)

    # --- 중간 결과 ---

    async def _run_partial(self, req: PartialRequest | LanePartial) -> None:
        if req.seg_id in self._final_done:
            return
        speaker = getattr(req, "speaker", None)
        if isinstance(req, LanePartial):
            others = tuple(sorted({t.speaker for t in req.turns if t.speaker != speaker}))
            seg = SpeechSegment(
                seg_id=req.seg_id,
                t_start_ms=req.t_start_ms,
                t_end_ms=req.t_end_ms,
                pcm=req.pcm,
                speakers=(speaker, *others),
                speaker_turns=req.turns,
            )
        else:
            seg = SpeechSegment(
                seg_id=req.seg_id,
                t_start_ms=req.t_start_ms,
                t_end_ms=req.t_start_ms + len(req.pcm) * 1000 // 16000,
                pcm=req.pcm,
            )
        if len(seg.pcm) < 1600:  # 0.1 s 미만은 돌리지 않는다
            return
        mask = speaker if getattr(self.stt, "supports_speaker_mask", False) else None
        try:
            tr = await self.stt.transcribe(
                seg,
                lang=self._lang_hint(speaker),
                prompt=self._prev_text.get(speaker) or None,
                final=False,
                speaker=mask,
            )
        except Exception as e:  # noqa: BLE001 — 중간 결과 실패는 경고만 남기고 넘긴다
            log.warning("partial STT 실패 seg=%s: %s", req.seg_id, e)
            return
        tr = self._guard(tr, req.seg_id, "partial")
        # 인식하는 동안 같은 발화의 확정 결과가 먼저 나갔으면 보내지 않는다
        if req.seg_id in self._final_done or not tr.text:
            return
        payload = {
            "seg_id": req.seg_id,
            "t_start_ms": req.t_start_ms,
            "text": tr.text,
            "lang": tr.lang,
        }
        if speaker is not None:
            payload["speaker"] = speaker
        self._emit("segment.partial", payload)

    async def _run_final(self, seg: SpeechSegment) -> None:
        if self.stream is not None:
            diar_ms = self.stream.last_latency_ms  # 레인 발화는 화자가 이미 정해져 있다
        else:
            diar_ms = await self._attach_speakers(seg)
        self._dump_segment(seg)
        if self.stream is not None:
            # 레인 발화: 이 레인의 화자 1명만 인식 (다른 화자는 마스크로만 쓴다)
            targets: list[str | None] = [
                seg.speakers[0] if getattr(self.stt, "supports_speaker_mask", False) else None
            ]
        elif seg.speakers and getattr(self.stt, "supports_speaker_mask", False):
            # 화자 마스크를 지원하는 엔진이면 화자마다 1회, 아니면 구간 전체 1회 (주 화자만 표시)
            targets = list(seg.speakers)
        else:
            targets = [None]
        results: list[tuple[str | None, Transcript, int]] = []
        for speaker in targets:
            got = await self._transcribe_final(seg, speaker)
            if got is not None:
                results.append((speaker, *got))
        self._final_done.add(seg.seg_id)
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
            self._queue_translation(seg.seg_id, shown, tr.text, tr.lang)

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
                majority = self._majority_lang(speaker)
                if tr.lang_prob < LANG_CONFIDENT and majority and majority != tr.lang:
                    tr = await self.stt.transcribe(
                        seg, lang=majority, prompt=prompt, final=True, speaker=speaker
                    )
                    stt_ms += tr.latency_ms
                elif tr.lang_prob >= LANG_CONFIDENT:
                    self._lang_votes.setdefault(speaker, deque(maxlen=LANG_VOTES)).append(tr.lang)
            tr, extra_ms = await self._recover_final(seg, tr, speaker, self.src_lang or tr.lang)
            stt_ms += extra_ms
        except Exception as e:
            log.exception("final STT 실패 seg=%s speaker=%s", seg.seg_id, speaker)
            who = f" ({speaker})" if speaker else ""
            self._emit(
                "error",
                error_payload("E_STT_FAILED", f"발화 {seg.seg_id}{who} 인식 실패: {e}", True),
            )
            return None
        # 환각 방지: 같은 화자의 직전과 똑같은 문장이 **짧은 발화**에서 또 나오면 프롬프트 되풀이로 보고 비운다 (§4.3).
        # 긴 발화는 실제로 같은 말을 반복한 것일 수 있어 남긴다 (합성음 E2E 에서 3.5 s 문장이 사라진 사례)
        if (
            tr.text
            and tr.text == self._prev_text.get(speaker)
            and seg.t_end_ms - seg.t_start_ms <= REPEAT_SUPPRESS_MAX_MS
        ):
            tr.text, tr.words = "", []
        if tr.text:
            self._prev_text[speaker] = tr.text
            if speaker is not None:
                self._prev_text[None] = tr.text  # 중간 결과 프롬프트용 (화자 무관 최신 문장)
        return tr, stt_ms

    def _majority_lang(self, speaker: str | None = None) -> str | None:
        """그 화자의 최근 확정 언어 다수결 (화자가 없으면 전체). 투표가 적으면 None."""
        votes = self._lang_votes.get(speaker)
        if votes is None and speaker is None:
            votes = deque(v for d in self._lang_votes.values() for v in d)
        if not votes or len(votes) < LANG_MIN_VOTES:
            return None
        return Counter(votes).most_common(1)[0][0]

    def _lang_hint(self, speaker: str | None = None) -> str | None:
        return self.src_lang or self._majority_lang(speaker)
