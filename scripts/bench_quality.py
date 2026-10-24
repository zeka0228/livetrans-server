"""품질 벤치 (STT · 번역). 서버 저장소 루트에서 실행. 결과는 runtime/bench/ 에 Markdown 으로 남긴다 (gitignore).

uv run python -X utf8 scripts/bench_quality.py stt [--dumps 30] [--seed 0]
    ① 합성음 정답 세트 (make_test_media 의 문장 + 추가 문장, Windows SAPI ko/en) → 엔진별 CER
    ② logs/audio 덤프 무작위 N개 → 엔진별 전사를 나란히 (정답 없음: 당시 서버 결과와 비교·눈으로 판단)
    엔진: 설정의 기본 엔진(dicow) + 구조 엔진(faster_whisper). GPU 를 쓰므로 서버는 내려 두고 돌린다.
uv run python -X utf8 scripts/bench_quality.py mt --models gemma2:9b,qwen2.5:7b [--n 40]
    ja→ko 20 + en→ko 20 문장(참조 번역 포함) → 모델별 chrF, 형식 실패, 문자 체계 오류, 지연. Ollama 가 모델을 바꿔 올린다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
import unicodedata
import wave
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("runtime/bench")

# ---------------------------------------------------------------- 정답 세트 (합성음) -------------------------
SYNTH_LINES = [
    ("ko", "안녕하세요. 오늘은 실시간 방송 번역기 모드 에이 테스트를 해 보겠습니다."),
    ("ko", "보시는 화면은 테스트 패턴이고, 자막과 음성이 영상에 맞춰 나오는지 확인하는 중입니다."),
    ("ko", "마지막 문장입니다. 수고하셨습니다."),
    ("ko", "지금부터 보스전 들어갈게요. 체력 물약 먼저 채우고 가겠습니다."),
    ("ko", "채팅 고마워요. 다음 방송은 금요일 저녁 여덟 시에 켤게요."),
    ("ko", "이 구간은 점프 타이밍이 어려워서 세 번쯤 죽을 것 같아요."),
    (
        "en",
        "Hello everyone, the weather is really nice today, so let's go for a walk after this stream.",
    ),
    ("en", "If you can hear the Korean voice over my sentence, the dubbing pipeline is working."),
    ("en", "Let me check the inventory first, then we head to the next checkpoint."),
    ("en", "Thanks for the follow, I will read the chat after this fight."),
    ("en", "This boss has three phases and the second one is the hardest."),
    ("en", "I think we should save the game here before trying the hidden route."),
]

# ---------------------------------------------------------------- 번역 세트 (참조 포함) -------------------------
MT_SET = [
    # ja → ko (방송 구어체, 게임·일상)
    (
        "ja",
        "今日はこの新しいステージを最後までクリアしていきたいと思います",
        "오늘은 이 새 스테이지를 끝까지 깨 보려고 합니다",
    ),
    ("ja", "ちょっと待って、回復アイテムがもう一個もない", "잠깐만, 회복 아이템이 하나도 없어"),
    ("ja", "コメントありがとう、後でまとめて読むね", "댓글 고마워, 이따가 몰아서 읽을게"),
    (
        "ja",
        "このボス、二段階目から攻撃パターンが変わるんだよね",
        "이 보스, 2페이즈부터 공격 패턴이 바뀌거든",
    ),
    ("ja", "さっきのジャンプ、タイミング完全にミスった", "아까 그 점프, 타이밍 완전히 놓쳤어"),
    (
        "ja",
        "明日は配信お休みで、次は金曜日の夜八時からです",
        "내일은 방송 쉬고, 다음은 금요일 밤 여덟 시부터입니다",
    ),
    (
        "ja",
        "この武器、見た目はいいけど火力が全然足りない",
        "이 무기, 생긴 건 좋은데 화력이 전혀 안 나와",
    ),
    (
        "ja",
        "えっ、今の音なに？後ろから何か来てる？",
        "어? 방금 그 소리 뭐야? 뒤에서 뭐가 오고 있어?",
    ),
    (
        "ja",
        "初見さんいらっしゃい、ゆっくりしていってね",
        "처음 오신 분 어서 오세요, 편하게 있다 가세요",
    ),
    (
        "ja",
        "セーブしてから隠しルートに挑戦してみようか",
        "세이브하고 나서 숨겨진 루트에 도전해 볼까",
    ),
    (
        "ja",
        "この曲めっちゃ好きなんだけど、歌詞が全然覚えられない",
        "이 노래 진짜 좋아하는데, 가사가 전혀 안 외워져",
    ),
    (
        "ja",
        "画面がカクつくのは多分配信ソフトのせいだと思う",
        "화면이 끊기는 건 아마 방송 프로그램 때문인 것 같아",
    ),
    (
        "ja",
        "もう一回だけやらせて、今度こそいける気がする",
        "딱 한 번만 더 하게 해 줘, 이번엔 될 것 같아",
    ),
    (
        "ja",
        "この街のBGM、夜になると雰囲気変わるの好き",
        "이 마을 배경음악, 밤이 되면 분위기 바뀌는 게 좋아",
    ),
    (
        "ja",
        "敵の体力バーが見えないから残りがわからない",
        "적 체력 바가 안 보여서 남은 양을 모르겠어",
    ),
    ("ja", "スパチャありがとうございます、大事に使います", "슈퍼챗 감사합니다, 소중하게 쓸게요"),
    ("ja", "今の操作、キーボードだとちょっと難しいかも", "지금 조작은 키보드로는 조금 어려울지도"),
    (
        "ja",
        "次の配信ではみんなと一緒に協力プレイしたいな",
        "다음 방송에서는 다 같이 협동 플레이 하고 싶어",
    ),
    (
        "ja",
        "あ、落ちた。この崖、見えないところに穴あるじゃん",
        "아, 떨어졌다. 이 절벽, 안 보이는 데 구멍 있잖아",
    ),
    (
        "ja",
        "ここで一旦休憩入れて、十分後に再開します",
        "여기서 잠깐 휴식하고 10분 뒤에 다시 시작할게요",
    ),
    # en → ko
    (
        "en",
        "okay chat let me grab a potion before we go into the boss room",
        "자 채팅 여러분, 보스방 들어가기 전에 물약 하나 챙길게요",
    ),
    ("en", "that jump was way harder than it looked", "저 점프 보기보다 훨씬 어려웠어"),
    ("en", "thanks for the sub i really appreciate it", "구독 고마워요, 정말 감사해요"),
    ("en", "i think the second phase is where most people die", "2페이즈에서 대부분 죽는 것 같아"),
    (
        "en",
        "hold on my mic was muted for the last minute",
        "잠깐, 지난 1분 동안 마이크가 꺼져 있었네",
    ),
    (
        "en",
        "we are going to save here and try the hidden route",
        "여기서 세이브하고 숨겨진 루트를 가 볼게요",
    ),
    ("en", "this weapon looks cool but the damage is terrible", "이 무기 멋있는데 데미지가 끔찍해"),
    (
        "en",
        "no stream tomorrow the next one is friday at eight",
        "내일은 방송 없고 다음은 금요일 여덟 시예요",
    ),
    ("en", "did you hear that something is behind us", "저 소리 들었어? 뒤에 뭐가 있어"),
    (
        "en",
        "welcome in if you are new here make yourself at home",
        "처음 오셨으면 환영해요, 편하게 계세요",
    ),
    (
        "en",
        "my game keeps stuttering i think it is the capture software",
        "게임이 계속 끊기는데 캡처 프로그램 때문인 것 같아",
    ),
    ("en", "one more try i swear i can do it this time", "한 번만 더, 이번엔 진짜 할 수 있어"),
    (
        "en",
        "the music in this town changes at night and i love it",
        "이 마을 음악이 밤에 바뀌는데 너무 좋아",
    ),
    (
        "en",
        "i cannot see the boss health bar so i have no idea how much is left",
        "보스 체력 바가 안 보여서 얼마나 남았는지 모르겠어",
    ),
    (
        "en",
        "thank you for the donation i will put it toward a new mic",
        "후원 고마워요, 새 마이크 사는 데 보탤게요",
    ),
    ("en", "this part is kind of tricky on keyboard", "이 부분은 키보드로 하면 좀 까다로워"),
    (
        "en",
        "next stream we should do some co-op with the viewers",
        "다음 방송에서는 시청자분들이랑 협동 플레이 해요",
    ),
    (
        "en",
        "oh i fell there is a hole you cannot see from here",
        "아 떨어졌다, 여기서 안 보이는 구멍이 있어",
    ),
    (
        "en",
        "let us take a short break and come back in ten minutes",
        "잠깐 쉬고 10분 뒤에 돌아올게요",
    ),
    ("en", "the enemies respawn if you leave the area", "이 구역을 벗어나면 적이 다시 생겨요"),
]


# ---------------------------------------------------------------- 지표 ------------------------------------------
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\s\.,!?、。「」『』\"'…:;()\-~]+", "", s)
    return s


def cer(hyp: str, ref: str) -> float:
    h, r = norm(hyp), norm(ref)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
            prev = cur
    return d[len(h)] / len(r)


def chrf(hyp: str, ref: str, n_max: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015) 문자 n-gram F-score, 0~100."""
    h, r = norm(hyp), norm(ref)
    if not h or not r:
        return 0.0
    precs, recs = [], []
    for n in range(1, n_max + 1):
        hg = Counter(h[i : i + n] for i in range(len(h) - n + 1))
        rg = Counter(r[i : i + n] for i in range(len(r) - n + 1))
        if not hg or not rg:
            continue
        match = sum((hg & rg).values())
        precs.append(match / max(1, sum(hg.values())))
        recs.append(match / max(1, sum(rg.values())))
    if not precs:
        return 0.0
    p, rc = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + rc == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * rc / (beta**2 * p + rc)


# ---------------------------------------------------------------- STT -------------------------------------------
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    if sr != 16000:
        t_src = np.arange(len(pcm)) / sr
        t_dst = np.arange(int(len(pcm) * 16000 / sr)) / 16000
        pcm = np.interp(t_dst, t_src, pcm).astype(np.float32)
    return pcm


async def synth_lines(settings):
    from livetrans.core.voices import VoiceRegistry
    from livetrans.main import _load_tts

    engines = await _load_tts(settings)
    voices = VoiceRegistry(engines, settings.tts.defaults)
    await voices.refresh()
    out = []
    for lang, text in SYNTH_LINES:
        info = voices.resolve({"ko": "ko-KR", "en": "en-US", "ja": "ja-JP"}[lang])
        eng_name, _ = info.voice_ref.split(":", 1)
        clip = await engines[eng_name].synthesize(text, info.display_name)
        pcm = clip.pcm
        if clip.sample_rate != 16000:
            t_src = np.arange(len(pcm)) / clip.sample_rate
            t_dst = np.arange(int(len(pcm) * 16000 / clip.sample_rate)) / 16000
            pcm = np.interp(t_dst, t_src, pcm).astype(np.float32)
        out.append((lang, text, pcm))
    return out


def pick_dumps(n: int, seed: int):
    rng = random.Random(seed)
    items = []
    for d in sorted(Path("logs/audio").glob("*")):
        jsonl = Path("logs") / f"{d.name}.jsonl"
        texts = {}
        if jsonl.is_file():
            for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "segment.final":
                    p = ev["payload"]
                    texts[p["seg_id"]] = (p.get("text", ""), p.get("lang", ""), p.get("speaker"))
        for wav in sorted(d.glob("seg*.wav")):
            sid = int(wav.stem[3:])
            items.append((d.name, wav, *texts.get(sid, ("", "", None))))
    rng.shuffle(items)
    return items[:n]


async def bench_stt(args) -> None:
    from livetrans.core.config import SttConfig, load_settings
    from livetrans.core.gpu import prepare_cuda_dlls
    from livetrans.pipeline.segmenter import SpeechSegment
    from livetrans.plugins.stt import create_stt

    settings = load_settings()
    prepare_cuda_dlls()
    engines = {}
    primary = create_stt(settings.stt)
    await primary.load()
    engines[primary.name] = primary
    if settings.stt.rescue and settings.stt.rescue != settings.stt.engine:
        alt = create_stt(
            SttConfig(
                **(
                    settings.stt.model_dump()
                    | {
                        "engine": settings.stt.rescue,
                        "model": settings.stt.rescue_model,
                        "fallback": None,
                    }
                )
            )
        )
        await alt.load()
        engines[alt.name] = alt
    names = list(engines)
    lines = [
        f"# STT 벤치 {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"엔진: {', '.join(f'{n}({engines[n].model_id})' for n in names)}",
        "",
    ]

    async def run(engine, pcm, lang_hint):
        seg = SpeechSegment(seg_id=0, t_start_ms=0, t_end_ms=len(pcm) * 1000 // 16000, pcm=pcm)
        t0 = time.perf_counter()
        tr = await engine.transcribe(seg, lang=lang_hint, prompt=None, final=True)
        return tr, (time.perf_counter() - t0) * 1000

    # ① 합성음 정답 세트
    try:
        synth = await synth_lines(settings)
    except Exception as e:  # noqa: BLE001
        synth = []
        lines.append(f"합성음 세트 생략 (TTS 실패: {e})")
    if synth:
        lines += ["## ① 합성음 정답 세트 (CER, 낮을수록 좋음; 언어 자동 감지)", ""]
        lines.append("| 언어 | 문장 | " + " | ".join(f"{n} CER / 감지 / ms" for n in names) + " |")
        lines.append("|---|---|" + "---|" * len(names))
        agg = {n: {"ko": [], "en": []} for n in names}
        for lang, text, pcm in synth:
            cells = []
            for n in names:
                tr, ms = await run(engines[n], pcm, None)
                c = cer(tr.text, text)
                agg[n][lang].append(c)
                cells.append(f"{c:.2f} / {tr.lang} / {ms:.0f}<br>`{tr.text[:60]}`")
            lines.append(f"| {lang} | {text[:40]} | " + " | ".join(cells) + " |")
        lines += ["", "| 엔진 | ko 평균 CER | en 평균 CER |", "|---|---|---|"]
        for n in names:
            ko = agg[n]["ko"]
            en = agg[n]["en"]
            lines.append(
                f"| {n} | {sum(ko) / max(1, len(ko)):.3f} | {sum(en) / max(1, len(en)):.3f} |"
            )
        lines.append("")

    # ② 실방송 덤프
    dumps = pick_dumps(args.dumps, args.seed)
    if dumps:
        lines += [
            f"## ② 실방송 덤프 {len(dumps)}개 (정답 없음 — 당시 서버 결과와 나란히, 눈으로 판단)",
            "",
        ]
        lines.append(
            "| 세션/파일 | 길이 s | 당시 결과 (lang) | "
            + " | ".join(f"{n} (감지, ms)" for n in names)
            + " |"
        )
        lines.append("|---|---|---|" + "---|" * len(names))
        agree = Counter()
        empty = Counter()
        for sess, wav, text, lang, spk in dumps:
            pcm = read_wav(wav)
            cells = []
            outs = {}
            for n in names:
                tr, ms = await run(engines[n], pcm, None)
                outs[n] = tr.text
                if not tr.text.strip():
                    empty[n] += 1
                cells.append(f"`{tr.text[:70]}` ({tr.lang}, {ms:.0f})")
            if len(names) == 2 and norm(outs[names[0]]) == norm(outs[names[1]]):
                agree["same"] += 1
            lines.append(
                f"| {sess[-6:]}/{wav.name} | {len(pcm) / 16000:.1f} | `{text[:60]}` ({lang}{', ' + spk if spk else ''}) | "
                + " | ".join(cells)
                + " |"
            )
        lines += [
            "",
            f"두 엔진 결과 동일: {agree['same']}/{len(dumps)} · 빈 결과: "
            + ", ".join(f"{n} {empty[n]}" for n in names),
            "",
        ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{time.strftime('%Y%m%d_%H%M')}_stt.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "\n".join(
            line
            for line in lines
            if not line.startswith("| ") or "평균" in line or line.count("|") < 6
        )
    )
    print(f"\n→ {out}")


# ---------------------------------------------------------------- MT --------------------------------------------
async def bench_mt(args) -> None:
    from livetrans.core.config import load_settings
    from livetrans.plugins.mt.base import TranslationError
    from livetrans.plugins.mt.ollama_llm import OllamaTranslator, script_problem

    settings = load_settings()
    items = MT_SET[: args.n] if args.n else MT_SET
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lines = [
        f"# 번역 벤치 {time.strftime('%Y-%m-%d %H:%M')} — {len(items)} 문장 (ja→ko {sum(1 for i in items if i[0] == 'ja')}, en→ko {sum(1 for i in items if i[0] == 'en')})",
        "",
    ]
    summary = []
    per_model_out: dict[str, list[str]] = {}
    for model in models:
        cfg = settings.mt.model_copy(update={"model": model, "timeout_s": 60.0})
        mt = OllamaTranslator(cfg)
        t0 = time.perf_counter()
        try:
            await mt.load()
        except TranslationError as e:
            summary.append(f"| {model} | 로드 실패: {e} | | | | |")
            continue
        load_s = time.perf_counter() - t0
        scores = {"ja": [], "en": []}
        lat = []
        fails = 0
        script_bad = 0
        outs = []
        for src, text, ref in items:
            t1 = time.perf_counter()
            try:
                out = await mt.translate(text, src=src, tgt="ko", context=[], glossary={})
            except TranslationError:
                out, fails = "", fails + 1
            ms = (time.perf_counter() - t1) * 1000
            lat.append(ms)
            if out and script_problem(out, "ko"):
                script_bad += 1
            s = chrf(out, ref)
            scores[src].append(s)
            outs.append(f"| {src} | {text} | {ref} | {out} | {s:.0f} | {ms:.0f} |")
        await mt.close()
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat) // 2]
        p95 = lat_sorted[int(len(lat) * 0.95) - 1] if len(lat) >= 20 else lat_sorted[-1]
        ja = scores["ja"]
        en = scores["en"]
        summary.append(
            f"| {model} | {sum(ja) / max(1, len(ja)):.1f} | {sum(en) / max(1, len(en)):.1f} | {fails} / {script_bad} | {p50:.0f} / {p95:.0f} | {load_s:.1f} |"
        )
        per_model_out[model] = outs
        print(summary[-1], flush=True)
    lines += [
        "| 모델 | chrF ja→ko | chrF en→ko | 형식 실패 / 문자 체계 오류 | 지연 p50 / p95 ms | 첫 로드 s |",
        "|---|---|---|---|---|---|",
        *summary,
        "",
    ]
    for model, outs in per_model_out.items():
        lines += [
            f"## {model}",
            "",
            "| src | 원문 | 참조 | 출력 | chrF | ms |",
            "|---|---|---|---|---|---|",
            *outs,
            "",
        ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{time.strftime('%Y%m%d_%H%M')}_mt.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: 4 + len(summary)]))
    print(f"\n→ {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stt")
    s.add_argument("--dumps", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("mt")
    m.add_argument("--models", required=True, help="쉼표로 구분한 Ollama 모델 이름")
    m.add_argument("--n", type=int, default=0, help="앞에서 N 문장만 (0 = 전부)")
    args = ap.parse_args()
    asyncio.run(bench_stt(args) if args.cmd == "stt" else bench_mt(args))


if __name__ == "__main__":
    main()
