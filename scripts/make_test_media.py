"""모드 A E2E 용 테스트 영상 (명세 §12): testsrc 화면(타이머) + SAPI 한·영 대사(2화자)를 섞어 MP4(h264, keyint 1 s)로 만든다.

사용: uv run python -X utf8 scripts\\make_test_media.py [출력 폴더=runtime/test_media]
권리: 합성음·테스트 패턴만 쓰므로 공개해도 된다 (권리 원칙). Windows SAPI(Heami ko / Zira en) 가 필요하다.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livetrans.output.hls import find_ffmpeg

SR = 48000
LINES = [
    ("ko", "안녕하세요. 오늘은 실시간 방송 번역기 모드 에이 테스트를 해 보겠습니다."),
    (
        "en",
        "Hello everyone. The weather is really nice today, so let's go for a walk after this stream.",
    ),
    (
        "ko",
        "지금 보시는 화면은 테스트 패턴이고, 자막과 음성이 영상에 맞춰 나오는지 확인하는 중입니다.",
    ),
    ("en", "If you can hear the Korean voice over my sentence, the dubbing pipeline is working."),
    ("ko", "마지막 문장입니다. 수고하셨습니다."),
]


def resample(pcm: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    t_src = np.arange(len(pcm)) / sr
    t_dst = np.arange(int(len(pcm) * target / sr)) / target
    return np.interp(t_dst, t_src, pcm).astype(np.float32)


async def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runtime/test_media")
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(None)
    if ffmpeg is None:
        sys.exit("ffmpeg 를 찾지 못했습니다 (tools.ffmpeg 또는 PATH)")
    from livetrans.plugins.tts.system import SystemTTS

    tts = SystemTTS()
    await tts.load()
    voices = {v.languages[0][:2]: v.display_name for v in await tts.list_voices() if v.languages}
    parts = [np.zeros(SR * 2, dtype=np.float32)]
    marks = []
    t = 2.0
    for lang, text in LINES:
        clip = await tts.synthesize(text, voices[lang])
        pcm = resample(clip.pcm, clip.sample_rate)
        if lang == "en":
            pcm = resample(pcm, int(SR * 0.8))  # 영어 화자 피치 낮춤 (화자 분할 테스트용)
        marks.append((t, t + len(pcm) / SR, lang, text))
        parts += [pcm * 0.8, np.zeros(int(SR * 1.5), dtype=np.float32)]
        t += len(pcm) / SR + 1.5
    await tts.close()
    audio = np.concatenate([*parts, np.zeros(SR * 2, dtype=np.float32)])
    wav = out_dir / "speech.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    dur = len(audio) / SR
    mp4 = out_dir / "test_broadcast.mp4"
    drawtext = "testsrc=size=1280x720:rate=30,drawtext=text='t=%{pts\\:hms}':fontsize=64:fontcolor=white:x=40:y=40:box=1:boxcolor=black@0.5"
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", drawtext,
            "-i", str(wav), "-t", f"{dur:.2f}",
            "-c:v", "libx264", "-preset", "veryfast", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", str(mp4),
        ],
        check=True,
    )  # fmt: skip
    print(f"made {mp4} ({dur:.1f}s)")
    for a, b, lang, text in marks:
        print(f"  {a:6.2f}-{b:6.2f} {lang}: {text[:40]}")


if __name__ == "__main__":
    asyncio.run(main())
