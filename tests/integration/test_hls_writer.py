"""HLS 출력 (실제 ffmpeg). ffmpeg 를 못 찾으면 skip."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from livetrans.core.audio import OUT_RATE
from livetrans.output.hls import HlsWriter, find_ffmpeg
from livetrans.pipeline.timeline import Cue

FFMPEG = find_ffmpeg(None)
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg 없음")


def tone_segment(freq=440.0):
    t = np.arange(OUT_RATE) / OUT_RATE
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, x], axis=1)


def test_hls_writer_produces_media_and_vtt(tmp_path: Path):
    w = HlsWriter(tmp_path / "s1", ffmpeg=FFMPEG, segment_s=1.0, langs=("ko", "ja"))
    w.start()
    assert w.alive
    cue = Cue(1, "tgt", 1200, 2500, "안녕하세요", "S0", shown_from_ms=1200)
    for k in range(4):
        w.write_segment(k, tone_segment(), {"tgt": [cue] if k in (1, 2) else [], "src": []})
    w.close()
    d = tmp_path / "s1"
    master = (d / "master.m3u8").read_text(encoding="utf-8")
    assert 'URI="sub_tgt.m3u8"' in master and 'LANGUAGE="ko"' in master and "media.m3u8" in master
    media = (d / "media.m3u8").read_text(encoding="utf-8")
    assert media.count(".ts") >= 2 and "#EXT-X-PROGRAM-DATE-TIME" in media
    vtt_list = (d / "sub_tgt.m3u8").read_text(encoding="utf-8")
    assert "#EXT-X-ENDLIST" in vtt_list and "sub_tgt_00001.vtt" in vtt_list
    vtt = (d / "sub_tgt_00001.vtt").read_text(encoding="utf-8")
    assert (
        vtt.startswith("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:") and "00:00:01.200 --> 00:00:02.500" in vtt
    )
    assert "<v S0>안녕하세요" in vtt
    assert (d / "sub_tgt_00000.vtt").read_text(encoding="utf-8").count("-->") == 0
    # 세그먼트에 AAC 오디오가 있다
    ffprobe = str(Path(FFMPEG).with_name("ffprobe" + Path(FFMPEG).suffix))
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(d / "media_00000.ts")],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    streams = json.loads(out.stdout)["streams"]
    assert any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in streams)
    w.remove()
    assert not d.exists()
