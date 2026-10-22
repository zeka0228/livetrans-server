import numpy as np

from livetrans.pipeline.vad import WINDOW_SAMPLES, StreamingSileroVAD


def test_streaming_matches_faster_whisper_batch():
    from faster_whisper.vad import get_vad_model

    rng = np.random.default_rng(0)
    t = np.arange(WINDOW_SAMPLES * 60) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 180 * t) * (t % 1.0 < 0.5)).astype(np.float32)
    audio += rng.normal(0, 0.01, audio.shape).astype(np.float32)

    batch = np.asarray(get_vad_model()(audio)).reshape(-1)
    vad = StreamingSileroVAD()
    stream = np.array(
        [vad(audio[i : i + WINDOW_SAMPLES]) for i in range(0, len(audio), WINDOW_SAMPLES)]
    )
    np.testing.assert_allclose(stream, batch, atol=1e-5)


def test_reset_restores_initial_state():
    vad = StreamingSileroVAD()
    w = np.full(WINDOW_SAMPLES, 0.2, dtype=np.float32)
    first = vad(w)
    vad(w)
    vad.reset()
    assert abs(vad(w) - first) < 1e-6
