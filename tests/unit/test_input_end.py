"""ffmpeg 입력 종료 분류 (S-23): 정상 종료(E_INPUT_ENDED) vs 끊김(E_CAPTURE_LOST)."""

from livetrans.plugins.source.ffmpeg_in import classify_input_end


def test_file_eof_and_srt_peer_close_are_normal():
    assert classify_input_end(0, "", "f.mp4") == (True, "입력이 끝났습니다 (f.mp4)")
    normal, why = classify_input_end(
        1, "[in#0/mpegts @ 0x1] Error during demuxing: I/O error", "srt://x"
    )
    assert normal and why.startswith("입력이 끝났습니다")
    assert classify_input_end(None, "End of file", "srt://x")[0] is True


def test_connection_failures_are_losses():
    normal, why = classify_input_end(1, "srt://127.0.0.1:9000: Connection refused", "srt://x")
    assert normal is False and "끊겼습니다" in why and "Connection refused" in why
    assert classify_input_end(69, "", "srt://x") == (False, "입력이 끊겼습니다 (srt://x) (rc=69)")
