from livetrans.plugins.stt.guard import compression_ratio, hallucination_reason, max_new_tokens_for


def test_repeated_syllables_are_hallucination():
    assert hallucination_reason("ho" * 200) is not None
    assert hallucination_reason("hohohohohoho") is not None  # 6회
    assert hallucination_reason("하하하하하하하하하") is not None
    assert (
        hallucination_reason("안녕하세요 안녕하세요 안녕하세요 안녕하세요 안녕하세요") is not None
    )  # 단어 5회


def test_normal_sentences_pass():
    for s in (
        "볼래? 오늘 날씨가 좋네요",
        "hello everyone this is a subtitle test",
        "こんにちは今日はいい天気ですね",
        "하하 진짜 웃기다",
        "ㅋㅋㅋㅋ",  # 5자 미만 반복은 통과
        "1 2 3 4 5 6 7 8 9 10",
        "네 네 네 그렇죠",  # 3회는 통과
        "",
    ):
        assert hallucination_reason(s) is None, s


def test_compression_ratio_catches_long_low_entropy_text():
    assert compression_ratio("abcabcabc" * 30) > 2.4
    assert hallucination_reason("la li lu le lo " * 20) is not None
    assert compression_ratio("오늘은 날씨가 맑고 바람이 적당히 불어서 산책하기 좋은 날입니다") < 2.4


def test_max_new_tokens_scales_with_duration():
    assert max_new_tokens_for(1000) == 31
    assert max_new_tokens_for(8000) == 136
    assert max_new_tokens_for(0) == 16
    assert max_new_tokens_for(60_000) == 440


def test_phrase_repeat_and_salvage():
    from livetrans.plugins.stt.guard import salvage_prefix

    loop = "oh yeah i am i am i am i am i am i am i am"
    assert hallucination_reason(loop).startswith("반복 구")
    assert salvage_prefix(loop) == "oh yeah i am"
    assert salvage_prefix("ok" * 100) == ""  # 살릴 앞부분이 없다
    assert (
        salvage_prefix(
            "なにでがいでるのにかいでるのにかいでるのにかいでるのにかいでるのにかいでるのにか"
        )
        == "なにでがいでるのにか"
    )
    assert salvage_prefix("la li lu le lo " * 20) == "la li lu le lo"  # 구 반복: 단위 1개는 남긴다
    assert salvage_prefix("정상 문장입니다") == ""
    assert hallucination_reason("we came to shop we came to shop") is None  # 2회 반복은 정상


def test_prompt_echo_is_stripped_only_when_output_is_longer_than_prompt():
    from livetrans.plugins.stt.dicow import strip_prompt_echo

    assert strip_prompt_echo("직전 문장 새 문장", "직전 문장") == "새 문장"
    assert strip_prompt_echo("직전 문장", "직전 문장") == "직전 문장"  # 똑같이 또 말한 것일 수 있다
    assert strip_prompt_echo("다른 문장", "직전 문장") == "다른 문장"
    assert strip_prompt_echo("문장", None) == "문장"
