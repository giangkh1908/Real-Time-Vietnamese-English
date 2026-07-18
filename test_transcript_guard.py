from backend.transcript_guard import TranscriptGuard


def test_rejects_known_vietnamese_subscribe_hallucination():
    guard = TranscriptGuard()

    result = guard.evaluate(
        "Hãy subscribe cho kênh La La School để không bỏ lỡ những video hấp dẫn",
        speech_duration_s=2.0,
    )

    assert not result.allowed
    assert result.reason == "denylisted_default_hallucination"


def test_rejects_mojibake_subscribe_hallucination():
    guard = TranscriptGuard()

    result = guard.evaluate(
        "HÃ£y subscribe cho kÃªnh La La School Äá»ƒ khÃ´ng bá» lá»¡ nhá»¯ng video háº¥p dáº«n",
        speech_duration_s=2.0,
    )

    assert not result.allowed


def test_allows_short_real_utterance_with_enough_speech():
    guard = TranscriptGuard()

    assert guard.evaluate("alo", speech_duration_s=0.7).allowed
    assert guard.evaluate("ok", speech_duration_s=0.7).allowed
    assert guard.evaluate("xin chào", speech_duration_s=0.7).allowed


def test_rejects_too_short_speech_before_translation():
    guard = TranscriptGuard()

    result = guard.evaluate("xin chào", speech_duration_s=0.1)

    assert not result.allowed
    assert result.reason.startswith("speech_too_short")
