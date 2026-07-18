from backend.boundary_arbitrator import (
    BoundaryArbitrator,
    BoundaryCandidate,
    BoundaryDecision,
    stable_prefix_ratio,
)


def candidate(text: str, pause_ms: int = 600, language: str = "vi") -> BoundaryCandidate:
    return BoundaryCandidate(
        pause_ms=pause_ms,
        utterance_ms=2400,
        speech_duration_s=1.6,
        text=text,
        language=language,
        stable_ratio=0.8,
        stable_repetitions=1,
    )


def test_waits_on_short_incomplete_vietnamese_phrase():
    result = BoundaryArbitrator().evaluate(candidate("Chúng tôi dự kiến"))

    assert result.decision == BoundaryDecision.WAIT_AND_RECHECK


def test_commits_complete_vietnamese_phrase():
    result = BoundaryArbitrator().evaluate(
        candidate("Chúng tôi dự kiến sẽ triển khai vào tháng Chín")
    )

    assert result.decision == BoundaryDecision.COMMIT


def test_waits_on_connector_tail():
    vi_result = BoundaryArbitrator().evaluate(candidate("Chúng tôi sẽ làm nếu"))
    en_result = BoundaryArbitrator().evaluate(candidate("We can approve it although", language="en"))

    assert vi_result.decision == BoundaryDecision.WAIT_AND_RECHECK
    assert en_result.decision == BoundaryDecision.WAIT_AND_RECHECK


def test_force_pause_commits_valid_text():
    result = BoundaryArbitrator().evaluate(candidate("Chúng tôi dự kiến", pause_ms=1200))

    assert result.decision == BoundaryDecision.COMMIT


def test_stable_prefix_ratio_uses_recent_hypotheses():
    ratio, repetitions = stable_prefix_ratio([
        "chúng tôi sẽ hoàn thành vào tháng",
        "chúng tôi sẽ hoàn thành dự án vào tháng",
        "chúng tôi sẽ hoàn thành dự án vào tháng chín",
    ])

    assert 0 < ratio < 1
    assert repetitions == 1
