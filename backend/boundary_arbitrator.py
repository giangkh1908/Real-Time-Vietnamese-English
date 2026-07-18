import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class BoundaryDecision(str, Enum):
    COMMIT = "commit"
    WAIT_AND_RECHECK = "wait_and_recheck"
    CONTINUE = "continue"


@dataclass(frozen=True)
class BoundaryCandidate:
    pause_ms: int
    utterance_ms: int
    speech_duration_s: float
    text: str
    language: str
    stable_ratio: float
    stable_repetitions: int
    forced: bool = False


@dataclass(frozen=True)
class BoundaryResult:
    decision: BoundaryDecision
    reason: str
    wait_ms: int = 0


class BoundaryArbitrator:
    """
    Lightweight endpointing for MVP realtime speech translation.

    VAD produces boundary candidates; this class decides whether a candidate is
    translation-safe enough to finalize now or should wait for more audio.
    """

    FAST_COMMIT_PAUSE_MS = 500
    WAIT_RECHECK_MS = 350
    FORCE_PAUSE_MS = 1200
    PREFERRED_MAX_UTTERANCE_MS = 6000
    HARD_MAX_UTTERANCE_MS = 12000

    _VI_CONNECTORS = {
        "neu",
        "boi vi",
        "mac du",
        "trong khi",
        "de",
        "rang",
        "khi",
        "tu",
        "khong chi",
        "cho den khi",
        "va",
        "nhung",
        "hoac",
    }

    _EN_CONNECTORS = {
        "if",
        "because",
        "although",
        "while",
        "so that",
        "that",
        "when",
        "until",
        "unless",
        "to",
        "and",
        "or",
        "but",
    }

    _VI_UNFINISHED_PHRASES = {
        "khong chi",
        "tu",
        "neu",
        "mac du",
        "boi vi",
        "du kien",
        "co the",
    }

    _EN_UNFINISHED_PHRASES = {
        "not only",
        "between",
        "from",
        "would like to",
        "going to",
    }

    def evaluate(self, candidate: BoundaryCandidate) -> BoundaryResult:
        text = candidate.text.strip()
        normalized = self._normalize(text)

        if candidate.utterance_ms >= self.HARD_MAX_UTTERANCE_MS:
            return BoundaryResult(BoundaryDecision.COMMIT, "hard_max_utterance")

        if not normalized:
            if candidate.pause_ms >= self.FORCE_PAUSE_MS:
                return BoundaryResult(BoundaryDecision.COMMIT, "force_pause_without_partial")
            return BoundaryResult(
                BoundaryDecision.WAIT_AND_RECHECK,
                "no_partial_text",
                self.WAIT_RECHECK_MS,
            )

        if self._has_incomplete_tail(normalized, candidate.language):
            if candidate.pause_ms >= self.FORCE_PAUSE_MS:
                return BoundaryResult(BoundaryDecision.COMMIT, "force_pause_after_incomplete_tail")
            return BoundaryResult(
                BoundaryDecision.WAIT_AND_RECHECK,
                "incomplete_tail",
                self.WAIT_RECHECK_MS,
            )

        if self._has_incomplete_number_or_unit(normalized):
            if candidate.pause_ms >= self.FORCE_PAUSE_MS:
                return BoundaryResult(BoundaryDecision.COMMIT, "force_pause_after_number_fragment")
            return BoundaryResult(
                BoundaryDecision.WAIT_AND_RECHECK,
                "number_or_unit_fragment",
                self.WAIT_RECHECK_MS,
            )

        if candidate.forced and candidate.utterance_ms >= self.PREFERRED_MAX_UTTERANCE_MS:
            return BoundaryResult(BoundaryDecision.COMMIT, "preferred_max_utterance")

        if candidate.pause_ms >= self.FORCE_PAUSE_MS:
            return BoundaryResult(BoundaryDecision.COMMIT, "force_pause")

        if candidate.pause_ms >= self.FAST_COMMIT_PAUSE_MS:
            if self._looks_complete(text, normalized, candidate):
                return BoundaryResult(BoundaryDecision.COMMIT, "complete_candidate")
            return BoundaryResult(
                BoundaryDecision.WAIT_AND_RECHECK,
                "candidate_needs_confirmation",
                self.WAIT_RECHECK_MS,
            )

        return BoundaryResult(BoundaryDecision.CONTINUE, "pause_too_short")

    def _looks_complete(
        self,
        raw_text: str,
        normalized: str,
        candidate: BoundaryCandidate,
    ) -> bool:
        words = normalized.split()
        if len(words) <= 2:
            return candidate.pause_ms >= self.FAST_COMMIT_PAUSE_MS
        has_terminal_punctuation = bool(re.search(r"[.!?。！？]\s*$", raw_text))
        stable_enough = candidate.stable_ratio >= 0.70 or candidate.stable_repetitions >= 2
        long_enough = candidate.speech_duration_s >= 1.0 and len(words) >= 4
        return has_terminal_punctuation or stable_enough or long_enough

    def _has_incomplete_tail(self, normalized: str, language: str) -> bool:
        connectors = self._VI_CONNECTORS if language == "vi" else self._EN_CONNECTORS
        unfinished = self._VI_UNFINISHED_PHRASES if language == "vi" else self._EN_UNFINISHED_PHRASES
        return any(normalized.endswith(f" {c}") or normalized == c for c in connectors | unfinished)

    def _has_incomplete_number_or_unit(self, normalized: str) -> bool:
        if re.search(r"\b\d+\s*$", normalized):
            return True
        return bool(re.search(r"\b(thang|ngay|nam|percent|phan tram|usd|vnd|dollars?)\s*$", normalized))

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text.casefold())
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def stable_prefix_ratio(hypotheses: list[str]) -> tuple[float, int]:
    texts = [h.strip() for h in hypotheses if h and h.strip()]
    if len(texts) < 2:
        return 0.0, len(texts)

    tokenized = [t.split() for t in texts[-3:]]
    latest = tokenized[-1]
    if not latest:
        return 0.0, 0

    prefix_len = 0
    for tokens in zip(*tokenized):
        if len(set(tokens)) != 1:
            break
        prefix_len += 1

    repetitions = 1
    for prev in reversed(texts[:-1]):
        if prev == texts[-1]:
            repetitions += 1
        else:
            break

    return prefix_len / len(latest), repetitions
