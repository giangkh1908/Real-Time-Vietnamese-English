import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptGuardResult:
    allowed: bool
    reason: str = "ok"


class TranscriptGuard:
    """
    Filters ASR finals before MT/TTS.

    The guard is intentionally conservative for realtime demos: suspicious ASR
    defaults are worse than dropping an utterance and continuing to listen.
    """

    MIN_SPEECH_S = 0.35
    MIN_SHORT_SPEECH_S = 0.50

    _SHORT_ALLOWED = {
        "ok",
        "okay",
        "alo",
        "hello",
        "hi",
        "yes",
        "no",
        "co",
        "khong",
        "xin chao",
    }

    _DENY_FRAGMENTS = {
        "hay subscribe",
        "subscribe cho kenh",
        "la la school",
        "khong bo lo nhung video hap dan",
        "nhung video hap dan",
        "please subscribe",
        "dont forget to subscribe",
        "thanks for watching",
        "thank you for watching",
    }

    def evaluate(
        self,
        text: str,
        speech_duration_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TranscriptGuardResult:
        raw = (text or "").strip()
        if not raw:
            return TranscriptGuardResult(False, "empty_transcript")

        variants = self._variants(raw)
        normalized = {self._normalize(v) for v in variants}
        normalized.discard("")

        if not normalized:
            return TranscriptGuardResult(False, "empty_after_normalize")

        if speech_duration_s is not None and speech_duration_s < self.MIN_SPEECH_S:
            return TranscriptGuardResult(False, f"speech_too_short:{speech_duration_s:.2f}s")

        if any(self._matches_denylist(n) for n in normalized):
            return TranscriptGuardResult(False, "denylisted_default_hallucination")

        shortest = min(normalized, key=len)
        word_count = len(shortest.split())
        if len(shortest) <= 1:
            return TranscriptGuardResult(False, "too_short_text")
        if word_count == 1 and shortest not in self._SHORT_ALLOWED:
            if speech_duration_s is None or speech_duration_s < self.MIN_SHORT_SPEECH_S:
                return TranscriptGuardResult(False, "single_word_without_enough_speech")

        if metadata:
            no_speech_prob = metadata.get("no_speech_prob")
            avg_logprob = metadata.get("avg_logprob")
            compression_ratio = metadata.get("compression_ratio")
            if isinstance(no_speech_prob, (int, float)) and no_speech_prob >= 0.80:
                return TranscriptGuardResult(False, f"high_no_speech_prob:{no_speech_prob:.2f}")
            if isinstance(avg_logprob, (int, float)) and avg_logprob <= -1.20:
                return TranscriptGuardResult(False, f"low_avg_logprob:{avg_logprob:.2f}")
            if isinstance(compression_ratio, (int, float)) and compression_ratio >= 2.60:
                return TranscriptGuardResult(False, f"high_compression_ratio:{compression_ratio:.2f}")

        return TranscriptGuardResult(True)

    def _matches_denylist(self, normalized: str) -> bool:
        return any(fragment in normalized for fragment in self._DENY_FRAGMENTS)

    def _variants(self, text: str) -> set[str]:
        variants = {text}
        for source_encoding in ("latin1", "cp1252"):
            try:
                variants.add(text.encode(source_encoding).decode("utf-8"))
            except UnicodeError:
                pass
        return variants

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text.casefold())
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
