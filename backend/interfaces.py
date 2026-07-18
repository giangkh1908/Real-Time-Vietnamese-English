from typing import AsyncIterator, Protocol

class ASREngine(Protocol):
    async def start_utterance(self) -> None:
        """Initialize ASR state for a new utterance."""
        ...

    async def feed_audio(self, pcm16_chunk: bytes) -> str | None:
        """
        Processes a PCM16 audio chunk.
        Returns a partial transcript if available, otherwise None.
        """
        ...

    def set_result_callback(self, callback) -> None:
        """Set a callback for partial transcript results."""
        ...

    def snapshot_audio(self) -> bytes:
        """
        Sync, race-safe: grab + clear the current utterance audio buffer.
        Called before scheduling the async finalize so the next utterance cannot
        clear the buffer out from under finalize.
        """
        ...

    async def finalize(self, audio_bytes: bytes = b"") -> tuple[str, str] | tuple[str, str, dict]:
        """
        Transcribe the snapshotted utterance audio (taken via snapshot_audio()).
        Returns (final_text, detected_lang), optionally with ASR metadata.
        """
        ...

class MTEngine(Protocol):
    def translate_stream(
        self, text: str, source_lang: str, target_lang: str,
        context: list[tuple[str, str]], glossary: dict | None
    ) -> AsyncIterator[str]:
        """
        Streams translation deltas for the given text.
        """
        ...

class TTSEngine(Protocol):
    sample_rate: int  # The output sample rate of the engine

    async def stream_pcm16(self, text: str) -> AsyncIterator[bytes]:
        """
        Streams PCM16 audio bytes for the given text.
        """
        ...
