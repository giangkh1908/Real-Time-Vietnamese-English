import asyncio
import time
import uuid
import logging
import os
from collections import deque
from .engine_factory import Engines
from .connection_manager import ConnectionManager
from .vad import VAD
from .transcript_guard import TranscriptGuard
from .boundary_arbitrator import (
    BoundaryArbitrator,
    BoundaryCandidate,
    BoundaryDecision,
    stable_prefix_ratio,
)

logger = logging.getLogger("session_state")

# Global state constants

# Max utterance duration: ép chốt utterance khi kéo dài quá N giây mà VAD chưa báo
# utterance_end. Lớn hơn = câu dài giữ nguyên 1 khối (đỡ garble/mất ngữ cảnh),
# partial translation live vẫn cho cảm giác realtime. 6s = cân bằng.
MAX_UTT_S = 6.0
MIN_FINALIZE_SPEECH_S = 0.35
PRE_ROLL_CHUNKS = 8
CANDIDATE_SILENCE_CHUNKS = 16


class SessionState:
    def __init__(self, session_id: str, manager: ConnectionManager, engines: Engines, languages: tuple[str, ...]):
        self.session_id = session_id
        self.manager = manager
        self.engines = engines
        self.languages = set(languages)

        self.source_lang = "vi"
        self.target_lang = "en"

        self.vad = VAD(self.engines.vad, silence_chunks_to_end=CANDIDATE_SILENCE_CHUNKS)
        self.transcript_guard = TranscriptGuard()
        self.boundary_arbitrator = BoundaryArbitrator()
        self.context_history: list[tuple[str, str]] = []
        self.glossary: dict | None = None
        self._session_asr: dict[str, object] = {}
        self._use_session_remote_asr = bool(os.getenv("ASR_REMOTE_URL"))

        self._asr_started = False
        self._last_status = None
        self._utt_start_time: float | None = None  # for max-duration cap
        self._utt_speech_bytes = 0
        self._pre_roll_chunks: deque[bytes] = deque(maxlen=PRE_ROLL_CHUNKS)
        self._partial_history: deque[str] = deque(maxlen=3)
        self._pending_boundary_task: asyncio.Task | None = None
        self._speech_generation = 0

        # Utterance tracking
        self.utterances: dict[str, dict] = {} # utt_id -> {text, translation, status}

        # Streaming state
        self._current_utt_id: str | None = None

        # Backpressure tracking
        self._dropped_chunks = 0

        # Partial translation preview: dịch partial transcript trong lúc nói
        # (debounce + cancel-in-flight) -> UI hiện bản dịch dần, chốt ở utterance_end.
        self._last_partial_text = ""
        self._partial_debounce_task: asyncio.Task | None = None
        self._partial_tr_task: asyncio.Task | None = None
        self._partial_debounce_s = 0.7

        # Pipeline serialization: chỉ 1 utterance finalize+MT tại lúc (shared ASR
        # engine + CPU contention + Ollama single-model queue). Nhiều utterance
        # concurrent làm mỗi cái chậm lại (test: F->tok utterance1 = 36s do contention).
        self._pipeline_lock = asyncio.Lock()

        self.audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=500)
        self.worker = asyncio.create_task(self._main_processing_loop())

    def _asr_engine(self):
        if self._use_session_remote_asr:
            lang = self.source_lang
            if lang not in self._session_asr:
                from .asr_remote import RemoteASR
                self._session_asr[lang] = RemoteASR(lang=lang)
            return self._session_asr[lang]
        return self.engines.asr_vi if self.source_lang == "vi" else self.engines.asr_en

    def _speech_duration_s(self) -> float:
        return self._utt_speech_bytes / 2 / 16000

    def _reset_utterance_counters(self):
        self._utt_speech_bytes = 0
        self._partial_history.clear()
        self._last_partial_text = ""
        self._cancel_pending_boundary()

    async def _feed_preroll(self):
        if not self._pre_roll_chunks:
            return
        for pre_chunk in list(self._pre_roll_chunks):
            await self._asr_engine().feed_audio(pre_chunk)
            self._utt_speech_bytes += len(pre_chunk)
        self._pre_roll_chunks.clear()

    def _cancel_pending_boundary(self):
        if self._pending_boundary_task and not self._pending_boundary_task.done():
            self._pending_boundary_task.cancel()
        self._pending_boundary_task = None

    def _utterance_ms(self) -> int:
        if self._utt_start_time is None:
            return int(self._speech_duration_s() * 1000)
        return int((time.perf_counter() - self._utt_start_time) * 1000)

    def _latest_boundary_text(self) -> str:
        if self._partial_history:
            return self._partial_history[-1]
        return self._last_partial_text

    def _boundary_candidate(self, pause_ms: int, forced: bool = False) -> BoundaryCandidate:
        stable_ratio, stable_repetitions = stable_prefix_ratio(list(self._partial_history))
        return BoundaryCandidate(
            pause_ms=pause_ms,
            utterance_ms=self._utterance_ms(),
            speech_duration_s=self._speech_duration_s(),
            text=self._latest_boundary_text(),
            language=self.source_lang,
            stable_ratio=stable_ratio,
            stable_repetitions=stable_repetitions,
            forced=forced,
        )

    async def _commit_current_utterance(self, reason: str):
        utt_id = self._current_utt_id
        audio_bytes = self._asr_engine().snapshot_audio()
        speech_duration_s = self._speech_duration_s()
        self._asr_started = False
        self._utt_start_time = None
        self._reset_utterance_counters()
        if speech_duration_s < MIN_FINALIZE_SPEECH_S:
            logger.info("[%s] Dropping utterance before finalize: speech too short (%.2fs, reason=%s)",
                        self.session_id, speech_duration_s, reason)
            return
        logger.info("[%s] Boundary commit: reason=%s speech=%.2fs",
                    self.session_id, reason, speech_duration_s)
        asyncio.create_task(self._finalize_and_pipeline(utt_id, audio_bytes, speech_duration_s))

    def _schedule_boundary_recheck(self, candidate: BoundaryCandidate, wait_ms: int, reason: str):
        self._cancel_pending_boundary()
        generation = self._speech_generation
        utt_id = self._current_utt_id
        self._pending_boundary_task = asyncio.create_task(
            self._boundary_recheck_after_wait(utt_id, generation, candidate.pause_ms, wait_ms, reason)
        )

    async def _boundary_recheck_after_wait(
        self,
        utt_id: str | None,
        generation: int,
        original_pause_ms: int,
        wait_ms: int,
        reason: str,
    ):
        try:
            await asyncio.sleep(wait_ms / 1000)
            if not self._asr_started or self._current_utt_id != utt_id:
                return
            if self._speech_generation != generation:
                logger.debug("[%s] Boundary candidate cancelled by resumed speech (reason=%s)",
                             self.session_id, reason)
                return
            candidate = self._boundary_candidate(original_pause_ms + wait_ms)
            result = self.boundary_arbitrator.evaluate(candidate)
            logger.info("[%s] Boundary recheck: decision=%s reason=%s pause=%dms text=%r",
                        self.session_id, result.decision, result.reason,
                        candidate.pause_ms, candidate.text[:80])
            if result.decision == BoundaryDecision.COMMIT:
                await self._commit_current_utterance(result.reason)
            elif result.decision == BoundaryDecision.WAIT_AND_RECHECK:
                self._schedule_boundary_recheck(candidate, result.wait_ms, result.reason)
        except asyncio.CancelledError:
            return

    async def enqueue_audio(self, chunk: bytes):
        try:
            self.audio_q.put_nowait(chunk)
        except asyncio.QueueFull:
            self._dropped_chunks += 1
            if self._dropped_chunks <= 5 or self._dropped_chunks % 100 == 0:
                logger.warning("[%s] Audio queue full, dropped %d chunks total",
                                self.session_id, self._dropped_chunks)

    async def on_control(self, msg: dict):
        msg_type = msg.get("type")
        if msg_type == "start_session":
            self.source_lang = msg.get("source_lang", "vi")
            self.target_lang = msg.get("target_lang", "en")
            self._asr_engine().set_result_callback(self._on_asr_partial)
            logger.info("[%s] Session started: %s -> %s", self.session_id, self.source_lang, self.target_lang)
        elif msg_type == "end_session":
            await self.shutdown()
        elif msg_type == "ping":
            await self.manager.push(self.session_id, {"type": "pong"})

    async def _update_status(self, status: str):
        if self._last_status != status:
            self._last_status = status
            await self.manager.push(self.session_id, {"type": "status", "state": status})

    # Debounced translation removed in favor of utterance-based flow

    async def _main_processing_loop(self):
        """
        Audio -> VAD -> ASR (partial via callback)
        """
        logger.info("[%s] Processing pipeline started", self.session_id)
        # Pipeline được serialize bằng self._pipeline_lock bên trong
        # _finalize_and_pipeline (không cần lock ở đây nữa).

        # Register ASR partial callback
        asr = self._asr_engine()
        asr.set_result_callback(self._on_asr_partial)

        try:
            while True:
                chunk = await self.audio_q.get()
                if chunk is None:
                    break

                # Measure VAD latency
                t0 = time.perf_counter()
                vad_state = self.vad.process(chunk)
                vad_ms = (time.perf_counter() - t0) * 1000

                if vad_state != "silence":
                    logger.debug("[%s] VAD state: %s (%.2fms)", self.session_id, vad_state, vad_ms)

                if vad_state == "speech_ongoing":
                    self._speech_generation += 1
                    self._cancel_pending_boundary()
                    if not self._asr_started:
                        try:
                            await self._asr_engine().start_utterance()
                        except Exception as e:
                            logger.exception("[%s] start_utterance failed: %s", self.session_id, e)
                            continue
                        self._asr_started = True
                        self._current_utt_id = str(uuid.uuid4())
                        self._utt_start_time = time.perf_counter()
                        self._reset_utterance_counters()
                        await self._feed_preroll()
                        logger.info("[%s] ASR started (%s)", self.session_id, self.source_lang)
                    elif self._utterance_ms() >= self.boundary_arbitrator.HARD_MAX_UTTERANCE_MS:
                        # Max-duration cap: ép chốt utterance hiện tại, chunk này + kế tiếp
                        # thuộc utterance mới (giữ trải nghiệm realtime, không đợi cả câu dài).
                        logger.info("[%s] ASR hard-duration cap (%dms)",
                                    self.session_id, self._utterance_ms())
                        utt_id = self._current_utt_id
                        audio_bytes = self._asr_engine().snapshot_audio()
                        speech_duration_s = self._speech_duration_s()
                        self._asr_started = False
                        self._reset_utterance_counters()
                        if speech_duration_s < MIN_FINALIZE_SPEECH_S:
                            logger.info("[%s] Dropping max-duration utterance before finalize: speech too short (%.2fs)",
                                        self.session_id, speech_duration_s)
                        else:
                            asyncio.create_task(self._finalize_and_pipeline(utt_id, audio_bytes, speech_duration_s))
                        # Start utterance mới ngay để không mất chunk hiện tại
                        try:
                            await self._asr_engine().start_utterance()
                        except Exception as e:
                            logger.exception("[%s] start_utterance (cap) failed: %s", self.session_id, e)
                        self._asr_started = True
                        self._current_utt_id = str(uuid.uuid4())
                        self._utt_start_time = time.perf_counter()
                        self._reset_utterance_counters()

                    await self._update_status("listening")

                    try:
                        # Measure ASR feed latency
                        t_asr = time.perf_counter()
                        await self._asr_engine().feed_audio(chunk)
                        self._utt_speech_bytes += len(chunk)
                        asr_ms = (time.perf_counter() - t_asr) * 1000
                        if asr_ms > 10: # Log if it takes significant time
                            logger.debug("[%s] ASR feed: %.2fms", self.session_id, asr_ms)
                    except Exception as e:
                        logger.exception("[%s] Error in ASR feed_audio: %s", self.session_id, e)

                elif vad_state == "utterance_end" and self._asr_started:
                    await self._update_status("silence")
                    candidate = self._boundary_candidate(pause_ms=CANDIDATE_SILENCE_CHUNKS * 32)
                    result = self.boundary_arbitrator.evaluate(candidate)
                    logger.info("[%s] Boundary candidate: decision=%s reason=%s pause=%dms stable=%.2f text=%r",
                                self.session_id, result.decision, result.reason,
                                candidate.pause_ms, candidate.stable_ratio, candidate.text[:80])
                    if result.decision == BoundaryDecision.COMMIT:
                        await self._commit_current_utterance(result.reason)
                    elif result.decision == BoundaryDecision.WAIT_AND_RECHECK:
                        self._schedule_boundary_recheck(candidate, result.wait_ms, result.reason)

                elif vad_state == "silence":
                    if not self._asr_started:
                        self._pre_roll_chunks.append(chunk)
                    await self._update_status("silence")

        except Exception as e:
            logger.exception("[%s] Critical error in processing loop: %s", self.session_id, e)

    async def _on_asr_partial(self, text: str):
        """Called by ASR whenever a partial transcript is ready."""
        logger.info("[%s] ASR partial: %s", self.session_id, text)
        if text and text.strip():
            self._partial_history.append(text.strip())
        await self.manager.push(self.session_id, {
            "type": "partial_transcript",
            "text": text
        })
        # Schedule a live translation preview of the partial (replaceable).
        self._last_partial_text = text
        self._schedule_partial_translation()

    def _schedule_partial_translation(self):
        """(Re)start the debounce timer for a partial translation preview."""
        if self._partial_debounce_task and not self._partial_debounce_task.done():
            self._partial_debounce_task.cancel()
        try:
            self._partial_debounce_task = asyncio.create_task(
                self._partial_debounce_then_translate())
        except RuntimeError:
            pass

    async def _partial_debounce_then_translate(self):
        try:
            await asyncio.sleep(self._partial_debounce_s)
        except asyncio.CancelledError:
            return
        text = self._last_partial_text
        if not text.strip():
            return
        # Cancel any in-flight partial translation before starting a new one.
        if self._partial_tr_task and not self._partial_tr_task.done():
            self._partial_tr_task.cancel()
        try:
            self._partial_tr_task = asyncio.create_task(self._do_partial_translate(text))
        except RuntimeError:
            pass

    async def _do_partial_translate(self, text: str):
        """Translate a partial transcript and stream a replaceable preview."""
        accumulated = ""
        try:
            async for delta in self.engines.mt.translate_stream(
                text, self.source_lang, self.target_lang,
                self.context_history, self.glossary
            ):
                accumulated += delta
                if accumulated.strip():
                    await self.manager.push(self.session_id, {
                        "type": "partial_translation",
                        "text": accumulated
                    })
        except asyncio.CancelledError:
            return
        except Exception as e:
            # Preview errors are non-fatal — the final translation is authoritative.
            logger.warning("[%s] partial translate failed: %s", self.session_id, e)

    def _cancel_partial_translation(self):
        """Cancel any pending/in-flight partial translation preview."""
        if self._partial_debounce_task and not self._partial_debounce_task.done():
            self._partial_debounce_task.cancel()
        if self._partial_tr_task and not self._partial_tr_task.done():
            self._partial_tr_task.cancel()

    async def _finalize_and_pipeline(
        self,
        utt_id: str | None = None,
        audio_bytes: bytes = b"",
        speech_duration_s: float | None = None,
    ):
        """
        Finalize ASR, translate the full utterance.
        Utterances are tracked in self.utterances for UI history.

        utt_id + audio_bytes are snapshotted by the caller (VAD utterance_end or
        max-duration cap) to avoid races: the finalize task runs async, and a new
        utterance's start_utterance() would otherwise overwrite self._current_utt_id
        and clear the shared ASR buffer.

        Serialized by self._pipeline_lock: chỉ 1 utterance finalize+MT tại lúc
        (shared ASR engine + CPU contention + Ollama single-model queue).
        """
        # Phase 1 (locked): ASR finalize + MT + push text. Chỉ 1 utterance tại lúc
        # (shared ASR engine + CPU contention + Ollama single-model queue).
        tts_text = ""
        tts_utt_id = utt_id
        async with self._pipeline_lock:
            try:
                logger.info("[%s] _finalize_and_pipeline starting (utt_id=%s, %d bytes)",
                            self.session_id, utt_id, len(audio_bytes))

                # Stop the live preview translation; the final translation below
                # is authoritative. Clear the preview box in the UI.
                self._cancel_partial_translation()
                await self.manager.push(self.session_id, {
                    "type": "partial_translation", "text": ""
                })

                # 1. Get final transcript from ASR (transcribe the snapshotted audio)
                t_start = time.perf_counter()
                asr_result = await self._asr_engine().finalize(audio_bytes)
                final_text, detected_lang = asr_result[0], asr_result[1]
                asr_metadata = asr_result[2] if len(asr_result) >= 3 else {}
                asr_finalize_ms = (time.perf_counter() - t_start) * 1000
                logger.info("[%s] ASR finalize: [%s] '%s' (%.2fms)", self.session_id, detected_lang, final_text, asr_finalize_ms)

                if not final_text or not final_text.strip():
                    return

                guard_result = self.transcript_guard.evaluate(
                    final_text,
                    speech_duration_s=speech_duration_s,
                    metadata=asr_metadata,
                )
                if not guard_result.allowed:
                    logger.warning("[%s] Transcript rejected before MT/TTS: reason=%s text=%r speech=%.2fs",
                                   self.session_id, guard_result.reason, final_text,
                                   speech_duration_s if speech_duration_s is not None else -1.0)
                    await self.manager.push(self.session_id, {
                        "type": "status",
                        "state": "silence",
                    })
                    return

                utt_id = utt_id or self._current_utt_id or str(uuid.uuid4())
                tts_utt_id = utt_id

                # Track utterance in state
                self.utterances[utt_id] = {
                    "text": final_text,
                    "translation": "",
                    "status": "transcribed"
                }

                # Send final transcript to UI
                await self.manager.push(self.session_id, {
                    "type": "final_transcript",
                    "lang": self.source_lang,
                    "text": final_text,
                    "utt_id": utt_id
                })

                # 2. Translate the finalized text
                self.utterances[utt_id]["status"] = "translating"

                t_mt_start = time.perf_counter()
                accumulated_translation = ""
                async for delta in self.engines.mt.translate_stream(
                    final_text, self.source_lang, self.target_lang, self.context_history, self.glossary
                ):
                    accumulated_translation += delta
                    await self.manager.push(self.session_id, {
                        "type": "translation_delta",
                        "utt_id": utt_id,
                        "text_delta": delta
                    })
                mt_ms = (time.perf_counter() - t_mt_start) * 1000
                logger.info("[%s] MT Translation complete (%.2fms)", self.session_id, mt_ms)

                # Update history and state
                self.context_history.append((final_text, accumulated_translation))
                self.context_history = self.context_history[-5:]
                self.utterances[utt_id]["translation"] = accumulated_translation
                self.utterances[utt_id]["status"] = "translated"

                await self.manager.push(self.session_id, {
                    "type": "translation_done",
                    "utt_id": utt_id,
                    "full_text": accumulated_translation
                })
                tts_text = accumulated_translation

            except Exception as e:
                logger.exception("[%s] Pipeline failure: %s", self.session_id, e)
                return

        # Phase 2 (unlocked): TTS stream. Chạy ngoài lock để utterance kế tiếp không
        # phải chờ synth xong; frontend tự interrupt audio chồng nhau trên mỗi tts_start.
        if tts_text.strip():
            await self._synthesize_and_stream_tts(tts_utt_id, tts_text)

    async def _synthesize_and_stream_tts(self, utt_id: str, text: str):
        """Synthesize the translation text with Piper and stream PCM16 to the client."""
        tts_engine = self.engines.tts.get(self.target_lang)
        if tts_engine is None:
            return
        try:
            await self.manager.push(self.session_id, {
                "type": "tts_start",
                "utt_id": utt_id,
                "sample_rate": tts_engine.sample_rate,
            })
            t_tts = time.perf_counter()
            n_chunks = 0
            async for pcm in tts_engine.stream_pcm16(text):
                await self.manager.push_bytes(self.session_id, pcm)
                n_chunks += 1
            tts_ms = (time.perf_counter() - t_tts) * 1000
            logger.info("[%s] TTS complete (%.2fms, %d chunks, %d chars)",
                        self.session_id, tts_ms, n_chunks, len(text))
        except Exception as e:
            logger.exception("[%s] TTS failure: %s", self.session_id, e)
        finally:
            await self.manager.push(self.session_id, {"type": "tts_end", "utt_id": utt_id})

    async def shutdown(self):
        self._cancel_pending_boundary()
        self._cancel_partial_translation()
        if self._asr_started:
            logger.info("[%s] Shutdown: Finalizing pending utterance", self.session_id)
            utt_id = self._current_utt_id
            audio_bytes = self._asr_engine().snapshot_audio()
            speech_duration_s = self._speech_duration_s()
            self._asr_started = False
            self._utt_start_time = None
            try:
                if speech_duration_s >= MIN_FINALIZE_SPEECH_S:
                    await self._finalize_and_pipeline(utt_id, audio_bytes, speech_duration_s)
            except Exception as e:
                logger.exception("[%s] Error finalizing during shutdown: %s", self.session_id, e)

        await self.audio_q.put(None)
        self.worker.cancel()
        try:
            await self.worker
        except asyncio.CancelledError:
            pass
        for asr in self._session_asr.values():
            close = getattr(asr, "close", None)
            if close:
                try:
                    await close()
                except Exception:
                    pass
