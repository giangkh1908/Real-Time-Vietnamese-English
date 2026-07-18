"""
RemoteASR — ASR engine that streams PCM16 audio to a remote GPU WebSocket ASR
service (e.g. the Colab notebook colab_asr_gpu.ipynb) and surfaces partial
transcripts via the same callback contract as the local faster-whisper engines.

DESIGN GOAL: never block the session processing loop on the network. The local
CPU engine decouples via a thread pool + local buffer; this engine decouples by
connecting in a BACKGROUND task and only ever put_nowait()-ing into an unbounded
send queue from start_utterance()/feed_audio()/snapshot_audio(). The single
writer task drains the queue over the WS when connected; if not yet connected,
chunks simply wait in the queue. finalize() (which runs in its own task, not the
loop) is the only place that may await the connection.

Protocol (JSON text + binary PCM16 chunks, all ordered by the WS):
  client -> server:
    {"action":"start","lang":"vi"}    # begin new utterance, clear server buffer
    <binary PCM16 16kHz chunk>        # append to current utterance buffer
    {"action":"finalize"}            # transcribe current buffer (final), then clear
  server -> client:
    {"type":"partial","text":"..."}  # unsolicited, while buffer grows
    {"type":"final","text":"...","lang":"vi"}  # in reply to finalize

Ordering: finalize requests and their responses stay FIFO (single WS). We pair
them with a deque of Futures — snapshot_audio() queues a Future + sends
finalize; the reader pops the oldest Future and sets the final result.
finalize() peeks deque[0] and awaits it. Because session_state serializes
utterance finalization (_pipeline_lock), finalize(A) completes (reader pops
F_A) before finalize(B) peeks, so deque[0] is always the right Future.
"""
import asyncio
import json
import logging
import os
from collections import deque
from typing import Tuple

from .interfaces import ASREngine

logger = logging.getLogger("asr_remote")

CONNECT_TIMEOUT = 15  # seconds
FINALIZE_TIMEOUT = 30  # seconds


class RemoteASR(ASREngine):
    def __init__(self, lang: str = "vi", url: str | None = None):
        self.lang = lang
        self.url = url or os.getenv("ASR_REMOTE_URL")
        if not self.url:
            raise RuntimeError("ASR_REMOTE_URL not set")
        self._ws = None
        self._send_q: asyncio.Queue = asyncio.Queue()  # unbounded; never blocks producers
        self._writer_task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        self._connect_task: asyncio.Task | None = None
        self._connected = False
        self._connected_event = asyncio.Event()
        self._result_callback = None
        self._pending_finals: deque = deque()

    # ---- connection (background, never blocks the loop) ----

    def _ensure_connect_bg(self):
        """Start a connect task if not connected and none in flight. Returns
        immediately (non-blocking). Safe to call from sync context too, but we
        only call it from async loop methods."""
        if self._connected or self._connect_task is not None:
            return
        try:
            self._connect_task = asyncio.create_task(self._do_connect())
        except RuntimeError:
            # No running loop (shouldn't happen in our call sites)
            pass

    async def _do_connect(self):
        try:
            import websockets
            logger.info("RemoteASR connecting to %s ...", self.url)
            ws = await asyncio.wait_for(
                websockets.connect(self.url, max_size=None, ping_interval=20),
                timeout=CONNECT_TIMEOUT,
            )
            self._ws = ws
            self._connected = True
            self._connected_event.set()
            self._writer_task = asyncio.create_task(self._writer_loop())
            self._reader_task = asyncio.create_task(self._reader_loop())
            logger.info("RemoteASR connected (lang=%s).", self.lang)
        except Exception as e:
            logger.error("RemoteASR connect failed: %s", e)
            self._clear_send_queue()
            self._fail_pending_finals()
        finally:
            self._connect_task = None

    async def _ensure_connected_blocking(self):
        """Used by finalize() (runs in its own task, may await). Returns True if
        connected (now or after waiting for the in-flight connect)."""
        if self._connected:
            return True
        self._ensure_connect_bg()
        if self._connect_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._connect_task), timeout=CONNECT_TIMEOUT)
            except Exception:
                pass
        return self._connected

    def _handle_disconnect(self):
        if not self._connected:
            return
        self._connected = False
        self._connected_event.clear()
        self._ws = None
        self._clear_send_queue()
        # Fail any pending finals so finalize() doesn't hang forever.
        self._fail_pending_finals()
        logger.warning("RemoteASR disconnected; will reconnect on next use.")

    def _fail_pending_finals(self):
        while self._pending_finals:
            fut = self._pending_finals.popleft()
            if not fut.done():
                fut.set_result(("", self.lang))

    def _clear_send_queue(self):
        cleared = 0
        while True:
            try:
                self._send_q.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        if cleared:
            logger.warning("RemoteASR cleared %d stale queued messages after disconnect.", cleared)

    async def close(self):
        for task in (self._writer_task, self._reader_task, self._connect_task):
            if task and not task.done():
                task.cancel()
        self._clear_send_queue()
        self._fail_pending_finals()
        ws = self._ws
        self._ws = None
        self._connected = False
        self._connected_event.clear()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    # ---- writer / reader ----

    async def _writer_loop(self):
        await self._connected_event.wait()
        try:
            while True:
                msg = await self._send_q.get()
                if msg is None:
                    break
                try:
                    await self._ws.send(msg)
                except Exception as e:
                    logger.warning("RemoteASR send error: %s", e)
                    self._handle_disconnect()
                    return
        except Exception as e:
            logger.exception("RemoteASR writer loop error: %s", e)
            self._handle_disconnect()

    async def _reader_loop(self):
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                t = data.get("type")
                if t == "partial":
                    text = data.get("text", "")
                    if text and self._result_callback:
                        try:
                            asyncio.create_task(self._result_callback(text))
                        except Exception as e:
                            logger.error("RemoteASR partial callback error: %s", e)
                elif t == "final":
                    try:
                        fut = self._pending_finals.popleft()
                    except IndexError:
                        logger.warning("RemoteASR final without pending future")
                        continue
                    if not fut.done():
                        fut.set_result((
                            data.get("text", ""),
                            data.get("lang", self.lang),
                            data.get("metadata", {}),
                        ))
        except Exception as e:
            logger.warning("RemoteASR reader loop ended: %s", e)
        self._handle_disconnect()

    # ---- ASREngine interface (all non-blocking from the loop's perspective) ----

    async def start_utterance(self) -> None:
        self._ensure_connect_bg()
        self._send_q.put_nowait(json.dumps({"action": "start", "lang": self.lang}))

    async def feed_audio(self, pcm16_chunk: bytes) -> str | None:
        if not self._connected:
            self._ensure_connect_bg()
        self._send_q.put_nowait(pcm16_chunk)
        return None

    def set_result_callback(self, callback) -> None:
        self._result_callback = callback

    def snapshot_audio(self) -> bytes:
        """Sync + race-safe. Audio lives on the server, so we return b"" (finalize
        ignores it). Queue a Future + send finalize in WS order so the server
        finalizes THIS utterance's buffer before the next start clears it."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending_finals.append(fut)
        self._send_q.put_nowait(json.dumps({"action": "finalize"}))
        return b""

    async def finalize(self, audio_bytes: bytes = b"") -> Tuple[str, str]:
        if not await self._ensure_connected_blocking():
            logger.error("RemoteASR not connected at finalize; returning empty.")
            self._clear_send_queue()
            self._fail_pending_finals()
            return "", self.lang
        if not self._pending_finals:
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            self._pending_finals.append(fut)
            self._send_q.put_nowait(json.dumps({"action": "finalize"}))
        fut = self._pending_finals[0]
        try:
            return await asyncio.wait_for(fut, timeout=FINALIZE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("RemoteASR finalize timed out (%ss)", FINALIZE_TIMEOUT)
            try:
                self._pending_finals.popleft()
            except IndexError:
                pass
            return "", self.lang
