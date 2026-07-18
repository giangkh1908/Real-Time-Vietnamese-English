"""VNAI — ASR (whisper-large-v3-turbo) + Ollama MT proxy. Portable, chạy được
trên Linux/Windows. Dùng cho GPU VPS (vd Windows RTX 5060 Ti qua RDP).

Chạy:
    pip install fastapi[standard] uvicorn[standard] numpy faster-whisper httpx
    python asr_server.py                 # bind 0.0.0.0:8000

Endpoint:
    GET  /health        -> {"ready": bool}
    WS   /asr           -> ASR (PCM16 16kHz chunks + {action:start|finalize})
    /api/{path}         -> proxy sang Ollama local (http://127.0.0.1:11434)
"""
import asyncio
import json
import logging
import os
import numpy as np
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr_server")

model = None
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_COMPUTE = os.getenv("ASR_COMPUTE", "float16")


@asynccontextmanager
async def lifespan(app):
    global model
    from faster_whisper import WhisperModel
    logger.info("Loading %s on %s (%s)...", ASR_MODEL, ASR_DEVICE, ASR_COMPUTE)
    model = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
    logger.info("ASR ready (%s).", ASR_MODEL)
    app.state.ready = True
    yield
    logger.info("Shutting down ASR server...")


app = FastAPI(lifespan=lifespan)
app.state.ready = False

INITIAL_TRANSCRIBE_BYTES = 6400     # ~400ms: partial đầu
PARTIAL_GROW_BYTES = 24000          # ~1.5s: partial tiếp theo chỉ chạy khi buffer thêm >=1.5s
PARTIAL_FALLBACK_MIN_LOGPROB = -1.0


def _segments_conf(segments):
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    if not segs:
        return text, 0.0, 1.0, 1.0
    total = 0
    wlp = 0.0
    nsp = 0.0
    cr = 1.0
    for s in segs:
        wc = max(1, len(s.text.split()))
        total += wc
        wlp += (getattr(s, "avg_logprob", 0.0) or 0.0) * wc
        nsp = max(nsp, getattr(s, "no_speech_prob", 0.0) or 0.0)
        cr = max(cr, getattr(s, "compression_ratio", 1.0) or 1.0)
    return text, (wlp / total if total else 0.0), nsp, cr


async def _transcribe(audio_bytes, lang, beam_size=1, fallback=False, prompt=""):
    if not audio_bytes:
        return "", lang, 0.0, 1.0, 1.0
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    loop = asyncio.get_event_loop()

    def run():
        kw = dict(
            language=lang, task="transcribe", beam_size=beam_size,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            no_speech_threshold=0.6, vad_filter=True,
        )
        if fallback:
            kw.update(temperature=[0.0, 0.2, 0.4],
                      compression_ratio_threshold=2.4, logprob_threshold=-1.0)
        if prompt:
            kw["initial_prompt"] = prompt
        segments, _ = model.transcribe(audio, **kw)
        return _segments_conf(segments)

    try:
        text, lp, nsp, cr = await loop.run_in_executor(None, run)
        return text, lang, lp, nsp, cr
    except Exception as e:
        logger.exception("transcribe error: %s", e)
        return "", lang, 0.0, 1.0, 1.0


@app.get("/health")
async def health():
    return {"ready": app.state.ready}


@app.websocket("/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    lang = "vi"
    partial_busy = [False]
    last_partial_len = 0
    last_partial_text = ""
    partial_task = None
    logger.info("ASR client connected.")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                buf.extend(msg["bytes"])
                grown = len(buf) - last_partial_len
                should_fire = (not partial_busy[0]) and (
                    (last_partial_len == 0 and len(buf) >= INITIAL_TRANSCRIBE_BYTES)
                    or (last_partial_len > 0 and grown >= PARTIAL_GROW_BYTES)
                )
                if should_fire:
                    partial_busy[0] = True
                    last_partial_len = len(buf)
                    audio = bytes(buf)

                    async def _p(audio=audio, lang=lang):
                        nonlocal last_partial_text
                        try:
                            text, _l, lp, nsp, _cr = await _transcribe(audio, lang, beam_size=1)
                            if text:
                                await ws.send_text(json.dumps({
                                    "type": "partial", "text": text,
                                    "avg_logprob": round(lp, 3),
                                    "no_speech_prob": round(nsp, 3),
                                }))
                                if lp >= PARTIAL_FALLBACK_MIN_LOGPROB:
                                    last_partial_text = text
                        except Exception as e:
                            logger.exception("partial error: %s", e)
                        finally:
                            partial_busy[0] = False

                    if partial_task and not partial_task.done():
                        partial_task.cancel()
                    partial_task = asyncio.create_task(_p())
            elif msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                action = data.get("action")
                if action == "start":
                    buf.clear()
                    lang = data.get("lang", "vi")
                    last_partial_len = 0
                    last_partial_text = ""
                    partial_busy[0] = False
                elif action == "finalize":
                    if partial_task and not partial_task.done():
                        partial_task.cancel()
                    text, det, lp, nsp, cr = await _transcribe(
                        bytes(buf), lang, beam_size=1, fallback=True)
                    if not text and last_partial_text:
                        text = last_partial_text
                        lp, nsp, cr = 0.0, 0.0, 1.0
                        logger.info("finalize empty -> fallback partial: %r", text[:80])
                    await ws.send_text(json.dumps({
                        "type": "final", "text": text, "lang": det,
                        "avg_logprob": round(lp, 3),
                        "no_speech_prob": round(nsp, 3),
                        "compression_ratio": round(cr, 3),
                    }))
                    buf.clear()
                    last_partial_len = 0
                    last_partial_text = ""
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("asr_ws error: %s", e)
    finally:
        logger.info("ASR client disconnected.")


@app.api_route("/api/{path:path}", methods=["GET", "POST"])
async def proxy_ollama(request: Request, path: str):
    url = f"{OLLAMA_URL}/api/{path}"
    body = await request.body()
    headers = {"Content-Type": request.headers.get("content-type", "application/json")}

    async def stream():
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream(request.method, url, content=body, headers=headers) as r:
                async for chunk in r.aiter_raw():
                    yield chunk

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)