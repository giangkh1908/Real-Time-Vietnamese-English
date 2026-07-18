import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from .connection_manager import ConnectionManager
from .engine_factory import build_engines, Engines
from .session_state import SessionState

# Load .env (e.g. ASR_REMOTE_URL) before build_engines() reads env vars.
# Explicit repo-root path + override=True so .env always wins (regardless of
# CWD or any stale OS env var from a previous shell `set`/`export`).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
for noisy_logger in ("websockets.client", "httpcore", "httpx", "piper.voice"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Global engines - loaded ONCE at startup
engines: Engines | None = None
manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines
    import os
    remote = os.getenv("ASR_REMOTE_URL", "")
    logger.info("Loading AI engines at startup... (ASR_REMOTE_URL=%s)",
                remote if remote else "<empty -> local CPU ASR>")
    engines = build_engines()
    logger.info("AI engines loaded successfully.")
    app.state.startup_complete = True
    yield
    logger.info("Shutting down...")
    app.state.startup_complete = False

app = FastAPI(lifespan=lifespan)
app.state.startup_complete = False

@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)

    logger.info("[ws_endpoint] startup_complete=%s, engines=%s",
                 app.state.startup_complete, engines is not None)

    if engines is None:
        logger.error("Engines not loaded!")
        await websocket.close(code=1011, reason="Server not ready")
        return

    session = SessionState(session_id, manager, engines, languages=("vi", "en"))

    chunk_count = 0
    try:
        while True:
            msg = await websocket.receive()

            # Handle disconnects
            if msg.get("type") == "websocket.disconnect":
                logger.info("[%s] WebSocket disconnected (received %d audio chunks)", session_id, chunk_count)
                break

            if "bytes" in msg:
                chunk_count += 1
                if chunk_count <= 3 or chunk_count % 100 == 0:
                    logger.info("[%s] Audio chunk #%d (%d bytes)", session_id, chunk_count, len(msg["bytes"]))
                # Binary audio data -> push to processing queue
                await session.enqueue_audio(msg["bytes"])
            elif "text" in msg:
                # JSON control messages
                try:
                    data = json.loads(msg["text"])
                    logger.info("[%s] Control message: %s", session_id, data.get("type"))
                    await session.on_control(data)
                except json.JSONDecodeError:
                    await manager.push(session_id, {"type": "error", "message": "Invalid JSON"})

    except WebSocketDisconnect:
        pass
    finally:
        await session.shutdown()
        manager.disconnect(session_id, websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
