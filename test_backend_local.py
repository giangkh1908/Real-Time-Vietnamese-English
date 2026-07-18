"""
Test script: Đọc Recording.m4a, gửi qua backend local (d:/VNAI/backend),
đo latency TẤT CẢ giai đoạn per-utterance, xuất kết quả dịch + latency ra file.

Pipeline đo (mỗi utterance):
  speech_start ──► first_partial ──► final_transcript ──► first_translation_token ──► translation_done
  |<- VAD->partial ->|  |<- VAD->final ->|  |<- ASRfinalize ->|  |<- MTstream ->| |<- E2E ->|

Metrics per utterance:
  - VAD -> partial      : first_partial - speech_start        (TTFT partial / độ nhạy realtime)
  - VAD -> final        : final - speech_start                (thời gian ra transcript chốt)
  - ASR finalize        : final - first_partial               (partial -> final)
  - Final -> first token: first_delta - final                (đợi MT token đầu)
  - MT stream duration  : done - first_delta                  (streaming dịch trọn câu)
  - Translation total   : done - final                       (MT end-to-end sau khi có final)
  - End-to-end          : done - speech_start                (từ nói xong đến dịch xong)

Aggregate: count / min / avg / max trên tất cả utterance.
"""
import asyncio
import time
import sys
import json
import statistics
from pathlib import Path
from datetime import datetime

# Windows console mặc định cp1252 -> lỗi Unicode khi in dấu tiếng Việt / box-drawing.
# Ép stdout UTF-8 để report hiển thị đúng dấu (thay vì ASCII replace '?' như trước).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Add backend to path (không import, chỉ để compatible nếu cần)
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# ============================================================
# Audio conversion
# ============================================================
def resample_linear(audio_float, src_sr, dst_sr):
    import numpy as np
    duration = len(audio_float) / src_sr
    num_samples = int(duration * dst_sr)
    old_indices = list(range(len(audio_float)))
    new_indices = [i * (len(audio_float) - 1) / (num_samples - 1) for i in range(num_samples)]
    resampled = np.interp(new_indices, old_indices, audio_float)
    return resampled

def read_m4a_to_pcm16(path, target_sr=16000):
    import av
    import numpy as np

    container = av.open(str(path))
    audio_stream = container.streams.audio[0]

    frames = []
    for frame in container.decode(audio=0):
        frames.append(frame)

    all_samples = []
    for frame in frames:
        if hasattr(frame, 'to_ndarray'):
            arr = frame.to_ndarray()
        else:
            arr = np.stack([np.frombuffer(plane, dtype=np.float32) for plane in frame.planes])
        if arr.ndim == 2:
            arr = arr.T
        all_samples.append(arr)

    if not all_samples:
        container.close()
        return b""

    samples = np.concatenate(all_samples)

    if samples.dtype == np.float32 or samples.dtype == np.float64:
        samples_int = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        samples_int = samples.astype(np.int16)

    if samples_int.ndim == 2:
        samples_int = samples_int.mean(axis=1).astype(np.int16)

    if audio_stream.rate != target_sr:
        samples_float = samples_int.astype(np.float32) / 32768.0
        resampled_float = resample_linear(samples_float, audio_stream.rate, target_sr)
        samples_int = (np.clip(resampled_float, -1.0, 1.0) * 32767).astype(np.int16)

    container.close()
    return samples_int.tobytes()

# ============================================================
# Latency tracking — đo TẤT CẢ giai đoạn per-utterance
# ============================================================
class LatencyTracker:
    def __init__(self):
        self.t0 = None
        self.events = []                 # timeline toàn cục: {name, ms, detail}
        self.utterances = {}             # utt_id -> dict(stage timestamps)
        self.speech_starts = []          # queue timestamp các đoạn speech (silence->listening)
        self._last_status = None
        self._current_partial_ms = None  # first partial của speech segment hiện tại

    def start(self):
        self.t0 = time.perf_counter()

    def now_ms(self):
        if self.t0 is None:
            self.t0 = time.perf_counter()
        return (time.perf_counter() - self.t0) * 1000

    def tick(self, name, detail=""):
        ms = self.now_ms()
        self.events.append({"name": name, "ms": round(ms, 1), "detail": detail})
        return ms

    # ---- WS event handlers ----
    def on_status(self, state):
        if state == "listening" and self._last_status != "listening":
            self.speech_starts.append(self.now_ms())
            self.tick("vad_speech_start", "VAD detected speech")
            # KHÔNG reset _current_partial_ms ở đây: VAD có thể flip-flop nhiều lần
            # trong cùng 1 utterance; partial đầu vẫn thuộc utterance hiện tại.
        self._last_status = state

    def on_partial(self, text):
        # partial_transcript KHÔNG có utt_id; gán cho speech segment gần nhất
        ms = self.now_ms()
        self.tick("partial_transcript", text[:60])
        if self._current_partial_ms is None and self.speech_starts:
            self._current_partial_ms = ms

    def on_final(self, text, utt_id):
        ms = self.now_ms()
        self.tick("final_transcript", text[:60])
        speech_start = self.speech_starts.pop(0) if self.speech_starts else ms
        self.utterances[utt_id] = {
            "speech_start": speech_start,
            "first_partial": self._current_partial_ms,
            "final": ms,
            "first_delta": None,
            "done": None,
            "text": text,
            "translation": "",
        }
        self._current_partial_ms = None

    def on_delta(self, utt_id, delta):
        u = self.utterances.get(utt_id)
        if u and u["first_delta"] is None:
            u["first_delta"] = self.now_ms()
            self.tick("mt_first_token", utt_id[:8])

    def on_done(self, utt_id, full):
        u = self.utterances.get(utt_id)
        if u:
            u["done"] = self.now_ms()
            u["translation"] = full
            self.tick("translation_done", full[:60])

    # ---- per-utterance metrics ----
    def utterance_metrics(self):
        rows = []
        for uid, u in self.utterances.items():
            sp = u["speech_start"]
            fp = u["first_partial"]
            fin = u["final"]
            fd = u["first_delta"]
            done = u["done"]
            row = {
                "utt_id": uid[:8],
                "text": u["text"],
                "translation": u["translation"],
                "vad_to_partial": (fp - sp) if fp else None,
                "vad_to_final": (fin - sp) if fin else None,
                "asr_finalize": (fin - fp) if (fin and fp) else None,
                "final_to_first_token": (fd - fin) if (fd and fin) else None,
                "mt_stream": (done - fd) if (done and fd) else None,
                "translation_total": (done - fin) if (done and fin) else None,
                "end_to_end": (done - sp) if (done and sp) else None,
            }
            rows.append(row)
        return rows

    def aggregate(self):
        rows = self.utterance_metrics()
        keys = ["vad_to_partial", "vad_to_final", "asr_finalize",
                "final_to_first_token", "mt_stream", "translation_total", "end_to_end"]
        agg = {}
        for k in keys:
            vals = [r[k] for r in rows if r[k] is not None]
            if vals:
                agg[k] = {"n": len(vals), "min": min(vals), "avg": statistics.mean(vals),
                          "max": max(vals), "p50": statistics.median(vals)}
            else:
                agg[k] = None
        return agg

    def report(self):
        rows = self.utterance_metrics()
        agg = self.aggregate()

        def p(s):
            print(s)

        p("\n" + "=" * 78)
        p("LATENCY REPORT (per-utterance)")
        p("=" * 78)

        p(f"\n{'#':<3} {'VAD->p':>7} {'VAD->f':>7} {'ASRfin':>7} {'F->tok':>7} {'MTstr':>7} {'MTtot':>7} {'E2E':>7}  text")
        p("-" * 78)
        for i, r in enumerate(rows, 1):
            def fmt(v):
                return f"{v:7.0f}" if v is not None else f"{'--':>7}"
            p(f"{i:<3} {fmt(r['vad_to_partial'])} {fmt(r['vad_to_final'])} {fmt(r['asr_finalize'])} "
              f"{fmt(r['final_to_first_token'])} {fmt(r['mt_stream'])} {fmt(r['translation_total'])} {fmt(r['end_to_end'])}  "
              f"{r['text'][:40]}")
        p("-" * 78)
        p("(ms) VAD->p=VAD→partial  VAD->f=VAD→final  ASRfin=partial→final  "
          "F->tok=final→first token  MTstr=stream  MTtot=total  E2E=end-to-end")

        p("\n" + "=" * 78)
        p("AGGREGATE (ms)")
        p("=" * 78)
        p(f"{'metric':<22} {'n':>3} {'min':>7} {'p50':>7} {'avg':>7} {'max':>7}")
        p("-" * 78)
        labels = {
            "vad_to_partial": "VAD -> partial",
            "vad_to_final": "VAD -> final",
            "asr_finalize": "ASR finalize",
            "final_to_first_token": "Final -> 1st token",
            "mt_stream": "MT stream",
            "translation_total": "Translation total",
            "end_to_end": "End-to-end",
        }
        for k in ["vad_to_partial", "vad_to_final", "asr_finalize",
                  "final_to_first_token", "mt_stream", "translation_total", "end_to_end"]:
            a = agg[k]
            if a:
                p(f"{labels[k]:<22} {a['n']:>3} {a['min']:>7.0f} {a['p50']:>7.0f} {a['avg']:>7.0f} {a['max']:>7.0f}")
            else:
                p(f"{labels[k]:<22} {'-':>3} {'-':>7} {'-':>7} {'-':>7} {'-':>7}")

        p("\n" + "=" * 78)
        p("TRANSCRIPTS & TRANSLATIONS")
        p("=" * 78)
        for i, r in enumerate(rows, 1):
            p(f"\n[Utterance {i}] (utt={r['utt_id']})")
            p(f"  Source:      {r['text']}")
            p(f"  Translation: {r['translation']}")

        return {"rows": rows, "agg": agg, "events": self.events}

# ============================================================
# WebSocket client
# ============================================================
async def send_audio_chunks(uri, pcm16_bytes, chunk_size=512, tracker: LatencyTracker = None, drain_timeout_s: float = 45.0, chunk_ms: int = 16):
    import websockets

    results = {"chunks_sent": 0, "audio_duration_ms": len(pcm16_bytes) / 2 / 16000 * 1000}

    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        tracker.tick("ws_connect")

        await ws.send(json.dumps({
            "type": "start_session",
            "source_lang": "vi",
            "target_lang": "en"
        }))
        tracker.tick("session_start", "vi->en")

        target_sr = 16000
        chunk_duration_s = chunk_size / target_sr
        send_start = time.perf_counter()

        # Track tiến độ để chờ động
        done_event = asyncio.Event()
        state = {"finals": 0, "dones": 0}

        async def receive_loop():
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    if isinstance(msg, bytes):
                        continue  # TTS bytes (nếu có) — bỏ qua
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    t = data.get("type")
                    if t == "status":
                        tracker.on_status(data.get("state", ""))
                    elif t == "partial_transcript":
                        tracker.on_partial(data.get("text", ""))
                    elif t == "final_transcript":
                        tracker.on_final(data.get("text", ""), data.get("utt_id"))
                        state["finals"] += 1
                    elif t == "translation_delta":
                        tracker.on_delta(data.get("utt_id"), data.get("text_delta", ""))
                    elif t == "translation_done":
                        tracker.on_done(data.get("utt_id"), data.get("full_text", ""))
                        state["dones"] += 1
                        # Mọi utterance đã có final đều đã done -> pipeline xong
                        if state["dones"] >= state["finals"] and state["finals"] > 0:
                            done_event.set()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Receive error: {e}")
                    break

        receive_task = asyncio.create_task(receive_loop())

        # Gửi chunks theo thời gian thực (chunk_duration_s mỗi chunk)
        total_sent = 0
        while total_sent < len(pcm16_bytes):
            chunk = pcm16_bytes[total_sent: total_sent + chunk_size]
            if len(chunk) < chunk_size:
                chunk = chunk + b'\x00' * (chunk_size - len(chunk))
            await ws.send(chunk)
            results["chunks_sent"] += 1
            total_sent += len(chunk)
            if results["chunks_sent"] % 200 == 0:
                print(f"  Sent {results['chunks_sent']} chunks ({total_sent/2/16000*1000:.0f}ms audio)")
            await asyncio.sleep(chunk_duration_s)

        # Gửi ~700ms silence để ép VAD báo utterance_end (6 windows * 32ms = 192ms cần)
        tracker.tick("audio_sent", "sending trailing silence")
        silence_ms = 700
        n_silence = max(1, int(silence_ms * target_sr / 1000) // (chunk_size // 2))
        silence_chunk = b'\x00' * chunk_size
        for _ in range(n_silence):
            await ws.send(silence_chunk)
            results["chunks_sent"] += 1
            await asyncio.sleep(chunk_duration_s)

        # Chờ động: pipeline hoàn tất (mọi final đã done) hoặc timeout
        tracker.tick("drain_wait", f"timeout={drain_timeout_s}s")
        try:
            await asyncio.wait_for(done_event.wait(), timeout=drain_timeout_s)
            tracker.tick("drain_complete", f"finals={state['finals']} dones={state['dones']}")
        except asyncio.TimeoutError:
            tracker.tick("drain_timeout", f"finals={state['finals']} dones={state['dones']}")

        await ws.send(json.dumps({"type": "end_session"}))
        tracker.tick("session_end")
        # Cho backend 1s xử lý end_session, rồi thu nốt message còn sót
        await asyncio.sleep(1.0)

        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

    results["send_wall_ms"] = (time.perf_counter() - send_start) * 1000
    results["finals"] = state["finals"]
    results["dones"] = state["dones"]
    return results

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default="Recording.m4a")
    parser.add_argument("--ws", "-w", default="ws://localhost:8000/ws/test-latency")
    parser.add_argument("--chunk-ms", type=int, default=16)
    parser.add_argument("--output", "-o", default="test_results.txt")
    args = parser.parse_args()

    path = Path(__file__).parent / args.file
    print(f"Reading {path}...")
    pcm16_bytes = read_m4a_to_pcm16(path)
    audio_duration_ms = len(pcm16_bytes) / 2 / 16000 * 1000
    print(f"PCM16: {len(pcm16_bytes)} bytes, {audio_duration_ms/1000:.2f}s @ 16kHz mono")

    chunk_size = int(16000 * args.chunk_ms / 1000) * 2

    tracker = LatencyTracker()
    tracker.start()
    tracker.tick("test_start", f"file={path.name}, {audio_duration_ms/1000:.1f}s")

    print(f"\nSending to {args.ws} with {args.chunk_ms}ms chunks...")
    t_start = time.perf_counter()
    results = await send_audio_chunks(args.ws, pcm16_bytes, chunk_size=chunk_size, tracker=tracker, chunk_ms=args.chunk_ms)
    t_end = time.perf_counter()

    report = tracker.report()

    # ---- Ghi file ----
    output_path = Path(__file__).parent / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Latency Test Results - {datetime.now().isoformat()}\n")
        f.write(f"File: {path.name} | Audio: {audio_duration_ms:.0f}ms | Backend: local (whisper-small vi + Ollama qwen2.5:1.5b)\n")
        f.write(f"Chunk: {args.chunk_ms}ms | Send wall time: {results['send_wall_ms']:.0f}ms | Total wall: {(t_end-t_start)*1000:.0f}ms\n")
        f.write("=" * 78 + "\n\n")

        f.write("PER-UTTERANCE LATENCY (ms)\n")
        f.write(f"{'#':<3} {'VAD->p':>7} {'VAD->f':>7} {'ASRfin':>7} {'F->tok':>7} {'MTstr':>7} {'MTtot':>7} {'E2E':>7}  text\n")
        f.write("-" * 78 + "\n")
        for i, r in enumerate(report["rows"], 1):
            def fmt(v):
                return f"{v:7.0f}" if v is not None else f"{'--':>7}"
            f.write(f"{i:<3} {fmt(r['vad_to_partial'])} {fmt(r['vad_to_final'])} {fmt(r['asr_finalize'])} "
                    f"{fmt(r['final_to_first_token'])} {fmt(r['mt_stream'])} {fmt(r['translation_total'])} {fmt(r['end_to_end'])}  "
                    f"{r['text'][:40]}\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("AGGREGATE (ms)\n")
        f.write("=" * 78 + "\n")
        f.write(f"{'metric':<22} {'n':>3} {'min':>7} {'p50':>7} {'avg':>7} {'max':>7}\n")
        f.write("-" * 78 + "\n")
        labels = {
            "vad_to_partial": "VAD -> partial",
            "vad_to_final": "VAD -> final",
            "asr_finalize": "ASR finalize",
            "final_to_first_token": "Final -> 1st token",
            "mt_stream": "MT stream",
            "translation_total": "Translation total",
            "end_to_end": "End-to-end",
        }
        for k in labels:
            a = report["agg"][k]
            if a:
                f.write(f"{labels[k]:<22} {a['n']:>3} {a['min']:>7.0f} {a['p50']:>7.0f} {a['avg']:>7.0f} {a['max']:>7.0f}\n")
            else:
                f.write(f"{labels[k]:<22} {'-':>3} {'-':>7} {'-':>7} {'-':>7} {'-':>7}\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("TRANSCRIPTS & TRANSLATIONS\n")
        f.write("=" * 78 + "\n")
        for i, r in enumerate(report["rows"], 1):
            f.write(f"\n[Utterance {i}] (utt={r['utt_id']})\n")
            f.write(f"  Source:      {r['text']}\n")
            f.write(f"  Translation: {r['translation']}\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("FULL EVENT TIMELINE\n")
        f.write("=" * 78 + "\n")
        f.write(f"{'time(ms)':>10}  {'event':<24} detail\n")
        f.write("-" * 78 + "\n")
        for e in report["events"]:
            f.write(f"{e['ms']:>10.1f}  {e['name']:<24} {e['detail']}\n")

    print(f"\nResults saved to: {output_path}")
    print(f"Utterances: {len(report['rows'])} | Events: {len(report['events'])}")

if __name__ == "__main__":
    asyncio.run(main())