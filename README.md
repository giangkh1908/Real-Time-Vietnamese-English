# VNAI — Dịch giọng nói thời gian thực (Tiếng Việt ↔ English) 🎙️🌐

Ứng dụng dịch giọng nói **2 chiều, thời gian thực**: nói tiếng Việt → ra text + bản dịch tiếng Anh + **phát audio tiếng Anh** (và ngược lại).

Kiến trúc **tách rời**: phần nặng (**ASR + MT/dịch**) chạy trên **GPU cloud** (RunPod/Colab), phần nhẹ (**VAD + TTS**) chạy **local CPU**, frontend là React. Backend local đóng vai trò *orchestrator* — nhận audio từ browser, chạy VAD, stream sang GPU để nhận transcript + dịch, rồi synth giọng và đẩy kết quả về browser. **Không cần GPU trên máy local.**

```
 🎙️ Mic (16kHz PCM16)
      │  WebSocket binary  (/ws/{session_id})
      ▼
 ┌──────────────── Local Backend (FastAPI, port 8000) ─────────────────┐
 │  VAD (Silero, CPU)  →  ASR (GPU cloud, WS)  →  MT (GPU cloud, HTTP)  │
 │                                                  → TTS (Piper, CPU)  │
 └──────────────────────────────┬────────────────────────────────────────┘
                                │  partial_transcript / final_transcript
                                │  partial_translation (live preview)
                                │  translation_delta / translation_done
                                │  tts_start / PCM16 binary / tts_end
                                ▼
                      Frontend (React + Vite, port 5173)
                      ─ 2 cột Source / Translation
                      ─ phát audio TTS realtime (gapless, interrupt)
```

## ✨ Tính năng
- **ASR** — `faster-whisper` `large-v3-turbo` trên **GPU cloud**. Backend tạo **2 engine** `asr_vi` + `asr_en` (chọn theo `source_lang`), cả 2 trỏ cùng 1 endpoint GPU — trên GPU là **1 model đa ngữ** nhận `lang` qua message `start`. Chống hallucinate (`condition_on_previous_text=False`, `hallucination_silence_threshold`, `no_speech_threshold`, `vad_filter=True`). Có partial transcript live + final `beam_size=1` + temperature fallback.
- **MT/dịch** — `gemma3:4b` (hoặc `qwen2.5:7b`…) qua **Ollama** trên GPU cloud, proxy qua cùng server uvicorn (`/api/*`). Streaming NDJSON, prompt dịch strict (chỉ dịch, không trả lời/trả lời câu hỏi). Kèm **partial translation preview** trong lúc nói (debounce 700ms + cancel-in-flight).
- **TTS** — **Piper** (`piper1-gpl`) offline local CPU. Voice `vi_VN-vais1000-medium` (vi) + `en_US-ryan-medium` (en). Stream PCM16 về browser phát **gapless + interrupt** khi có câu mới.
- **VAD** — Silero VAD local CPU. Silence ~1s để chốt utterance, cap `MAX_UTT_S=6s` chống câu dài vô tận / garble. **Adaptive endpointing**: pause dài hơn với câu dày, ngắn hơn với câu thưa.
- **Tunnel** — GPU cloud có IP/proxy công khai (RunPod) → bind thẳng `0.0.0.0:8000`, URL **ổn định** (không đổi mỗi phiên như Cloudflare quick tunnel). Có cell cloudflared dự phòng nếu port bị chặn.

## 📋 Yêu cầu
- **Python 3.10+** (test 3.11/3.12) + `venv`.
- **Node.js 18+** + npm (frontend).
- **GPU cloud**: instance GPU dùng template `pytorch/pytorch:latest` (CUDA sẵn). Lần đầu tải ~3GB `whisper-large-v3-turbo` + ~3-5GB model Ollama. Không cần GPU local.

## 🗂 Cấu trúc
```
VNAI/
├── backend/
│   ├── main.py              # FastAPI WS /ws/{session_id}: nhận audio + control, điều phối session
│   ├── session_state.py      # pipeline VAD→ASR→MT→TTS, partial preview, guards chống hallucinate
│   ├── engine_factory.py     # build engines theo .env (remote GPU / local CPU / fake)
│   ├── asr_remote.py         # WS client → GPU cloud ASR (non-blocking, send-queue có bound, reconnect)
│   ├── llm_engine.py         # OllamaTranslationEngine (prompt strict, streaming, timeout có bound)
│   ├── tts_en.py             # PiperTTSEngine (generic, dùng cho cả vi+en)
│   ├── vad.py                # Silero VAD (window 512 samples @16kHz, silence→utterance_end)
│   ├── interfaces.py         # Protocol ASREngine/MTEngine/TTSEngine + ASRFinal (confidence metadata)
│   ├── connection_manager.py # 1 session_id = 1 WS (supersede WS cũ), push JSON/binary
│   ├── models/               # voice Piper: *.onnx + *.onnx.json (vi + en)
│   └── ...
├── gpu_cloud/
│   └── asr_server.py         # Server GPU: WS /asr (whisper-large-v3-turbo) + proxy /api/* → Ollama
├── gpu_cloud_asr.ipynb      # Notebook triển khai GPU cloud (RunPod/Colab) — chạy cell 1→7
├── colab_asr_gpu.ipynb       # (cũ) Notebook Colab — giữ làm tham khảo
├── frontend/                # React + Vite + AudioWorklet (16kHz PCM16)
│   └── src/hooks/useTranslatorSocket.ts  # WS + phát audio TTS gapless
├── .env.example             # template cấu hình
├── .env                     # (gitignored) cấu hình thực tế
└── requirements.txt
```

## 🚀 Cài đặt

### 1. Backend (local)
```powershell
cd D:\VNAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # hoặc: .venv\Scripts\activate (cmd)
pip install -r requirements.txt
```

Model Piper TTS (vi+en) cần nằm trong `backend/models/`. Nếu thiếu, tải 4 file:
```powershell
cd backend\models
$BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
curl -L -f -o en_US-ryan-medium.onnx          "$BASE/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
curl -L -f -o en_US-ryan-medium.onnx.json      "$BASE/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
curl -L -f -o vi_VN-vais1000-medium.onnx       "$BASE/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx"
curl -L -f -o vi_VN-vais1000-medium.onnx.json  "$BASE/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"
```

### 2. Frontend
```powershell
cd frontend
npm install
```

### 3. Cấu hình `.env`
```powershell
Copy-Item .env.example .env
```
Sửa `.env` (xem `.env.example` để biết ý nghĩa từng biến):
```env
# Để trống cả ASR_REMOTE_URL + MT_BASE_URL = chạy 100% local CPU.
# Điền URL GPU cloud = offload ASR + MT lên GPU (khuyến nghị).

ASR_REMOTE_URL=ws://<IP-cloud>:8000/asr        # hoặc wss://<pod>-8000.proxy.runpod.net/asr
MT_BASE_URL=http://<IP-cloud>:8000            # hoặc https://<pod>-8000.proxy.runpod.net
MT_MODEL=gemma3:4b                              # đúng với MT_MODEL_NAME ở cell 6 notebook
USE_FAKE_PIPELINE=
```

---

## ▶️ Cách chạy (3 phần)

Phần **GPU cloud** phải lên trước (nó là server mà backend local sẽ gọi). Thứ tự: GPU cloud → cập nhật `.env` → backend local → frontend.

### Bước 1 — Khởi động GPU cloud (ASR + MT)
Mở `gpu_cloud_asr.ipynb` trên instance GPU (template `pytorch/pytorch:latest`), chạy tuần tự:
1. **Cell 1** — kiểm tra GPU (CUDA ready).
2. **Cell 2** — cài deps (`fastapi`/`uvicorn`/`faster-whisper`/`httpx`…; torch đã có trong image).
3. **Cell 3** — ghi `asr_server.py` vào `/workspace` (WS `/asr` + proxy `/api/*` → Ollama, header chống buffer).
4. **Cell 4** — (re)khởi động uvicorn nền (`0.0.0.0:8000`). Lần đầu download whisper ~1-2 phút.
5. **Cell 5** — xem log đến khi hiện `ASR ready (whisper-large-v3-turbo)` (chạy lại đến khi thấy).
6. **Cell 6** — cài + chạy Ollama, `ollama pull <MT_MODEL_NAME>` (lần đầu ~2-3 phút), warm-up load vào VRAM.
7. **Cell 7** — in ra 3 giá trị cho `.env`:
   - `ASR_REMOTE_URL` → `ws://…/asr` (hoặc `wss://…proxy.runpod.net/asr`)
   - `MT_BASE_URL`  → `http://…:8000` (hoặc `https://…proxy.runpod.net`)
   - `MT_MODEL`      → đúng `MT_MODEL_NAME` cell 6
   - *(dự phòng)* **Cell 7b** — Cloudflare Tunnel nếu port 8000 không ra được internet (URL `trycloudflare` đổi mỗi lần chạy).
   - *(test)* **Cell 8** — test nhanh WS ASR bằng tone 440Hz.

> RunPod: URL **ổn định** (proxy `https://<pod>-8000.proxy.runpod.net`, TLS sẵn, cố định theo pod). Chỉ cần cập nhật `.env` khi instance restart / lấy IP mới.

### Bước 2 — Cập nhật `.env`
Dán 3 giá trị từ cell 7 vào `D:\VNAI\.env` (như mẫu trên).

### Bước 3 — Khởi động backend local
```powershell
cd D:\VNAI
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Log khi OK: `Successfully loaded Piper TTS model: …vi_VN-vais1000-medium.onnx` + `…en_US-ryan-medium.onnx`, rồi `AI engines loaded successfully.`
> Backend local và GPU cloud **cùng port 8000 nhưng khác máy** — không xung đột.

### Bước 4 — Khởi động frontend
```powershell
cd frontend
npm run dev
```
Mở `http://localhost:5173`, chọn chiều dịch (🇻🇳 Tiếng Việt → English hoặc 🇺🇸 English → Tiếng Việt), bấm **Start Mic**, cho phép micro, bắt đầu nói.

---

## 🧪 Các chế độ chạy

| Chế độ | ASR | MT | TTS | Khi nào dùng |
|---|---|---|---|---|
| **GPU cloud (khuyến nghị)** | large-v3-turbo @ GPU | gemma3:4b @ GPU (Ollama) | Piper local | chạy thật, chất lượng cao |
| **Local CPU** | whisper-medium local | qwen2.5:1.5b local Ollama | Piper local | không có GPU cloud (chậm) |
| **Fake** | giả lập | giả lập | giả lập | test UI/luồng, không cần model |

- **GPU cloud**: điền `ASR_REMOTE_URL` + `MT_BASE_URL` = URL cloud (Bước 1-2). `MT_MODEL=gemma3:4b`.
- **Local CPU**: để `ASR_REMOTE_URL=` và `MT_BASE_URL=` trống. Cần Ollama local (`ollama serve`) + model `qwen2.5:1.5b` (`ollama pull qwen2.5:1.5b`).
- **Fake**: set `USE_FAKE_PIPELINE=1` (test UI không cần GPU/model).

---

## 🔧 Luồng pipeline — backend nhận gì, xử lý sao, tại sao

Đây là phần quan trọng nhất: **backend local là orchestrator**. Nó không tự transcribe/dịch — nó điều phối audio giữa browser, VAD local, GPU cloud (ASR+MT) và TTS local. Mọi xử lý AI chạy dưới dạng **background task** để không nghẽn vòng lặp `receive()` của WebSocket.

### Bước A — Thu âm & gửi lên backend (frontend)
- Browser dùng `AudioWorklet` thu micro ở **16kHz mono PCM16** (định dạng mà faster-whisper + Silero VAD đều dùng trực tiếp, không cần resample).
- Mỗi chunk ~32ms (512 samples @ 16kHz) được gửi **binary** qua WS `/ws/{session_id}`.
- Control message dạng JSON: `start_session` (chọn `source_lang`/`target_lang`), `ping`, `end_session`.
- **Tại sao 16kHz PCM16**: là chuẩn đầu vào của Whisper và Silero VAD → backend feed thẳng, 0 conversion, latency thấp.

### Bước B — Backend nhận & phân phối (`main.py`)
- `ws_endpoint` accept WS, đăng ký vào `ConnectionManager`. **1 session_id chỉ giữ 1 WS**: WS mới đến → đóng WS cũ + `shutdown()` pipeline cũ (chống 2 pipeline song song trên cùng mic, 2 kết nối GPU song song).
- Tạo `SessionState` cho session. Mọi audio chunk → `session.enqueue_audio(chunk)` (push vào `audio_q` có bound 500 — **backpressure**: đầy thì drop + đếm, không nghẽn receive).
- `SessionState` spawn 1 **worker task** `_main_processing_loop` drain `audio_q` → chạy VAD.
- **Tại sao queue + worker**: tách `receive()` (network) khỏi xử lý (VAD/ASR). Nếu VAD/ASR chậm, network vẫn nhận tiếp vào queue, không drop ở socket.

### Bước C — VAD (local CPU, `vad.py` + `session_state.py`)
- Mỗi chunk → Silero VAD (window **512 samples @16kHz**) → state ∈ `silence` / `speech_ongoing` / `utterance_end`.
- `speech_ongoing` lần đầu → `start_utterance()` (báo GPU bắt đầu utterance mới, clear buffer server) + flush **pre-roll ~300ms** (ring buffer luôn giữ audio gần nhất → không mất âm đầu vì Silero fire trễ ~100-200ms).
- Im lặng đủ lâu (`silence_chunks_to_end` ≈ 960ms, **adaptive**: câu dày → 1440ms, câu thưa → 640ms) → `utterance_end`.
- Cap `MAX_UTT_S=6s`: nói liên tục >6s mà chưa im → ép chốt utterance hiện tại, mở utterance mới ngay (overlap ~100ms không cắt đôi từ).
- **Tại sao VAD local chứ không trên GPU**: VAD rẻ, quyết định *khi nào chốt câu* cần độ trễ thấp + cục bộ. Đẩy lên GPU qua network sẽ chậm + phụ thuộc mạng. GPU chỉ việc nặng (transcribe).

### Bước D — ASR (GPU cloud, `asr_remote.py` + `asr_server.py`)
- `SessionState._asr_engine()` chọn `asr_vi` hoặc `asr_en` theo `source_lang` (2 engine, cùng URL GPU).
- `feed_audio(chunk)` → `put_nowait` vào **send-queue có bound (600)** → 1 writer task drain qua WS lên GPU. **Non-blocking**: chưa connect thì chunk chờ trong queue, không treo loop.
- Trên GPU (`asr_server.py`): nhận PCM → tích lũy buffer. Fire partial `beam_size=1` khi buffer đủ (lần đầu ~400ms, sau đó mỗi +1.5s). Partial trả về kèm `avg_logprob` + `no_speech_prob` (confidence).
- `utterance_end` (hoặc cap 6s) → `snapshot_audio()` (sync, race-safe: queue Future + gửi `finalize` theo thứ tự WS) → GPU transcribe final với **temperature fallback** `[0.0, 0.2, 0.4]` + threshold; nếu rỗng → fallback partial cuối.
- `finalize()` await Future (timeout 30s — không treo vĩnh viễn). Reader task ghép `final` với Future tương ứng theo FIFO.
- **Tại sao non-blocking + send-queue có bound**: GPU có thể stall/lag. Queue có bound drop-oldest (realtime > completeness) thay vì grow memory vô hạn. Reconnect backoff (8 lần, jitter) không treo loop.
- **Tại sao 2 engine cùng URL**: tách routing theo ngôn ngữ ở backend, để sau này có thể trỏ `asr_vi`/`asr_en` sang 2 model chuyên biệt (vd PhoWhisper cho vi) mà không đổi code pipeline. Hiện GPU dùng 1 model đa ngữ (large-v3-turbo).

### Bước E — Partial transcript → UI + partial translation preview
- Mỗi partial từ GPU → callback `_on_asr_partial`:
  - **LocalAgreement-2**: `confirmed = LCP(partial trước, partial này)` (locked, chỉ grow); phần còn lại = suffix mutable. UI render confirmed (solid) + suffix (italic) → không nhấp nháy/rewind.
  - **Drop low-confidence** partial (`avg_logprob < -1.0`) → không push UI, không update LCP → hallucinated partial không flicker.
  - Push `partial_transcript` lên UI.
  - Schedule **partial translation preview**: debounce 700ms → dịch preview `partial_translation` (thay thế, cancel-in-flight cái cũ).
- **Tại sao debounce + cancel-in-flight**: dịch mỗi partial sẽ spam MT. Debounce gộp; cancel-in-flight bỏ bản dịch cũ khi có partial mới → preview luôn theo kịp, không lag.

### Bước F — Finalize → guards → final transcript (`_finalize_and_pipeline`)
- Vào `_asr_lock` (chỉ wrap phần ASR + guards + push transcript; MT/TTS chạy **không lock**):
  - Hủy partial translation preview (bản dịch final dưới đây mới chính xác), clear preview box.
  - `finalize()` lấy `ASRFinal` (text + lang + confidence).
  - **Guards chống hallucinate** (lọc rác trước khi hiển thị/dịch):
    - **Micro-utterance** (< ~256ms speech + empty) → drop (ho/khách).
    - **Low-confidence** (`no_speech_prob > 0.8` hoặc `avg_logprob < -1.2`) → drop.
    - **Boilerplate YouTube** ("cảm ơn các bạn đã theo dõi", "subscribe cho kênh", "thank you for watching", tên kênh cụ thể…) → drop (Whisper prior mạnh về mấy câu này trên silence).
    - **Repeat** (final dài ≥4 từ trùng 1 final trong 8 utterance gần đây) → drop (bắt boilerplate lặp).
  - Push `final_transcript` (kèm `utt_id`) lên UI.

### Bước G — MT/dịch (GPU cloud, `llm_engine.py`)
- **STAGE 2 (không lock)**: `mt.translate_stream(final_text, src, tgt, context, glossary)` → POST `/api/chat` (proxy `/api/*` → Ollama local trên GPU). Streaming NDJSON.
- Mỗi delta → push `translation_delta` (UI append theo `utt_id`). Xong → `translation_done` + lưu `context_history` (5 câu gần nhất, OFF mặc định — model nhỏ hay echo bản dịch cũ).
- Prompt **strict**: "Output ONLY the translation — no explanations; if input is a question, translate it, NEVER answer". Chống model trả lời câu hỏi thay vì dịch.
- Timeout có bound (connect 10s, read 120s) → Ollama/tunnel chết thì báo `Error: …`, không treo "đang dịch…" mãi.
- **Tại sao tách lock**: trước đây 1 lock bao cả ASR+MT → finalize utterance N+1 phải chờ MT utterance N xong. Tách ra: Ollama có queue 1-model riêng, frontend append theo `utt_id` nên order chỉ quan trọng trong utt (đã preserve). Latency giảm.

### Bước H — TTS (local CPU, `_synthesize_and_stream_tts`)
- **STAGE 3 (không lock)**: `tts[target_lang].stream_pcm16(translation)` (Piper).
- Push `tts_start` (sample_rate) → stream PCM16 binary (`push_bytes`) → `tts_end`.
- **Tại sao chạy ngoài lock**: utterance kế không phải chờ synth xong. Frontend tự **interrupt** audio chồng nhau trên mỗi `tts_start` mới (gapless).
- **Tại sao TTS local**: Piper nhẹ, offline, CPU đủ; đẩy lên GPU chỉ tăng latency mạng mà không cần. Resample 48kHz cho browser.

### Bước I — Browser phát audio
- `useTranslatorSocket.ts`: nhận PCM16 binary → decode Int16→Float32 → schedule gapless qua `AudioBufferSourceNode`. `tts_start` mới → interrupt audio cũ. `AudioContext.resume()` trên user gesture (nút Start Mic).

### Tóm tắt flow dữ liệu
```
Mic → WS binary → audio_q → VAD ──speech_ongoing──► feed_audio → GPU ASR (partial)
                                       │                              │
                                       │                   partial_transcript ──► UI
                                       │                   partial_translation ─► UI (debounce preview)
                                       │
                                  utterance_end / cap 6s
                                       ▼
                          snapshot + finalize → GPU ASR (final) → guards
                                       │
                              final_transcript ──► UI
                                       ▼
                          MT stream (GPU Ollama) → translation_delta ─► UI → translation_done
                                       ▼
                          TTS (local Piper) → tts_start + PCM16 ─► UI (phát gapless) → tts_end
```

---

## 🩛 Khắc phục sự cố
- **`HTTP 502 Bad Gateway` / WS không connect**: server GPU chưa ready hoặc instance tắt → chạy lại cell 4-5 (chờ `ASR ready`) + cell 7 (URL mới nếu IP đổi) → cập nhật `.env` → restart backend.
- **`đang dịch…` treo mãi**: Ollama trên GPU chưa lên / model chưa pull xong / tunnel buffer. `asr_server.py` đã thêm header `Cache-Control: no-cache` + `X-Accel-Buffering: no`. Backend có timeout (connect 10s, read 120s) nên không treo vĩnh viễn — sẽ báo `Error: …`.
- **Hallucinate ASR** ("hello", text chả liên quan): đã chống bằng `condition_on_previous_text=False` + `hallucination_silence_threshold=2.0` + `no_speech_threshold=0.6` + boilerplate/repeat guard. Nếu vẫn còn → tăng `no_speech_threshold` trong `gpu_cloud/asr_server.py` (cell 3 notebook).
- **Câu bị băm thành nhiều mảnh**: do `MAX_UTT_S` quá nhỏ. Hiện `6.0`. Nếu vẫn bị → tăng trong `backend/session_state.py`.
- **Latency cao**: `large-v3` chậm. Đảm bảo dùng `large-v3-turbo` (đã mặc định trong `asr_server.py`, nhanh hơn ~8×).
- **Audio TTS không phát**: browser suspend `AudioContext` → đã `ctx.resume()` trên `tts_start` (cần 1 user gesture trước — nút Start Mic đã là gesture). Kiểm DevTools Console lỗi decode PCM.
- **Giọng vi (vais1000) nghe robot**: đổi sang `vi_VN-vivos-x_low` (tải file tương ứng + sửa `VI_VOICE` trong `backend/engine_factory.py`), hoặc cân nhắc Edge-TTS cho vi (neural, chất lượng cao hơn, nhưng cần mạng).
- **URL cloud đổi**: RunPod URL cố định theo pod — chỉ đổi khi instance restart. Cloudflare (cell 7b) thì đổi mỗi lần chạy cell → copy URL mới vào `.env` → restart backend.
- **Port 8000 không ra internet** (GPU cloud): chạy cell 7b (cloudflared) thay vì cell 7, dùng URL `trycloudflare`.

## 📦 Ghi chú
- `.env` gitignored (chứa URL cloud, không phải secret nhưng nên giữ local).
- License Piper (`piper1-gpl`) = GPL v3.0; từng voice có license riêng (xem `MODEL_CARD` trên HF).
- GPU cloud: giới hạn thời gian session, GPU có thể bị recycle → giữ notebook mở + cell server chạy nền. Cell "Dừng tất cả" để tắt uvicorn + ollama gọn.
- Mọi xử lý AI trong `SessionState` chạy background task, không nghẽn `receive()`. Send-queue có bound + reconnect backoff → chịu được GPU lag/mạng chập chờn mà không treo.