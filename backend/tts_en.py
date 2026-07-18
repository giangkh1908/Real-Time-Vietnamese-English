import asyncio
from pathlib import Path
from typing import AsyncIterator
from .interfaces import TTSEngine
from .audio_utils import resample_pcm16

# All models live in backend/models/ (downloaded manually)
MODELS_DIR = Path(__file__).resolve().parent / "models"

class PiperTTSEngine(TTSEngine):
    def __init__(self, model_path: str = str(MODELS_DIR / "en_US-ryan-medium.onnx")):
        self.model_path = model_path
        self.sample_rate = 48000  # Standardized output rate
        self._native_sr = 22050    # Typical Piper output rate
        self.voice = None

        try:
            # piper1-gpl (OHF-Voice): `from piper import PiperVoice`.
            # rhasspy cũ dùng `from piper.voice import PiperVoice` — đã archive.
            from piper import PiperVoice
            self.voice = PiperVoice.load(model_path)
            try:
                self._native_sr = self.voice.config.sample_rate
            except Exception:
                self._native_sr = 22050  # fallback; thực tế dùng audio_chunk.sample_rate
            self.sample_rate = self._native_sr
            print(f"Successfully loaded Piper TTS model: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load Piper model from {model_path}: {e}")
            print("PiperTTSEngine will use fallback (silence).")

    async def stream_pcm16(self, text: str) -> AsyncIterator[bytes]:
        if self.voice is None:
            # Fallback: Emit silence blocks if model is not loaded
            total_samples = int(self.sample_rate * len(text.split()) * 0.35)
            chunk_size = 9600
            while total_samples > 0:
                n = min(chunk_size, total_samples)
                await asyncio.sleep(0.05)
                yield b"\x00\x00" * n
                total_samples -= n
            return

        loop = asyncio.get_event_loop()
        def generate_audio():
            # piper-tts 1.x API: synthesize() yields AudioChunk objects
            return self.voice.synthesize(text)

        try:
            chunks = await loop.run_in_executor(None, generate_audio)
            for audio_chunk in chunks:
                pcm16_native = audio_chunk.audio_int16_bytes
                if audio_chunk.sample_rate == self.sample_rate:
                    yield pcm16_native
                else:
                    pcm16_out = await loop.run_in_executor(
                        None, resample_pcm16, pcm16_native, audio_chunk.sample_rate, self.sample_rate
                    )
                    yield pcm16_out
        except Exception as e:
            print(f"Error during Piper synthesis: {e}")
            yield b"\x00\x00" * 9600
