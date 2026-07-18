import os
from pathlib import Path
from dataclasses import dataclass
from .interfaces import ASREngine, MTEngine, TTSEngine
from .fakes import FakeASR, FakeMT, FakeTTS
from .vad import VADModel

@dataclass
class Engines:
    asr_vi: ASREngine   # Vietnamese ASR
    asr_en: ASREngine   # English ASR
    mt: MTEngine
    tts: dict[str, TTSEngine]  # key: "vi" or "en"
    vad: VADModel

# Piper voice models live in backend/models/ (downloaded manually, see plan).
_MODELS_DIR = Path(__file__).resolve().parent / "models"
VI_VOICE = str(_MODELS_DIR / "vi_VN-vais1000-medium.onnx")
EN_VOICE = str(_MODELS_DIR / "en_US-ryan-medium.onnx")


def _piper_tts() -> dict[str, TTSEngine]:
    """Piper (piper1-gpl) cho cả vi + en, chạy local CPU."""
    from .tts_en import PiperTTSEngine
    return {
        "vi": PiperTTSEngine(VI_VOICE),
        "en": PiperTTSEngine(EN_VOICE),
    }


def build_engines() -> Engines:
    # Check environment variables for mock mode
    fake = os.getenv("USE_FAKE_PIPELINE", "").lower() in ("1", "true", "yes")

    if fake:
        return Engines(
            asr_vi=FakeASR(),
            asr_en=FakeASR(),
            mt=FakeMT(),
            tts={"vi": FakeTTS(), "en": FakeTTS()},
            vad=VADModel(),
        )

    # Remote ASR mode: offload ASR to a GPU service (e.g. Colab notebook
    # colab_asr_gpu.ipynb) via WebSocket. VAD/MT/TTS still run locally.
    remote_asr_url = os.getenv("ASR_REMOTE_URL")
    if remote_asr_url:
        from .llm_engine import OllamaTranslationEngine
        # SessionState creates one RemoteASR per browser session/language. Keep
        # placeholders here so no shared RemoteASR state is accidentally reused.
        return Engines(
            asr_vi=FakeASR(),
            asr_en=FakeASR(),
            mt=OllamaTranslationEngine(),
            tts=_piper_tts(),
            vad=VADModel(),
        )

    # Real Implementations (local CPU ASR)
    from .asr_vietnamese import VietnameseASR
    from .asr_english import EnglishASR
    from .llm_engine import OllamaTranslationEngine

    return Engines(
        asr_vi=VietnameseASR(),
        asr_en=EnglishASR(),
        mt=OllamaTranslationEngine(),
        tts=_piper_tts(),
        vad=VADModel(),
    )
