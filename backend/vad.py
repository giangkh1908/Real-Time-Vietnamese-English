import logging
import numpy as np
import torch
from typing import Tuple

logger = logging.getLogger("vad")

class VADModel:
    """
    Shared VAD model loaded once at startup.
    Forced to CPU for maximum stability across different environments.
    """
    def __init__(self, threshold: float = 0.55):
        logger.info("Loading Silero VAD model on CPU for stability...")
        try:
            self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
            self.model.to('cpu')
            self.model.eval()
            logger.info("Silero VAD model loaded successfully on CPU.")
        except Exception as e:
            logger.error("Failed to load Silero VAD model: %s", e)
            raise e
        self.threshold = threshold

class VAD:
    """
    Per-session VAD state manager. Handles windowing and state transitions.
    """
    def __init__(self, vad_model: VADModel, silence_chunks_to_end: int = 30):
        self.vad_model = vad_model
        self.silence_chunks_to_end = silence_chunks_to_end
        self.silence_count = 0
        self.is_speaking = False
        self.buffer = np.array([], dtype=np.float32)

    def process(self, chunk: bytes) -> str:
        """
        Processes a PCM16 chunk and returns the VAD state.
        States: "silence", "speech_ongoing", "utterance_end"
        """
        # 1. Normalization
        audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_data) == 0:
            return "silence"

        # 2. Simple Gain Control (Automatic Leveling). Do not amplify very low
        # energy noise into speech; that path produced false utterances.
        max_val = np.max(np.abs(audio_data))
        rms = float(np.sqrt(np.mean(np.square(audio_data))))
        if not self.is_speaking and rms < 0.003:
            return "silence"
        if 0.01 <= max_val < 0.1:
            audio_data = audio_data * (0.1 / max_val)

        # 3. Windowing for Silero (MUST be exactly 512 samples for 16kHz)
        self.buffer = np.append(self.buffer, audio_data)
        WINDOW_SIZE = 512

        if len(self.buffer) < WINDOW_SIZE:
            return "silence" if not self.is_speaking else "speech_ongoing"

        # Process as many windows as possible in this chunk
        probs = []
        while len(self.buffer) >= WINDOW_SIZE:
            window = self.buffer[:WINDOW_SIZE]
            self.buffer = self.buffer[WINDOW_SIZE:]

            tensor_window = torch.from_numpy(window).unsqueeze(0).to('cpu').float()
            with torch.no_grad():
                probs.append(self.vad_model.model(tensor_window, 16000).item())

        speech_prob = max(probs) if probs else 0.0

        # 4. State Machine
        if speech_prob > self.vad_model.threshold:
            self.silence_count = 0
            if not self.is_speaking:
                self.is_speaking = True
            return "speech_ongoing"
        else:
            if self.is_speaking:
                self.silence_count += 1
                if self.silence_count >= self.silence_chunks_to_end:
                    self.is_speaking = False
                    self.silence_count = 0
                    return "utterance_end"
            return "silence"

    def reset(self):
        self.is_speaking = False
        self.silence_count = 0
        self.buffer = np.array([], dtype=np.float32)
