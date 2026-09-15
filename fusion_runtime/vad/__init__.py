"""
VAD and Turn Detection
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator
import asyncio
import numpy as np


@dataclass
class VADResult:
    is_speech: bool
    confidence: float
    audio: bytes  # Pass-through audio
    start_sample: int = 0
    end_sample: int = 0


@dataclass
class SpeechSegment:
    """A detected speech segment with audio and timestamps."""
    audio: bytes
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    confidence: float


class VADBase(ABC):
    """Base class for Voice Activity Detection."""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = getattr(config, 'sample_rate', 16000)
    
    @abstractmethod
    async def process_stream(
        self, 
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[SpeechSegment]:
        """Process audio stream and yield speech segments."""
        pass
    
    @abstractmethod
    async def reset(self):
        pass


class SileroVAD(VADBase):
    """Silero VAD implementation with proper speech segment framing."""
    
    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._get_speech_timestamps = None
        self._state = None
        self._min_speech_duration = getattr(config, 'min_speech_duration_ms', 250)
        self._min_silence_duration = getattr(config, 'min_silence_duration_ms', 100)
        self._speech_pad = getattr(config, 'speech_pad_ms', 30)
    
    async def _load_model(self):
        import torch
        self._model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self._get_speech_timestamps = utils[0]
        self._state = None
    
    async def process_stream(
        self, 
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[SpeechSegment]:
        import torch
        
        if self._model is None:
            await self._load_model()
        
        # Accumulate audio for VAD analysis
        buffer = bytearray()
        chunk_samples = 512  # 32ms at 16kHz (Silero's expected chunk)
        bytes_per_chunk = chunk_samples * 2  # 16-bit
        total_samples = 0
        
        async for chunk in audio_stream:
            buffer.extend(chunk)
            
            # Process when we have enough data
            while len(buffer) >= bytes_per_chunk:
                chunk_bytes = bytes(buffer[:bytes_per_chunk])
                buffer = buffer[bytes_per_chunk:]
                
                # Convert to float32 tensor
                audio_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                
                # Run VAD
                with torch.no_grad():
                    speech_prob = self._model(audio_tensor, self.sample_rate).item()
                
                total_samples += chunk_samples
        
        # After stream ends, run VAD on accumulated audio to get segments
        # Note: In production, you'd use streaming VAD with state
        # For now, collect all audio and segment at the end
        if len(buffer) > 0:
            # Process remaining
            pass
        
        # Yield the full buffer as one segment for now
        # TODO: Implement streaming segment detection with get_speech_timestamps
        if total_samples > 0:
            yield SpeechSegment(
                audio=bytes(buffer),
                start_sample=0,
                end_sample=total_samples,
                start_time=0,
                end_time=total_samples / self.sample_rate,
                confidence=0.9,
            )
    
    async def reset(self):
        self._state = None


async def vad_stream_segments(
    audio_stream: AsyncIterator[bytes],
    vad: SileroVAD,
    min_segment_duration: float = 0.5
) -> AsyncIterator[SpeechSegment]:
    """
    Convenience function: run VAD on stream and yield speech segments.
    Uses Silero's get_speech_timestamps on accumulated windows.
    """
    import torch
    
    if vad._model is None:
        await vad._load_model()
    
    # Accumulate into windows for segment detection
    window_duration = 30.0  # seconds
    window_samples = int(window_duration * vad.sample_rate)
    audio_buffer = np.array([], dtype=np.float32)
    sample_offset = 0
    
    async for chunk in audio_stream:
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buffer = np.concatenate([audio_buffer, audio_np])
        
        # When window is full, detect segments
        if len(audio_buffer) >= window_samples:
            # Run VAD timestamps detection
            audio_tensor = torch.from_numpy(audio_buffer).unsqueeze(0)
            with torch.no_grad():
                speech_timestamps = vad._get_speech_timestamps(
                    audio_tensor,
                    vad._model,
                    sampling_rate=vad.sample_rate,
                    min_speech_duration_ms=vad._min_speech_duration,
                    min_silence_duration_ms=vad._min_silence_duration,
                    speech_pad_ms=vad._speech_pad,
                )
            
            # Yield each speech segment
            for ts in speech_timestamps:
                start = ts['start']
                end = ts['end']
                segment_audio = audio_buffer[start:end]
                segment_bytes = (segment_audio * 32768).astype(np.int16).tobytes()
                
                yield SpeechSegment(
                    audio=segment_bytes,
                    start_sample=sample_offset + start,
                    end_sample=sample_offset + end,
                    start_time=(sample_offset + start) / vad.sample_rate,
                    end_time=(sample_offset + end) / vad.sample_rate,
                    confidence=0.9,
                )
            
            # Keep tail for overlap
            overlap = int(1.0 * vad.sample_rate)  # 1 second overlap
            if len(audio_buffer) > overlap:
                audio_buffer = audio_buffer[-overlap:]
                sample_offset += len(audio_buffer) - overlap
    
    # Process remaining
    if len(audio_buffer) > 0:
        audio_tensor = torch.from_numpy(audio_buffer).unsqueeze(0)
        with torch.no_grad():
            speech_timestamps = vad._get_speech_timestamps(
                audio_tensor,
                vad._model,
                sampling_rate=vad.sample_rate,
                min_speech_duration_ms=vad._min_speech_duration,
                min_silence_duration_ms=vad._min_silence_duration,
                speech_pad_ms=vad._speech_pad,
            )
        
        for ts in speech_timestamps:
            start = ts['start']
            end = ts['end']
            segment_audio = audio_buffer[start:end]
            segment_bytes = (segment_audio * 32768).astype(np.int16).tobytes()
            
            yield SpeechSegment(
                audio=segment_bytes,
                start_sample=sample_offset + start,
                end_sample=sample_offset + end,
                start_time=(sample_offset + start) / vad.sample_rate,
                end_time=(sample_offset + end) / vad.sample_rate,
                confidence=0.9,
            )


@dataclass
class TurnState:
    """VAD-derived signal shared between the audio pipeline and turn
    detection: how long (ms) it's been since Silero last saw real speech.

    Updated frame-by-frame by `PipelineOrchestrator._apply_vad`; read by
    the turn detector so it can tell a genuine pause apart from the STT's
    own rolling-window transcript merely ending in punctuation (Whisper
    does that constantly on short windows, even mid-sentence).
    `vad_active` is False until real VAD frames have been scored (e.g. the
    Silero model failed to load) — callers should skip silence-gating in
    that case rather than silently never reaching the threshold.
    """
    silence_ms: float = 0.0
    vad_active: bool = False


class TurnDetectorBase(ABC):
    """Base class for End-of-Turn *content* detection.

    This answers exactly one question — "does this accumulated transcript
    read like a finished utterance on its own?" — and nothing about timing.
    Silence timing is handled separately, by an always-running watcher in
    `PipelineOrchestrator._llm_stage` (see its docstring for why this had
    to be split out: reacting to silence only when a *new* STT result
    arrives means the silence being measured is the pause *before* that
    new speech, not a pause *after* the user actually stopped talking —
    which is backwards, and was cutting turns off mid-sentence on any
    ordinary thinking-pause). The watcher uses this signal only to pick
    between a short confirmation pause (confidently complete) and a
    longer, safer one (ambiguous) — it doesn't own the decision.
    """

    def __init__(self, config):
        self.config = config

    def looks_complete(self, text: str) -> bool:
        """Default heuristic: ends with sentence-ending punctuation.
        Subclasses (e.g. a real EoT model) can override with something
        smarter — text + prosody, not just the trailing character."""
        return text.strip().endswith((".", "!", "?", "。", "！", "？"))

    async def reset(self):
        pass


class PunctuationTurnDetector(TurnDetectorBase):
    """Text ending in sentence punctuation counts as a finished thought, so a
    shorter pause ends the turn. Uses TurnDetectorBase's `looks_complete`."""
    pass


class FireRedEOTTurnDetector(TurnDetectorBase):
    """FireRedChat EoT model (bilingual EN/ZH).

    `looks_complete` currently falls back to the same punctuation heuristic
    as PunctuationTurnDetector — a real implementation would load
    `self.config.model_path` (an ONNX EoT model taking text + prosodic
    features) and override this method with its actual prediction."""
    pass


def create_vad(config) -> VADBase:
    from fusion_runtime.config import Provider
    if config.provider == Provider.SILERO:
        return SileroVAD(config)
    raise ValueError(f"Unknown VAD provider: {config.provider}")


def create_turn_detector(config) -> TurnDetectorBase:
    from fusion_runtime.config import Provider
    if config.provider == Provider.PUNCTUATION:
        return PunctuationTurnDetector(config)
    elif config.provider == Provider.FIRE_RED_EOT:
        return FireRedEOTTurnDetector(config)
    raise ValueError(f"Unknown turn detector: {config.provider}")