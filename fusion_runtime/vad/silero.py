"""Silero VAD engine."""
from typing import AsyncIterator
import numpy as np

from fusion_runtime.vad.base import SpeechSegment, VADBase


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
