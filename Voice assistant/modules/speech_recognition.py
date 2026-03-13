"""
BUDDY Voice Assistant - Speech Recognition
===========================================
Efficient speech-to-text using faster-whisper.
Optimized for Raspberry Pi with INT8 quantization.
"""
from __future__ import annotations

import time
import io
import wave
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available. Install with: pip install faster-whisper")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class SpeechRecognizer:
    """
    Speech-to-text using faster-whisper.
    Optimized for low latency on Raspberry Pi.
    """
    
    def __init__(self, lazy_load: bool = True):
        """
        Initialize speech recognizer.
        
        Args:
            lazy_load: If True, load model on first use (saves startup time)
        """
        self.model_name = config.whisper.model
        self.compute_type = config.whisper.compute_type
        self.cpu_threads = config.whisper.cpu_threads
        self.language = config.whisper.language
        self.beam_size = config.whisper.beam_size
        
        self._model: Optional[WhisperModel] = None
        
        if not lazy_load:
            self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        if self._model is not None:
            return
        
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper is not installed")
        
        print(f"Loading Whisper model: {self.model_name} ({self.compute_type})...")
        start_time = time.time()
        
        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=str(config.whisper.model_dir)
        )
        
        load_time = time.time() - start_time
        print(f"✓ Whisper model loaded in {load_time:.1f}s")
    
    @property
    def model(self) -> WhisperModel:
        """Get the Whisper model (lazy loading)."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Audio sample rate
        
        Returns:
            Transcribed text
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize to float32 in range [-1, 1]
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        return self.transcribe_array(audio_float, sample_rate)
    
    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe numpy audio array to text.
        
        Args:
            audio_array: Float32 audio array in range [-1, 1]
            sample_rate: Audio sample rate (must be 16000 for Whisper)
        
        Returns:
            Transcribed text
        """
        if sample_rate != 16000:
            # Resample to 16kHz if needed
            from scipy import signal
            audio_array = signal.resample(
                audio_array,
                int(len(audio_array) * 16000 / sample_rate)
            )
        
        start_time = time.time()
        
        # Transcribe with optimized settings for speed
        segments, info = self.model.transcribe(
            audio_array,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,  # Filter out non-speech
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            word_timestamps=False,  # Disable for speed
            condition_on_previous_text=False,  # Disable for speed
        )
        
        # Collect all segment texts
        text = " ".join(segment.text.strip() for segment in segments)
        
        transcribe_time = time.time() - start_time
        
        if config.assistant.debug:
            print(f"[STT] Transcribed in {transcribe_time:.2f}s: '{text}'")
        
        return text.strip()
    
    def transcribe_file(self, filepath: str) -> str:
        """
        Transcribe audio from a WAV file.
        
        Args:
            filepath: Path to WAV file
        
        Returns:
            Transcribed text
        """
        with wave.open(filepath, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
        
        # Convert to numpy
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        return self.transcribe_array(audio_float, sample_rate)
    
    def transcribe_stream(
        self,
        audio_generator,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5
    ) -> str:
        """
        Transcribe streaming audio (for real-time applications).
        
        This collects audio chunks and transcribes in batches.
        For true real-time streaming, consider using whisper_streaming.
        
        Args:
            audio_generator: Generator yielding audio chunks
            sample_rate: Audio sample rate
            chunk_duration: Duration of each chunk in seconds
        
        Returns:
            Transcribed text
        """
        all_audio = []
        
        for chunk in audio_generator:
            # Convert chunk to numpy
            chunk_array = np.frombuffer(chunk, dtype=np.int16)
            all_audio.append(chunk_array)
        
        # Combine all chunks
        combined = np.concatenate(all_audio)
        audio_float = combined.astype(np.float32) / 32768.0
        
        return self.transcribe_array(audio_float, sample_rate)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model": self.model_name,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "language": self.language,
            "loaded": self._model is not None
        }
    
    def warmup(self):
        """
        Warm up the model with a dummy transcription.
        This reduces latency on the first real transcription.
        """
        print("Warming up speech recognition...")
        
        # Create 1 second of silence
        dummy_audio = np.zeros(16000, dtype=np.float32)
        
        start_time = time.time()
        _ = self.transcribe_array(dummy_audio)
        warmup_time = time.time() - start_time
        
        print(f"✓ Speech recognition warmed up in {warmup_time:.2f}s")


# Benchmark function
def benchmark_whisper(duration: float = 3.0, iterations: int = 3):
    """
    Benchmark Whisper transcription speed.
    
    Args:
        duration: Audio duration to test (seconds)
        iterations: Number of test iterations
    """
    print(f"\nBenchmarking Whisper ({config.whisper.model})...")
    print(f"Audio duration: {duration}s, Iterations: {iterations}")
    print("-" * 40)
    
    recognizer = SpeechRecognizer(lazy_load=False)
    
    # Generate test audio (random noise)
    test_audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.1
    
    times = []
    for i in range(iterations):
        start = time.time()
        _ = recognizer.transcribe_array(test_audio)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    rtf = avg_time / duration  # Real-Time Factor
    
    print("-" * 40)
    print(f"Average time: {avg_time:.2f}s")
    print(f"Real-Time Factor: {rtf:.2f}x")
    print(f"Speed: {'✓ Real-time capable' if rtf < 1 else '✗ Slower than real-time'}")


# Test the speech recognizer
if __name__ == "__main__":
    print("Testing Speech Recognition...")
    
    recognizer = SpeechRecognizer()
    
    # Print model info
    info = recognizer.get_model_info()
    print(f"\nModel Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Warm up
    recognizer.warmup()
    
    # Test with recording
    from audio_utils import AudioRecorder
    
    recorder = AudioRecorder()
    
    print("\n🎤 Speak something (recording for 5 seconds)...")
    audio = recorder.record_for_duration(5.0)
    
    print("Transcribing...")
    text = recognizer.transcribe(audio)
    
    print(f"\n📝 Transcription: '{text}'")
    
    recorder.cleanup()
    
    # Run benchmark
    benchmark_whisper()
