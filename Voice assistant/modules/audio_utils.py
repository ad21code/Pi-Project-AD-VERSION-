"""
BUDDY Voice Assistant - Audio Utilities
========================================
Handles audio recording and playback optimized for Raspberry Pi.
Uses PyAudio for cross-platform compatibility.
"""
from __future__ import annotations

import wave
import struct
import threading
import queue
from typing import Generator, Optional, Callable
from pathlib import Path

import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Audio features disabled.")

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("Warning: webrtcvad not available. Voice activity detection disabled.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class AudioRecorder:
    """
    Efficient audio recording with voice activity detection.
    Optimized for low-latency streaming on Raspberry Pi.
    """
    
    def __init__(self):
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = config.audio.chunk_size
        self.channels = config.audio.channels
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        
        # Voice Activity Detection
        self.vad = None
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(config.audio.vad_aggressiveness)
        
        # PyAudio instance (lazy initialization)
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream = None
        
        # Recording state
        self._is_recording = False
        self._audio_queue: queue.Queue = queue.Queue()
    
    @property
    def pyaudio(self) -> pyaudio.PyAudio:
        """Lazy initialization of PyAudio."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio
    
    def list_devices(self):
        """List available audio input devices."""
        print("\nAvailable Audio Input Devices:")
        print("-" * 40)
        for i in range(self.pyaudio.get_device_count()):
            info = self.pyaudio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
                print(f"      Channels: {info['maxInputChannels']}, "
                      f"Rate: {int(info['defaultSampleRate'])}Hz")
        print("-" * 40)
    
    def get_input_device_index(self) -> Optional[int]:
        """Get the input device index from config or default."""
        if config.audio.mic_device_index >= 0:
            return config.audio.mic_device_index
        return None  # Use default device
    
    def _open_stream(self):
        """Open audio input stream."""
        if self._stream is not None:
            return
        
        device_index = self.get_input_device_index()
        
        self._stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for non-blocking audio capture."""
        if self._is_recording:
            self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def start_recording(self):
        """Start non-blocking audio recording."""
        self._is_recording = True
        self._audio_queue = queue.Queue()
        self._open_stream()
        self._stream.start_stream()
    
    def stop_recording(self):
        """Stop audio recording."""
        self._is_recording = False
        if self._stream is not None:
            self._stream.stop_stream()
    
    def get_audio_chunks(self) -> Generator[bytes, None, None]:
        """Generator that yields audio chunks."""
        while self._is_recording or not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                continue
    
    def record_until_silence(
        self,
        max_duration: float = None,
        silence_threshold: float = None,
        on_speech_start: Optional[Callable] = None
    ) -> bytes:
        """
        Record audio until silence is detected.
        Uses VAD for efficient silence detection.
        
        Args:
            max_duration: Maximum recording duration in seconds
            silence_threshold: Seconds of silence to stop recording
            on_speech_start: Callback when speech is first detected
        
        Returns:
            Recorded audio as bytes
        """
        max_duration = max_duration or config.audio.max_recording_duration
        silence_threshold = silence_threshold or config.audio.silence_threshold
        
        # Calculate frame parameters
        frame_duration_ms = 30  # VAD requires 10, 20, or 30ms frames
        frame_size = int(self.sample_rate * frame_duration_ms / 1000) * 2  # *2 for 16-bit
        
        self._is_recording = True
        self._audio_queue = queue.Queue()
        
        # Open stream in blocking mode for simpler control
        device_index = self.get_input_device_index()
        stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frame_size // 2  # frames, not bytes
        )
        
        frames = []
        silence_frames = 0
        speech_started = False
        max_frames = int(max_duration * self.sample_rate / (frame_size // 2))
        silence_frame_threshold = int(silence_threshold * self.sample_rate / (frame_size // 2))
        
        try:
            for _ in range(max_frames):
                if not self._is_recording:
                    break
                
                data = stream.read(frame_size // 2, exception_on_overflow=False)
                frames.append(data)
                
                # Voice activity detection
                if self.vad is not None:
                    try:
                        is_speech = self.vad.is_speech(data, self.sample_rate)
                    except Exception:
                        is_speech = True  # Assume speech on VAD error
                else:
                    # Fallback: check audio energy
                    is_speech = self._check_audio_energy(data)
                
                if is_speech:
                    if not speech_started:
                        speech_started = True
                        if on_speech_start:
                            on_speech_start()
                    silence_frames = 0
                else:
                    if speech_started:
                        silence_frames += 1
                        if silence_frames >= silence_frame_threshold:
                            break
        
        finally:
            stream.stop_stream()
            stream.close()
            self._is_recording = False
        
        return b''.join(frames)
    
    def _check_audio_energy(self, data: bytes, threshold: int = 500) -> bool:
        """Fallback voice detection using audio energy."""
        shorts = struct.unpack(f'{len(data)//2}h', data)
        energy = sum(abs(s) for s in shorts) / len(shorts)
        return energy > threshold
    
    def record_for_duration(self, duration: float) -> bytes:
        """Record audio for a fixed duration."""
        device_index = self.get_input_device_index()
        stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        num_chunks = int(self.sample_rate / self.chunk_size * duration)
        
        for _ in range(num_chunks):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        return b''.join(frames)
    
    def save_wav(self, audio_data: bytes, filepath: str):
        """Save audio data to WAV file."""
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
    
    def audio_to_numpy(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array for Whisper."""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Normalize to float32 in range [-1, 1]
        return audio_array.astype(np.float32) / 32768.0
    
    def cleanup(self):
        """Clean up PyAudio resources."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None


class AudioPlayer:
    """
    Audio playback optimized for Raspberry Pi.
    Supports streaming playback for low latency.
    """
    
    def __init__(self):
        self.sample_rate = config.audio.sample_rate
        self._pyaudio: Optional[pyaudio.PyAudio] = None
    
    @property
    def pyaudio(self) -> pyaudio.PyAudio:
        """Lazy initialization of PyAudio."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio
    
    def get_output_device_index(self) -> Optional[int]:
        """Get the output device index from config or default."""
        if config.audio.speaker_device_index >= 0:
            return config.audio.speaker_device_index
        return None
    
    def play_wav(self, filepath: str):
        """Play a WAV file."""
        with wave.open(filepath, 'rb') as wf:
            stream = self.pyaudio.open(
                format=self.pyaudio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                output_device_index=self.get_output_device_index()
            )
            
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            
            stream.stop_stream()
            stream.close()
    
    def play_audio(self, audio_data: bytes, sample_rate: int = None, channels: int = 1):
        """Play raw audio data."""
        sample_rate = sample_rate or self.sample_rate
        
        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
            output_device_index=self.get_output_device_index()
        )
        
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
    
    def play_beep(self, frequency: int = 440, duration: float = 0.2):
        """Play a simple beep sound."""
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        
        # Generate sine wave
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        wave_data = np.sin(2 * np.pi * frequency * t)
        
        # Apply fade in/out to avoid clicks
        fade_samples = int(sample_rate * 0.01)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        wave_data[:fade_samples] *= fade_in
        wave_data[-fade_samples:] *= fade_out
        
        # Convert to 16-bit PCM
        audio_data = (wave_data * 32767).astype(np.int16).tobytes()
        
        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=self.get_output_device_index()
        )
        
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
    
    def play_notification(self, sound_type: str = "wake"):
        """Play a notification sound."""
        sounds_dir = config.assistant.sounds_dir
        sound_file = sounds_dir / f"{sound_type}.wav"
        
        if sound_file.exists():
            self.play_wav(str(sound_file))
        else:
            # Fallback to beep
            if sound_type == "wake":
                self.play_beep(880, 0.15)  # High beep
            elif sound_type == "error":
                self.play_beep(220, 0.3)   # Low beep
            else:
                self.play_beep(440, 0.2)   # Normal beep
    
    def cleanup(self):
        """Clean up resources."""
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None


# Test the audio utilities
if __name__ == "__main__":
    print("Testing Audio Utilities...")
    
    recorder = AudioRecorder()
    player = AudioPlayer()
    
    # List available devices
    recorder.list_devices()
    
    # Play a test beep
    print("\nPlaying test beep...")
    player.play_beep(880, 0.2)
    
    # Test recording
    print("\nRecording for 3 seconds...")
    audio = recorder.record_for_duration(3.0)
    print(f"Recorded {len(audio)} bytes")
    
    # Play back
    print("Playing back recording...")
    player.play_audio(audio)
    
    # Cleanup
    recorder.cleanup()
    player.cleanup()
    
    print("\nAudio test complete!")
