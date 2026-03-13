"""
BUDDY Voice Assistant - Wake Word Detection
============================================
Efficient wake word detection optimized for Raspberry Pi.
Uses openWakeWord for open-source detection.
"""

import time
import threading
from typing import Callable, Optional
from pathlib import Path
import struct

import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from openwakeword.model import Model as OWWModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    print("Warning: openwakeword not available. Using fallback detection.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class WakeWordDetector:
    """
    Wake word detection using openWakeWord.
    Runs continuously with minimal CPU usage.
    """
    
    def __init__(self):
        self.sample_rate = 16000  # openWakeWord requires 16kHz
        self.chunk_size = 1280    # 80ms chunks (optimal for openWakeWord)
        self.sensitivity = config.wake_word.sensitivity
        self.wake_phrase = config.wake_word.phrase
        
        # Initialize openWakeWord model
        self.model: Optional[OWWModel] = None
        if OPENWAKEWORD_AVAILABLE:
            try:
                # Load pre-trained wake word models
                self.model = OWWModel(
                    wakeword_models=["hey_jarvis"],  # Use built-in model as base
                    inference_framework="onnx"
                )
                print(f"✓ Wake word model loaded")
            except Exception as e:
                print(f"Warning: Could not load wake word model: {e}")
        
        # PyAudio
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream = None
        
        # Detection state
        self._is_listening = False
        self._paused = False  # Pause detection without killing the thread
        self._detection_callback: Optional[Callable] = None
        self._listen_thread: Optional[threading.Thread] = None
        
        # For fallback detection (energy-based with safeguards)
        self._energy_threshold = 5000
        self._energy_buffer = []
        self._warmup_frames = 0
        self._warmup_required = 30  # Collect baseline before detecting
        self._consecutive_spikes = 0
        self._consecutive_required = 3  # Require sustained energy, not a single spike
        self._last_trigger_time = 0.0  # Timestamp-based cooldown
    
    @property
    def pyaudio(self) -> pyaudio.PyAudio:
        """Lazy initialization of PyAudio."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio
    
    def start_listening(self, on_wake_word: Callable):
        """
        Start listening for wake word in background.
        
        Args:
            on_wake_word: Callback function when wake word detected
        """
        if self._is_listening:
            return
        
        self._detection_callback = on_wake_word
        self._is_listening = True
        
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        
        print(f"👂 Listening for wake word...")
    
    def stop_listening(self):
        """Stop wake word detection."""
        self._is_listening = False
        
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=1.0)
            self._listen_thread = None
    
    def _listen_loop(self):
        """Main listening loop running in background thread."""
        try:
            # Open audio stream
            self._stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            while self._is_listening:
                try:
                    # Read audio chunk (always read to avoid buffer overflow)
                    audio_data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Skip detection while paused
                    if self._paused:
                        continue
                    
                    # Check for wake word
                    if self._detect_wake_word(audio_data):
                        if self._detection_callback:
                            self._detection_callback()
                
                except Exception as e:
                    if self._is_listening:
                        print(f"Audio error: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Wake word listener error: {e}")
        
        finally:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
    
    def _detect_wake_word(self, audio_data: bytes) -> bool:
        """
        Detect wake word in audio chunk.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
        
        Returns:
            True if wake word detected
        """
        if self.model is not None:
            # Use openWakeWord
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Run prediction
            prediction = self.model.predict(audio_array)
            
            # Check all wake word scores
            for key, score in prediction.items():
                if score > self.sensitivity:
                    print(f"🎯 Wake word detected! ({key}: {score:.2f})")
                    return True
            
            return False
        else:
            # Fallback: Simple energy-based detection
            # This is a placeholder - in production, use a proper wake word model
            return self._fallback_detection(audio_data)
    
    def _fallback_detection(self, audio_data: bytes) -> bool:
        """
        Fallback wake word detection using audio energy.
        This is NOT recommended for production - use openWakeWord or Porcupine.
        
        Safeguards:
        - Requires a warmup period to establish baseline energy levels
        - Requires multiple consecutive high-energy frames (sustained speech)
        - Uses a high energy threshold and multiplier to avoid ambient noise triggers
        """
        # Convert to samples
        samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
        energy = sum(abs(s) for s in samples) / len(samples)
        
        # Track energy buffer for baseline
        self._energy_buffer.append(energy)
        if len(self._energy_buffer) > 30:
            self._energy_buffer.pop(0)
        
        # Warmup phase: collect baseline energy before allowing detection
        self._warmup_frames += 1
        if self._warmup_frames < self._warmup_required:
            return False
        
        # Cooldown: prevent multiple triggers in quick succession
        if time.time() - self._last_trigger_time < 2.0:
            return False
        
        avg_energy = sum(self._energy_buffer) / len(self._energy_buffer)
        
        # Require energy to be significantly above both the threshold and average
        is_spike = energy > avg_energy * 5 and energy > self._energy_threshold
        
        if is_spike:
            self._consecutive_spikes += 1
        else:
            self._consecutive_spikes = 0
        
        # Only trigger after multiple consecutive high-energy frames (sustained speech)
        if self._consecutive_spikes >= self._consecutive_required:
            self._consecutive_spikes = 0
            self._last_trigger_time = time.time()
            return True
        
        return False
    
    def pause_listening(self):
        """Temporarily pause wake word detection without killing the thread."""
        self._paused = True
    
    def resume_listening(self):
        """Resume wake word detection."""
        if self._detection_callback is not None:
            # Reset detection state to avoid stale data triggering false positives
            self._energy_buffer = []
            self._consecutive_spikes = 0
            self._warmup_frames = 0
            self._last_trigger_time = 0.0
            self._paused = False
            if self._listen_thread is None or not self._listen_thread.is_alive():
                self._is_listening = True
                self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
                self._listen_thread.start()
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_listening()
        
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None


class SimpleWakeWordDetector:
    """
    Simplified wake word detector using keyword spotting.
    Uses the speech recognizer to detect wake phrases.
    This is a fallback for when openWakeWord is not available.
    """
    
    def __init__(self, speech_recognizer):
        self.speech_recognizer = speech_recognizer
        self.wake_phrases = ["hey buddy", "hey body", "hey birdie", "a buddy"]  # Include common misheard variations
        self._is_listening = False
        self._callback = None
    
    def start_listening(self, on_wake_word: Callable):
        """Start listening for wake word."""
        self._callback = on_wake_word
        self._is_listening = True
        
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()
    
    def _listen_loop(self):
        """Continuously listen and check for wake phrases."""
        from .audio_utils import AudioRecorder
        
        recorder = AudioRecorder()
        
        while self._is_listening:
            try:
                # Record short audio snippet
                audio = recorder.record_for_duration(2.0)
                
                # Transcribe
                text = self.speech_recognizer.transcribe(audio)
                
                if text:
                    text_lower = text.lower()
                    for phrase in self.wake_phrases:
                        if phrase in text_lower:
                            if self._callback:
                                self._callback()
                            break
                
            except Exception as e:
                print(f"Wake word detection error: {e}")
                time.sleep(0.5)
    
    def stop_listening(self):
        """Stop listening."""
        self._is_listening = False


# Test wake word detection
if __name__ == "__main__":
    print("Testing Wake Word Detection...")
    print("Say 'Hey Jarvis' or make a loud sound (fallback mode)")
    
    detector = WakeWordDetector()
    
    wake_detected = threading.Event()
    
    def on_wake():
        print("\n🎤 WAKE WORD DETECTED!")
        wake_detected.set()
    
    detector.start_listening(on_wake)
    
    try:
        # Wait for wake word or timeout
        if wake_detected.wait(timeout=30):
            print("Wake word test successful!")
        else:
            print("No wake word detected (timeout)")
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        detector.stop_listening()
        detector.cleanup()
