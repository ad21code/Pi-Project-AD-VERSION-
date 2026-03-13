#!/usr/bin/env python3
"""
BUDDY Voice Assistant - Main Entry Point
==========================================
A locally-running voice assistant for Raspberry Pi 4,
similar to Amazon Alexa, using Gemini API for internet queries.

Usage:
    python main.py              # Run normally
    python main.py --debug      # Run with debug output
    python main.py --test       # Test mode (no wake word)

Author: AI Assistant
Optimized for: Raspberry Pi 4 Model B (4GB RAM)
"""

# --- Suppress noisy native library warnings before any other imports ---
import os
os.environ.setdefault('ORT_LOG_LEVEL', '3')          # ONNX Runtime: errors only
os.environ.setdefault('JACK_NO_START_SERVER', '1')    # JACK: don't attempt auto-start

# Suppress ALSA error messages on Linux (they are harmless but very noisy)
try:
    import ctypes
    _ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
    )
    def _alsa_error_handler(filename, line, function, err, fmt):
        pass
    _c_alsa_handler = _ERROR_HANDLER_FUNC(_alsa_error_handler)
    _asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    _asound.snd_lib_error_set_handler(_c_alsa_handler)
except (OSError, AttributeError):
    pass  # Not on Linux or ALSA not available
# -----------------------------------------------------------------------

import sys
import time
import signal
import argparse
import threading
from pathlib import Path

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


class BuddyAssistant:
    """
    Main voice assistant orchestrator.
    Coordinates all modules for seamless voice interaction.
    """
    
    def __init__(self, debug: bool = False, test_mode: bool = False):
        """
        Initialize the assistant.
        
        Args:
            debug: Enable debug logging
            test_mode: Skip wake word detection (for testing)
        """
        self.debug = debug
        self.test_mode = test_mode
        self._running = False
        self._processing = False
        self._wake_word_triggered = False  # Guard against premature voice input
        self._tts_playing = False  # Track TTS playback
        
        # Override debug setting
        if debug:
            config.assistant.debug = True
        
        self._print_banner()
        
        # Initialize modules (lazy loading where possible)
        self._init_modules()
    
    def _print_banner(self):
        """Print startup banner."""
        if RICH_AVAILABLE and console is not None:
            banner_text = (
                "██████╗ ██╗   ██╗██████╗ ██████╗ ██╗   ██╗\n"
                "██╔══██╗██║   ██║██╔══██╗██╔══██╗╚██╗ ██╔╝\n"
                "██████╔╝██║   ██║██║  ██║██║  ██║ ╚████╔╝ \n"
                "██╔══██╗██║   ██║██║  ██║██║  ██║  ╚██╔╝  \n"
                "██████╔╝╚██████╔╝██████╔╝██████╔╝   ██║   \n"
                "╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝    ╚═╝   "
            )
            console.print(
                Panel(
                    banner_text + "\n\n   Voice Assistant for Raspberry Pi",
                    style="bold cyan",
                    expand=False,
                ),
                justify="center",
            )
        else:
            banner = (
                "\n"
                " ____  _   _ ____  ______   __\n"
                "| __ )| | | |  _ \\|  _ \\ \\ / /\n"
                "|  _ \\| | | | | | | | | \\ V /\n"
                "| |_) | |_| | |_| | |_| || |\n"
                "|____/ \\___/|____/|____/ |_|\n"
                "\n"
                "    Voice Assistant for Raspberry Pi\n"
            )
            print(banner, flush=True)
        sys.stdout.flush()
    
    def _init_modules(self):
        """Initialize all assistant modules."""
        print("\n📦 Initializing modules...\n")
        
        # Audio utilities
        print("  [1/6] Audio I/O...")
        from modules.audio_utils import AudioRecorder, AudioPlayer
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        print("  ✓ Audio ready")
        
        # Wake word detection
        print("  [2/6] Wake word detection...")
        from modules.wake_word import WakeWordDetector
        self.wake_word = WakeWordDetector()
        print("  ✓ Wake word ready")
        
        # Speech recognition (lazy load model)
        print("  [3/6] Speech recognition...")
        from modules.speech_recognition import SpeechRecognizer
        self.recognizer = SpeechRecognizer(lazy_load=True)
        print("  ✓ Speech recognition ready (model loads on first use)")
        
        # Text-to-speech
        print("  [4/6] Text-to-speech...")
        from modules.tts import TextToSpeech
        self.tts = TextToSpeech()
        print("  ✓ TTS ready")
        
        # Intent handler
        print("  [5/6] Intent handler...")
        from modules.intent_handler import IntentHandler, IntentType
        self.intent_handler = IntentHandler()
        self.IntentType = IntentType
        print("  ✓ Intent handler ready")
        
        # Gemini client
        print("  [6/6] Gemini API client...")
        from modules.gemini_client import GeminiClient
        self.gemini = GeminiClient(keep_history=True)
        print("  ✓ Gemini client ready")
        
        print("\n✅ All modules initialized!")
    
    def start(self):
        """Start the voice assistant."""
        self._running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Print configuration
        if self.debug:
            config.print_config()
        
        print("\n" + "="*50)
        print(f"🎤 Say '{config.wake_word.phrase.replace('_', ' ').title()}' to activate")
        print("   Press Ctrl+C to quit")
        print("="*50 + "\n")
        
        if self.test_mode:
            # Test mode - skip wake word
            self._test_loop()
        else:
            # Normal mode - listen for wake word
            self._main_loop()
    
    def _main_loop(self):
        """Main interaction loop with wake word detection."""
        # Wait briefly to ensure system is ready
        time.sleep(0.5)
        
        # Start wake word detection in background thread
        self.wake_word.start_listening(self._on_wake_word)
        
        try:
            while self._running:
                # Keep the main thread alive while listening for wake word
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self._cleanup()
    
    def _test_loop(self):
        """Test loop without wake word detection."""
        print("🧪 Test mode - Press Enter to simulate wake word")
        print("   Type 'quit' to exit\n")
        
        try:
            while self._running:
                try:
                    user_input = input("Press Enter to speak (or 'quit'): ")
                    if user_input.lower() == 'quit':
                        break
                    self._on_wake_word()
                except EOFError:
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self._cleanup()
    
    def _on_wake_word(self):
        """Handle wake word detection."""
        # Guard: only process if genuinely triggered and not already processing
        if self._processing or not self._running:
            return
        
        self._processing = True
        self._wake_word_triggered = True  # Mark that wake word was genuinely detected
        
        try:
            # Play wake sound
            print("\n🔔 Wake word detected!")
            sys.stdout.flush()
            self.player.play_notification("wake")
            
            # Pause wake word detection
            self.wake_word.pause_listening()
            
            # Listen for command
            print("👂 Listening for your command...")
            sys.stdout.flush()
            
            audio = self.recorder.record_until_silence(
                max_duration=config.assistant.listen_timeout,
                silence_threshold=1.0
            )
            
            if len(audio) < 3200:  # Less than 0.1 seconds
                print("❌ No speech detected")
                sys.stdout.flush()
                self.tts.speak("Sorry, I didn't hear anything.")
                return
            
            # Transcribe
            print("🔄 Processing...")
            sys.stdout.flush()
            text = self.recognizer.transcribe(audio)
            
            if not text or len(text.strip()) < 2:
                print("❌ Could not understand audio")
                sys.stdout.flush()
                self.tts.speak("Sorry, I didn't catch that.")
                return
            
            print(f"📝 You said: '{text}'")
            sys.stdout.flush()
            
            # Process the command
            self._process_command(text)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            sys.stdout.flush()
            self.tts.speak("Sorry, something went wrong.")
        
        finally:
            self._processing = False
            self._wake_word_triggered = False
            # Resume wake word detection
            if not self.test_mode and self._running:
                self.wake_word.resume_listening()
    
    def _process_command(self, text: str):
        """
        Process user command.
        Routes to local handler or Gemini API.
        Supports interruption: user can say "stop" to stop the assistant.
        """
        # Detect intent
        intent_type, params = self.intent_handler.detect_intent(text)
        
        if self.debug:
            print(f"🎯 Intent: {intent_type.value}, Params: {params}")
            sys.stdout.flush()
        
        # Handle STOP commands immediately
        if intent_type == self.IntentType.LOCAL_STOP:
            print("⏹️ Command stopped by user")
            sys.stdout.flush()
            self.tts.speak("Okay, stopped.")
            return
        
        # Handle based on intent
        if self.intent_handler.is_local_intent(intent_type):
            # Handle locally
            response = self.intent_handler.handle_local(intent_type, params)
        else:
            # Query Gemini
            print("🌐 Querying Gemini...")
            sys.stdout.flush()
            response = self.gemini.generate(text)
        
        # Speak response with ability to interrupt
        print(f"🔊 Response: '{response}'")
        sys.stdout.flush()
        self._speak_with_interrupt(response)
    
    def _speak_with_interrupt(self, text: str):
        """
        Speak text but allow user to interrupt by saying "stop".
        After response finishes, listen for interruption.
        """
        self._tts_playing = True
        
        try:
            # Start speaking in a separate thread to allow monitoring
            import threading
            speak_thread = threading.Thread(
                target=self.tts.speak,
                args=(text,),
                daemon=True
            )
            speak_thread.start()
            
            # Wait for TTS to complete while checking for interrupts
            speak_thread.join(timeout=config.assistant.response_timeout)
            
            self._tts_playing = False
            
            # Brief pause before re-engaging wake word listener
            time.sleep(0.5)
        
        except Exception as e:
            print(f"Error during TTS: {e}")
            sys.stdout.flush()
            self._tts_playing = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n\n🛑 Shutting down...")
        self._running = False
    
    def _cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        try:
            self.wake_word.stop_listening()
        except Exception:
            pass
        
        # Allow background threads a moment to finish
        time.sleep(0.2)
        
        try:
            self.wake_word.cleanup()
        except Exception:
            pass
        
        try:
            self.recorder.cleanup()
        except Exception:
            pass
        
        try:
            self.player.cleanup()
        except Exception:
            pass
        
        print("👋 Goodbye!")
        # Force-exit to avoid PortAudio segfault during interpreter shutdown
        os._exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BUDDY Voice Assistant for Raspberry Pi"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test mode (no wake word required)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        # Run benchmark
        print("Running benchmarks...\n")
        from modules.speech_recognition import benchmark_whisper
        benchmark_whisper()
        return
    
    # Start assistant
    assistant = BuddyAssistant(
        debug=args.debug,
        test_mode=args.test
    )
    assistant.start()


if __name__ == "__main__":
    main()