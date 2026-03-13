#!/usr/bin/env python3
"""
BUDDY Voice Assistant - Unit Tests for Wake Word Detection
==========================================================
Tests the wake word detection improvements including:
- Warmup period prevents premature triggers
- Consecutive frame requirement for sustained speech detection
- Energy threshold and multiplier rejection
- Timestamp-based cooldown between triggers
- Pause/resume state reset
- False trigger comparison (old vs new)

Run: python test_wake_word_unit.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

results = {}


def test_pass(name, detail=""):
    results[name] = True
    detail_str = f" ({detail})" if detail else ""
    print(f"  ✅ PASS: {name}{detail_str}")


def test_fail(name, error=""):
    results[name] = False
    print(f"  ❌ FAIL: {name} - {error}")


class FallbackDetectionTester:
    """Isolated tester that mirrors the fallback detection logic from wake_word.py."""

    def __init__(self):
        self._energy_threshold = 5000
        self._energy_buffer = []
        self._warmup_frames = 0
        self._warmup_required = 30
        self._consecutive_spikes = 0
        self._consecutive_required = 3
        self._last_trigger_time = 0.0

    def detect(self, energy):
        self._energy_buffer.append(energy)
        if len(self._energy_buffer) > 30:
            self._energy_buffer.pop(0)

        self._warmup_frames += 1
        if self._warmup_frames < self._warmup_required:
            return False

        if time.time() - self._last_trigger_time < 2.0:
            return False

        avg_energy = sum(self._energy_buffer) / len(self._energy_buffer)
        is_spike = energy > avg_energy * 5 and energy > self._energy_threshold

        if is_spike:
            self._consecutive_spikes += 1
        else:
            self._consecutive_spikes = 0

        if self._consecutive_spikes >= self._consecutive_required:
            self._consecutive_spikes = 0
            self._last_trigger_time = time.time()
            return True

        return False

    def reset(self):
        self._energy_buffer = []
        self._warmup_frames = 0
        self._consecutive_spikes = 0
        self._last_trigger_time = 0.0


class OldDetection:
    """The OLD detection logic before improvements for comparison."""

    def __init__(self):
        self._energy_threshold = 2000
        self._energy_buffer = []

    def detect(self, energy):
        self._energy_buffer.append(energy)
        if len(self._energy_buffer) > 10:
            self._energy_buffer.pop(0)
        avg_energy = sum(self._energy_buffer) / len(self._energy_buffer)
        if energy > avg_energy * 3 and energy > self._energy_threshold:
            return True
        return False


def test_warmup_period():
    """No triggers during warmup period (30 frames)."""
    det = FallbackDetectionTester()
    for i in range(30):
        if det.detect(500):
            test_fail("Warmup period", "Triggered during warmup")
            return
    test_pass("Warmup period", "No trigger during 30-frame warmup")


def test_ambient_noise_rejection():
    """Low-energy ambient noise should never trigger."""
    det = FallbackDetectionTester()
    for i in range(100):
        energy = 200 + (i % 10) * 60  # 200-740
        if det.detect(energy):
            test_fail("Ambient noise rejection", f"Triggered at frame {i}")
            return
    test_pass("Ambient noise rejection", "100 frames, energy 200-740")


def test_single_spike_rejection():
    """A single energy spike should not trigger detection."""
    det = FallbackDetectionTester()
    for i in range(30):
        det.detect(200)
    if det.detect(50000):
        test_fail("Single spike rejection", "Triggered on single spike")
        return
    if det.detect(200):
        test_fail("Single spike rejection", "Triggered after spike on ambient")
        return
    test_pass("Single spike rejection", "energy=50000")


def test_two_consecutive_not_enough():
    """Two consecutive spikes should not trigger (need 3)."""
    det = FallbackDetectionTester()
    for i in range(30):
        det.detect(200)
    det.detect(50000)
    if det.detect(50000):
        test_fail("Two consecutive spikes", "Triggered on only 2")
        return
    test_pass("Two consecutive spikes", "Need 3, got 2 → no trigger")


def test_three_consecutive_triggers():
    """Three consecutive high-energy frames should trigger."""
    det = FallbackDetectionTester()
    for i in range(30):
        det.detect(200)
    det.detect(50000)
    det.detect(50000)
    if not det.detect(50000):
        test_fail("Three consecutive spikes", "Did NOT trigger on 3 consecutive")
        return
    test_pass("Three consecutive spikes", "Sustained speech → trigger")


def test_interrupted_sequence():
    """Interrupted spike sequence should reset count."""
    det = FallbackDetectionTester()
    for i in range(30):
        det.detect(200)
    det.detect(50000)
    det.detect(50000)
    det.detect(200)  # Interruption
    det.detect(50000)
    if det.detect(50000):
        test_fail("Interrupted sequence", "Triggered despite interruption")
        return
    test_pass("Interrupted sequence", "Reset after ambient interruption")


def test_below_threshold():
    """Energy above average*5 but below absolute threshold (5000)."""
    det = FallbackDetectionTester()
    for i in range(30):
        det.detect(200)
    for _ in range(10):
        if det.detect(4000):
            test_fail("Below threshold", "Triggered below 5000 threshold")
            return
    test_pass("Below threshold", "4000 < 5000 → no trigger")


def test_cooldown_period():
    """Cooldown prevents re-trigger within 2 seconds."""
    det = FallbackDetectionTester()
    for i in range(30):
        det.detect(200)
    # First trigger
    for _ in range(3):
        det.detect(50000)
    # Try again immediately
    for _ in range(3):
        if det.detect(50000):
            test_fail("Cooldown period", "Triggered within cooldown")
            return
    test_pass("Cooldown period", "2s cooldown prevents re-trigger")


def test_pause_resume_reset():
    """Pause/resume should reset detection state."""
    det = FallbackDetectionTester()
    for i in range(30):
        det.detect(300)
    det.detect(50000)
    det.detect(50000)
    # Simulate resume
    det.reset()
    for i in range(30):
        if det.detect(300):
            test_fail("Pause/resume reset", "Triggered during post-resume warmup")
            return
    test_pass("Pause/resume reset", "State fully reset on resume")


def test_false_trigger_comparison():
    """Compare old vs new detection false trigger rates."""
    import random

    old_det = OldDetection()
    old_triggers = 0
    for i in range(200):
        random.seed(42 + i)
        energy = random.randint(200, 800)
        if i % 30 == 0:
            energy = random.randint(2500, 4000)
        if old_det.detect(energy):
            old_triggers += 1

    new_det = FallbackDetectionTester()
    new_triggers = 0
    for i in range(200):
        random.seed(42 + i)
        energy = random.randint(200, 800)
        if i % 30 == 0:
            energy = random.randint(2500, 4000)
        if new_det.detect(energy):
            new_triggers += 1

    if new_triggers <= old_triggers:
        test_pass("False trigger comparison",
                  f"OLD={old_triggers} → NEW={new_triggers}")
    else:
        test_fail("False trigger comparison",
                  f"NEW({new_triggers}) > OLD({old_triggers})")


def test_intent_handler():
    """Test intent handler works correctly."""
    try:
        from modules.intent_handler import IntentHandler, IntentType
        handler = IntentHandler()

        cases = [
            ("What time is it?", IntentType.LOCAL_TIME),
            ("What's the date today?", IntentType.LOCAL_DATE),
            ("Set a timer for 5 minutes", IntentType.LOCAL_TIMER),
            ("Help me", IntentType.LOCAL_HELP),
            ("Stop", IntentType.LOCAL_STOP),
            ("What is the capital of France?", IntentType.INTERNET_QUERY),
        ]

        all_ok = True
        failed_cases = []
        for text, expected in cases:
            actual, _ = handler.detect_intent(text)
            if actual != expected:
                all_ok = False
                failed_cases.append(f"'{text}' → {actual.value} (expected {expected.value})")

        if all_ok:
            test_pass("Intent handler", f"{len(cases)} test cases correct")
        else:
            test_fail("Intent handler", "; ".join(failed_cases))
    except Exception as e:
        test_fail("Intent handler", str(e))


def test_config_loading():
    """Test configuration loads correctly."""
    try:
        from config import config
        assert config.wake_word.phrase == "hey_buddy"
        assert config.whisper.model == "tiny.en"
        assert config.audio.sample_rate == 16000
        test_pass("Config loading", "All config values correct")
    except Exception as e:
        test_fail("Config loading", str(e))


def test_syntax_validation():
    """Validate all source files parse without syntax errors."""
    import ast
    files = [
        "main.py", "config.py", "test_components.py",
        "modules/wake_word.py", "modules/audio_utils.py",
        "modules/speech_recognition.py", "modules/tts.py",
        "modules/intent_handler.py", "modules/gemini_client.py",
    ]
    all_ok = True
    for f in files:
        try:
            with open(PROJECT_ROOT / f) as fh:
                ast.parse(fh.read())
        except SyntaxError as e:
            test_fail(f"Syntax: {f}", str(e))
            all_ok = False
    if all_ok:
        test_pass("Syntax validation", f"{len(files)} files OK")


def main():
    print("\n" + "=" * 60)
    print("   BUDDY Voice Assistant - Wake Word Unit Tests")
    print("=" * 60)

    test_warmup_period()
    test_ambient_noise_rejection()
    test_single_spike_rejection()
    test_two_consecutive_not_enough()
    test_three_consecutive_triggers()
    test_interrupted_sequence()
    test_below_threshold()
    test_cooldown_period()
    test_pause_resume_reset()
    test_false_trigger_comparison()
    test_intent_handler()
    test_config_loading()
    test_syntax_validation()

    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"\n  Total: {len(results)}  Passed: {passed}  Failed: {failed}")

    if failed == 0:
        print(f"\n  🎉 ALL {len(results)} TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed:")
        for name, ok in results.items():
            if not ok:
                print(f"    ❌ {name}")

    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
