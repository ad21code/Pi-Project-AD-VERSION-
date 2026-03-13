# BUDDY - Raspberry Pi Voice Assistant

A locally-running voice assistant similar to Amazon Alexa, optimized for **Raspberry Pi 4 Model B (4GB RAM)**. Uses Gemini API for internet knowledge queries.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BUDDY Voice Assistant Architecture                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  Microphone  │───►│  Wake Word   │───►│  Voice Activity Detection    │   │
│  │   (USB/I2S)  │    │  (Porcupine) │    │  (webrtcvad)                 │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘   │
│                                                    │                         │
│                                                    ▼                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Speech-to-Text (faster-whisper)                   │   │
│  │                     Model: tiny.en (39MB, optimized for Pi)           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                    │                         │
│                                                    ▼                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Intent Detection                              │   │
│  │            (Local commands vs. Gemini API queries)                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                           │                        │                         │
│              ┌────────────┴────────────┐           │                         │
│              ▼                         ▼           │                         │
│  ┌──────────────────┐    ┌──────────────────┐      │                         │
│  │  Local Commands  │    │    Gemini API    │◄─────┘                         │
│  │  (time, volume)  │    │  (internet info) │                                │
│  └──────────────────┘    └──────────────────┘                                │
│              │                         │                                     │
│              └────────────┬────────────┘                                     │
│                           ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Text-to-Speech (Piper TTS)                       │   │
│  │                 Voice: en_US-lessac-medium (fast, natural)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                    │                         │
│                                                    ▼                         │
│                                          ┌──────────────┐                    │
│                                          │   Speaker    │                    │
│                                          │  (3.5mm/USB) │                    │
│                                          └──────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Recommended Libraries & Rationale

### Speech-to-Text: **faster-whisper** (Recommended)

| Option | Size | Speed on Pi4 | RAM | Recommendation |
|--------|------|--------------|-----|----------------|
| Whisper Tiny (OpenAI) | 39MB | ~8-12s | ~1GB | ❌ Too slow |
| whisper.cpp | 39MB | ~3-5s | ~300MB | ✅ Good alternative |
| **faster-whisper** | 39MB | **~2-4s** | **~200MB** | ✅ **Best choice** |

**Why faster-whisper?**
- Uses CTranslate2 for 4x faster inference
- Lower memory footprint via INT8 quantization
- Streaming support for real-time processing
- Python-native API

### Text-to-Speech: **Piper TTS** (Recommended)

| Option | Quality | Speed | RAM | Recommendation |
|--------|---------|-------|-----|----------------|
| espeak-ng | Robotic | Instant | ~5MB | ❌ Poor quality |
| pyttsx3 | Variable | Fast | ~20MB | ❌ Platform issues |
| Coqui TTS | Excellent | ~5-10s | ~1GB | ❌ Too heavy |
| **Piper TTS** | **Natural** | **~0.5-1s** | **~100MB** | ✅ **Best choice** |

**Why Piper?**
- Designed for Raspberry Pi
- Real-time streaming synthesis
- Multiple natural voices
- Low CPU/RAM usage

### Wake Word: **Porcupine** or **openWakeWord**

| Option | Accuracy | CPU | Free | Recommendation |
|--------|----------|-----|------|----------------|
| Snowboy | Good | Low | ✅ | ❌ Discontinued |
| **Porcupine** | Excellent | Very Low | Limited free | ✅ Best accuracy |
| **openWakeWord** | Good | Low | ✅ | ✅ Open source |

---

## Performance Optimizations

### 1. Memory Management
```python
# Load models once at startup
whisper_model = None  # Lazy loading
piper_voice = None    # Singleton pattern

# Use generators for streaming
def stream_audio():
    while recording:
        yield audio_chunk  # Don't store in memory
```

### 2. CPU Optimization
```python
# Use threading for non-blocking I/O
import threading

# Limit Whisper to 2 CPU threads (leave room for other processes)
model = WhisperModel("tiny.en", compute_type="int8", cpu_threads=2)

# Use voice activity detection to reduce processing
import webrtcvad
vad = webrtcvad.Vad(2)  # Aggressiveness level 2
```

### 3. Async Processing
```python
import asyncio

async def process_query(text):
    # Non-blocking Gemini API call
    response = await gemini_client.generate_content_async(text)
    return response
```

### 4. Audio Streaming
```python
# Stream TTS output instead of generating full audio first
async def stream_speech(text):
    for audio_chunk in piper.synthesize_stream(text):
        play_audio_chunk(audio_chunk)
```

---

## Project Structure

```
Voice assistant/
├── .env                    # API keys (never commit!)
├── .env.example            # Template for .env
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── config.py               # Configuration management
├── main.py                 # Main orchestration script
├── modules/
│   ├── __init__.py
│   ├── wake_word.py        # Wake word detection
│   ├── speech_recognition.py # STT with faster-whisper
│   ├── intent_handler.py   # Intent detection & routing
│   ├── gemini_client.py    # Gemini API integration
│   ├── tts.py              # Text-to-speech with Piper
│   └── audio_utils.py      # Audio I/O utilities
├── models/                 # Downloaded model files
│   └── .gitkeep
├── sounds/                 # Notification sounds
│   ├── wake.wav
│   └── error.wav
└── logs/                   # Log files
    └── .gitkeep
```

---

## Installation on Raspberry Pi

### 1. System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install audio dependencies
sudo apt install -y \
    portaudio19-dev \
    python3-pyaudio \
    libsndfile1 \
    ffmpeg \
    espeak-ng

# Install Piper TTS
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar -xzf piper_arm64.tar.gz
sudo mv piper /usr/local/bin/

# Download Piper voice
mkdir -p ~/piper-voices
wget -O ~/piper-voices/en_US-lessac-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O ~/piper-voices/en_US-lessac-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### 2. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your API key
nano .env
```

### 4. Test Audio

```bash
# Test microphone
arecord -d 3 test.wav
aplay test.wav

# List audio devices
arecord -l
aplay -l
```

### 5. Run the Assistant

```bash
# Activate virtual environment
source venv/bin/activate

# Run assistant
python main.py
```

---

## Usage

1. Run the assistant on Raspberry Pi:
   ```bash
   python main.py
   ```

2. Say **"Hey Buddy"** to activate (banner will display when ready)

3. Wait for confirmation beep and "Listening for your command..." message

4. Ask your question:
   - "What time is it?" (local)
   - "Set a timer for 5 minutes" (local)
   - "What's the weather in London?" (Gemini)
   - "Explain quantum computing" (Gemini)

5. Listen to the response

6. **Control commands:**
   - Say **"stop"** during response to interrupt and listen again
   - Say **"help"** to hear available features
   - Press **Ctrl+C** to quit the assistant

**Interruption feature:** While the assistant is speaking, you can say "stop" to cancel the current response and return to wake word listening state.

---

## Troubleshooting

### Audio Issues
```bash
# Check ALSA configuration
cat /proc/asound/cards

# Set default audio device
sudo nano /etc/asound.conf
```

### Memory Issues
```bash
# Monitor memory usage
htop

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Model Loading Slow
```bash
# Pre-download models
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en')"
```

---

## Future Improvements

1. **Smart Home Integration**: Add Home Assistant / MQTT support
2. **Offline Fallback**: Use smaller local LLM when internet unavailable
3. **Multi-Language**: Support additional languages with multilingual Whisper
4. **Custom Wake Words**: Train custom wake word with openWakeWord
5. **Conversation Context**: Maintain conversation history for follow-up questions
6. **Voice Profiles**: Recognize different users by voice
7. **Skills System**: Plugin architecture for custom commands

---

## License

MIT License - Feel free to modify and distribute.
