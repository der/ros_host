# Marvin Base Station

Socket.io-based base station nodes for Marvin the droid. All nodes communicate via a central hub process using python-socketio.

## Architecture

- **Hub**: Standalone ASGI server (uvicorn + python-socketio) on port 5000
- **Rooms**: Equivalent of ROS2 topics - join to subscribe, emit to publish
- **Messages**: Pydantic models sent as dicts over socket.io, with binary attachments for audio data
- **Nodes**: Separate asyncio processes connecting to the hub

## Setup

Dropped uv in favour of plain venv due to problems with pywhispercpp.

```
. venv/bin/activate
```

Updates:
```
pip install .
```

For ASR node, install `pywhispercpp` with GPU support:

```bash
GGML_VULKAN=1 python -m pip install git+https://github.com/absadiki/pywhispercpp
```

## Running

Run all the nodes in the background with foreground listener on `/events`.

```bash
. start
```

## Manual commands.

Basic eye and neck controls:
```
./sender --topic /marvin/neck --message '{"pan": 50, "tilt": 50, "speed": 1000}'
./sender --topic /marvin/neck --message '{"pan": 0, "tilt": 0}'
```
and
```
./sender --topic /marvin/eyes --message '{"wide": True}'
./sender --topic /marvin/eyes --message '{"x": 50}'
```

Display camera view:

```
./camera_show
```

## Rooms & Data Flow

```
audio_stream → ASR node → text_stream → LLM node → llm_response → TTS node → speech_stream → player
```

| Room | Direction | Content |
|------|-----------|---------|
| `audio_stream` | capture → ASR | Audio chunks (AudioMessage) |
| `text_stream` | ASR → LLM | Transcribed text |
| `llm_response` | LLM → TTS | LLM response sentences |
| `speech_stream` | TTS → player | Synthesized audio chunks |
| `events` | any → any | Debug/stop/interrupt signals |

## Nodes

### capture_node
Captures audio from default mic (16kHz, mono, 16-bit, 512-sample chunks) and publishes to `audio_stream`. Utterance start/end events come from an external source.

### player_node
Subscribes to an audio room and plays through default speaker using PyAudio. Handles dynamic format changes.

### asr_node
Subscribes to `audio_stream`, buffers audio between start/end utterance events, transcribes using pywhispercpp, publishes to `text_stream` and `events`.

### asr_gemma_node
Alternative ASR using Gemma multimodal model instead of whisper.

### tts_node
Subscribes to `text_stream`, synthesizes speech using pocket-tts, publishes audio chunks to `speech_stream`. Handles stop/interrupt events.

### llm_node
Subscribes to `text_stream`, streams responses from Ollama via pydantic_ai, publishes sentence-by-sentence to `llm_response`. Handles stop/interrupt events.

## CLI Tools

```bash
# Listen on any room
listener --room text_stream

# Send text to any room
sender --room text_stream --message "hello"
```

## Configuration

All nodes accept `--hub-url` (env: `HUB_URL`, default: `http://localhost:5000`) plus node-specific options. Run with `--help` for details.

## Audio Message Format

| Field | Type | Description |
|-------|------|-------------|
| `info.num_channels` | int | Channel count (default: 1) |
| `info.sample_rate` | int | Sample rate (default: 16000) |
| `info.chunk_size` | int | Samples per chunk (default: 512) |
| `info.format` | str | Format identifier |
| `data.int16_data` | list[int] | Audio samples as int16 |
| `data.float32_data` | list[float] | Audio samples as float32 |
| `event` | str | `start_utterance`, `end_utterance`, `break`, or empty |
