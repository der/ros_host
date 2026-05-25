# IMPORTANT! General principles

1. Don't assume. Don't hide confusion. Surface tradeoffs.
2. Minimum code that solves the problem. Nothing speculative.
3. Touch only what you must. Clean up only your own mess.
4. Define success criteria. Loop until verified.
5. If adding new libraries ensure they are not GPL.

# Project specific

- Use pyproject.toml with src directory style of layout.
- Python >= 3.11 required.
- No CI, Docker, Makefile, or pre-commit. Lint/typecheck/test are manual.

## Dev commands

```bash
uv pip install ".[dev]"
uv run ruff check src/       # E, F, I, UP, B, SIM; ignore E501
uv run pyright src/          # standard strictness
uv run pytest                # --asyncio-mode=auto is automatic
```

Run lint + typecheck + tests before committing.

## Architecture

Socket.io hub-broker: nodes connect as clients to a central hub. Rooms act like ROS2 topics — join to subscribe, emit to publish.

```
audio_stream → ASR → text_stream → LLM → llm_response → TTS → speech_stream → player
```

- **Hub** (`src/hub/server.py`): port 5000. `skip_sid` is commented out — publishers receive their own messages.
- **Nodes** (in `src/audio_base/`, `src/llm_support/`): each is a standalone asyncio script, run directly with `python src/<pkg>/<node>.py`. No setuptools entry points.
- **Messages** (`src/messages/`): Pydantic models. `BaseNode` wraps `socketio.AsyncClient` with `handler(room)` decorator and `publish(room, msg)`.

## Running

```bash
./start                   # hub + asr + tts + llm (background logs/) + listener foreground
./start_hub.sh            # hub only
./sender --topic /marvin/neck --message '{"pan": 50, "tilt": 50, "speed": 1000}'
./listener --topic /events
```

Hub must be running before any node.

## Testing

- `tests/conftest.py` starts a test hub on port 5001 in a daemon thread. Uses session-scoped fixture.
- `pytest` discovers and runs everything. No special flags needed.
- Integration test (`test_text_send_receive`) exercises real socket.io clients against the test hub.

## Threading model

Nodes mix asyncio with threads for I/O and blocking work:
- Audio capture/playback callbacks run in separate threads, feed `threading.Queue`.
- ASR/LLM inference uses `asyncio.to_thread()` or `run_in_executor()`.
- Results drain to an async `_publisher_loop` that emits via socket.io.


## RPC support

```
# Any node serving RPC — register on the raw sio client
self.sio.on('rpc_request', self._handle_rpc)
async def _handle_rpc(self, data):
    result = process(data['message'])
    return result  # ACK

# Any caller node
response = await node.call('/rpc/some_service', {'key': 'value'})
```

## GPU ASR (optional)

```bash
GGML_VULKAN=1 pip install git+https://github.com/absadiki/pywhispercpp
```