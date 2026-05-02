"""TTS node for Marvin speech project.

Ported from ROS2 to socket.io. Subscribes to text_stream room and publishes
synthesised audio chunks to speech_stream using pocket-tts in streaming mode.
"""

import argparse
import asyncio
import logging
import os
import sys
import threading
from queue import Queue, Empty

import numpy as np
from scipy.signal import resample_poly
from pocket_tts import TTSModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from messages.base import BaseNode
from messages.audio import AudioInfo, AudioData, AudioMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("tts_node")

# Sentinel value to signal end of synthesis
_SENTINEL = object()


class TTSNode(BaseNode):
    def __init__(
        self,
        hub_url: str,
        input_topic: str = "text_stream",
        output_topic: str = "speech_stream",
        voice: str = "alba",
    ):
        super().__init__(hub_url=hub_url, node_name="tts_node")
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.voice_path = voice

        # TTS model settings
        self.tts_model = None
        self.tts_sample_rate = 24000
        self.output_sample_rate = 16000
        self.chunk_size = 1280  # 80ms at 16kHz
        self.voice_state = None

        # Text input queue (asyncio)
        self._text_queue: asyncio.Queue = asyncio.Queue()
        self.stop = False
        self.is_running = False

        # Audio output queue (thread-safe, for communication between TTS thread and event loop)
        self._audio_queue: Queue = Queue(maxsize=128)

        # Register handlers
        self.handler(self.input_topic)(self.text_callback)
        self.handler("events")(self.event_callback)

    async def text_callback(self, message: dict):
        """Queue incoming text for synthesis."""
        text = message.get("data", "")
        text = text.strip()
        if text:
            self.stop = False
            await self._text_queue.put(text)
            logger.info(f"Queued text: {text!r}")

    async def event_callback(self, message: dict):
        event = message.get("data", "")
        event = event.strip()
        if event == "stop":
            logger.info("Stop event received")
            if self.is_running:
                logger.info("Stopping current TTS synthesis")
                self.stop = True
            # Clear text queue
            while not self._text_queue.empty():
                try:
                    self._text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def _synthesize_thread(self, text: str):
        """Run in a dedicated thread. Puts AudioMessage dicts into _audio_queue."""
        chunk_index = 0
        for audio_tensor in self.tts_model.generate_audio_stream(
            self.voice_state, text
        ):
            if self.stop:
                chunk_samples = self.chunk_size
                audio_buffer = np.zeros(chunk_samples, dtype=np.int16)
            else:
                # Resample 24kHz -> 16kHz (ratio 2/3)
                audio_f32 = audio_tensor.numpy().astype(np.float32)
                audio_16k = resample_poly(audio_f32, 2, 3).astype(np.float32)

                # Convert float32 [-1, 1] to int16
                audio_buffer = np.clip(audio_16k, -1.0, 1.0)
                audio_buffer = (audio_buffer * 32767).astype(np.int16)
                chunk_samples = len(audio_buffer)

            msg = AudioMessage(
                info=AudioInfo(
                    num_channels=1,
                    sample_rate=self.output_sample_rate,
                    chunk_size=chunk_samples,
                    format=f"16kmono-{chunk_samples}",
                ),
                data=AudioData(int16_data=audio_buffer.tolist()),
                event="break" if self.stop else "",
            )

            self._audio_queue.put(msg.model_dump())
            chunk_index += 1

            if self.stop:
                logger.info("TTS synthesis stopped mid-stream")
                break

        # Signal end of synthesis
        self._audio_queue.put(_SENTINEL)
        logger.info(f"Synthesis complete: {chunk_index} chunks generated")

    async def _publisher_loop(self):
        """Publish audio chunks from the thread-safe queue to the hub."""
        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(
                    None, self._audio_queue.get, True, 0.1
                )
            except Empty:
                continue

            if item is _SENTINEL:
                self.is_running = False
                return

            await self.publish(self.output_topic, item)

    async def _synthesis_worker(self):
        """Main worker: pulls text from queue, spawns synthesis thread, publishes results."""
        while True:
            try:
                text = await asyncio.wait_for(self._text_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            logger.info(f"Synthesising: {text!r}")
            self.is_running = True
            self._audio_queue = Queue(maxsize=128)

            # Start synthesis in a thread
            thread = threading.Thread(
                target=self._synthesize_thread, args=(text,), daemon=True
            )
            thread.start()

            # Publish chunks as they arrive
            await self._publisher_loop()

            # Wait for thread to finish
            thread.join(timeout=5.0)

    async def run(self):
        # Load model
        logger.info("Loading pocket_tts model...")
        self.tts_model = TTSModel.load_model()
        self.tts_sample_rate = self.tts_model.sample_rate
        logger.info(f"pocket_tts model loaded (sample_rate={self.tts_sample_rate})")

        # Load voice
        logger.info(f"Loading voice: {self.voice_path}")
        self.voice_state = self.tts_model.get_state_for_audio_prompt(self.voice_path)
        logger.info("Voice loaded")

        # Connect to hub
        await self.sio.connect(self.hub_url)
        await self._connected.wait()

        # Subscribe to rooms
        await self.subscribe(self.input_topic)
        await self.subscribe("events")
        logger.info(f"TTS node ready: {self.input_topic} -> {self.output_topic}")

        # Start synthesis worker
        worker_task = asyncio.create_task(self._synthesis_worker())

        try:
            while self.sio.connected:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            await self.sio.disconnect()


def main():
    parser = argparse.ArgumentParser(description="TTS node")
    parser.add_argument("--hub-url", default=None, help="Hub URL")
    parser.add_argument("--input-topic", default="/text_stream", help="Input text room")
    parser.add_argument("--output-topic", default="/speech_stream", help="Output audio room")
    parser.add_argument("--voice", default="alba", help="Voice to use")
    args = parser.parse_args()

    hub_url = args.hub_url or os.environ.get("HUB_URL", "http://localhost:5000")
    node = TTSNode(
        hub_url=hub_url,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        voice=args.voice,
    )
    asyncio.run(node.run())


if __name__ == "__main__":
    main()
