"""Audio player node for Marvin speech project.

Ported from ROS2 to socket.io. Subscribes to audio room and plays through
default speaker using PyAudio.
"""

import argparse
import asyncio
import logging
import os
from queue import Empty, Queue

import numpy as np
import pyaudio

from messages.base import BaseNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("player_node")


class AudioPlayerNode(BaseNode):
    def __init__(
        self,
        hub_url: str,
        topic: str = "audio_stream",
        device_index: int = 0,
        buffer_size: int = 64,
    ):
        super().__init__(hub_url=hub_url, node_name="player_node")
        self.topic = topic
        self.device_index = device_index
        self.buffer_size = buffer_size

        # Audio state
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 512
        self.format: str = ""
        self.config_received = False

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False

        # Buffering
        self.audio_queue: Queue = Queue(maxsize=self.buffer_size)
        self.chunks_received = 0
        self.chunks_played = 0
        self.buffer_underruns = 0

        self.handler(self.topic)(self.audio_chunk_callback)

    async def audio_chunk_callback(self, message: dict):
        """Handle incoming audio chunk messages."""
        self.chunks_received += 1

        info = message.get("info", {})
        data = message.get("data", {})
        event = message.get("event", "")

        fmt = info.get("format", "")
        if fmt != self.format:
            logger.info(
                f"Audio format set to: {fmt}, "
                f"{info.get('sample_rate')}Hz, {info.get('num_channels')}ch, "
                f"{info.get('chunk_size')} samples/chunk"
            )
            self.format = fmt
            self.sample_rate = info.get("sample_rate", self.sample_rate)
            self.channels = info.get("num_channels", self.channels)
            self.chunk_size = info.get("chunk_size", self.chunk_size)

            # Reinitialize audio stream with new format
            if self.stream is not None:
                self._cleanup_stream()
            self._init_audio_stream()
            self.config_received = True

        # Convert to numpy
        int16_list = data.get("int16_data", [])
        if not int16_list:
            return
        audio_data = np.array(int16_list, dtype=np.int16)

        # Test for events
        if event:
            logger.info(f"Received audio event: {event}")

        # Try to add to queue
        try:
            self.audio_queue.put_nowait(audio_data.tobytes())
        except Exception:
            # Queue is full, drop the oldest chunk
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data.tobytes())
                self.buffer_underruns += 1
            except Empty:
                pass

    def _init_audio_stream(self):
        """Initialize the audio output stream."""
        try:
            device_index = self.device_index if self.device_index >= 0 else None

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_device_index=device_index,
                stream_callback=self._audio_callback,
            )

            self.stream.start_stream()
            self.is_playing = True
            logger.info("Audio output stream started successfully")

        except Exception as e:
            logger.error(f"Failed to initialize audio stream: {e}")
            self.is_playing = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function for playing audio chunks."""
        if status:
            logger.warning(f"Audio stream status: {status}")

        try:
            audio_data = self.audio_queue.get_nowait()
            self.chunks_played += 1
            return (audio_data, pyaudio.paContinue)
        except Empty:
            # No audio data available, return silence
            silence = b"\x00" * (frame_count * self.channels * 2)
            return (silence, pyaudio.paContinue)

    def _cleanup_stream(self):
        """Clean up the audio stream."""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            self.is_playing = False

    def _log_status(self):
        """Log periodic status information."""
        queue_size = self.audio_queue.qsize()
        status = "playing" if self.is_playing else "stopped"
        logger.info(
            f"Audio player status: {status}, "
            f"queue: {queue_size}/{self.buffer_size}, "
            f"received: {self.chunks_received}, "
            f"played: {self.chunks_played}, "
            f"underruns: {self.buffer_underruns}"
        )

    async def run(self):
        # Connect to hub
        await self.sio.connect(self.hub_url)
        await self._connected.wait()

        # Subscribe
        await self.subscribe(self.topic)

        # Initialize audio stream
        self._init_audio_stream()
        self.config_received = True

        # Status logging timer
        async def status_loop():
            while True:
                await asyncio.sleep(5.0)
                self._log_status()

        status_task = asyncio.create_task(status_loop())

        logger.info(f"Audio player node initialized for topic {self.topic}")

        try:
            while self.sio.connected:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass
            self._cleanup_stream()
            self.audio.terminate()
            await self.sio.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Audio player node")
    parser.add_argument("--hub-url", default=None, help="Hub URL")
    parser.add_argument("--topic", default="/audio_stream", help="Audio room to listen to")
    parser.add_argument("--device-index", type=int, default=0, help="Audio output device index")
    parser.add_argument("--buffer-size", type=int, default=64, help="Number of chunks to buffer")
    args = parser.parse_args()

    hub_url = args.hub_url or os.environ.get("HUB_URL", "http://localhost:5000")
    node = AudioPlayerNode(
        hub_url=hub_url,
        topic=args.topic,
        device_index=args.device_index,
        buffer_size=args.buffer_size,
    )
    asyncio.run(node.run())


if __name__ == "__main__":
    main()
