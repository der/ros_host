"""Audio capture node for Marvin speech project.

Ported from ROS2 to socket.io. Captures audio from default mic and publishes
chunks to audio_stream room.
"""

import argparse
import asyncio
import logging
import os
import sys
import threading
from queue import Queue, Empty

import numpy as np
import pyaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from messages.base import BaseNode
from messages.audio import AudioInfo, AudioData, AudioMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("capture_node")


class AudioCaptureNode(BaseNode):
    def __init__(
        self,
        hub_url: str,
        topic: str = "audio_stream",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,
        device_index: int = -1,
    ):
        super().__init__(hub_url=hub_url, node_name="capture_node")
        self.topic = topic
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index

        # PyAudio
        self.audio = None
        self.stream = None
        self.is_streaming = False

        # Queue for passing audio from PyAudio callback to asyncio publisher
        self._audio_queue: Queue = Queue(maxsize=256)
        self._stop_event = threading.Event()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - puts audio data into queue for async publishing."""
        if status:
            logger.warning(f"Audio stream status: {status}")

        audio_data = np.frombuffer(in_data, dtype=np.int16)

        msg = AudioMessage(
            info=AudioInfo(
                num_channels=self.channels,
                sample_rate=self.sample_rate,
                chunk_size=len(audio_data),
                format=f"{self.sample_rate // 1000}kmono-{self.chunk_size}",
            ),
            data=AudioData(int16_data=audio_data.tolist()),
            event="",
        )

        try:
            self._audio_queue.put_nowait(msg.model_dump())
        except Exception:
            # Queue full, drop oldest
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(msg.model_dump())
            except Empty:
                pass

        return (None, pyaudio.paContinue)

    def _init_audio_stream(self):
        """Initialize the audio input stream."""
        try:
            device_index = self.device_index if self.device_index >= 0 else None

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index,
                stream_callback=self._audio_callback,
            )

            self.stream.start_stream()
            self.is_streaming = True
            logger.info("Audio stream started successfully")

        except Exception as e:
            logger.error(f"Failed to initialize audio stream: {e}")
            self.is_streaming = False

    def _cleanup(self):
        """Clean up resources."""
        self._stop_event.set()
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if self.audio is not None:
            self.audio.terminate()
        logger.info("Audio capture cleanup completed")

    async def _publisher_loop(self):
        """Publish audio chunks from the thread-safe queue to the hub."""
        while not self._stop_event.is_set():
            try:
                item = await asyncio.get_event_loop().run_in_executor(
                    None, self._audio_queue.get, True, 0.5
                )
                await self.publish(self.topic, item)
            except Empty:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Publisher error: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Publisher error: {e}")
                await asyncio.sleep(1)

    async def _status_loop(self):
        """Log periodic status information."""
        while not self._stop_event.is_set():
            await asyncio.sleep(5.0)
            status = "streaming" if self.is_streaming else "stopped"
            logger.info(f"Audio capture status: {status}")

    async def run(self):
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Connect to hub
        await self.sio.connect(self.hub_url)
        await self._connected.wait()

        # Initialize audio stream
        self._init_audio_stream()

        logger.info(
            f"Audio capture node sending to {self.topic}: "
            f"{self.sample_rate}Hz, {self.channels}ch, "
            f"{self.chunk_size} samples/chunk"
        )

        # Start publisher and status loops
        publisher_task = asyncio.create_task(self._publisher_loop())
        status_task = asyncio.create_task(self._status_loop())

        try:
            while self.sio.connected:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self._cleanup()
            publisher_task.cancel()
            status_task.cancel()
            try:
                await publisher_task
            except asyncio.CancelledError:
                pass
            try:
                await status_task
            except asyncio.CancelledError:
                pass
            await self.sio.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Audio capture node")
    parser.add_argument("--hub-url", default=None, help="Hub URL")
    parser.add_argument("--topic", default="/audio_stream", help="Room to publish to")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--chunk-size", type=int, default=512, help="Samples per chunk")
    parser.add_argument("--device-index", type=int, default=-1, help="Audio input device index")
    args = parser.parse_args()

    hub_url = args.hub_url or os.environ.get("HUB_URL", "http://localhost:5000")
    node = AudioCaptureNode(
        hub_url=hub_url,
        topic=args.topic,
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk_size=args.chunk_size,
        device_index=args.device_index,
    )
    asyncio.run(node.run())


if __name__ == "__main__":
    main()
