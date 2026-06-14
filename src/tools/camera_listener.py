"""Camera listener: fetches images via RPC and displays them."""

import argparse
import asyncio
import io
import logging

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from messages.base import BaseNode
from messages.image import ImageMessage
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("node")

class CameraListener(BaseNode):
    def __init__(self, hub_url: str, topic: str):
        super().__init__(hub_url=hub_url, node_name=f"listener:{topic}")
        self.topic = topic
        self.handler(topic)(self.on_message)
        self.model = YOLO("yolov26m.pt")

    async def on_message(self, message: dict):
        try:
            msg = ImageMessage(**message)
        except Exception as e:
            logger.error("Error processing image message: %s, error: %s", message, e)
            msg = ImageMessage(error=str(e))
        if msg.error:
            logger.error("Camera server error: %s", msg.error)
        elif msg.data:
            print("Received image data, displaying...")
            image = np.asarray(Image.open(io.BytesIO(msg.data)))
            results = self.model.predict(image, conf=0.5)
            annotated_image = results[0].plot()
            plt.imshow(annotated_image)
            plt.show(block=False)
            plt.pause(0.001)

    async def run(self):
        await self.sio.connect(self.hub_url)
        await self._connected.wait()
        await self.subscribe(self.topic)
        try:
            while self.sio.connected:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.sio.disconnect()

def main() -> None:
    parser = argparse.ArgumentParser(description="Camera listener — fetches and displays camera frames via RPC")
    parser.add_argument("--topic", default="/marvin/camera", help="Image topic")
    parser.add_argument("--hub-url", default=None, help="Hub URL (default: http://localhost:5000)")
    args = parser.parse_args()

    listener = CameraListener(hub_url=args.hub_url or "http://localhost:5000", topic=args.topic)
    asyncio.run(listener.run())


if __name__ == "__main__":
    main()
