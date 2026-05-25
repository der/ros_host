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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("node")


async def run(topic: str, resolution: str, hub_url: str, interval: float) -> None:
    client = BaseNode(hub_url=hub_url, node_name="camera_listener")
    await client.sio.connect(client.hub_url)
    await client._connected.wait()

    try:
        while True:
            response = await client.call(topic, {"resolution": resolution})
            if response is None:
                logger.error("No response from camera server")
            else:
                try:
                    msg = ImageMessage(**response)
                    if msg.error:
                        logger.error("Camera server error: %s", msg.error)
                    elif msg.data:
                        print("Received image data, displaying...")
                        image = np.asarray(Image.open(io.BytesIO(msg.data)))
                        plt.imshow(image)
                        plt.show(block=False)
                        plt.pause(0.001)
                except Exception as e:
                    logger.error("Error processing image message: %s, error: %s", response, e)
            await asyncio.sleep(interval)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await client.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera listener — fetches and displays camera frames via RPC")
    parser.add_argument("--topic", default="/marvin/camera", help="RPC topic to call")
    parser.add_argument("--resolution", default="full", choices=["full", "lores"], help="Resolution to request")
    parser.add_argument("--hub-url", default=None, help="Hub URL (default: http://localhost:5000)")
    parser.add_argument("--interval", type=float, default=0.1, help="Seconds between frames (default: 0.1)")
    args = parser.parse_args()

    asyncio.run(run(
        topic=args.topic,
        resolution=args.resolution,
        hub_url=args.hub_url or "http://localhost:5000",
        interval=args.interval,
    ))


if __name__ == "__main__":
    main()
