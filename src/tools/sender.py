"""CLI tool to send text messages to a room."""

import argparse
import asyncio
import logging
import sys

from messages.base import BaseNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("node")
logger = logging.getLogger("node")


class Sender(BaseNode):
    def __init__(self, hub_url: str, room: str):
        super().__init__(hub_url=hub_url, node_name=f"sender:{room}")
        self.room = room

    async def send(self, text: str):
        await self.publish(self.room, {"data": text})
        logger.info(f"Sent: {text}")


def main():
    parser = argparse.ArgumentParser(description="Send text to a room")
    parser.add_argument("--room", required=True, help="Room to send to")
    parser.add_argument("--hub-url", default=None, help="Hub URL (default: http://localhost:5000)")
    parser.add_argument("--message", default=None, help="Message to send (or use stdin)")
    args = parser.parse_args()

    hub_url = args.hub_url or "http://localhost:5000"
    sender = Sender(hub_url=hub_url, room=args.room)

    async def run():
        await sender.sio.connect(hub_url)
        await sender._connected.wait()

        if args.message:
            await sender.send(args.message)
        else:
            print("Interactive mode. Type messages (empty line to send, Ctrl+C to quit):")
            loop = asyncio.get_event_loop()
            while True:
                try:
                    text = await loop.run_in_executor(None, sys.stdin.readline)
                    text = text.strip()
                    if text:
                        await sender.send(text)
                except KeyboardInterrupt:
                    break

        await sender.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
