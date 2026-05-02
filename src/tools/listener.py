"""CLI tool to listen on a room and print incoming messages."""

import argparse
import asyncio
import logging

from messages.base import BaseNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


class Listener(BaseNode):
    def __init__(self, hub_url: str, room: str):
        super().__init__(hub_url=hub_url, node_name=f"listener:{room}")
        self.room = room
        self.handler(room)(self.on_message)

    async def on_message(self, message: dict):
        text = message.get("data", message)
        print(f"[{self.room}] {text}")

    async def run(self):
        await self.sio.connect(self.hub_url)
        await self._connected.wait()
        await self.subscribe(self.room)
        try:
            while self.sio.connected:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.sio.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Listen on a room")
    parser.add_argument("--room", required=True, help="Room to listen to")
    parser.add_argument("--hub-url", default=None, help="Hub URL (default: http://localhost:5000)")
    args = parser.parse_args()

    hub_url = args.hub_url or "http://localhost:5000"
    listener = Listener(hub_url=hub_url, room=args.room)
    asyncio.run(listener.run())


if __name__ == "__main__":
    main()
