"""RPC trial: server that squares a number received via RPC call."""

import argparse
import asyncio
import logging

from messages.base import BaseNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


class SquareServer(BaseNode):
    def __init__(self, hub_url: str, topic: str):
        super().__init__(hub_url=hub_url, node_name=f"square_server:{topic}")
        self.sio.on("rpc_request", self._handle_rpc)

    async def _handle_rpc(self, data):
        num = data.get("message", {}).get("number")
        if num is None:
            return {"error": "missing 'number' in message"}
        return {"square": num ** 2}


def main():
    parser = argparse.ArgumentParser(description="RPC square server")
    parser.add_argument("--topic", default="/rpc/square", help="Room to listen on")
    parser.add_argument("--hub-url", default=None, help="Hub URL (default: http://localhost:5000)")
    args = parser.parse_args()

    server = SquareServer(hub_url=args.hub_url or "http://localhost:5000", topic=args.topic)

    async def run():
        await server.sio.connect(server.hub_url)
        await server._connected.wait()
        await server.subscribe(args.topic)
        running = asyncio.Event()
        try:
            await running.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await server.sio.disconnect()

    asyncio.run(run())


if __name__ == "__main__":
    main()