"""RPC trial: client that calls the square server and prints the result."""

import argparse
import asyncio
import logging

from messages.base import BaseNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("node")


class SquareClient(BaseNode):
    def __init__(self, hub_url: str):
        super().__init__(hub_url=hub_url, node_name="square_client")


def main():
    parser = argparse.ArgumentParser(description="RPC square client")
    parser.add_argument("--topic", default="/rpc/square", help="Room to send RPC to")
    parser.add_argument("--hub-url", default=None, help="Hub URL (default: http://localhost:5000)")
    parser.add_argument("number", type=float, help="Number to square")
    args = parser.parse_args()

    client = SquareClient(hub_url=args.hub_url or "http://localhost:5000")

    async def run():
        await client.sio.connect(client.hub_url)
        await client._connected.wait()
        response = await client.call(args.topic, {"number": args.number})
        print(f"{args.number} ** 2 = {response}")
        await client.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()