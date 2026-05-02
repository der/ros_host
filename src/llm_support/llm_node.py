"""LLM node for Marvin speech project.

Ported from ROS2 to socket.io. Subscribes to text_stream room and responds
on llm_response room using pydantic_ai with Ollama.
"""

import argparse
import asyncio
import logging
import os
import re
import sys

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.ollama import OllamaProvider

from messages.base import BaseNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("llm_node")

# Matches the position just after sentence-ending punctuation followed by whitespace
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class LLMNode(BaseNode):
    def __init__(
        self,
        hub_url: str,
        text_topic: str = "text_stream",
        response_topic: str = "llm_response",
        model_name: str = "granite4:tiny-h",
    ):
        super().__init__(hub_url=hub_url, node_name="llm_node")
        self.text_topic = text_topic
        self.response_topic = response_topic
        self.model_name = model_name

        self.is_running = False
        self.stop = False

        # Register handlers
        self.handler(self.text_topic)(self.message_callback)
        self.handler("events")(self.event_callback)

    async def _setup(self):
        """Initialize the LLM agent."""
        ollama_model = OpenAIChatModel(
            model_name=self.model_name,
            provider=OllamaProvider(base_url="http://localhost:11434/v1"),
            profile=ModelProfile(
                supports_thinking=True,
                thinking_tags=("<|channel>thought", "<channel|>"),
            ),
        )

        self.agent = Agent(
            ollama_model,
            output_type=str,
            system_prompt=(
                "You are small droid called Marvin with speech, vision and movement capabilities."
                "Respond to questions VERY BRIEFLY in plain text that the droid can speak aloud."
                'If the user just says "Marvin" then respond with "Hi"'
            ),
            model_settings={"thinking": False},
        )

    async def event_callback(self, message: dict):
        msg = message.get("data", "").strip()
        if msg in ("interrupt", "stop"):
            if self.is_running:
                logger.info("Interrupt event received, stopping current LLM response")
                self.stop = True

    async def message_callback(self, message: dict):
        text = message.get("data", "").strip()
        logger.info(f"Processing message: {text}")

        if re.match(r"^stop[.!?]*$", text, re.IGNORECASE):
            if self.is_running:
                logger.info("Stop command received, stopping current LLM response")
                self.stop = True
            await self.publish("events", {"data": "stop"})
            return

        # Run agent in a new task to avoid blocking the message handler
        asyncio.create_task(self._stream_agent(text))

    async def _publish(self, text: str):
        text = text.strip()
        if text:
            logger.info(f"LLM response: {text}")
            await self.publish(self.response_topic, {"data": text})
            await self.publish("events", {"data": f"llm/{text}"})

    async def _stream_agent(self, text: str):
        self.is_running = True
        self.stop = False
        buffer = ""
        try:
            async with self.agent.run_stream(text) as response:
                async for chunk in response.stream_text(delta=True):
                    if self.stop:
                        logger.info("LLM response streaming stopped")
                        self.stop = False
                        return
                    buffer += chunk
                    # Publish any complete sentences found in the buffer
                    parts = _SENTENCE_SPLIT_RE.split(buffer)
                    # The last element may be an incomplete sentence — keep it in the buffer
                    for sentence in parts[:-1]:
                        await self._publish(sentence)
                    buffer = parts[-1]
            # Publish any remaining text after streaming completes
            await self._publish(buffer)
        finally:
            self.is_running = False

    async def run(self):
        await self._setup()

        # Connect to hub
        await self.sio.connect(self.hub_url)
        await self._connected.wait()

        # Subscribe to rooms
        await self.subscribe(self.text_topic)
        await self.subscribe("events")

        logger.info(
            f"LLM Node started, listening on: {self.text_topic}, "
            f"publishing to: {self.response_topic}, using model: {self.model_name}"
        )

        try:
            while self.sio.connected:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.sio.disconnect()


def main():
    parser = argparse.ArgumentParser(description="LLM node")
    parser.add_argument("--hub-url", default=None, help="Hub URL")
    parser.add_argument("--text-topic", default="/text_stream", help="Input text room")
    parser.add_argument("--response-topic", default="/llm_response", help="Output response room")
    parser.add_argument("--model-name", default="granite4:tiny-h", help="Ollama model name")
    args = parser.parse_args()

    hub_url = args.hub_url or os.environ.get("HUB_URL", "http://localhost:5000")
    node = LLMNode(
        hub_url=hub_url,
        text_topic=args.text_topic,
        response_topic=args.response_topic,
        model_name=args.model_name,
    )
    asyncio.run(node.run())


if __name__ == "__main__":
    main()
