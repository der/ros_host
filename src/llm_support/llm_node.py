"""LLM node for Marvin speech project.

Ported from ROS2 to socket.io. Subscribes to text_stream room and responds
on llm_response room using pydantic_ai with Ollama.
"""

import argparse
import asyncio
import logging
import os
import re
from datetime import datetime

from pydantic_ai import Agent, SystemPromptPart, BinaryContent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.ollama import OllamaProvider

from messages.base import BaseNode, EventMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("llm_node")

# Matches the position just after sentence-ending punctuation followed by whitespace
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Tools for use by the agent, extract to separate file if this grows
def get_time() -> str:
    """Get the current date and time.
       If asked the time always call this fresh, don't rely on your previous answer, since time is always changing."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class LLMNode(BaseNode):
    def __init__(
        self,
        hub_url: str,
        text_topic: str = "text_stream",
        response_topic: str = "llm_response",
        model_name: str = "gemma4:26b",
    ):
        super().__init__(hub_url=hub_url, node_name="llm_node")
        self.text_topic = text_topic
        self.response_topic = response_topic
        self.model_name = model_name
        self.messages: list[ModelMessage] | None = None

        self.is_running = False
        self.stop = False

        # Register handlers
        self.handler(self.text_topic)(self.message_callback)
        self.handler("/events")(self.event_callback)

    async def _setup(self):
        """Initialize the LLM agent."""
        if self.model_name.startswith("gemma4"):
            ollama_model = OpenAIChatModel(
                model_name=self.model_name,
                provider=OllamaProvider(base_url="http://localhost:11434/v1"),
                profile=ModelProfile(
                    supports_thinking=True,
                    thinking_tags=("<|channel>thought", "<channel|>"),
                ),
            )
        else:
            ollama_model = OpenAIChatModel(
                model_name=self.model_name,
                provider=OllamaProvider(base_url="http://localhost:11434/v1"),
            )

        async def move_neck(pan: int, tilt: int) -> None:
            """Move your robot neck to the specified pan and tilt positions.
            These are values between -100 and 100 representing the percentage 
            of the full range of motion in each direction. 
            Positive tilts are up, negative are down."""
            logger.info(f"Moving neck to pan: {pan}, tilt: {tilt}")
            await self.publish("/marvin/neck", {"pan": pan, "tilt": tilt, "speed": 1000})
            return None
        
        async def move_robot(speed: int = 50, dir: str = 'f', dist: int | None = 50) -> None:
            """Move your robot at the specified speed and direction for an optional distance.
            Speed is a percentage from 0 to 100. Direction can be 'f' for forward, 'b' for backward, 
            'sl'/'sr' for slide left/right, 'rl'/'rr' for rotate left/right, 'tr'/'tl' for turn right/left while moving forward, or 's' for stop.
            Distance is how far to move in centimeters, for rotations use a small value like 20."""
            logger.info(f"Moving motor with speed: {speed}, direction: {dir}, distance: {dist}")
            await self.publish("/marvin/motor", {"speed": speed, "dir": dir, "dist": dist})
            return None
        
        async def get_view() -> BinaryContent | str:
            """Get a description of what you see through your camera."""
            logger.info("Getting view from camera")
            image_data = await self.call("/marvin/camera", {"resolution": "full"})
            if image_data is None:
                logger.error("No response from camera server")
                return "I couldn't get a view from the camera."
            elif "error" in image_data and image_data["error"]:
                logger.error(f"Camera server error: {image_data['error']}")
                return "I couldn't get a view from the camera."
            elif "data" in image_data and image_data["data"]:
                logger.info("Received image data from camera")
                format = image_data.get("format", "image/jpeg")
                return BinaryContent(data=image_data["data"], media_type=format)
            else:
                logger.error(f"Unexpected camera response: {image_data}")
                return "I couldn't get a view from the camera."

        
        self.agent = Agent(
            ollama_model,
            output_type=str,
            system_prompt=(
                "You are small droid called Marvin with speech, vision and movement capabilities."
                "Respond to questions VERY BRIEFLY in plain text that the droid can speak aloud."
                'If the user just says "Marvin" then respond with "Hi"'
            ),
            tools=[get_time, move_neck, move_robot, get_view],
            model_settings={"thinking": False},
            history_processors=[self.keep_recent_messages]
        )

    async def event_callback(self, message: dict):
        msg = EventMessage(**message)
        if msg.message in ("interrupt", "stop"):
            logger.info("Interrupt event received, stopping current LLM response")
            self.stop = True

    async def message_callback(self, message: str):
        text = message.strip()
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
            await self.publish_event("llm", text)

    async def _stream_agent(self, text: str):
        self.is_running = True
        self.stop = False
        buffer = ""
        try:
            async with self.agent.run_stream(text, message_history = self.messages) as response:
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
                if self.messages is None:
                    self.messages = response.all_messages()
                else:
                    self.messages += response.new_messages()
            # Publish any remaining text after streaming completes
            await self._publish(buffer)
        finally:
            self.is_running = False

    async def keep_recent_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """
        Keep only recent messages while preserving AI model message ordering rules.

        Most AI models require proper sequencing of:
        - Tool/function calls and their corresponding returns
        - User messages and model responses
        - Multi-turn conversations with proper context

        This means we cannot cut conversation history in a way that:
        - Leaves tool calls without their corresponding returns
        - Separates paired messages inappropriately
        - Breaks the logical flow of multi-turn interactions

        Reference: https://github.com/pydantic/pydantic-ai/issues/2050
        """
        message_window = 15

        if len(messages) <= message_window:
            return messages

        # Find system prompt if it exists
        system_prompt = None
        system_prompt_index = None
        for i, msg in enumerate(messages):
            if isinstance(msg, ModelRequest) and any(isinstance(part, SystemPromptPart) for part in msg.parts):
                system_prompt = msg
                system_prompt_index = i
                break

        # Start at target cut point and search backward (upstream) for a safe cut
        target_cut = len(messages) - message_window

        for cut_index in range(target_cut, -1, -1):
            first_message = messages[cut_index]

            # Skip if first message has tool returns (orphaned without calls)
            if any(isinstance(part, ToolReturnPart) for part in first_message.parts):
                continue

            # Skip if first message has tool calls (violates AI model ordering rules)
            if isinstance(first_message, ModelResponse) and any(
                isinstance(part, ToolCallPart) for part in first_message.parts
            ):
                continue

            # Found a safe cut point
            result = messages[cut_index:]

            # If we cut off the system prompt, prepend it back
            if system_prompt is not None and system_prompt_index is not None and cut_index > system_prompt_index:
                result = [system_prompt] + result

            return result

        # No safe cut point found, keep all messages
        return messages

    async def run(self):
        await self._setup()

        # Connect to hub
        await self.sio.connect(self.hub_url)
        await self._connected.wait()

        # Subscribe to rooms
        await self.subscribe(self.text_topic)
        await self.subscribe("/events")

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
    parser.add_argument("--model-name", default="gemma4:26b", help="Ollama model name")
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
