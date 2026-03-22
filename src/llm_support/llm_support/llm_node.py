import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import asyncio
import threading

# Matches the position just after sentence-ending punctuation followed by whitespace
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')

        self.declare_parameter('text_topic', 'text_stream')
        self.declare_parameter('response_topic', 'llm_response')
        self.declare_parameter('model_name', 'granite4:tiny-h')

        self.topic_name = self.get_parameter('text_topic').value
        self.response_topic = self.get_parameter('response_topic').value
        self.model_name = self.get_parameter('model_name').value

        ollama_model = OpenAIChatModel(
            model_name=self.model_name,
            provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        )

        self.agent = Agent(
            ollama_model,
            output_type=str,
            system_prompt=(
                'You are small droid called Marvin with speech, vision and movement capabilities.'
                'Respond to questions VERY BRIEFLY in plain text that the droid can speak aloud.'
                'If the user just says "Marvin" then respond with "Hi"'
            ),
        )

        self.is_running = False
        self.stop = False

        self.subscription = self.create_subscription(
            String,
            self.topic_name,
            self.message_callback,
            10,
        )
        self.publisher = self.create_publisher(String, self.response_topic, 10)
        self.get_logger().info(
            f'LLM Node started, listening on topic: {self.topic_name}, '
            f'publishing to: {self.response_topic}, using model: {self.model_name}'
        )

    def message_callback(self, message: String):
        self.get_logger().info(f'Processing message: {message.data}')
        text = message.data.strip()
        if text == '<start>':
            if self.is_running:
                self.get_logger().info('Stop command received, stopping current LLM response')
                self.stop = True
            self._publish('<start>')
            return
        if re.match(r'^stop[.!?]*$', text, re.IGNORECASE):
            if self.is_running:
                self.get_logger().info('Stop command received, stopping current LLM response')
                self.stop = True
            self._publish('stopping')
            return
        else:
            thread = threading.Thread(target=self._run_agent, args=(text,), daemon=True)
            thread.start()

    def _run_agent(self, text: str):
        self.is_running = True
        asyncio.run(self._stream_agent(text))
        self.is_running = False

    def _publish(self, text: str):
        text = text.strip()
        if text:
            self.get_logger().info(f'LLM response: {text}')
            msg = String()
            msg.data = text
            self.publisher.publish(msg)

    async def _stream_agent(self, text: str):
        buffer = ''
        async with self.agent.run_stream(text) as response:
            async for chunk in response.stream_text(delta=True):
                if self.stop:
                    self.get_logger().info('LLM response streaming stopped')
                    self.stop = False
                    return
                buffer += chunk
                # Publish any complete sentences found in the buffer
                parts = _SENTENCE_SPLIT_RE.split(buffer)
                # The last element may be an incomplete sentence — keep it in the buffer
                for sentence in parts[:-1]:
                    self._publish(sentence)
                buffer = parts[-1]
        # Publish any remaining text after streaming completes
        self._publish(buffer)


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
