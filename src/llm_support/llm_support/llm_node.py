import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import asyncio
import threading


class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')

        self.declare_parameter('text_topic', 'text_stream')
        self.declare_parameter('model_name', 'granite4:tiny-h')

        self.topic_name = self.get_parameter('text_topic').value
        self.model_name = self.get_parameter('model_name').value

        ollama_model = OpenAIChatModel(
            model_name=self.model_name,
            provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        )

        self.agent = Agent(
            ollama_model,
            output_type=str,
            system_prompt=(
                'You are small, helpful droid called Marvin with speech, vision and movement capabilities.'
                'Respond to questions very briefly in plain text that the droid can speak aloud.'
                'If the user just says "Marvin" then respond with "Hi"'
            ),
        )

        self.subscription = self.create_subscription(
            String,
            self.topic_name,
            self.message_callback,
            10,
        )
        self.get_logger().info(
            f'LLM Node started, listening on topic: {self.topic_name}, using model: {self.model_name}'
        )

    def message_callback(self, message: String):
        self.get_logger().info(f'Processing message: {message.data}')
        thread = threading.Thread(target=self._run_agent, args=(message.data,), daemon=True)
        thread.start()

    def _run_agent(self, text: str):
        asyncio.run(self._stream_agent(text))

    async def _stream_agent(self, text: str):
        async with self.agent.run_stream(text) as response:
            async for chunk in response.stream_text():
                self.get_logger().info(f'LLM stream: {chunk}')


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
