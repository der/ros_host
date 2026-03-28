"""
TTS node for Marvin speech project.
Subscribes to text_stream (String) and publishes synthesised audio
chunks to speech_stream using pocket_tts in streaming mode.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.signal import resample_poly
from threading import Thread
from queue import Queue, Empty

from pocket_tts import TTSModel

from audio_msg.msg import Audio, AudioInfo, AudioData
from std_msgs.msg import String


class TTSNode(Node):
    """ROS2 node for text-to-speech using pocket_tts."""

    def __init__(self):
        super().__init__('tts_node')

        # Declare parameters
        self.declare_parameter('input_topic', 'text_stream')
        self.declare_parameter('output_topic', 'speech_stream')
        self.declare_parameter(
            'voice',
            'alba'  # 'hf://kyutai/tts-voices/alba-mackenna/casual.wav'
        )

        # Get parameters
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.voice_path = self.get_parameter('voice').value

        print(f'Debug output topics: input={self.input_topic}, output={self.output_topic}, voice={self.voice_path}')

        # Load model
        self.get_logger().info('Loading pocket_tts model...')
        self.tts_model = TTSModel.load_model()
        self.tts_sample_rate = self.tts_model.sample_rate  # 24000 Hz
        self.output_sample_rate = 16000
        self.chunk_size = 1280  # 80ms at 16kHz
        self.get_logger().info(
            f'pocket_tts model loaded (sample_rate={self.tts_sample_rate})'
        )

        # Load voice state
        self.get_logger().info(f'Loading voice: {self.voice_path}')
        self.voice_state = self.tts_model.get_state_for_audio_prompt(self.voice_path)
        self.get_logger().info('Voice loaded')

        # Publisher
        self.audio_publisher = self.create_publisher(Audio, self.output_topic, 10)

        # Subscriber
        self.text_subscriber = self.create_subscription(
            String,
            self.input_topic,
            self.text_callback,
            10
        )

        # Worker thread to keep synthesis off the subscriber callback thread
        self._queue: Queue = Queue()
        self.stop = False
        self.is_running = False
        self._worker = Thread(target=self._synthesis_worker, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f'TTS node ready: {self.input_topic} -> {self.output_topic}'
        )

    def text_callback(self, msg: String):
        """Queue incoming text for synthesis."""
        text = msg.data.strip()
        if text == '<start>':
            self.get_logger().info('Start command received, stopping current synthesis if running')
            if self.is_running:
                self.get_logger().info('Stopping current TTS synthesis')
                self.stop = True
            self._queue.empty()  # Clear any pending text
            return
        if text:
            self.stop = False
            self._queue.put(text)

    def _synthesis_worker(self):
        """Background thread that synthesises text and publishes audio chunks."""
        while rclpy.ok():
            try:
                text = self._queue.get(timeout=0.5)
            except Empty:
                continue

            self.get_logger().info(f'Synthesising: {text!r}')
            self.is_running = True

            chunk_index = 0
            for audio_tensor in self.tts_model.generate_audio_stream(
                self.voice_state, text
            ):
                if self.stop:
                    chunk_samples = self.chunk_size
                    audio_buffer = np.zeros(chunk_samples, dtype=np.int16)
                else:
                    # Resample 24kHz -> 16kHz (ratio 2/3)
                    audio_f32 = audio_tensor.numpy().astype(np.float32)
                    audio_16k = resample_poly(audio_f32, 2, 3).astype(np.float32)

                    # Convert float32 [-1, 1] to int16
                    audio_buffer = np.clip(audio_16k, -1.0, 1.0)
                    audio_buffer = (audio_buffer * 32767).astype(np.int16)
                    chunk_samples = len(audio_buffer)  # 1280

                msg = Audio()
                msg.info = AudioInfo(
                    num_channels=1,
                    sample_rate=self.output_sample_rate,
                    chunk_size=chunk_samples,
                    format=f'16kmono-{chunk_samples}',
                )
                msg.data = AudioData(int16_data=audio_buffer.tolist())
                msg.event = 'break' if self.stop else ''

                self.audio_publisher.publish(msg)
                chunk_index += 1

                if self.stop:
                    self.get_logger().info('TTS synthesis stopped')
                    break

            self.is_running = False
            self.get_logger().info(
                f'Synthesis complete: {chunk_index} chunks published'
            )

    def destroy_node(self):
        """Clean shutdown."""
        self._queue.put(None)  # unblock worker
        self._worker.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    """Main function to run the TTS node."""
    rclpy.init(args=args)

    node = TTSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down TTS node...')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
