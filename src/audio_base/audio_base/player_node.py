"""
Audio player node for Marvin speech project.
Subscribes to AudioStamped messages and plays them using PyAudio.
"""

import rclpy
from rclpy.node import Node
import pyaudio
import numpy as np
from threading import Lock
from queue import Queue, Empty

from audio_msg.msg import AudioStamped, Audio, AudioInfo, AudioData
from std_msgs.msg import Header


class AudioPlayerNode(Node):
    """ROS2 node for audio playback using PyAudio."""

    def __init__(self):
        super().__init__('audio_player')
        
        # Default audio parameters (will be updated from stream)
        self.sample_rate = 16000
        self.channels = 1
        self.bits_per_sample = 16
        self.chunk_size = 512
        self.format: str = ""
        
        # Declare parameters
        self.declare_parameter('device_index', -1)  # -1 for default device
        self.declare_parameter('buffer_size', 64)   # Number of chunks to buffer
        self.declare_parameter('topic', 'audio_stream')
        
        # Get parameters
        self.device_index = self.get_parameter('device_index').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.topic = self.get_parameter('topic').value
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Audio stream and buffering
        self.stream = None
        self.audio_queue = Queue(maxsize=self.buffer_size)
        self.is_playing = False
        self.stream_lock = Lock()
        self.config_received = False
        
        # Subscribers
        self.chunk_subscriber = self.create_subscription(
            Audio,
            self.topic,
            self.audio_chunk_callback,
            10
        )
        
        # Statistics
        self.chunks_received = 0
        self.chunks_played = 0
        self.buffer_underruns = 0
        
        # Timer for status logging
        self.create_timer(5.0, self.log_status)
        
        self.get_logger().info(f'Audio player node initialized for topic {self.topic}')
        
        # Initialize or reinitialize audio stream
        with self.stream_lock:
            if self.stream is not None:
                self.cleanup_stream()
            
            self.init_audio_stream()
            self.config_received = True

    def audio_chunk_callback(self, msg):
        """Handle incoming audio chunk messages."""
        self.chunks_received += 1

        if msg.info.format != self.format:
            self.get_logger().info(
                f'Audio format set to: {msg.info.format}, '
                f'{msg.info.sample_rate}Hz, {msg.info.num_channels}ch, '
                f'{msg.info.chunk_size} samples/chunk'
            )
            self.format = msg.info.format
            self.sample_rate = msg.info.sample_rate
            self.channels = msg.info.num_channels
            self.chunk_size = msg.info.chunk_size
            
            # Reinitialize audio stream with new format
            with self.stream_lock:
                if self.stream is not None:
                    self.cleanup_stream()
                self.init_audio_stream()

        # Convert message data to numpy array
        audio_data = np.array(msg.data.int16_data, dtype=np.int16)

        # Test for events
        if msg.event != '':
            self.get_logger().info(f'Received audio event: {msg.event}')
        
        # Try to add to queue
        try:
            self.audio_queue.put_nowait(audio_data.tobytes())
        except:
            # Queue is full, drop the oldest chunk
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data.tobytes())
                self.buffer_underruns += 1
            except Empty:
                pass

    def init_audio_stream(self):
        """Initialize the audio output stream."""
        try:
            device_index = self.device_index if self.device_index >= 0 else None
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_device_index=device_index,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            self.is_playing = True
            
            self.get_logger().info('Audio output stream started successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize audio stream: {e}')
            self.is_playing = False

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function for playing audio chunks."""
        if status:
            self.get_logger().warn(f'Audio stream status: {status}')
        
        try:
            # Get audio data from queue
            audio_data = self.audio_queue.get_nowait()
            self.chunks_played += 1
            return (audio_data, pyaudio.paContinue)
        except Empty:
            # No audio data available, return silence
            silence = b'\x00' * (frame_count * self.channels * 2)  # 2 bytes per sample for int16
            return (silence, pyaudio.paContinue)

    def cleanup_stream(self):
        """Clean up the audio stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self.is_playing = False

    def log_status(self):
        """Log periodic status information."""
        queue_size = self.audio_queue.qsize()
        status = "playing" if self.is_playing else "stopped"
        
        self.get_logger().info(
            f'Audio player status: {status}, '
            f'queue: {queue_size}/{self.buffer_size}, '
            f'received: {self.chunks_received}, '
            f'played: {self.chunks_played}, '
            f'underruns: {self.buffer_underruns}'
        )

    def cleanup(self):
        """Clean up resources."""
        with self.stream_lock:
            try:
                self.cleanup_stream()
            except Exception as e:
                pass
        
        if self.audio is not None:
            self.audio.terminate()
        
        self.get_logger().info('Audio player cleanup completed')

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def main(args=None):
    """Main function to run the audio player node."""
    rclpy.init(args=args)
    
    node = AudioPlayerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down audio player node...')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()