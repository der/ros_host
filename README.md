# Marvin ROS2

ROS2 nodes to run a base station machine to provide control and "intelligence" for Marvin the droid.

Use via rosbridge so Marvin doesn't need to run ROS2 itself.

## Set up

```
. /opt/ros/jazzy/setup.bash
python -m venv ./venv  --system-site-packages
. ./venv/bin/activate
touch ./venv/COLCON_IGNORE
rosdep install -i --from-path src --rosdistro jazzy -y
colcon build --symlink-install
. install/local_setup.bash
sudo apt install ros-jazzy-rosbridge-server
```

For asr_node also install `pywhispercpp` with appropriate GPU support. E.g. 

```
GGML_CUDA=1 pip install git+https://github.com/absadiki/pywhispercpp
```

But for some reason this isn't actually about to use CUDA.

## Running

- `start_bridge.sh`
- `start_asr.sh`
- `start_llm.sh`

## Packages

### audio_msg

Audio message formats for sending from Marvin to base station. Similar to `audio_common` but had initial issues with that, and want to include an `event` field in messages to mark start/end of utterances.

Default to use is `Audio` format which has:
   - `info`  - `num_channels`, `sample_rate`, `chunk_size`
   - `data` - `float32_data`, `int16_data`
   - `event` - string for associated event, supports at least `start_utterance` and `end_utterance`
  
### audio_base

Provides the following nodes:

`capture_node` - capture audio stream from default mic (default mono, 16Khz, 16bit int, 512 sample chunks) and publishes to topic `audio_stream`. All those can be set by parameters, especially `topic` to set the topic.

`player_node` - consumes messages from topic (default `audio_stream` as above) to play through default speaker, takes audio settings from info header in the stream messages

`asr_node` - consumes messages from topic (default `audio_stream` as above) and at start of utterance event captures data to submit to ASR, at end of utterance event runs fasterwhisper ASR and sends the result to `text_stream`

### llm_support

`llm_node` subscribes to `text_stream` and @@
