. ./setup.sh
ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
ros2 run audio_base asr_node &
ros2 run llm_support llm_node --ros-args -p model_name:=gemma4:26b &
ros2 run audio_base tts_node --ros-args -p input_topic:=llm_response &
