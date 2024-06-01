#!/bin/bash

# Terminal 0 commands - dummy yolov8 publisher node
gnome-terminal -- bash -c "
source ./venv/local/bin/activate;
colcon build --packages-select gimbal;
source install/setup.bash;
export PYTHONPATH=\$PYTHONPATH:/home/nicholas_tyy/GitHub_repos/Robomasters/ultralytics_prediction/gimbal_ws/src/gimbal/gimbal;
ros2 run gimbal yolov8_publisher;
exec bash"
