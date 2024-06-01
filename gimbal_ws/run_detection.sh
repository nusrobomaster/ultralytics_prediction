#!/bin/bash

# Terminal commands
gnome-terminal -- bash -c "
source ./venv/local/bin/activate;
colcon build --packages-select gimbal;
source install/setup.bash;
export PYTHONPATH=\$PYTHONPATH:/home/nicholas_tyy/GitHub_repos/Robomasters/ultralytics_prediction/gimbal_ws/src/gimbal/gimbal;
ros2 run gimbal main;
exec bash"

