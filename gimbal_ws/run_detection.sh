#!/bin/bash

# Terminal 1 commands
gnome-terminal -- bash -c "
source ./venv/local/bin/activate;
colcon build --packages-select gimbal;
source install/setup.bash;
ros2 run gimbal dummy_gimbal_orientation_publisher;
exec bash"

# Terminal 2 commands
gnome-terminal -- bash -c "
source ./venv/local/bin/activate;
colcon build --packages-select gimbal;
source install/setup.bash;
export PYTHONPATH=\$PYTHONPATH:/home/nicholas_tyy/GitHub_repos/Robomasters/ultralytics_prediction/gimbal_ws/src/gimbal/gimbal;
ros2 run gimbal main;
exec bash"

