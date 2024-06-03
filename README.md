# Object tracking using Yolov8n and Supervision

## Run the code using ROS2:

```
# Terminal commands
gnome-terminal -- bash -c "
source ./venv/local/bin/activate;
colcon build --packages-select gimbal;
source install/setup.bash;
export PYTHONPATH=\$PYTHONPATH:/<full path to repo>/ultralytics_prediction/gimbal_ws/src/gimbal/gimbal;
ros2 run gimbal main;
exec bash"
```

Or simply edit run_detection.sh with the correct paths and run it.

## Run the code using python3: (for testing without ROS2)
```
cd armour_plate_tracker
python3 main.py
```

## Notes
#### Current state:
- YOLOv8n is used for object detection, supervision library for object tracking
- There can be multiple bbox detections for different objects, but there will only be 1 target, which is the one with shortest euclidean distance to camera.
- Logic is in `run()` in class Main in main.py

#### Rostopics
- /front_camera/rgb_footage
    - rgb image from realsense camera
- /front_camera/depth_map
    - depth map from realsense camera
- /front_camera/detections_output
    - output of YOLOv8n in Detection2DArray format
- /front_camera/target_information
    - contains [euclidean_dist, pitch, yaw]
- *same topics for back camera once the nodes have been added in main.py*
- **/decision**
    - contains [pitch_to_turn, yaw_to_turn] to be sent to gimbal