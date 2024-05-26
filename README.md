# Object tracking using Yolov8n and Supervision

## Run the code using ROS2:
```
// in 1 terminal
cd gimbal_ws
colcon build --packages-select gimbal
source install/setup.bash
ros2 run gimbal dummy_gimbal_orientation_publisher (replace with actual publisher node)

// in another terminal
// make sure you are in the gimbal_ws folder
source install/setup.bash
export PYTHONPATH=$PYTHONPATH:{full_path_to_repo}/ultralytics_prediction/gimbal_ws/src/gimbal/gimbal
ros2 run gimbal main
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
- Gimbal adjustments are now published to a rostopic.Modify the name of the rostopic to match the name of the actual published topic for gimbal orientation. If not it will be using a dummy set of values from the dummy publisher.
- Logic is in `run()` in class Main in main.py

## Others
### Multithreading for frame capturing and inference

Running on Intel Realsense D435I & performing inference using YOLOv8n:
- <code>ultralytics_multithreading_realsense.py</code> runs with ~68 FPS
- <code>ultralytics_vanilla_realsense.py</code> runs with ~29 FPS 