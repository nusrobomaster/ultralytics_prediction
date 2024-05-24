# Object tracking using Yolov8n and Supervision
## Run the code:
```
cd armour_plate_tracker
python3 main.py
```

## Notes
#### Current state:
- YOLOv8n is used for object detection, supervision library for object tracking
- There can be multiple bbox detections for different objects, but there will only be 1 target, which is the one with shortest euclidean distance to camera.
- **Gimbal adjustments are done relative to the position of the camera, not the map frame**. We will need to write a ros2 subscriber to track gimbal orientation. Then the calculated gimbal orientation is `new_gimbal_yaw = original_gimbal_yaw + yaw_adjustment + yaw_offset` and `new_gimbal_pitch = original_gimbal_pitch + pitch_adjustment + pitch_offset`, where `yaw_offset` and `pitch_offset` can be modified in `calculate_gimbal_offsets()` in `main.py`.
- Logic is in `run()` in class Main in main.py

## Others
### Multithreading for frame capturing and inference

Running on Intel Realsense D435I & performing inference using YOLOv8n:
- <code>ultralytics_multithreading_realsense.py</code> runs with ~68 FPS
- <code>ultralytics_vanilla_realsense.py</code> runs with ~29 FPS 