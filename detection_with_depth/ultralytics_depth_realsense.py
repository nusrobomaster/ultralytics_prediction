from ultralytics import YOLO
import time
import pyrealsense2 as rs
import numpy as np
import cv2
from spatial_location_calculator import SpatialLocationCalculator

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

spatial_location_calculator = SpatialLocationCalculator(IMAGE_WIDTH, IMAGE_HEIGHT)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Start streaming
config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)

profile = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

conf_threshold = 0.5

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    num_frames_processed = 0
    last_time = start = time.time()

    elapsed_time_lst = []

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            num_frames_processed += 1

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run YOLOv8 inference
            results = model(color_image, verbose=False)

            # Get bounding box coordinates and depth
            for det in results[0].boxes:
                # print(det)
                confidence = det.conf
                if confidence > conf_threshold:
                    # det.xyxy has the format tensor([xmin, ymin, xmax, ymax])
                    xmin, ymin, xmax, ymax = map(int, det.xyxy[0].tolist())
                    
                    depth_value = spatial_location_calculator.calc_location((xmin, ymin, xmax, ymax), depth_image)[2]
                    distance = spatial_location_calculator.calc_distance((xmin, ymin, xmax, ymax), depth_image)
                    
                    # Find centroid coordinates
                    centroid_x = (xmin + xmax) // 2
                    centroid_y = (ymin + ymax) // 2

                    # # Get depth information
                    # depth_value = depth_frame.get_distance(centroid_x, centroid_y)

                    # Draw bounding box
                    object_str = "cls: {}".format(det.cls[0])
                    confidence_str = "conf: {:.2f}".format(confidence.tolist()[0])
                    depth_str = "distance: {:.2f} meters".format(depth_value)
                    cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(color_image, str(object_str), (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    cv2.putText(color_image, confidence_str, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    cv2.putText(color_image, depth_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    # Print coordinates and depth
                    print("Object {} detected at ({}, {}) with depth: {:.2f} meters".format(object_str, centroid_x, centroid_y, depth_value))
                    print("Euclidean distance away: {:.2f}".format(distance))

            elapsed_time_lst.append(time.time() - last_time)
            # print(elapsed_time_lst[-1])
            last_time = time.time()
            
            # Display the resulting frame
            cv2.imshow('Object Detection', color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        pipeline.stop()
        end = time.time()

        elapsed = end - start
        fps = num_frames_processed / elapsed

        print("FPS: {} / {:.2f} = {:.2f}".format(num_frames_processed, elapsed, fps))
        print("Average elapsed time: ", sum(elapsed_time_lst) / len(elapsed_time_lst))
    