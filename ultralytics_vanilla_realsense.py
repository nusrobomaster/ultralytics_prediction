from ultralytics import YOLO
import time
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Start streaming
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    num_frames_processed = 0
    last_time = start = time.time()

    elapsed_time_lst = []

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            num_frames_processed += 1

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            color_image = model.track(color_image, imgsz=640, verbose=False)[0].plot()
            
            elapsed_time_lst.append(time.time() - last_time)
            # print(elapsed_time_lst[-1])
            last_time = time.time()
            
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        pipeline.stop()
        end = time.time()

        elapsed = end - start
        fps = num_frames_processed / elapsed

        print("FPS: {} / {:.2f} = {:.2f}".format(num_frames_processed, elapsed, fps))
        print("Average elapsed time: ", sum(elapsed_time_lst) / len(elapsed_time_lst))
    