from ultralytics import YOLO
import time
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Start streaming
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

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
            color_image = model.track(color_image, imgsz=640, verbose=False)[0].plot()
            
            elapsed_time_lst.append(time.time() - last_time)
            # print(elapsed_time_lst[-1])
            last_time = time.time()
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
        
            cv2.imshow('RealSense', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        pipeline.stop()
        end = time.time()

        elapsed = end - start
        fps = num_frames_processed / elapsed

        print("FPS: {} / {:.2f} = {:.2f}".format(num_frames_processed, elapsed, fps))
        print("Average elapsed time: ", sum(elapsed_time_lst) / len(elapsed_time_lst))
    