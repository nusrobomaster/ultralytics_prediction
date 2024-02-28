from ultralytics import YOLO
import time
import threading
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseStream:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.pipeline.start(config)
        
        self.frame = None
        self.stopped = False
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
    
    def start(self):
        self.stopped = False
        self.thread.start()
    
    def update(self):
        try:
            while True:
                if self.stopped:
                    break
                
                # Wait for a coherent pair of frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert images to numpy arrays
                self.frame = np.asanyarray(color_frame.get_data())
                
        finally:
            self.pipeline.stop()
    
    def read(self):
        # Return the most recent frame
        return self.frame
    
    def stop(self):
        # Stop the thread
        self.stopped = True
        # Wait for the thread to ensure it has exited
        if self.thread.is_alive():
            self.thread.join()


if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    
    num_frames_processed = 0
    last_time = start = time.time()

    elapsed_time_lst = []

    realsense_stream = RealSenseStream()
    realsense_stream.start()

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            color_image = realsense_stream.read()
            
            if color_image is None:
                print(color_image)
                pass
            
            num_frames_processed += 1

            # run inference on image
            color_image = model.track(color_image, imgsz=640, verbose=False)[0].plot()
            
            elapsed_time_lst.append(time.time() - last_time)
            # print(elapsed_time_lst[-1])
            last_time = time.time()
            
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        realsense_stream.stop()
        end = time.time()

        elapsed = end - start
        fps = num_frames_processed / elapsed

        print("FPS: {} / {:.2f} = {:.2f}".format(num_frames_processed, elapsed, fps))
        print("Average elapsed time: ", sum(elapsed_time_lst) / len(elapsed_time_lst))
    