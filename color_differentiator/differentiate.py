from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Start streaming
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def is_blue_or_red(hsv_image, bbox):
    x1, y1, x2, y2 = bbox
    roi_extracted = hsv_image[y1:y2, x1:x2]
    roi_extracted_hue = roi_extracted[:,:,0] # only take the hue channel
    inRange = roi_extracted_hue > 0.5 # filter out the zero pixels
    mean_hue = np.mean(roi_extracted_hue[inRange])

    print("Mean hue: ", mean_hue)
    cv2.putText(roi_extracted, "average Hue: " + str(mean_hue),
                (2, roi_extracted.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
    cv2.imshow("average_hue", roi_extracted)
    
    if mean_hue > 50 and mean_hue < 150:
        return "Blue"
    elif mean_hue > 0 and mean_hue < 50:
        return "Red"
    return "Neither"

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    # model = YOLO("yolov8n_040624_imgsz_640_2.pt")

    while True:
        # get color frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # perform prediction using model
        color_image = np.asanyarray(color_frame.get_data())
        results = model(color_image, classes=67, verbose=False)[0]

        # Convert color image to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Check color of detected objects
        for result in results:
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                color = is_blue_or_red(hsv_image, (x1, y1, x2, y2))
                cv2.putText(color_image, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # visualisation
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
