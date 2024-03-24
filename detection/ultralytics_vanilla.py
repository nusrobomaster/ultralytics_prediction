from ultralytics import YOLO
import cv2
import time


if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    num_frames_processed = 0
    last_time = start = time.time()

    elapsed_time_lst = []

    while True:
        num_frames_processed += 1

        ret, frame = cap.read()
        if not ret:
            print(f"[Exiting] Failed to read from stream {0}")
            break

        frame = model.track(frame, imgsz=640, verbose=False)[0].plot()

        elapsed_time_lst.append(time.time() - last_time)
        print(elapsed_time_lst[-1])
        last_time = time.time()

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()

    elapsed = end - start
    fps = num_frames_processed / elapsed

    print("FPS: {} / {:.2f} = {:.2f}".format(num_frames_processed, elapsed, fps))
    
    print("Average elapsed time: ", sum(elapsed_time_lst) / len(elapsed_time_lst))