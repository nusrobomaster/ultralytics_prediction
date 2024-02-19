from ultralytics import YOLO
import threading
import cv2
import time

class WebCamStream:
    def __init__(self, stream_id = 0):
        self.stream_id = stream_id
        self.vcap = cv2.VideoCapture(self.stream_id)
        if not self.vcap.isOpened():
            raise ValueError(f"Failed to open stream {self.stream_id}")
        
        self.ret, self.frame = self.vcap.read()
        if not self.ret:
            print(f"[Exiting] Failed to read from stream {self.stream_id}")
            exit()
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()

    def update(self):
        while True:
            if self.stopped:
                break
            self.ret, self.frame = self.vcap.read()
            if not self.ret:
                print(f"[Exiting] Failed to read from stream {self.stream_id}")
                self.stopped = True
                break
        self.vcap.release()

    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    webcam_stream = WebCamStream(stream_id=0)
    webcam_stream.start()

    num_frames_processed = 0
    last_time = start = time.time()

    elapsed_time_lst = []

    while True:
        num_frames_processed += 1

        if webcam_stream.stopped:
            break
        frame = webcam_stream.read()

        frame = model.track(frame, imgsz=640, verbose=False)[0].plot()

        elapsed_time_lst.append(time.time() - last_time)
        print(elapsed_time_lst[-1])
        last_time = time.time()

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()
    webcam_stream.stop()

    elapsed = end - start
    fps = num_frames_processed / elapsed

    print("FPS: {} / {:.2f} = {:.2f}".format(num_frames_processed, elapsed, fps))

    print("Average elapsed time: ", sum(elapsed_time_lst) / len(elapsed_time_lst))