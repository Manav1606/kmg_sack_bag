import cv2
import time
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, Button, Label, Canvas, Frame
from PIL import Image, ImageTk
import threading


# Load the YOLO model with ByteTrack enabled
model = YOLO("/Users/mac/transline/on site improvised/on site only plus 12k images/best.pt")

class VideoCaptureBuffer:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.buffer_frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.is_rtsp = isinstance(video_source, str) and video_source.startswith("rtsp")

        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.buffer_frame = frame
                time.sleep(0.01)
            else:
                if not self.is_rtsp:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_source)
                    time.sleep(1)

    def read(self):
        with self.lock:
            frame = self.buffer_frame
        return frame is not None, frame

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

class SackbagDetectorApp:
    def __init__(self, video_path, conf_threshold, iou_threshold, image_size):
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.cap = VideoCaptureBuffer(video_path)

        self.counter_left_to_right = 0
        self.counter_right_to_left = 0
        self.tracked_positions = {}
        self.counted_ids = set()
        self.direction_state = {}
        self.last_seen = {}
        self.current_id = 1
        self.is_running = False
        self.frame_count = 0

        # Set your angled line coordinates here
        self.line_x1 = 391
        self.line_y1 = 2
        self.line_x2 = 591
        self.line_y2 = 593
        self.min_movement_threshold = 1         
        self.distance_threshold = 200                     
        self.max_inactive_frames = 3
        self.frame_skip_interval = 1

        self.init_gui()
# cross profuct of AB and AP
    '''def get_side(self, x1, y1, x2, y2, px, py):
       return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)'''

    def init_gui(self):
        self.window = Tk()
        self.window.title("Sackbag Detection Control Panel")

        control_frame = Frame(self.window)
        control_frame.pack(side="bottom", fill="x", pady=10)

        self.start_button = Button(control_frame, text="Start", command=self.start_detection,
                                   font=("Helvetica", 15, "bold"), bg= "#00FF08", fg="green",
                                   relief="raised", padx=20, pady=10)
        self.start_button.pack(side="left", padx=10)

        self.stop_button = Button(control_frame, text="Stop", command=self.stop_detection,
                                  font=("Helvetica", 15, "bold"), bg="#FF1100", fg="red",
                                  relief="raised", padx=20, pady=10)
        self.stop_button.pack(side="right", padx=10)

        self.counter_label = Label(control_frame, text="IN: 0   OUT: 0",
                                font=("Helvetica", 20, "bold"), fg="white",
                                padx=10, pady=5)
        self.counter_label.pack()

        self.canvas = Canvas(self.window, width=1200, height=640, bg="green")
        self.canvas.pack(expand=True, anchor="center")

        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.mainloop()

    def start_detection(self):
        self.is_running = True
        self.counter_left_to_right = 0
        self.counter_right_to_left = 0
        self.tracked_positions = {}
        self.counted_ids = set()
        self.direction_state = {}
        self.last_seen = {}
        self.current_id = 1
        self.frame_count = 0
        self.update_frame()

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

    def stop_detection(self):
        self.is_running = False
        self.counter_label.config(text=f"IN: {self.counter_left_to_right}   OUT: {self.counter_right_to_left}")
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def update_frame(self):
        if not self.is_running:
            return

        if self.frame_count % self.frame_skip_interval != 0:
            self.frame_count += 1
            self.window.after(33, self.update_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (1200, 640))
        self.frame_count += 1
        results = model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            tracker="bytetrack.yaml",
            verbose=False
        )

        for r in results:
            for box in r.boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                distances = [
                    np.linalg.norm(np.array((cx, cy)) - np.array((prev_cx, prev_cy)))
                    for prev_cx, prev_cy in self.tracked_positions.values()
                ]

                if distances and min(distances) < self.distance_threshold:
                    obj_id = next(
                        id_ for id_, (prev_cx, prev_cy) in self.tracked_positions.items()
                        if np.linalg.norm(np.array((cx, cy)) - np.array((prev_cx, prev_cy))) < self.distance_threshold
                    )
                else:
                    obj_id = self.current_id
                    self.current_id += 1

                if obj_id in self.counted_ids:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# comparing the current side with prev side to decide if the object crossed the line 
                '''if obj_id in self.tracked_positions:
                    prev_cx, prev_cy = self.tracked_positions[obj_id]
                    self.last_seen[obj_id] = self.frame_count

                    prev_side = self.get_side(self.line_x1, self.line_y1, self.line_x2, self.line_y2, prev_cx, prev_cy)
                    curr_side = self.get_side(self.line_x1, self.line_y1, self.line_x2, self.line_y2, cx, cy)

                    if prev_side < 0 and curr_side >= 0:
                        self.counter_left_to_right += 1
                        self.counted_ids.add(obj_id)
                    elif prev_side > 0 and curr_side <= 0:
                        self.counter_right_to_left += 1
                        self.counted_ids.add(obj_id)
                else:
                    self.last_seen[obj_id] = self.frame_count

                self.tracked_positions[obj_id] = (cx, cy)'''

        inactive_ids = [id_ for id_, last_frame in self.last_seen.items() if self.frame_count - last_frame > self.max_inactive_frames]
        for id_ in inactive_ids:
            self.tracked_positions.pop(id_, None)
            self.last_seen.pop(id_, None)
            self.direction_state.pop(id_, None)

        self.counter_label.config(text=f"IN: {self.counter_left_to_right}   OUT: {self.counter_right_to_left}")
        cv2.line(frame, (self.line_x1, self.line_y1), (self.line_x2, self.line_y2), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk

        if self.is_running:
            self.window.after(33, self.update_frame)

    def close(self):
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()

if __name__ == "__main__":
    video_path = "/Users/mac/transline/on site improvised/tests/vid4.mp4"
    conf_threshold = 0.2
    iou_threshold = 0.3
    image_size = 640
    app = SackbagDetectorApp(video_path, conf_threshold, iou_threshold, image_size)