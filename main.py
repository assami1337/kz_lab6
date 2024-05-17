import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Scale, ttk
from PIL import Image, ImageTk

# Параметры для детектирования хороших признаков
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Параметры для вычисления оптического потока методом Лукаса-Канаде
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player with Optical Flow Algorithms")

        # Frame for video display
        self.video_frame = ttk.Frame(root)
        self.video_frame.pack(pady=10)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Frame for controls
        self.controls_frame = ttk.LabelFrame(root, text="Controls")
        self.controls_frame.pack(pady=10, fill="x")

        self.load_button = ttk.Button(self.controls_frame, text="Load Video", command=self.load_video)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.horn_schunck_button = ttk.Button(self.controls_frame, text="Horn-Schunck", command=self.horn_schunck)
        self.horn_schunck_button.grid(row=0, column=1, padx=5, pady=5)

        self.lucas_kanade_button = ttk.Button(self.controls_frame, text="Lucas-Kanade", command=self.lucas_kanade)
        self.lucas_kanade_button.grid(row=0, column=2, padx=5, pady=5)

        self.reset_button = ttk.Button(self.controls_frame, text="Reset", command=self.reset_algorithm)
        self.reset_button.grid(row=0, column=3, padx=5, pady=5)

        # Frame for parameters
        self.params_frame = ttk.LabelFrame(root, text="Parameters")
        self.params_frame.pack(pady=10, fill="x")

        self.lamda_scale = Scale(self.params_frame, from_=0.1, to=10.0, resolution=0.1, label="λ", orient=tk.HORIZONTAL)
        self.lamda_scale.pack(fill="x", padx=10, pady=5)

        self.threshold_scale = Scale(self.params_frame, from_=1, to=100, resolution=1, label="T", orient=tk.HORIZONTAL)
        self.threshold_scale.pack(fill="x", padx=10, pady=5)

        self.k_scale = Scale(self.params_frame, from_=3, to=20, resolution=1, label="k", orient=tk.HORIZONTAL)
        self.k_scale.pack(fill="x", padx=10, pady=5)

        self.cap = None
        self.algorithm = None
        self.prev_gray = None
        self.max_width = 640
        self.max_height = 480

        self.update_video()

    def load_video(self):
        video_path = filedialog.askopenfilename()
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.prev_gray = None
            self.update_video()

    def update_video(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.algorithm:
                    frame = self.algorithm(frame)

                frame = self.resize_frame(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                self.video_label.config(image=frame)
                self.video_label.image = frame

            self.root.after(30, self.update_video)

    def resize_frame(self, frame):
        height, width, _ = frame.shape
        if width > self.max_width or height > self.max_height:
            scale_w = self.max_width / width
            scale_h = self.max_height / height
            scale = min(scale_w, scale_h)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        return frame

    def horn_schunck_algorithm(self, frame):
        alpha = self.lamda_scale.get()  # Use the alpha parameter instead of lambda to avoid confusion
        num_iter = self.threshold_scale.get()

        if self.prev_gray is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame

        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        u = np.zeros(next_gray.shape)
        v = np.zeros(next_gray.shape)

        avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                               [1 / 6, 0, 1 / 6],
                               [1 / 12, 1 / 6, 1 / 12]])

        Ix = cv2.Sobel(next_gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(next_gray, cv2.CV_64F, 0, 1, ksize=5)
        It = next_gray - self.prev_gray

        for _ in range(num_iter):
            u_avg = cv2.filter2D(u, -1, avg_kernel)
            v_avg = cv2.filter2D(v, -1, avg_kernel)
            u = u_avg - Ix * ((Ix * u_avg + Iy * v_avg + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2))
            v = v_avg - Iy * ((Ix * u_avg + Iy * v_avg + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2))

        magnitude, angle = cv2.cartToPolar(u, v)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        self.prev_gray = next_gray

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def lucas_kanade_algorithm(self, frame):
        k = self.k_scale.get()

        if self.prev_gray is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame

        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **feature_params)
        if p0 is None:
            return frame

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, next_gray, p0, None, winSize=(k, k), maxLevel=2,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        flow_image = frame.copy()

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)
            flow_image = cv2.line(flow_image, (a, b), (c, d), (0, 255, 0), 2)
            flow_image = cv2.circle(flow_image, (a, b), 5, (0, 0, 255), -1)

        self.prev_gray = next_gray
        return flow_image

    def horn_schunck(self):
        self.algorithm = self.horn_schunck_algorithm

    def lucas_kanade(self):
        self.algorithm = self.lucas_kanade_algorithm

    def reset_algorithm(self):
        self.algorithm = None
        self.prev_gray = None


if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()
