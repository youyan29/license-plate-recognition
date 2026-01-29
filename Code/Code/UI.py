import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk, RIGHT, TOP, LEFT, BOTH
from PlateRecognizer import PlateRecognizer
import cv2
from PIL import Image, ImageTk
import time
import numpy as np


class LicensePlateRecognitionUI(ttk.Frame):
    image_path = ""
    display_height = 600
    display_width = 600
    last_update_time = 0
    processing_thread = None
    is_thread_running = False
    camera_capture = None
    plate_color_mapping = {"green": ("green", "#55FF55"), "yellow": ("yellow", "#FFFF00"), "blue": ("blue", "#6666FF")}

    def __init__(self, root_window):
        ttk.Frame.__init__(self, root_window)
        # Main frames
        original_image_frame = ttk.Frame(self)
        processed_image_frame = ttk.Frame(self)
        control_panel_frame = ttk.Frame(self)

        root_window.title("License Plate Recognition")
        root_window.state("zoomed")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")

        # Pack main frames
        original_image_frame.pack(side=LEFT, expand=1, fill=BOTH)
        processed_image_frame.pack(side=LEFT, expand=1, fill=BOTH)
        control_panel_frame.pack(side=RIGHT, expand=0)

        # Original image section
        ttk.Label(original_image_frame, text='Original Image:').pack(anchor="nw")
        self.original_image_label = ttk.Label(original_image_frame)
        self.original_image_label.pack(anchor="nw")

        # Grayscale image section
        self.grayscale_header_label = ttk.Label(processed_image_frame, text='')
        self.grayscale_header_label.pack(anchor="nw")
        self.processed_image_label = ttk.Label(processed_image_frame)
        self.processed_image_label.pack(anchor="nw")

        # Control panel
        ttk.Label(control_panel_frame, text='Detected Plate:').pack(anchor="nw")
        self.plate_image_label = ttk.Label(control_panel_frame)
        self.plate_image_label.pack(anchor="nw")

        ttk.Label(control_panel_frame, text='Recognition Result:').pack(anchor="nw", pady=(10, 0))
        self.result_text_label = ttk.Label(control_panel_frame, text="")
        self.result_text_label.pack(anchor="nw")

        self.plate_color_label = ttk.Label(control_panel_frame, text="", width="20")
        self.plate_color_label.pack(anchor="nw", pady=(10, 0))

        load_image_button = ttk.Button(control_panel_frame, text="Load Image", width=20, command=self.load_image)
        load_image_button.pack(anchor="se", pady="5", side=tk.BOTTOM)

        # Initialize PlateRecognizer from the separate module
        self.plate_recognizer = PlateRecognizer()
        self.plate_recognizer.train_svm()

    def load_image_file(self, filename):
        """Moved from PlateRecognizer to here since it's UI-related"""
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    def convert_to_tk_image(self, opencv_image):
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        img_width = tk_image.width()
        img_height = tk_image.height()

        if img_width > self.display_width or img_height > self.display_height:
            width_ratio = self.display_width / img_width
            height_ratio = self.display_height / img_height
            scale_factor = min(width_ratio, height_ratio)

            img_width = int(img_width * scale_factor)
            if img_width <= 0: img_width = 1
            img_height = int(img_height * scale_factor)
            if img_height <= 0: img_height = 1
            pil_image = pil_image.resize((img_width, img_height), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(image=pil_image)
        return tk_image

    def display_results(self, plate_text, plate_image, plate_color):
        if plate_text:
            plate_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            plate_pil = Image.fromarray(plate_rgb)
            self.tk_plate_image = ImageTk.PhotoImage(image=plate_pil)
            self.plate_image_label.configure(image=self.tk_plate_image, state='enable')
            self.result_text_label.configure(text=str(plate_text))
            self.last_update_time = time.time()
            try:
                color_info = self.plate_color_mapping[plate_color]
                self.plate_color_label.configure(text=color_info[0], background=color_info[1], state='enable')
            except:
                self.plate_color_label.configure(state='disabled')
        elif self.last_update_time + 8 < time.time():
            self.plate_image_label.configure(state='disabled')
            self.result_text_label.configure(text="")
            self.plate_color_label.configure(state='disabled')

    def load_image(self):
        self.is_thread_running = False
        supported_formats = [
            ("JPEG ", "*.jpg *.jpeg")
        ]
        self.image_path = askopenfilename(title="Select image for recognition", filetypes=supported_formats)
        if self.image_path:
            # Load and display original image
            input_image = self.load_image_file(self.image_path)
            self.tk_input_image = self.convert_to_tk_image(input_image)
            self.original_image_label.configure(image=self.tk_input_image)

            # Display grayscale image label
            self.grayscale_header_label.configure(text='Grayscale Image:')

            # Convert and display grayscale image
            grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            grayscale_3channel = cv2.merge([grayscale_image, grayscale_image, grayscale_image])
            self.tk_grayscale_image = self.convert_to_tk_image(grayscale_3channel)
            self.processed_image_label.configure(image=self.tk_grayscale_image)

            # Plate recognition logic
            scale_factors = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
            for scale in scale_factors:
                plate_text, plate_img, plate_color = self.plate_recognizer.recognize_plate(input_image, scale)
                if plate_text:
                    break
            self.display_results(plate_text, plate_img, plate_color)

    @staticmethod
    def video_processing_thread(self):
        self.is_thread_running = True
        last_recognition_time = time.time()
        while self.is_thread_running:
            _, frame = self.camera_capture.read()
            self.tk_input_image = self.convert_to_tk_image(frame)
            self.original_image_label.configure(image=self.tk_input_image)
            if time.time() - last_recognition_time > 2:
                plate_text, plate_img, plate_color = self.plate_recognizer.recognize_plate(frame)
                self.display_results(plate_text, plate_img, plate_color)
                last_recognition_time = time.time()
        print("Thread stopped")


def on_window_close():
    print("Closing application")
    if recognition_ui.is_thread_running:
        recognition_ui.is_thread_running = False
        recognition_ui.processing_thread.join(2.0)
    main_window.destroy()


if __name__ == '__main__':
    main_window = tk.Tk()
    recognition_ui = LicensePlateRecognitionUI(main_window)
    main_window.protocol('WM_DELETE_WINDOW', on_window_close)
    main_window.mainloop()