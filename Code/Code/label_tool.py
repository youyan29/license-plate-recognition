import os
import cv2
import pandas as pd
import numpy as np
from tkinter import Tk, Label, Button, Entry, filedialog
from PIL import Image, ImageTk

class LabelTool:
    def __init__(self, root, image_folder="C:\\Users\\29602\\Desktop\\SAT_FYP\\Code\\imgs", csv_file="Data.csv"):
        self.root = root
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort()
        self.current_index = 0

        # 加载或初始化标签 CSV
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = pd.DataFrame({"filename": self.image_files, "ground_truth": [""] * len(self.image_files)})

        # UI 元素
        self.image_label = Label(root)
        self.image_label.pack()

        self.filename_label = Label(root, text="", font=('Arial', 14))
        self.filename_label.pack(pady=5)

        self.entry = Entry(root, width=30, font=('Arial', 16))
        self.entry.pack(pady=5)

        Button(root, text="保存并下一张", command=self.save_and_next).pack(side="left", padx=10, pady=10)
        Button(root, text="上一张", command=self.prev_image).pack(side="left", padx=10, pady=10)
        Button(root, text="保存所有", command=self.save_csv).pack(side="left", padx=10, pady=10)

        self.load_image()

    def load_image(self):
        if 0 <= self.current_index < len(self.image_files):
            filename = self.image_files[self.current_index]
            self.filename_label.config(text=f"{filename}  ({self.current_index + 1}/{len(self.image_files)})")

            img_path = os.path.join(self.image_folder, filename)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(img)
            pil_img.thumbnail((800, 600))
            self.tk_img = ImageTk.PhotoImage(pil_img)

            self.image_label.config(image=self.tk_img)

            # 载入已存在的标签（如有）
            current_label = self.df.loc[self.df["filename"] == filename, "ground_truth"].values[0]
            self.entry.delete(0, "end")
            self.entry.insert(0, current_label)

    def save_and_next(self):
        self.save_label()
        if self.current_index + 1 < len(self.image_files):
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        self.save_label()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def save_label(self):
        plate_text = self.entry.get().strip()
        filename = self.image_files[self.current_index]
        self.df.loc[self.df["filename"] == filename, "ground_truth"] = plate_text
        self.save_csv()

    def save_csv(self):
        self.df.to_csv(self.csv_file, index=False, encoding='utf-8-sig')
        print("[✓] 数据已保存到", self.csv_file)


if __name__ == "__main__":
    root = Tk()
    root.title("车牌标注工具")
    app = LabelTool(root)
    root.mainloop()
