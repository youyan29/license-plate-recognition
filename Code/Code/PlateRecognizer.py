import cv2
import numpy as np
from numpy.linalg import norm
import os
from SVMModel import SVMModel

CHARACTER_SIZE = 20
MAX_IMAGE_WIDTH = 1000
MIN_PLATE_AREA = 2000
PROVINCE_START_ID = 1000

PROVINCE_MAPPING = [
    "zh_chuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]

def load_image_file(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

def clamp_point_coordinates(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

def find_histogram_peaks(threshold, histogram):
    rising_edge = -1
    is_peak = False
    if histogram[0] > threshold:
        rising_edge = 0
        is_peak = True
    peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - rising_edge > 2:
                is_peak = False
                peaks.append((rising_edge, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            rising_edge = i
    if is_peak and rising_edge != -1 and i - rising_edge > 4:
        peaks.append((rising_edge, i))
    return peaks

def split_characters(image, peaks):
    character_images = []
    for peak in peaks:
        character_images.append(image[:, peak[0]:peak[1]])
    return character_images

def correct_skew(image):
    moments = cv2.moments(image)
    if abs(moments['mu02']) < 1e-2:
        return image.copy()
    skew = moments['mu11'] / moments['mu02']
    transform_matrix = np.float32([[1, skew, -0.5 * CHARACTER_SIZE * skew], [0, 1, 0]])
    corrected_image = cv2.warpAffine(image, transform_matrix, (CHARACTER_SIZE, CHARACTER_SIZE),
                                     flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return corrected_image

def extract_hog_features(digit_images):
    feature_vectors = []
    for img in digit_images:
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y)
        bins_count = 16
        quantized_angles = np.int32(bins_count * angle / (2 * np.pi))

        angle_blocks = (quantized_angles[:10, :10], quantized_angles[10:, :10],
                        quantized_angles[:10, 10:], quantized_angles[10:, 10:])
        mag_blocks = (magnitude[:10, :10], magnitude[10:, :10],
                      magnitude[:10, 10:], magnitude[10:, 10:])

        histograms = [np.bincount(b.ravel(), m.ravel(), bins_count) for b, m in zip(angle_blocks, mag_blocks)]
        combined_hist = np.hstack(histograms)

        epsilon = 1e-7
        combined_hist /= combined_hist.sum() + epsilon
        combined_hist = np.sqrt(combined_hist)
        combined_hist /= norm(combined_hist) + epsilon

        feature_vectors.append(combined_hist)
    return np.float32(feature_vectors)

class PlateRecognizer:
    def __init__(self):
        self.configuration = {
            "configurations": [
                {
                    "enabled": 1,
                    "blur_size": 3,
                    "morphology_rows": 4,
                    "morphology_cols": 19,
                    "col_threshold": 10,
                    "row_threshold": 21
                },
                {
                    "enabled": 0,
                    "blur_size": 3,
                    "morphology_rows": 5,
                    "morphology_cols": 12,
                    "col_threshold": 10,
                    "row_threshold": 18
                }
            ]
        }

        # Choose first enabled configuration
        for config in self.configuration["configurations"]:
            if config["enabled"]:
                self.current_config = config.copy()
                break
        else:
            raise RuntimeError('No valid configuration found')

    def __del__(self):
        self.save_training_data()

    def train_svm(self):
        self.char_model = SVMModel(C=1, gamma=0.5)
        self.province_model = SVMModel(C=1, gamma=0.5)

        if os.path.exists("svm.dat"):
            self.char_model.load_model("svm.dat")
        else:
            training_images = []
            training_labels = []

            for root, dirs, files in os.walk("train\\chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                char_code = ord(os.path.basename(root))
                for filename in files:
                    file_path = os.path.join(root, filename)
                    char_img = cv2.imread(file_path)
                    char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
                    training_images.append(char_img)
                    training_labels.append(char_code)

            training_images = list(map(correct_skew, training_images))
            training_features = extract_hog_features(training_images)
            training_labels = np.array(training_labels)
            self.char_model.train_model(training_features, training_labels)

        if os.path.exists("svmchinese.dat"):
            self.province_model.load_model("svmchinese.dat")
        else:
            training_images = []
            training_labels = []
            for root, dirs, files in os.walk("train\\charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                province_id = PROVINCE_MAPPING.index(pinyin) + PROVINCE_START_ID + 1
                for filename in files:
                    file_path = os.path.join(root, filename)
                    char_img = cv2.imread(file_path)
                    char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
                    training_images.append(char_img)
                    training_labels.append(province_id)

            training_images = list(map(correct_skew, training_images))
            training_features = extract_hog_features(training_images)
            training_labels = np.array(training_labels)
            self.province_model.train_model(training_features, training_labels)

    def save_training_data(self):
        if not os.path.exists("svm.dat"):
            self.char_model.save_model("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.province_model.save_model("svmchinese.dat")

    def locate_plate_area(self, plate_hsv, lower_hue, upper_hue, plate_color):
        rows, cols = plate_hsv.shape[:2]
        left = cols
        right = 0
        top = 0
        bottom = rows

        row_threshold = self.current_config["row_threshold"]
        col_threshold = cols * 0.8 if plate_color != "green" else cols * 0.5

        for i in range(rows):
            count = 0
            for j in range(cols):
                H = plate_hsv.item(i, j, 0)
                S = plate_hsv.item(i, j, 1)
                V = plate_hsv.item(i, j, 2)
                if lower_hue < H <= upper_hue and S > 34 and V > 46:
                    count += 1
            if count > col_threshold:
                if bottom > i:
                    bottom = i
                if top < i:
                    top = i

        for j in range(cols):
            count = 0
            for i in range(rows):
                H = plate_hsv.item(i, j, 0)
                S = plate_hsv.item(i, j, 1)
                V = plate_hsv.item(i, j, 2)
                if lower_hue < H <= upper_hue and S > 34 and V > 46:
                    count += 1
            if count > rows - row_threshold:
                if left > j:
                    left = j
                if right < j:
                    right = j
        return left, right, top, bottom

    def recognize_plate(self, input_image, resize_ratio=1):
        if isinstance(input_image, str):
            image = load_image_file(input_image)
        else:
            image = input_image

        height, width = image.shape[:2]
        if width > MAX_IMAGE_WIDTH:
            resize_factor = MAX_IMAGE_WIDTH / width
            image = cv2.resize(image, (MAX_IMAGE_WIDTH, int(height * resize_factor)),
                               interpolation=cv2.INTER_LANCZOS4)

        if resize_ratio != 1:
            image = cv2.resize(image, (int(width * resize_ratio), int(height * resize_ratio)),
                               interpolation=cv2.INTER_LANCZOS4)
            height, width = image.shape[:2]

        blur_size = self.current_config["blur_size"]
        if blur_size > 0:
            image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

        original_image = image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((20, 20), np.uint8)
        opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        opened_image = cv2.addWeighted(gray_image, 1, opened_image, -1, 0)

        _, threshold_image = cv2.threshold(opened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge_image = cv2.Canny(threshold_image, 100, 200)

        kernel_size = (self.current_config["morphology_rows"], self.current_config["morphology_cols"])
        kernel = np.ones(kernel_size, np.uint8)
        closed_edges = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel)
        opened_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel)

        try:
            contours, _ = cv2.findContours(opened_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _, contours, _ = cv2.findContours(opened_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_PLATE_AREA]
        plate_candidates = []

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            if rect_width < rect_height:
                rect_width, rect_height = rect_height, rect_width
            aspect_ratio = rect_width / rect_height

            if 2 < aspect_ratio < 5.5:
                plate_candidates.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

        plate_images = []
        for rect in plate_candidates:
            angle = 1 if -1 < rect[2] < 1 else rect[2]
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)

            box = cv2.boxPoints(rect)
            top_point = right_point = [0, 0]
            left_point = bottom_point = [width, height]

            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if bottom_point[1] > point[1]:
                    bottom_point = point
                if top_point[1] < point[1]:
                    top_point = point
                if right_point[0] < point[0]:
                    right_point = point

            if left_point[1] <= right_point[1]:
                new_right = [right_point[0], top_point[1]]
                dst_points = np.float32([left_point, top_point, new_right])
                src_points = np.float32([left_point, top_point, right_point])
                M = cv2.getAffineTransform(src_points, dst_points)
                warped = cv2.warpAffine(original_image, M, (width, height))
                clamp_point_coordinates(new_right)
                clamp_point_coordinates(top_point)
                clamp_point_coordinates(left_point)
                plate_img = warped[int(left_point[1]):int(top_point[1]), int(left_point[0]):int(new_right[0])]
                plate_images.append(plate_img)

            elif left_point[1] > right_point[1]:
                new_left = [left_point[0], top_point[1]]
                dst_points = np.float32([new_left, top_point, right_point])
                src_points = np.float32([left_point, top_point, right_point])
                M = cv2.getAffineTransform(src_points, dst_points)
                warped = cv2.warpAffine(original_image, M, (width, height))
                clamp_point_coordinates(right_point)
                clamp_point_coordinates(top_point)
                clamp_point_coordinates(new_left)
                plate_img = warped[int(right_point[1]):int(top_point[1]), int(new_left[0]):int(right_point[0])]
                plate_images.append(plate_img)

        plate_colors = []
        for i, plate_img in enumerate(plate_images):
            green = yellow = blue = black = white = 0
            plate_hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
            if plate_hsv is None:
                continue

            rows, cols = plate_hsv.shape[:2]
            pixel_count = rows * cols

            for row in range(rows):
                for col in range(cols):
                    H = plate_hsv.item(row, col, 0)
                    S = plate_hsv.item(row, col, 1)
                    V = plate_hsv.item(row, col, 2)
                    if 11 < H <= 34 and S > 34:
                        yellow += 1
                    elif 35 < H <= 99 and S > 34:
                        green += 1
                    elif 99 < H <= 124 and S > 34:
                        blue += 1
                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                    elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                        white += 1

            color = "none"
            lower_hue = upper_hue = 0

            if yellow * 2 >= pixel_count:
                color = "yellow"
                lower_hue = 11
                upper_hue = 34
            elif green * 2 >= pixel_count:
                color = "green"
                lower_hue = 35
                upper_hue = 99
            elif blue * 2 >= pixel_count:
                color = "blue"
                lower_hue = 100
                upper_hue = 124
            elif black + white >= pixel_count * 0.7:
                color = "bw"

            plate_colors.append(color)

            if lower_hue == 0:
                continue

            left, right, top, bottom = self.locate_plate_area(plate_hsv, lower_hue, upper_hue, color)
            if bottom == top and left == right:
                continue

            need_refinement = False
            if bottom >= top:
                bottom = 0
                top = rows
                need_refinement = True
            if left >= right:
                left = 0
                right = cols
                need_refinement = True

            plate_images[i] = plate_img[bottom:top, left:right] if color != "green" or bottom < (
                        top - bottom) // 4 else plate_img[
                                                bottom - (top - bottom) // 4:top, left:right]

            if need_refinement:
                plate_img = plate_images[i]
                plate_hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
                left, right, top, bottom = self.locate_plate_area(plate_hsv, lower_hue, upper_hue, color)
                if bottom == top and left == right:
                    continue
                if bottom >= top:
                    bottom = 0
                    top = rows
                if left >= right:
                    left = 0
                    right = cols

            plate_images[i] = plate_img[bottom:top, left:right] if color != "green" or bottom < (
                        top - bottom) // 4 else plate_img[
                                                bottom - (top - bottom) // 4:top, left:right]

        recognition_result = []
        plate_roi = None
        plate_color = None

        for i, color in enumerate(plate_colors):
            if color in ("blue", "yellow", "green"):
                plate_img = plate_images[i]
                gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

                if color == "green" or color == "yellow":
                    gray_plate = cv2.bitwise_not(gray_plate)

                _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                row_hist = np.sum(binary_plate, axis=1)
                min_row = np.min(row_hist)
                avg_row = np.sum(row_hist) / row_hist.shape[0]
                row_thresh = (min_row + avg_row) / 2
                row_peaks = find_histogram_peaks(row_thresh, row_hist)

                if len(row_peaks) == 0:
                    continue

                largest_peak = max(row_peaks, key=lambda x: x[1] - x[0])
                binary_plate = binary_plate[largest_peak[0]:largest_peak[1]]

                rows, cols = binary_plate.shape[:2]
                binary_plate = binary_plate[1:rows - 1]
                col_hist = np.sum(binary_plate, axis=0)
                min_col = np.min(col_hist)
                avg_col = np.sum(col_hist) / col_hist.shape[0]
                col_thresh = (min_col + avg_col) / 5
                col_peaks = find_histogram_peaks(col_thresh, col_hist)

                if len(col_peaks) <= 6:
                    continue

                max_peak = max(col_peaks, key=lambda x: x[1] - x[0])
                max_width = max_peak[1] - max_peak[0]

                if col_peaks[0][1] - col_peaks[0][0] < max_width / 3 and col_peaks[0][0] == 0:
                    col_peaks.pop(0)

                current_width = 0
                for j, peak in enumerate(col_peaks):
                    if peak[1] - peak[0] + current_width > max_width * 0.6:
                        break
                    else:
                        current_width += peak[1] - peak[0]

                if j > 0:
                    merged_peak = (col_peaks[0][0], col_peaks[j][1])
                    col_peaks = col_peaks[j + 1:]
                    col_peaks.insert(0, merged_peak)

                dot_peak = col_peaks[2]
                if dot_peak[1] - dot_peak[0] < max_width / 3:
                    dot_img = binary_plate[:, dot_peak[0]:dot_peak[1]]
                    if np.mean(dot_img) < 255 / 5:
                        col_peaks.pop(2)

                if len(col_peaks) <= 6:
                    continue

                char_images = split_characters(binary_plate, col_peaks)

                for j, char_img in enumerate(char_images):
                    if np.mean(char_img) < 255 / 5:
                        continue

                    char_img_orig = char_img
                    border = char_img.shape[1] // 3
                    char_img = cv2.copyMakeBorder(char_img, 0, 0, border, border,
                                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    char_img = cv2.resize(char_img, (CHARACTER_SIZE, CHARACTER_SIZE))
                    char_img = cv2.GaussianBlur(char_img, (1, 1), 0)
                    _, char_img = cv2.threshold(char_img, 0, 255,
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    char_features = extract_hog_features([char_img])

                    if j == 0:
                        prediction = self.province_model.predict_chars(char_features)
                        char = PROVINCE_MAPPING[int(prediction[0]) - PROVINCE_START_ID]
                    else:
                        prediction = self.char_model.predict_chars(char_features)
                        char = chr(prediction[0])

                    if char == "1" and j == len(char_images) - 1:
                        if char_img_orig.shape[0] / char_img_orig.shape[1] >= 8:
                            continue

                    recognition_result.append(char)

                plate_roi = plate_img
                plate_color = color
                break

        return recognition_result, plate_roi, plate_color