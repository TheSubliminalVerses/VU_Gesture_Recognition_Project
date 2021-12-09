from tqdm import tqdm
import os
import cv2 as cv
import numpy as np

print("Building Dataset. Please wait...")


class BuildData:
    def __init__(self, img_size):
        self.IMG_SIZE = img_size
        self.p_stop_and_play = "Gesture_Dataset/gesture_stop_and_play"
        self.p_mute = "Gesture_Dataset/gesture_mute"
        self.p_volume_down = "Gesture_Dataset/gesture_volume_down"
        self.p_volume_up = "Gesture_Dataset/gesture_volume_up"
        self.p_full_screen = "Gesture_Dataset/gesture_full_screen"
        self.p_5s_fw = "Gesture_Dataset/gesture_5s_forward"
        self.p_5s_bck = "Gesture_Dataset/gesture_5s_backward"
        self.p_10s_fw = "Gesture_Dataset/gesture_10s_forward"
        self.p_10s_bck = "Gesture_Dataset/gesture_10s_backwards"
        self.p_null = "Gesture_Dataset/gesture_null"

        self.LABELS = {self.p_stop_and_play: 0, self.p_mute: 1, self.p_volume_down: 2, self.p_volume_up: 3,
                       self.p_full_screen: 4, self.p_5s_fw: 5, self.p_5s_bck: 6, self.p_10s_fw: 7, self.p_10s_bck: 8,
                       self.p_null: 9}

        self.training_data = []

        self.count_dict = {"play/stop": 0, "mute": 0, "vol_down": 0, "vol_up": 0, "full_scr": 0, "5s_fw": 0, "5s_bck": 0,
                           "10s_fw": 0, "10s_bck": 0, "null": 0}

    def build_dataset(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    pth = os.path.join(label, f)
                    img = cv.imread(pth, cv.IMREAD_GRAYSCALE)
                    img = cv.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(10)[self.LABELS[label]]])

                    if label == self.p_stop_and_play:
                        self.count_dict["play/stop"] += 1
                    elif label == self.p_mute:
                        self.count_dict["mute"] += 1
                    elif label == self.p_volume_down:
                        self.count_dict["vol_down"] += 1
                    elif label == self.p_volume_up:
                        self.count_dict["vol_up"] += 1
                    elif label == self.p_full_screen:
                        self.count_dict["full_scr"] += 1
                    elif label == self.p_5s_fw:
                        self.count_dict["5s_fw"] += 1
                    elif label == self.p_5s_bck:
                        self.count_dict["5s_bck"] += 1
                    elif label == self.p_10s_fw:
                        self.count_dict["10s_fw"] += 1
                    elif label == self.p_10s_bck:
                        self.count_dict["10s_bck"] += 1
                    elif label == self.p_null:
                        self.count_dict["null"] += 1

                except Exception:
                    print("Could not load image!")
                    exit(1)

        np.random.shuffle(self.training_data)
        np.save("train_dataset/training_data.npy", self.training_data)
        print(f"Play/Stop gestures: {self.count_dict['play/stop']}")
        print(f"Mute gestures: {self.count_dict['mute']}")
        print(f"Volume Down gestures: {self.count_dict['vol_down']}")
        print(f"Volume Up gestures: {self.count_dict['vol_up']}")
        print(f"Full Screen gestures: {self.count_dict['full_scr']}")
        print(f"5s Forward gestures: {self.count_dict['5s_fw']}")
        print(f"5s Backward gestures: {self.count_dict['5s_bck']}")
        print(f"10s Forward gestures: {self.count_dict['10s_fw']}")
        print(f"10s Backward gestures: {self.count_dict['10s_bck']}")
        print(f"Null gestures: {self.count_dict['null']}")


if __name__ == "__main__":
    data = BuildData(50)
    data.build_dataset()

    print("Build finished!")

