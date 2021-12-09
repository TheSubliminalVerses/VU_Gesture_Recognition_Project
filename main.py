import cv2 as cv
import time
import handTracking
import convNet
import torch
import numpy as np
import pyautogui
import CvImgManip
import matplotlib.pyplot as plt


class Camera:
    def __init__(self, scale, capture_device, interact_mode, init_weight=0.4):
        self.scale = scale
        self.device = capture_device
        self.weight = init_weight
        self.interact_mode = interact_mode
        self.weight = init_weight
        self.dim = [100, 360, 350, 700]  # top, right, bottom, left
        self.batch_size = 100
        self.batch_x = []
        self.tensor_x = []
        self.net = convNet.Net()  # Creates an instance of convNet.py module, this is needed for load_network_data fun
        self.cuda_device = torch.device("cuda:0")  # define gpu
        self.track = handTracking.Tracking()
        self.ImageManipulator = CvImgManip.ImageManip()  # Create an instance of CvImgManip.py to create binary images
        self.bg = None

        # Labels translate nn predicted class max number into gesture names
        self.LABELS = {0: "Play/Stop", 1: "Mute", 2: "Volume Down", 3: "Volume Up",
                       4: "Full Screen", 5: "5s Fast Forward", 6: "5s Backward", 7: "10s Forward", 8: "10s Backward",
                       9: "null"}

        self.wmp_labels = {0: "k", 1: "mute", 2: "volumedown", 3: "volumeup"}

    def __repr__(self):
        return f"Camera recording object, scaled by x{self.scale}. Utilizing cv2 image capture."

    @staticmethod
    def open_media():
        """Open media that the user wants to target - Video Player"""
        media_player = pyautogui.getWindowsWithTitle("Video Player")[0]
        media_player.maximize()
        media_player.activate()

    @staticmethod
    def open_youtube():
        """Opens media that the users wants to target - YouTube"""
        t_window = pyautogui.getWindowsWithTitle("YouTube")[0]
        window = pyautogui.getActiveWindow()
        window.minimizpe()
        t_window.maximize()
        t_window.activate()

    def load_network_data(self):
        """Loads the saved state dict created by convNet.py module. This file ends with .pt/.pth.
        After training network with convNet.py save the state !state dict! of the model. NOT THE ACTUAL MODEL CLASS"""
        try:
            self.net.load_state_dict(torch.load("Brain.pt"))  # load state dict
            self.net.eval()  # adjust weights
            self.net.to(self.cuda_device)  # send net to gpu

        except Exception:
            print("Failed to load the model!")
            exit(1)

        print("Model loaded successfully!")

    def to_net(self, frame) -> None:
        """Main gesture recognition function. Contains the actual network to make predictions based on the loaded
        state dict."""
        img = cv.resize(frame, (50, 50))
        to_tensor = np.array(img)
        self.tensor_x = torch.tensor([i for i in to_tensor], device=self.cuda_device, dtype=torch.float).view(-1, 1, 50,
                                                                                                              50)
        self.tensor_x = self.tensor_x / 255.0
        self.tensor_x.to(self.cuda_device)

        with torch.no_grad():
            net_out = self.net(self.tensor_x)
            p_class = torch.argmax(net_out)
            # print(p_class)

            if p_class.item() in self.LABELS:
                print(f"Predicted class {self.LABELS[p_class.item()]}")
                if self.LABELS[p_class.item()] != "null":
                    pyautogui.press(self.wmp_labels[p_class.item()])
            else:
                return None

    @staticmethod
    def resize(frame, scale=0.75):
        """This function resizes the output frame by a given scaling factor"""
        width = int(frame.shape[0] * scale)
        height = int(frame.shape[1] * scale)

        return cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)

    def deploy(self):
        """Deploys the images capture from the camera.Sends the video stream directly to ImageManip module,
        which extracts features from the images by applying a threshold."""
        if not self.device.isOpened():
            print("Can't open camera!")
            exit(1)

        print("Camera Active!")

        self.load_network_data()  # loads neural network state dict (loads weights, features etc...)
        time.sleep(1)

        if self.interact_mode == "Media Player":
            self.open_media()
        elif self.interact_mode == "YouTube":
            self.open_youtube()

        while True:
            ret, frame = self.device.read()  # spawn a video stream

            if not ret:
                print("Can't receive frame (stream end?). Exiting...")
                break

            clone = frame.copy()  # creates a copy to send to the image manip module
            frame = cv.flip(frame, 1)
            cv.rectangle(frame, (self.dim[3], self.dim[0]), (self.dim[1], self.dim[2]), (0, 255, 0))

            to_net = self.ImageManipulator.convert(clone)  # sends cloned image to be thresholded

            cv.imshow("Video", frame)

            if to_net is not None:
                self.to_net(to_net)

            if cv.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                break

    def cleanup(self):
        """Cleanup is called after user presses 'q' frees memory allocated by cv2."""
        self.device.release()
        cv.destroyAllWindows()


def device_select(device_name: str) -> int:
    """Provides the ability to enter the name of the devices. Translates them into an integer."""
    device_dict = {"webcam": 0, "external": 1}
    if device_name in device_dict:
        return device_dict[device_name]
    else:
        print("Error! No such device!")
        exit(1)


if __name__ == "__main__":
    # m = input("Enter mode\n>>")
    dev = input("Enter device to capture on\n>> ")
    print("Activating camera! Please wait...")
    cap = cv.VideoCapture(device_select(dev))
    camera = Camera(1.5, cap, None)
    camera.deploy()
