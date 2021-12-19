import cv2 as cv
import imutils


# Module for turning grayscale images into binary images
class ImageManip:
    def __init__(self):
        self.device = cv.VideoCapture(0)
        self.bg = None
        self.dim = [100, 360, 350, 700]  # top - [0], right - [1], bottom - [2], left - [3]
        self.frames = 0

    def convert(self, frame):
        """Function for converting grayscale images to binary images by using a threshold. Returns a threshold back
        to the main function."""
        frame = imutils.resize(frame, width=700)
        frame = cv.flip(frame, 1)

        roi = frame[self.dim[0]:self.dim[2], self.dim[1]:self.dim[3]]  # focus only on our area of interest

        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)  # turn image into grayscale
        gray = cv.GaussianBlur(gray, (7, 7), 0)  # apply slight blur (for edge detection)

        # calibrate the background values for initial 30 frames
        if self.frames < 30:
            self.get_run_avg(gray, 0.5)
        else:
            hand = self.segment(gray, 65)  # return a binary image

            if hand is not None:
                (thresh, seg) = hand
                cv.imshow("Thresh", thresh)  # shows the binary image

                return thresh  # return thresh back to main function this goes to the network
            else:
                return None

        self.frames += 1

    def segment(self, frame, threshold=25):
        """This function is used to turn given frame to binary using a given threshold."""
        # get absolute difference between background and the hand
        diff = cv.absdiff(self.bg.astype("uint8"), frame)
        thresh = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]  # function to convert to binary

        # Return contours for given binary images, using edge detection.
        contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return
        else:
            seg = max(contours, key=cv.contourArea)
            return thresh, seg

    def get_run_avg(self, frame, weight):
        """Calibration function to calibrate background values. For comparing them to the hand"""
        if self.bg is None:
            self.bg = frame.copy().astype("float")  # convert to float
            return

        cv.accumulateWeighted(frame, self.bg, weight)

    def make_data(self):
        """Function for generating binary image datasets, similar to convert function."""
        while True:
            _, frame = self.device.read()
            frame = cv.flip(frame, 1)
            clone = frame.copy()

            roi = frame[self.dim[0]:self.dim[2], self.dim[1]:self.dim[3]]

            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (7, 7), 0)

            if self.frames < 30:
                self.get_run_avg(blur, 0.5)
            else:
                hand = self.segment(blur, 65)

                if hand is not None:
                    thresh, seg = hand

                    cv.imshow("Threshold", thresh)
                    # save the gestures to specified directories
                    cv.imwrite(f"Gesture_Dataset/gesture_full_screen/gesture_full_screen{self.frames}.jpeg", thresh)

            self.frames += 1
            cv.rectangle(clone, (self.dim[3], self.dim[0]), (self.dim[1], self.dim[2]), (0, 255, 0), 2)
            print(self.frames)
            cv.imshow("Video", clone)

            if self.frames == 30:
                print("Done Calibrating, place hand!")

            if cv.waitKey(1) & 0xFF == ord('q') or self.frames == 2500:
                self.cleanup()
                break

    def cleanup(self):
        """Frees memory allocated by cv2."""
        self.device.read()
        cv.destroyAllWindows()


if __name__ == "__main__":
    cap = ImageManip()
    cap.make_data()
