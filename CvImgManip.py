import cv2 as cv
import imutils


class ImageManip:
    def __init__(self):
        self.device = cv.VideoCapture(0)
        self.bg = None
        self.dim = [100, 360, 350, 700]  # top, right, bottom, left
        self.frames = 0

    def convert(self, frame):
        frame = imutils.resize(frame, width=700)
        frame = cv.flip(frame, 1)

        roi = frame[self.dim[0]:self.dim[2], self.dim[1]:self.dim[3]]

        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (7, 7), 0)

        if self.frames < 30:
            self.get_run_avg(gray, 0.5)
        else:
            hand = self.segment(gray, 65)

            if hand is not None:
                (thresh, seg) = hand
                cv.imshow("Thresh", thresh)

                return thresh
            else:
                return None

        self.frames += 1

    def segment(self, frame, threshold=25):
        diff = cv.absdiff(self.bg.astype("uint8"), frame)
        thresh = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]

        contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return
        else:
            seg = max(contours, key=cv.contourArea)
            return thresh, seg

    def get_run_avg(self, frame, weight):
        if self.bg is None:
            self.bg = frame.copy().astype("float")
            return

        cv.accumulateWeighted(frame, self.bg, weight)

    def make_data(self):
        while True:
            _, frame = self.device.read()
            frame = cv.flip(frame, 1)
            clone = frame.copy()

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (7, 7), 0)

            if self.frames < 30:
                self.get_run_avg(blur, 0.5)
            else:
                hand = self.segment(blur, 65)

                if hand is not None:
                    thresh, seg = hand

                    cv.imshow("Threshold", thresh)
                    cv.imwrite(f"Gesture_Dataset/gesture_volume_down/gesture_volume_down{self.frames}.jpeg", thresh)

            self.frames += 1
            print(self.frames)
            cv.imshow("Video", clone)

            if cv.waitKey(1) & 0xFF == ord('q') or self.frames == 2500:
                self.cleanup()
                break

    def cleanup(self):
        self.device.read()
        cv.destroyAllWindows()

