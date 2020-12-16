import cv2
from hand_classifier.hand_cnn import HandCNN
import numpy as np


def main():
    hand_cnn = HandCNN("hand_classifier/models/model_final.hdf5")
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(25)

        if key & 0xFF == ord('q'):
            break

        cv2.imshow("Captured", frame)
        # OpenCV follows BGR color convention
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        predictions = hand_cnn.predict_img(img_rgb)
        label = hand_cnn.LABELS[np.argmax(predictions)]
        prob = np.max(predictions)
        print(label, " ", prob)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
