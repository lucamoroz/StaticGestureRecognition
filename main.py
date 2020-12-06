import cv2
from hand_classifier.hand_cnn import HandCNN
from PIL import Image
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

        # OpenCV follows BGR color convention while PIL follows RGB color convention
        cv2.imshow("Captured", frame)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        preds = hand_cnn.predict_img(pil_img=pil_img)
        label = hand_cnn.LABELS[np.argmax(preds)]
        prob = np.max(preds)
        print(label, " ", prob)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
