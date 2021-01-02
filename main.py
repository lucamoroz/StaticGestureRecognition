import argparse
import json
import sys
import time

import cv2

from action_taker import ActionTaker
from hand_classifier.hand_cnn import HandCNN
import numpy as np


hand_cnn = HandCNN()
action_taker = ActionTaker()


def main(args=None):
    global hand_cnn
    global action_taker

    if not args:
        args = parse_args(sys.argv[1:])

    with open(args.commands) as f:
        commands = json.loads(f.read())
        if set(commands) != set(HandCNN.LABELS):
            raise RuntimeError("Commands should define actions for the following labels: ", HandCNN.LABELS)

    hand_cnn = HandCNN(args.model)
    action_taker = ActionTaker(commands=commands, history_len=args.history_len, min_confidence=args.min_conf)

    capture_frames(args)


def capture_frames(args):
    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)

    prev = 0

    print("Running...\n")

    while cap.isOpened():
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if not ret:
            break

        if args.debug:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(25)

            if key & 0xFF == ord('q'):
                break

        if time_elapsed > 1. / args.fps:
            prev = time.time()

            on_new_frame(frame, args)

    cap.release()
    cv2.destroyAllWindows()


def on_new_frame(frame, args):
    # OpenCV follows BGR color convention
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predictions = hand_cnn.predict_img(img_rgb)
    label = hand_cnn.LABELS[np.argmax(predictions)]
    prob = np.max(predictions)

    if args.debug:
        print(label, " ", prob)

    action_taker.on_new_state(label, prob)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Run user-defined commands on gesture recognition.")
    parser.add_argument("--fps", help="number of processed frames per second, -1 to disable", default=3, type=int)
    parser.add_argument("--min_conf", help="minimum gesture prediction confidence in [0,1].", type=float, default=0.98)
    parser.add_argument("--commands", help="path to user defined commands.", type=str, default="commands.json")
    parser.add_argument("--model", help="path to classifier model", type=str, default="hand_classifier/models/model_final.hdf5")
    parser.add_argument("--debug", help="show frames and recognized gestures.", action="store_true")
    parser.add_argument("--history_len", help="required number of recognitions of the same gesture required to take an action", default=3, type=int)
    parser.add_argument("--video", help="can be either a path to a video or the id of a local device", default="0")

    return parser.parse_args(args)


if __name__ == '__main__':
    main()
