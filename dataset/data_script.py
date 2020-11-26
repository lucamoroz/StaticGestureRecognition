"""
Script used to get training/testing data.
It uses a video (either live or an existing file) to add training images for a given class.

TODO check if it's possible to filter video frames by noise
"""
import argparse
import sys
import os
import time
import cv2


def main(args=None):
    if not args:
        args = parse_args(sys.argv[1:])
    print("Running with args:", args)

    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)

    base_path = args.path + args.class_name
    try:
        os.makedirs(base_path, exist_ok=True)
    except OSError:
        print("Failed creating directory at ", base_path)
        return
    base_path = base_path + "/" + args.prefix

    counter = 0

    frame_rate = 2
    prev = 0
    while cap.isOpened():
        time_elapsed = time.time() - prev

        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)

        if time_elapsed > 1. / args.fps:
            prev = time.time()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            img_path = base_path + "_" + str(counter) + ".jpg"
            if os.path.isfile(img_path):
                print("Error: file {} already exists!".format(img_path))
                return
            cv2.imwrite(img_path, frame)
            counter += 1
            print("Saved {} images to: {}".format(counter, img_path))

    cap.release()
    cv2.destroyAllWindows()


def parse_args(args):
    parser = argparse.ArgumentParser(description="Add image data.")
    parser.add_argument("--class_name", help="name of the class the video contains image about", required=True)
    parser.add_argument("--prefix", help="image prefix when saving video frames", required=True)
    parser.add_argument("--fps", help="number of frames per second", default=2, type=int)
    parser.add_argument("--video", help="can be either a path to a video or the id of a local device", default="0")
    parser.add_argument("--path", help="path to the folder containing the dataset", default="./")

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
