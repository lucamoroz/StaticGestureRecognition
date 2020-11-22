import xml.etree.ElementTree as ET
from glob import glob
import os


def prepare_ouhands(ouhands_path, destination_ouhands, destination_csv):
    lines = []

    # Parse annotated images
    tree = ET.parse(ouhands_path + "/dlib/dlib_train.xml")
    root = tree.getroot()
    images = root[1]
    assert images.tag == "images"

    for img in images:
        img_name = img.attrib['file']
        for box in img:
            x1 = int(box.attrib['top'])
            y1 = int(box.attrib['left'])
            width = int(box.attrib['width'])
            height = int(box.attrib['height'])
            x2 = x1 + width
            y2 = y1 + height
            lines = lines + ["{},{},{},{},{},hand\n".format(destination_ouhands + "/train/hand_data/colour/" + img_name, x1, y1, x2, y2)]

    # Parse negative data
    for img_path in glob(ouhands_path + "/train/negative_data/colour/*.png"):
        head, tail = os.path.split(img_path)
        lines = lines + ["{},,,,,\n".format(destination_ouhands + "/train/negative_data/colour/" + tail)]

    with open(destination_csv, 'w', newline='') as csv_file:
        csv_file.writelines(lines)


if __name__ == "__main__":
    prepare_ouhands("/home/datasets/ml/OUHANDS", "/floyd/input/ouhands", "ouhands.csv") # /floyd/input/ouhands
