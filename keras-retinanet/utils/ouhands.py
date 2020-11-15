import xml.etree.ElementTree as ET


def prepare_ouhands(dlib_file, images_path, destination_csv):
    tree = ET.parse(dlib_file)
    root = tree.getroot()
    images = root[1]
    assert images.tag == "images"

    with open(destination_csv, 'w', newline='') as csvfile:
        for img in images:
            img_name = img.attrib['file']
            for box in img:
                x1 = int(box.attrib['top'])
                y1 = int(box.attrib['left'])
                width = int(box.attrib['width'])
                height = int(box.attrib['height'])
                x2 = x1 + width
                y2 = y1 + height

                csvfile.writelines("{},{},{},{},{},hand\n".format(images_path + img_name, x1, y1, x2, y2))


if __name__ == "__main__":
    prepare_ouhands(
        "/home/datasets/ml/OUHANDS/dlib/dlib_train.xml",
        "/home/datasets/ml/OUHANDS/train/hand_data/colour/",
        "../dataset/ouhands.csv")
