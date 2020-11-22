import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# use this to change which GPU to use
gpu = "0"

# set the modified tf session as backend in keras
setup_gpu(gpu)


model = models.load_model("keras-retinanet/snapshots/resnet50_csv_02.h5", "resnet50")
# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)
labels_to_names = {0: 'hand'}

# load image
image = read_image_bgr('/home/datasets/ml/OUHANDS/train/hand_data/colour/A-jha-0001.png')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
# boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
boxes, scores = model.predict_on_batch(np.expand_dims(image, axis=0))

print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score in zip(boxes[0], scores[0]):
    print(box)
    print(score)

    # scores are sorted so we can break
    if score < 0.5:
        break

    b = box.astype(int)
    draw_box(draw, b, color=(0, 255, 0))

    caption = "hand {:.3f}".format(score[0])
    draw_caption(draw, b, caption)

cv2.imshow("res", draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
