import numpy as np
from sklearn.model_selection import train_test_split
from os.path import normpath, basename
from glob import glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam


class HandCNN:

    CONST_MODELS_PATH = "models"

    def __init__(self, load=False):
        if load:
            self.model = keras.models.load_model(self.CONST_MODELS_PATH)

    def train(self, data_path: str):
        """ The folder data_path should contain one folder per class, each one containing images of that class."""

        img_height = 224
        img_width = 224
        batch_size = 32
        epochs = 1
        data_augment = True

        # Classes inferred by the sub-folders
        data_gen = ImageDataGenerator(
            preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
            validation_split=0.2)

        if data_augment:
            data_gen = ImageDataGenerator(
                preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
                validation_split=0.2,
                height_shift_range=0.2,
                width_shift_range=0.2,
                rotation_range=20,
                brightness_range=[0.2, 1.0],
                zoom_range=[0.5, 1.0])

        train_generator = data_gen.flow_from_directory(
            data_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='rgb',
            class_mode='categorical',
            subset='training',
            shuffle=True)

        validation_generator = data_gen.flow_from_directory(
            data_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='rgb',
            class_mode='categorical',
            subset='validation',
            shuffle=True)

        model = self.get_model(train_generator.num_classes)

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // train_generator.batch_size,  # End epoch when all images have been used
            validation_data=validation_generator,
            validation_steps=validation_generator.n // validation_generator.batch_size,
            epochs=epochs)

        model.save(self.CONST_MODELS_PATH)
        self.model = model

    @staticmethod
    def get_model(num_classes, learning_rate=0.01):

        # Note: input is 224x224x3
        base_model = keras.applications.MobileNetV2(weights="imagenet", include_top=False)

        # TODO - try to freeze/not freeze the pretrained part
        for layer in base_model.layers:
            layer.trainable = False

        last = base_model.output
        last = GlobalAveragePooling2D()(last)
        last = Dense(1024, activation='relu')(last)
        last = Dense(1024, activation='relu')(last)
        last = Dense(512, activation='relu')(last)

        predictions = Dense(num_classes, activation='softmax')(last)

        model = Model(inputs=base_model.inputs, outputs=predictions)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=learning_rate),
                      metrics=['accuracy'])

        return model

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        return self.model.predict(img_tensor)

    def _load_dataset(self, data_path: str):
        """
        Loads the dataset from the given folder and returns 2 tuples, the first containing training data and
        the second validation data.
        The folder must contain a sub-folder for each hand pose, the sub-folder must contain samples of that pose.
        """

        poses_path = glob(data_path + "*/")
        poses = [basename(normpath(p)) for p in poses_path]

        print("Loading dataset for poses: ", poses)

        X = []
        y = []

        for p in poses_path:
            pose_name = basename(normpath(p))
            new_pose_imgs = glob(p + "*.png")
            X = X + new_pose_imgs
            y = y + [pose_name for _ in range(len(new_pose_imgs))]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

        return (X_train, y_train), (X_val, y_val)

if __name__ == "__main__":
    load = True

    if load:
        handCNN = HandCNN(load=True)

    else:
        handCNN = HandCNN(load=False)
        handCNN.train("/home/datasets/ml/TinyHands/carlos_r/")

    print(handCNN.predict("/home/datasets/ml/TinyHands/carlos_r/fist/img_CO_01_puno_0000_1_001_045.png"))
