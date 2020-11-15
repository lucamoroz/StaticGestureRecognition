import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.optimizers import Adam


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

if __name__ == "__main__":
    load = True

    if load:
        handCNN = HandCNN(load=True)

    else:
        handCNN = HandCNN(load=False)
        handCNN.train("/home/datasets/ml/TinyHands/carlos_r/")

    print(handCNN.predict("/home/datasets/ml/TinyHands/carlos_r/fist/img_CO_01_puno_0000_1_001_045.png"))
