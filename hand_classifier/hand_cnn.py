import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import imagenet_utils, MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import os


class HandCNN:

    CONST_MODELS_PATH = "models/"

    def __init__(self, path=None):
        if path:
            self.model = keras.models.load_model(path)

    def train(self, data_path: str, epochs: int, batch_size: int):
        """ The folder data_path should contain one folder per class, each one containing images of that class."""

        img_height = 224
        img_width = 224

        data_augment = False

        # Classes inferred by the sub-folders
        data_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2)

        if data_augment:
            data_gen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
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

        base_path = self.CONST_MODELS_PATH + str(datetime.now()) + "/"
        os.makedirs(base_path, exist_ok=True)
        filepath = base_path + "checkpoint-model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = \
            ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='auto', period=1)

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // train_generator.batch_size,  # End epoch when all images have been used
            validation_data=validation_generator,
            validation_steps=validation_generator.n // validation_generator.batch_size,
            epochs=epochs,
            callbacks=[checkpoint])

        model.save(base_path + "model_final.hdf5")
        self.model = model

        return history

    @staticmethod
    def get_model(num_classes, learning_rate=0.01):

        # Note: input is 224x224x3
        # TODO try with input size 64x64
        base_model = MobileNet(
            alpha=1,                # Keep default number of filters in each layer
            weights="imagenet",
            include_top=False)

        # TODO - try to freeze/not freeze the pretrained part
        for layer in base_model.layers:
            layer.trainable = True

        last = base_model.output
        last = GlobalAveragePooling2D()(last)
        last = Dense(1024, activation='relu')(last)
        last = Dense(1024, activation='relu')(last)
        last = Dense(512, activation='relu')(last)

        predictions = Dense(num_classes, activation='softmax')(last)

        model = Model(inputs=base_model.inputs, outputs=predictions)

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(lr=learning_rate),
                      metrics=['accuracy'])

        return model

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # Rescale to range [0, 1] as expected by MobileNet
        img_tensor /= 255.

        return self.model.predict(img_tensor)


def main():
    train = False
    if train:
        handCNN = HandCNN()
        handCNN.train(data_path="../dataset/testdataset/", epochs=2, batch_size=16)
    else:
        handCNN = HandCNN(path="models/model_final.hdf5")

    prediction = handCNN.predict("../dataset/testdataset/fist/low_light1_0.jpg")
    print("{} -> class: {}".format(prediction, prediction.argmax(axis=-1)))

    return
    tf.debugging.set_log_device_placement(True)
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found, device name: ' + device_name)
    print('Found GPU at: {}'.format(device_name))

    with tf.device('GPU:0'):
        handCNN = HandCNN(load=False)
        handCNN.train("/floyd/input/tinyhands/carlos_r/")


if __name__ == '__main__':
    main()
