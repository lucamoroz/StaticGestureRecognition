import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, AlphaDropout
from keras.applications import imagenet_utils, MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import os


class HandCNN:

    CONST_MODELS_PATH = "models/"
    LABELS = ["fist", "palm", "pointer", "spok", "thumb_down", "thumb_up"]

    def __init__(self, path=None):
        if path:
            self.model = keras.models.load_model(path)
            print(self.model.summary())

    def train(self, data_path: str, epochs: int, batch_size: int):
        """ The folder data_path should contain one folder per class, each one containing images of that class."""

        img_height = 224
        img_width = 224

        # TODO add "nothing" data
        # TODO train with data augment!!! try cutout?

        # Classes inferred by the sub-folders
        data_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
            height_shift_range=0.2,  # Fraction of total height
            width_shift_range=0.2,
            rotation_range=15,  # Allow small rotations only as some gestures are orientation-dependant
            shear_range=10,
            brightness_range=[0.3, 1.0],
            channel_shift_range=30,  # Randomly shift one color channel value
            horizontal_flip=True,  # Ok as long as we use gestures where horizontal orientation doesn't matter
            vertical_flip=False,
        )

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

        checkpoint_callback = self._get_checkpoint_callback(base_path)
        early_stop_callback = self._get_early_stop_callback()

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // train_generator.batch_size,  # End epoch when all images have been used
            validation_data=validation_generator,
            validation_steps=validation_generator.n // validation_generator.batch_size,
            epochs=epochs,
            callbacks=[checkpoint_callback, early_stop_callback])

        model.save(base_path + "model_final.hdf5")
        self.model = model

        return history

    @staticmethod
    def _get_checkpoint_callback(base_path):
        filepath = base_path + "checkpoint-model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        return ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='auto', period=1)

    @staticmethod
    def _get_early_stop_callback():
        return EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=2,  # Stop if validation loss increases for 2 epochs
            verbose=1
        )

    @staticmethod
    def get_model(num_classes, learning_rate=0.001):

        # TODO perform parameter tuning on alpha - remember to use float notation (e.g. 1.0, 1.3)
        #      otherwise pretrained model can't be found
        base_model = MobileNetV2(
            # Use default input shape (224x224x3)
            input_shape=(224, 224, 3),
            # Width multiplier to controls the number of filters. Available pretrained: 0.35, 0.5, 0.75, 1.0, 1.3, 1.4
            alpha=1.0,
            weights="imagenet",
            include_top=False)

        for layer in base_model.layers:
            layer.trainable = True

        last = GlobalAveragePooling2D()(base_model.output)
        # TODO weight decay regularization with param = 0.01?
        last = Dense(400, kernel_initializer="lecun_normal", activation="selu")(last)
        # TODO try one of these: dropout, batch normalization, SELU
        # AlphaDropout should be used with selu activation - see:
        # https://mlfromscratch.com/activation-functions-explained/#selu
        last = AlphaDropout(0.2)(last)
        predictions = Dense(num_classes, activation="softmax")(last)

        model = Model(inputs=base_model.inputs, outputs=predictions)

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(lr=learning_rate),  # TODO check SGDR
                      metrics=["accuracy"])

        return model

    def predict_img_path(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        return self.predict_img(img)

    def predict_img(self, pil_img):
        """ Returns predictions on a PIL image.
        """
        img = pil_img.resize((224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # Rescale to range [0, 1] as expected by MobileNet
        img_tensor /= 255.

        return self.model.predict(img_tensor)


def main():
    # TODO add a visualization method (class activation maps)
    # train = True
    # if train:
    #     handCNN = HandCNN()
    #     history = handCNN.train(data_path="../dataset/testdataset/", epochs=3, batch_size=16)
    #     save_history_graphs(history)
    # else:
    #     handCNN = HandCNN(path="models/model_final.hdf5")
    #
    # prediction = handCNN.predict("../dataset/testdataset/fist/low_light1_0.jpg")
    # print("{} -> class: {}".format(prediction, prediction.argmax(axis=-1)))
    # return

    # tf.debugging.set_log_device_placement(True)
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found, device name: ' + device_name)
    print('Found GPU at: {}'.format(device_name))

    with tf.device('GPU:0'):
        handCNN = HandCNN()
        history = handCNN.train(data_path="/floyd/input/handposes", epochs=15, batch_size=32)
        save_history_graphs(history)


def save_history_graphs(history):
    import matplotlib.pyplot as plt
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy_history.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_history.png')

if __name__ == '__main__':
    main()
