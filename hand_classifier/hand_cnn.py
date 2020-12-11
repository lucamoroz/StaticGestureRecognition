import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, AlphaDropout
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import os


class HandCNN:
    # Used to map the predictions to labels
    LABELS = ["fist", "palm", "pointer", "spok", "thumb_down", "thumb_up"]
    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def __init__(self, path=None):
        if path:
            self.model = keras.models.load_model(path)
            print(self.model.summary())

    def train(self,
              data_path: str,
              epochs: int,
              batch_size: int,
              learning_rate=0.0001,
              img_width=IMG_WIDTH,
              img_height=IMG_HEIGHT,
              checkpoints_callback=True,
              early_stop_callback=True):

        train_generator, validation_generator = self._get_generators(data_path, batch_size, img_height, img_width)

        model = self.get_model(train_generator.num_classes, img_height, img_width)
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(lr=learning_rate),
                      metrics=["accuracy"])

        callbacks = []
        if checkpoints_callback:
            os.makedirs("models/checkpoints/", exist_ok=True)
            callbacks += [self._get_checkpoint_callback("models/checkpoints/")]

        if early_stop_callback:
            callbacks += [self._get_early_stop_callback()]

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // train_generator.batch_size,  # End epoch when all images have been used
            validation_data=validation_generator,
            validation_steps=validation_generator.n // validation_generator.batch_size,
            epochs=epochs,
            callbacks=callbacks)

        self.model = model

        return history

    def save_model(self, path="models/", file_name="model_final.hdf5"):
        os.makedirs(path, exist_ok=True)
        self.model.save(path + file_name)

    @staticmethod
    def _get_generators(data_path: str, batch_size: int, img_height: int, img_width: int):
        # Classes inferred by the sub-folders
        data_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
            height_shift_range=0.1,  # Fraction of total height
            width_shift_range=0.1,
            rotation_range=10,  # Allow small rotations only as some gestures are orientation-dependant
            brightness_range=[0.5, 1.0],
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

        return train_generator, validation_generator

    @staticmethod
    def _get_checkpoint_callback(base_path):
        filepath = base_path + "checkpoint-model-{epoch:02d}-{val_accuracy:.4f}.hdf5"
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
    def get_model(num_classes, img_height, img_width):

        base_model = MobileNetV2(
            # Use default input shape (224x224x3)
            input_shape=(img_height, img_width, 3),
            # Width multiplier to controls the number of filters. Available pretrained: 0.35, 0.5, 0.75, 1.0, 1.3, 1.4
            alpha=1.0,
            weights="imagenet",
            include_top=False)

        for layer in base_model.layers:
            layer.trainable = True

        last = GlobalAveragePooling2D()(base_model.output)
        last = Dense(320, kernel_initializer="lecun_normal", activation="selu")(last)
        # AlphaDropout should be used with SELU activation - see:
        # https://mlfromscratch.com/activation-functions-explained/#selu
        last = AlphaDropout(0.2)(last)
        predictions = Dense(num_classes, activation="softmax")(last)

        model = Model(inputs=base_model.inputs, outputs=predictions)

        return model

    def predict_img_path(self, img_path):
        img = image.load_img(img_path, target_size=(self.IMG_HEIGHT, self.IMG_WIDTH))
        return self.predict_img(img)

    def predict_img(self, pil_img):
        """ Returns predictions on a PIL image.
        """
        img = pil_img.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # Rescale to range [0, 1] as expected by MobileNet
        img_tensor /= 255.

        return self.model.predict(img_tensor)

