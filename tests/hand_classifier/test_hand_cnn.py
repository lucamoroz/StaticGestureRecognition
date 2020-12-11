import warnings
warnings.simplefilter('ignore')
import pytest
import numpy as np
import keras
from hand_classifier.hand_cnn import HandCNN

parameters = []

for num_classes in [3, 6]:
    parameters.append((num_classes))


@pytest.mark.parametrize("n_classes", parameters)
def test_model(n_classes):
    warnings.simplefilter('ignore')
    inputs = np.zeros((1, 224, 224, 3), dtype=np.float32)
    targets = np.zeros((1, n_classes), np.float32)

    model = HandCNN.get_model(n_classes, 224, 224)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=1e-5))

    model.fit(inputs, targets, batch_size=1)


def test_predictions():
    warnings.simplefilter('ignore')
    hand_cnn = HandCNN()
    hand_cnn.train("tests/hand_classifier/testdataset/", batch_size=1, epochs=1, learning_rate=0.01, save=False)
    res = hand_cnn.predict_img_path("tests/hand_classifier/testdataset/fist/low_light1_0.jpg")

    assert len(res[0]) == len(hand_cnn.LABELS)
    np.testing.assert_almost_equal(np.sum(res[0]), 1, 5)
