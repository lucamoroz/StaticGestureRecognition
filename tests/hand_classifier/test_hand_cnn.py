import warnings

warnings.simplefilter('ignore')
import pytest
import numpy as np
import keras
from hand_classifier.hand_cnn import HandCNN


@pytest.mark.parametrize("n_classes", [3, 6])
def test_model(n_classes):
    warnings.simplefilter('ignore')
    inputs = np.zeros((1, 224, 224, 3), dtype=np.float32)
    targets = np.zeros((1, n_classes), np.float32)

    model = HandCNN.get_model(n_classes, 224, 224)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=1e-5))

    model.fit(inputs, targets, batch_size=1)


@pytest.mark.parametrize("img_path", ["tests/hand_classifier/testdataset/fist/closeup1_0.jpg",
                                      "tests/hand_classifier/testdataset/spok/closeup1_0.jpg",
                                      "tests/hand_classifier/testdataset/palm/closeup1_0.jpg"])
def test_predictions(img_path):
    warnings.simplefilter('ignore')
    hand_cnn = HandCNN()
    hand_cnn.LABELS = ["fist", "palm", "pointer", "spok", "thumb_down", "thumb_up"]
    hand_cnn.train("tests/hand_classifier/testdataset/", batch_size=1, epochs=1, learning_rate=0.01,
                   checkpoints_callback=False)
    res = hand_cnn.predict_img_path(img_path)

    assert len(res[0]) == len(hand_cnn.LABELS)
    np.testing.assert_almost_equal(np.sum(res[0]), 1, 5)
