import warnings

warnings.simplefilter('ignore')
import pytest
import numpy as np
import keras
from hand_classifier.hand_cnn import HandCNN


@pytest.mark.parametrize("img_shape, target_shape", [((512, 512, 3), (224, 224, 3)), ((820, 430, 3), (96, 96, 3)), ((400, 800, 3), (114, 114, 3))])
def test_preprocessing(img_shape, target_shape):
    # Test size and normalization
    warnings.simplefilter('ignore')
    input_img = np.random.random_sample(img_shape) * 255

    preprocessed_img = HandCNN.preprocess_input(input_img, target_shape[0], target_shape[1])

    assert (np.asarray(preprocessed_img) < -1).sum() == 0, "preprocessed image contains values below 1"
    assert (np.asarray(preprocessed_img) > 1).sum() == 0, "preprocessed image contains values above 1"
    assert preprocessed_img.shape == target_shape, "preprocessed image doesn't have target shape"


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
