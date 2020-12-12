import warnings
warnings.simplefilter('ignore')
import pytest
from hand_classifier.hand_cnn import HandCNN


@pytest.mark.parametrize("batch_size", [1, 2])
def test_train(batch_size):
    warnings.simplefilter('ignore')
    hand_cnn = HandCNN()
    hand_cnn.train("tests/hand_classifier/testdataset/", batch_size=batch_size, epochs=2, learning_rate=0.01, checkpoints_callback=False)


