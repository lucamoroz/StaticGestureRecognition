import warnings
warnings.simplefilter('ignore')
from hand_classifier.hand_cnn import HandCNN


def test_train():
    warnings.simplefilter('ignore')
    hand_cnn = HandCNN()
    hand_cnn.train("tests/hand_classifier/testdataset/", batch_size=1, epochs=2, learning_rate=0.01, checkpoints=False)


