from hand_cnn import HandCNN
import tensorflow as tf


def main(floyd=True):
    if floyd:
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found, device name: ' + device_name)
        print('Found GPU at: {}'.format(device_name))

        with tf.device('GPU:0'):
            hand_cnn = HandCNN()
            history = hand_cnn.train(data_path="/floyd/input/handposes", epochs=15, batch_size=32, learning_rate=0.0001)
            hand_cnn.save_model()
            save_history_graphs(history)
    else:
        # Train locally
        hand_cnn = HandCNN()
        history = hand_cnn.train("tests/hand_classifier/testdataset/", batch_size=1, epochs=2, learning_rate=0.01)
        hand_cnn.save_model()
        save_history_graphs(history)


def save_history_graphs(history):
    import matplotlib.pyplot as plt

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
    main(False)
