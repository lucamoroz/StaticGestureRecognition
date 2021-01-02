import argparse
import sys

from hand_cnn import HandCNN


def main(args=None):
    if not args:
        args = parse_args(sys.argv[1:])

    # Train locally
    hand_cnn = HandCNN(args.model_path)
    history = hand_cnn.train(args.dataset, batch_size=8, epochs=4, learning_rate=0.0001, checkpoints_callback=False)
    hand_cnn.save_model(path="models/", file_name="retrained.hdf5")
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


def parse_args(args):
    parser = argparse.ArgumentParser(description="Retrain predictions layer.")
    parser.add_argument("--model_path", help="path to the already trained model", required=True)
    parser.add_argument("--dataset", help="path to training dataset", required=True)

    return parser.parse_args(args)


if __name__ == '__main__':
    main()
