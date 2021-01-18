# StaticGestureRecognition
Project from Applied Deep Learning 2020 - TUWien

The project consists in a static hand gesture recognizer that maps gestures to user-defined commands.

NOTE: to see the deliverables for assignment 2, please refer to the branch "assignment2".

# Requirements

## Install dependencies
1. `virtualenv venv`. Note: the program was tested with python 3.7, you can choose an interpreter using for example `virtualenv -p=/usr/bin/python3.7`
2. `pip install requirements.txt`

## Get the model
1. Download released [model](https://github.com/lucamoroz/StaticGestureRecognition/releases/download/0.9/model_final.hdf5)
2. Copy model to: hand_classifier/models/model_final.hdf5

# Run
To run on the webcam:
`python main.py`

To run the application in debug mode (and see the video stream and prediction confidence) run `python main.py --debug`.

There are multiple options available, to see all of them run `python main.py --help`. 

# Change commands
You can change the commands executed by modifying the file commands.json, which associates a command to each gesture.

The commands are passed to the underlying system and executed.

The default commands use `xdotool key [keys]` that emulates pressing the keys.

# Test
From the project root folder, run:

`py.test`

# Train on your hands
There is a dedicated python file that can be used to retrain the classification layer to fir your dataset.

1. Collect your dataset. You can use the script `datset/data_script.py` to quickly add images to a datset, see `dataset/README.md` for more info.
2. Change the labels of `hand_classifier/HandCNN.LABELS` and `commands.json` according to your dataset classes.
3. Retrain the classification layer of the pretrained model: `python hand_classifier/retrain_top.py --dataset [PATH_TO_DATASET] --model [PATH_TO_TRAINED_MODEL]` 
