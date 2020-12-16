# StaticGestureRecognition
Project from Applied Deep Learning 2020 - TUWien

The project consists in a static hand gesture recognizer (implemented) that maps gestures to user-defined commands (to implement).

# Requirements

## Install dependencies
1. `virtualenv venv`
2. `pip install requirements.txt`

## Get the model
1. Download released [model](https://github.com/lucamoroz/StaticGestureRecognition/releases/download/0.9/model_final.hdf5)
2. Copy model to: hand_classifier/models/model_final.hdf5

# Run
To run on the webcam:
`python main.py`

Check the console to see the predicted gesture, press q to exit.

# Test
From the project root folder, run:

`py.test`

To run tests using multiple workers use:
`py.test -n [N_WORKERS]`