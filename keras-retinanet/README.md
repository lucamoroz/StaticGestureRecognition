floyd run --gpu --data lucamoro/datasets/ouhands/1:ouhands --env keras --follow "cd keras-retinanet && pip install . --user && pip install tensorflow==2.3.0 pip install keras==2.4 && python setup.py build_ext --inplace && python keras_retinanet/bin/train.py --steps 4 csv dataset/ouhands.csv dataset/classes.csv"


python keras_retinanet/bin/train.py --steps 50 --batch-size 32 --gpu 0 csv dataset/ouhands.csv dataset/classes.csv

Maybe env tensorflow 2.2
Define batch and steps