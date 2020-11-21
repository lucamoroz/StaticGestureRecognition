pip install pandas pycocotools opencv-python requests scikit-image
python train.py --dataset csv --csv_train dataset/ouhands.csv --csv_classes dataset/classes.csv

floyd run --gpu --data lucamoro/datasets/ouhands/1:ouhands --env pytorch-1.5 --follow "cd pytorch-retinanet && pip install pandas pycocotools opencv-python requests scikit-image && python train.py --epochs 3 --dataset csv --csv_train dataset/ouhands.csv --csv_classes dataset/classes.csv"
