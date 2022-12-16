import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = "logs/train_fully"
_C.MODEL = CN()
_C.MODEL.NAME = "deeplabv3+_resnet50"
_C.MODEL.NUM_CLASSES = 7
_C.MODEL.ATROUS = [6, 12, 18]
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.WEIGHTS = ""
_C.MODEL.FREEZE_BN = False

_C.INPUT = CN()
_C.INPUT.MULTI_SCALES = [0.5, 0.75, 1, 1.5, 2]
_C.INPUT.CROP_SIZE = (512, 512)
_C.INPUT.FLIP_PROB = 0.5
_C.INPUT.DSBN = 2
_C.INPUT.SDA = 3

_C.DATASETS = CN()
_C.DATASETS.IMGDIR = "/home/kc/luantt/kaggle_data/dataset-medium/image-chips"
_C.DATASETS.LBLDIR = "/home/kc/luantt/kaggle_data/dataset-medium/label-chips"
_C.DATASETS.TRAIN_LIST: "/home/kc/luantt/kaggle_data/dataset-medium/train.txt"
_C.DATASETS.VALID_LIST: "/home/kc/luantt/kaggle_data/dataset-medium/valid.txt"

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.STOP_ITER = 40000

_C.SOLVER.LR_METHOD = "poly"
_C.SOLVER.LR = 0.005
_C.SOLVER.POWER = 0.9

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4

_C.SOLVER.BATCH_SIZE = 8 

